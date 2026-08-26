#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <openssl/sha.h>
#include <secp256k1.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kTriggerBytes = 256;
constexpr std::size_t kPayloadBytes = 192;
constexpr std::size_t kWireBytes = 768;

struct Trigger {
    std::uint64_t iteration;
    std::uint64_t amount;
    std::array<std::uint8_t, 32> nonce;
    std::array<std::uint8_t, 16> marker;
    std::array<std::uint8_t,
               kTriggerBytes - 2 * sizeof(std::uint64_t) - 32 - 16> padding;
};
static_assert(sizeof(Trigger) == kTriggerBytes);

struct WireFrame {
    std::uint32_t length;
    std::array<char, kWireBytes - sizeof(std::uint32_t)> bytes;
};
static_assert(sizeof(WireFrame) == kWireBytes);

struct TcpPair {
    int client = -1;
    int server = -1;

    ~TcpPair() {
        if (client >= 0) close(client);
        if (server >= 0) close(server);
    }

    TcpPair() = default;
    TcpPair(const TcpPair &) = delete;
    TcpPair &operator=(const TcpPair &) = delete;

    TcpPair(TcpPair &&other) noexcept
        : client(std::exchange(other.client, -1)),
          server(std::exchange(other.server, -1)) {}

    TcpPair &operator=(TcpPair &&other) noexcept {
        if (this == &other) return *this;
        if (client >= 0) close(client);
        if (server >= 0) close(server);
        client = std::exchange(other.client, -1);
        server = std::exchange(other.server, -1);
        return *this;
    }
};

struct Sample {
    std::size_t iteration;
    double parse_us;
    double build_hash_us;
    double sign_us;
    double serialize_us;
    double send_us;
    double rho_send_call_us;
    double rho_send_done_us;
    double rho_sink_us;
};

std::uint64_t now_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        throw std::runtime_error("clock_gettime failed");
    }
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

double elapsed_us(std::uint64_t start, std::uint64_t end) {
    return static_cast<double>(end - start) / 1000.0;
}

void read_full(int fd, void *buffer, std::size_t length) {
    auto *cursor = static_cast<std::uint8_t *>(buffer);
    while (length > 0) {
        const ssize_t count = recv(fd, cursor, length, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("socket read failed");
        cursor += count;
        length -= static_cast<std::size_t>(count);
    }
}

void write_full(int fd, const void *buffer, std::size_t length) {
    const auto *cursor = static_cast<const std::uint8_t *>(buffer);
    while (length > 0) {
        const ssize_t count = send(fd, cursor, length, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("socket write failed");
        cursor += count;
        length -= static_cast<std::size_t>(count);
    }
}

TcpPair make_loopback_pair() {
    TcpPair pair;
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) throw std::runtime_error("listener socket failed");

    try {
        int enabled = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
            listen(listener, 1) != 0) {
            throw std::runtime_error("loopback listener setup failed");
        }

        socklen_t address_length = sizeof(address);
        if (getsockname(listener, reinterpret_cast<sockaddr *>(&address),
                        &address_length) != 0) {
            throw std::runtime_error("getsockname failed");
        }

        pair.client = socket(AF_INET, SOCK_STREAM, 0);
        if (pair.client < 0 ||
            connect(pair.client, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) != 0) {
            throw std::runtime_error("loopback connect failed");
        }
        pair.server = accept(listener, nullptr, nullptr);
        if (pair.server < 0) throw std::runtime_error("loopback accept failed");

        setsockopt(pair.client, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
        setsockopt(pair.server, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
        close(listener);
        return pair;
    } catch (...) {
        close(listener);
        throw;
    }
}

std::uint64_t xorshift64(std::uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void fill_trigger(Trigger &trigger, std::size_t iteration, std::uint64_t &state) {
    trigger = {};
    trigger.iteration = iteration;
    trigger.amount = xorshift64(state);
    static constexpr std::array<std::uint8_t, 16> marker = {
        'A', 'M', 'B', 'U', 'S', 'H', '-', 'T', 'A', 'R', 'G', 'E', 'T', '-', '0', '1'};
    trigger.marker = marker;
    for (std::size_t offset = 0; offset < trigger.nonce.size(); offset += 8) {
        const std::uint64_t value = xorshift64(state);
        std::memcpy(trigger.nonce.data() + offset, &value, sizeof(value));
    }
}

std::size_t hex_encode(const std::uint8_t *input, std::size_t input_length,
                       char *output, std::size_t output_capacity) {
    static constexpr char alphabet[] = "0123456789abcdef";
    if (output_capacity < input_length * 2) {
        throw std::runtime_error("hex output buffer is too small");
    }
    for (std::size_t index = 0; index < input_length; ++index) {
        output[2 * index] = alphabet[input[index] >> 4];
        output[2 * index + 1] = alphabet[input[index] & 0x0f];
    }
    return input_length * 2;
}

void build_payload(const Trigger &trigger,
                   std::array<std::uint8_t, kPayloadBytes> &payload) {
    payload.fill(0);
    static constexpr std::array<std::uint8_t, 12> tag = {
        'F', 'R', 'O', 'N', 'T', 'R', 'U', 'N', '-', 'F', '-', '1'};
    std::copy(tag.begin(), tag.end(), payload.begin());
    std::memcpy(payload.data() + 16, trigger.nonce.data(), trigger.nonce.size());
    std::memcpy(payload.data() + 48, &trigger.amount, sizeof(trigger.amount));
    std::memcpy(payload.data() + 56, &trigger.iteration, sizeof(trigger.iteration));
    for (std::size_t index = 64; index < payload.size(); ++index) {
        payload[index] = static_cast<std::uint8_t>(
            trigger.nonce[index % trigger.nonce.size()] ^
            static_cast<std::uint8_t>(index + trigger.amount));
    }
}

WireFrame make_prebuilt_frame() {
    WireFrame frame{};
    const std::string request =
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_sendRawTransaction\","
        "\"params\":[\"0x02f86b0180843b9aca008252089400000000000000000000000000000000"
        "000000008080c001a0feedfacefeedfacefeedfacefeedfacefeedfacefeedface"
        "feedfacefeedfacea0cafebabecafebabecafebabecafebabecafebabecafebabe"
        "cafebabecafebabe\"]]}";
    frame.length = static_cast<std::uint32_t>(request.size());
    std::memcpy(frame.bytes.data(), request.data(), request.size());
    return frame;
}

void serialize_request(const std::array<std::uint8_t, kPayloadBytes> &payload,
                       const std::array<std::uint8_t, 64> *signature,
                       WireFrame &frame) {
    static constexpr char prefix[] =
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_sendRawTransaction\","
        "\"params\":[\"0x";
    static constexpr char suffix[] = "\"]]}";

    frame = {};
    std::size_t cursor = 0;
    const std::size_t prefix_length = sizeof(prefix) - 1;
    std::memcpy(frame.bytes.data() + cursor, prefix, prefix_length);
    cursor += prefix_length;
    cursor += hex_encode(payload.data(), payload.size(),
                         frame.bytes.data() + cursor, frame.bytes.size() - cursor);
    if (signature != nullptr) {
        cursor += hex_encode(signature->data(), signature->size(),
                             frame.bytes.data() + cursor,
                             frame.bytes.size() - cursor);
    }
    const std::size_t suffix_length = sizeof(suffix) - 1;
    if (cursor + suffix_length > frame.bytes.size()) {
        throw std::runtime_error("wire frame is too small");
    }
    std::memcpy(frame.bytes.data() + cursor, suffix, suffix_length);
    cursor += suffix_length;
    frame.length = static_cast<std::uint32_t>(cursor);
}

void write_csv(const std::string &path, const std::string &mode,
               const std::vector<Sample> &samples) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create output CSV");
    output << "iteration,mode,parse_us,build_hash_us,sign_us,serialize_us,send_us,"
              "rho_send_call_us,rho_send_done_us,rho_sink_us\n";
    output << std::fixed << std::setprecision(3);
    for (const auto &sample : samples) {
        output << sample.iteration << ',' << mode << ',' << sample.parse_us << ','
               << sample.build_hash_us << ',' << sample.sign_us << ','
               << sample.serialize_us << ',' << sample.send_us << ','
               << sample.rho_send_call_us << ',' << sample.rho_send_done_us << ','
               << sample.rho_sink_us << '\n';
    }
}

struct Options {
    std::string mode = "signed";
    std::string output = "rho.csv";
    std::size_t iterations = 50000;
    std::size_t warmup = 5000;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) throw std::runtime_error("missing option value");
        const std::string value = argv[++index];
        if (argument == "--mode") options.mode = value;
        else if (argument == "--output") options.output = value;
        else if (argument == "--iterations") options.iterations = std::stoull(value);
        else if (argument == "--warmup") options.warmup = std::stoull(value);
        else throw std::runtime_error("unknown option: " + argument);
    }
    if (options.mode != "prebuilt" && options.mode != "build" &&
        options.mode != "signed") {
        throw std::runtime_error("mode must be prebuilt, build, or signed");
    }
    return options;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        TcpPair feed = make_loopback_pair();
        TcpPair submission = make_loopback_pair();
        const std::size_t total = options.warmup + options.iterations;
        std::vector<Sample> samples;
        samples.reserve(options.iterations);
        std::exception_ptr bot_error;
        std::exception_ptr sink_error;

        std::thread sink([&] {
            try {
                WireFrame frame{};
                for (std::size_t index = 0; index < total; ++index) {
                    read_full(submission.server, &frame, sizeof(frame));
                    const std::uint64_t received_ns = now_ns();
                    write_full(submission.server, &received_ns, sizeof(received_ns));
                }
            } catch (...) {
                sink_error = std::current_exception();
            }
        });

        std::thread bot([&] {
            try {
                using ContextPtr = std::unique_ptr<secp256k1_context,
                                                   decltype(&secp256k1_context_destroy)>;
                ContextPtr context(
                    secp256k1_context_create(SECP256K1_CONTEXT_SIGN),
                    &secp256k1_context_destroy);
                std::array<std::uint8_t, 32> secret_key{};
                secret_key.back() = 1;
                std::array<unsigned char, 32> randomization{};
                randomization.front() = 7;
                if (!context ||
                    !secp256k1_context_randomize(context.get(), randomization.data()) ||
                    !secp256k1_ec_seckey_verify(context.get(), secret_key.data())) {
                    throw std::runtime_error("secp256k1 setup failed");
                }

                const WireFrame prebuilt = make_prebuilt_frame();
                static constexpr std::array<std::uint8_t, 16> expected_marker = {
                    'A', 'M', 'B', 'U', 'S', 'H', '-', 'T', 'A', 'R', 'G', 'E', 'T', '-', '0', '1'};

                for (std::size_t index = 0; index < total; ++index) {
                    Trigger trigger{};
                    read_full(feed.server, &trigger, sizeof(trigger));
                    const std::uint64_t visible_ns = now_ns();

                    volatile std::uint8_t decision = 0;
                    for (std::size_t byte = 0; byte < trigger.marker.size(); ++byte) {
                        decision |= trigger.marker[byte] ^ expected_marker[byte];
                    }
                    if (decision != 0) throw std::runtime_error("unexpected trigger");
                    const std::uint64_t parsed_ns = now_ns();

                    WireFrame frame{};
                    std::array<std::uint8_t, kPayloadBytes> payload{};
                    std::array<std::uint8_t, 32> digest{};
                    std::array<std::uint8_t, 64> compact_signature{};
                    std::uint64_t built_ns = parsed_ns;
                    std::uint64_t signed_ns = parsed_ns;
                    std::uint64_t serialized_ns = parsed_ns;

                    if (options.mode == "prebuilt") {
                        frame = prebuilt;
                    } else {
                        build_payload(trigger, payload);
                        SHA256(payload.data(), payload.size(), digest.data());
                        built_ns = now_ns();
                        signed_ns = built_ns;
                        if (options.mode == "signed") {
                            secp256k1_ecdsa_signature signature{};
                            if (!secp256k1_ecdsa_sign(
                                    context.get(), &signature, digest.data(),
                                    secret_key.data(), secp256k1_nonce_function_rfc6979,
                                    nullptr)) {
                                throw std::runtime_error("secp256k1 signing failed");
                            }
                            secp256k1_ecdsa_signature_serialize_compact(
                                context.get(), compact_signature.data(), &signature);
                            signed_ns = now_ns();
                            serialize_request(payload, &compact_signature, frame);
                        } else {
                            serialize_request(payload, nullptr, frame);
                        }
                        serialized_ns = now_ns();
                    }

                    const std::uint64_t send_call_ns = now_ns();
                    write_full(submission.client, &frame, sizeof(frame));
                    const std::uint64_t send_done_ns = now_ns();
                    std::uint64_t sink_ns = 0;
                    read_full(submission.client, &sink_ns, sizeof(sink_ns));

                    if (index >= options.warmup) {
                        samples.push_back(Sample{
                            index - options.warmup,
                            elapsed_us(visible_ns, parsed_ns),
                            elapsed_us(parsed_ns, built_ns),
                            elapsed_us(built_ns, signed_ns),
                            elapsed_us(signed_ns, serialized_ns),
                            elapsed_us(send_call_ns, send_done_ns),
                            elapsed_us(visible_ns, send_call_ns),
                            elapsed_us(visible_ns, send_done_ns),
                            elapsed_us(visible_ns, sink_ns),
                        });
                    }
                    const std::uint8_t acknowledgement = 1;
                    write_full(feed.server, &acknowledgement,
                               sizeof(acknowledgement));
                }
            } catch (...) {
                bot_error = std::current_exception();
            }
        });

        std::uint64_t random_state = 0x9e3779b97f4a7c15ULL;
        for (std::size_t index = 0; index < total; ++index) {
            Trigger trigger{};
            fill_trigger(trigger, index, random_state);
            write_full(feed.client, &trigger, sizeof(trigger));
            std::uint8_t acknowledgement = 0;
            read_full(feed.client, &acknowledgement, sizeof(acknowledgement));
            if (acknowledgement != 1) {
                throw std::runtime_error("invalid bot acknowledgement");
            }
        }

        bot.join();
        sink.join();
        if (bot_error) std::rethrow_exception(bot_error);
        if (sink_error) std::rethrow_exception(sink_error);
        write_csv(options.output, options.mode, samples);
        std::cout << options.mode << ": wrote " << samples.size()
                  << " samples to " << options.output << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "rho_benchmark: " << error.what() << '\n';
        return 1;
    }
}
