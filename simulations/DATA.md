# Network inputs

The simulation inputs are included. Experiments read the local files below.

| Input | Contents | Historical/execution split |
| --- | --- | --- |
| `data-ping/aws-individual_pings.csv` | RTT samples across 21 AWS location profiles | Alternating samples, converted to RTT/2 |
| `data-ping/wp-hourly-2020-07-19-20.npz` | 48 hourly directed mean-delay matrices for 220 WonderProxy locations, with an imputation mask | Even/odd hours; values already converted to RTT/2 |
| `data-ping/wp-servers-2020-07-19.csv` | WonderProxy location coordinates and labels | — |
| `data-owd/esnet-29node-history.json` | Directed OWAMP histograms | November 16, 2026, 04:50–05:00 UTC |
| `data-owd/esnet-29node-snapshot.json` | Directed OWAMP histograms | November 17, 2026, 04:50–05:00 UTC |
| `data-regional/layouts.json` | 20 fixed deployments per AWS/WP dataset, region and reporter identities | Fixed before test execution |

The WP archive marks 5,550 of 2,323,200 path-hour slots as imputed. These slots
use the nearest available hourly measurement on the same directed path.

ESnet data source: **ESnet Stardust — perfSONAR Endpoint Pair Explorer**:
<https://dashboard.stardust.es.net/d/kzuX1fD4k/perfsonar-endpoint-pair-explorer>.
Measurements were retrieved on November 17, 2026, through its public read-only
data-source endpoint:

```text
https://dashboard.stardust.es.net/api/datasources/proxy/uid/edi4g4kcotd6oa/stardust_perfsonar*/_search
```

The two `*-query.json` files contain the Elasticsearch query bodies. The GET
parameters were `source=<JSON query>` and
`source_content_type=application/json`. Each file stores the latest returned
histogram per directed link within its query window: 812 off-diagonal links
among 29 endpoints. Links are not filled using RTT or shortest paths. The
queries filter for IPv4 and successful `latencybg` tests; the stored records
use `powstream`. Histogram values are in milliseconds, with packet counts as
weights. Record timestamps and `result.max_clock_error` are retained.

The ESnet experiment samples each message independently from these histograms,
weighted by packet counts. It does not replay cross-link temporal correlation.

The fixed regional layouts have group sizes 5/4/4/4/4, one Byzantine identity
per group, and one measured location profile per group. Replica identities
remain distinct. The `low`/`high` fields parameterize Byzantine-owned receipt
claims; the regional ordering rule takes no historical interval or learned
jitter parameter as input.
