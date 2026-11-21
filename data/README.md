# Source extract

The benchmark uses the [NYC Open Data catalog](https://data.cityofnewyork.us/d/43nn-pn8j) and the
nine-field Socrata query saved in [`reports/reference/snapshot.json`](../reports/reference/snapshot.json).
Run `python -m inspection_risk fetch` from the repository root. The cache and SHA-256 manifest are
stored under `data/raw/`, which is excluded from Git. Do not check in the downloaded file or
row-level inspection predictions. The reference query was captured on November 21, 2026 UTC.

The upstream source is mutable. Repeat the original benchmark with the exact cached CSV and
manifest, not a fresh download; verify it against the reference SHA-256. If the exact bytes are
unavailable, a fresh run is a new as-of study, not an exact reproduction. If the rolling dataset
has lost required dates or classes, use the archived input or define a new evaluation window.
