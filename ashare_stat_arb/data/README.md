# Local A-Share Data

Do not commit licensed or large market data to Git.

This directory is reserved for local point-in-time A-share inputs. The free
BaoStock download command writes a compressed panel, a JSON manifest, and a
Parquet request cache here. The empirical dataset must satisfy the fields and
audits in `../RESEARCH_SPEC.md`. Only this README is tracked; local data files
are ignored.
