# Study overview

- Data source: NASA POWER hourly meteorology/radiation, 2015–2025.
- Fixed extraction points: 15.
- Direct split: training/scaler fitting 2015–2021; validation/configuration 2022–2023; independent test 2024–2025.
- Forecast horizons: H24 and H48.
- Common eligibility warm-up: 168 h.
- H24 independent-test windows: 257,775; target pairs: 6,186,600; daylight cases: 3,103,207.
- H48 independent-test windows: 257,415; target pairs: 12,355,920; daylight cases: 6,199,242.
- Repeated neural systems: seeds 1, 2, 3; summary variability is sample SD (ddof=1).
- Daylight: calculated solar elevation > 0.
