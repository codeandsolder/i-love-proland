# Terrain amplification laboratory

Research-only comparison of five elevation fields derived from the same stock
Kerbin MapSO region. No change to the scatter or GPU library. This example
requires a locally extracted 4096×2048, one-byte stock MapSO. Do not commit
the input or imagery derived from it.

Run on stable Rust with the existing compiler cache:
```sh
cargo +stable test --locked --example terrain_amplification --target-dir /path/to/build/target
cargo +stable run --locked --release --example terrain_amplification --target-dir /path/to/build/target -- /path/to/KerbinHeight.map /path/to/build
```

The example chooses a mountainous inland region deterministically, starting
with a 65², 1 km-spaced grid. The 1025² output covers 64 km at 62.5 m
spacing. PPMs use identical elevation/slope colors and illumination, not
different texture or lighting settings. Timings exclude analysis and output.

- **coarse**: clamped cubic interpolation.
- **noise**: seeded gradient noise at four successive refinement scales.
- **erosion**: same noise, six steps of depression-aware multiple-flow
  accumulation, empirical stream-power incision and talus relaxation per
  scale. No sediment deposition simulation.
- **directed**: compact-support slope-oriented wave packets, inspired by
  [Grenier et al. (2024)](https://doi.org/10.1111/cgf.14992), independently
  evaluated on each refinement grid.
- **hybrid**: erosion at the first two scales, directed waves at the last two.

This is a planar regional experiment, not a canonical spherical surface.
Neither the gradient orientation nor the erosion drainage is independent
of neighboring data. Boundary taper is for the comparison, not proof of
global seam continuity. Initial experiments do not establish stock PQS
parity, physical hydrology, or CPU/GPU equivalence. A future library needs
a canonical body-fixed point field with global drainage constraints and
explicit hierarchy; render and scatter must query that same surface.
