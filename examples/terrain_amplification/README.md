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

## Biome-guided landscape and exemplar experiments

Pass an additional 4096×2048 tightly packed RGB8 image of the native Kerbin
biome map (its original palette, south-to-north rows, matching MapSO).
The generator decodes the classes into rainfall, rock/uplift, aridity,
temperature and vegetation priors before spatial smoothing. The following
modes compare the earlier geology experiments and the newer exemplar method:

```sh
cargo +stable run --locked --release --example terrain_amplification --target-dir /path/to/build/target -- /path/to/KerbinHeight.map /path/to/build/landscape /path/to/biome.rgb
cargo +stable run --locked --release --example terrain_amplification --target-dir /path/to/build/target -- /path/to/KerbinHeight.map /path/to/build/exemplars /path/to/biome.rgb /path/to/alps-513-i16le.raw
cargo +stable run --locked --release --example terrain_amplification --target-dir /path/to/build/target -- /path/to/KerbinHeight.map /path/to/build/natural /path/to/biome.rgb /path/to/alps-513-i16le.raw natural
```

The optional DEM is exactly 513×513 signed little-endian 16-bit elevations
in metres, at 62.5 m spacing and with south-to-north rows. The reference
used in this experiment is Copernicus DEM GLO-30, tile N46 E007, reprojected
to EPSG:32632 and centered at longitude 7.62°, latitude 46.43°.
Average-resample to the target grid, round, then flip the raster rows.
Source and attribution terms:
<https://copernicus-dem-30m.s3.amazonaws.com/readme.html>.
Reference raw SHA-256:
`a9b0d44926eda8d537276b42b417492a3c5a6aed382eb03418e1e06d81c12beb`.
This is a digital surface model, not strictly bare-earth terrain. Neither
the source data nor derived atlas is included in the repository.

The current preferred visual experiment is `gentle-natural`: softened
biome-guided base relief, erosion, deterministic overlapping broad
terrain residuals selected by local geometry, protected river corridors,
and a depression-depth constraint. The separate `natural` variant
includes coupled uplift and incision. Outputs include shaded plan views,
perspective reference renders, albedo, land-cover masks, hydrology masks,
and river-segment CSVs. The common Kerbin camera is selected from the
native source, not optimized for each result.

This remains a regional research example. The final depression correction
can lift individual samples by hundreds of metres; future fitting should
respect drainage constraints during synthesis instead. River widths in
the preview are raster coverage, not an in-game water mesh. Both test
regions overlap, and neither establishes spherical or LOD continuity.
