# i-love-proland

Deterministic, topology-aware object scatter in Rust. Inspired by the ideas in
[INRIA Proland](https://gitlab.inria.fr/neyret/proland), implemented independently
without importing its C++, GLSL, renderer, textures, assets, or runtime.

Proland's original authors deserve credit for its procedural planetary landscape
and vegetation research. The official project relicensed to BSD-3-Clause in
[commit 65cfba5 (2015-08-23)](https://gitlab.inria.fr/neyret/proland/-/commit/65cfba5).
This is a separate, new Rust project: **not** an official Proland port,
a complete Proland replacement, or affiliated with INRIA.

## Available now

- **Stable canonical identities:** world seed + independent object-layer ID +
  fixed global cell + candidate serial. Objects exist independent of camera,
  render-tile partition, draw order, concurrent workers or terrain LOD.
- **Two topologies:** an infinite XY plane (including negative coordinates)
  and a six-face sphere. The spherical generator uses Jacobian-aware rejection,
  yielding uniform expected population per square metre across all six faces.
- **Two placement policies:** a true Poisson point process with arbitrary
  positive density, and a jittered one-per-cell stratified distribution for
  evenly spaced large objects. The latter rejects unrepresentable densities
  explicitly rather than silently dropping instances.
- **Any object type:** each layer holds stable numeric variant IDs and
  positive weights. Weighted selection is independent of variant array order.
  Trees, grass, rocks, coral, buildings, debris or anything else use the exact
  same API. Each layer has an independent random stream, so adding rocks
  does not relocate trees.
- **Composable habitat masks:** deterministic callback returning a fractional
  acceptance probability (biome, slope, elevation, land-cover, site-mask,
  game-state exclusion). Applying a stricter mask removes a subset of stable
  candidate IDs instead of moving the surviving objects.
- **Explicit resource budgets** with errors rather than partial silent output.
- **No dependencies** and no reliance on a global random generator.

### Quick start

```bash
cargo +stable test --all-targets
cargo +stable run --example groves
```

```rust
use i_love_proland::{
    Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let generator = Generator::new(
        Domain::Planar { cell_size_m: 8.0 },
        GeneratorConfig {
            world_seed: 2026,
            cells_per_tile: 16,
            ..GeneratorConfig::default()
        },
    )?;
    let oaks = Layer::new(
        0x4f414b, // forever-stable kind key
        0.012,    // expected oak count per m² before biome masking
        Distribution::Poisson,
        vec![
            Variant { id: 1001, weight: 3.0 },
            Variant { id: 1002, weight: 1.0 },
        ],
    )?;
    for tree in generator.tile_with_habitat(
        TileKey::planar(-1, 2),
        &oaks,
        |candidate| {
            // Replace with a *canonical*, camera-independent biome mask.
            if candidate.cell.y > 0 { 0.8 } else { 0.0 }
        },
    )? {
        println!("{:032x} {:?} model={}",
            tree.candidate.id, tree.candidate.anchor, tree.variant_id);
    }
    Ok(())
}
```

### View-dependent streaming without changing world placement

The renderer can request only the canonical cells covered by a camera/terrain
patch. Queries are bounded and lazy, support negative coordinates and work
independently of render-tile size. For example:

```rust
let rect = generator.planar_cell_rect([-150.0, -100.0], [150.0, 100.0])?;
for tile in generator.tiles_covering_cells(None, rect, 256)? {
    let visible = generator.tile_with_habitat_in_cells(
        tile, rect, &oaks, |_| 1.0,
    )?;
    // Supply visible anchors to your renderer or streaming cache.
}
```

The explicit tile limit prevents accidentally enumerating an entire planet.
For spherical worlds, build a canonical CellRect on each intersected cube face
and call tiles_covering_cells(Some(face), rect, limit). The same cell seeds
are used regardless of which view or render tile requested them.

```bash
cargo +stable run --locked --example streaming
```

### Sparse saved-world edits

`WorldEdits<T>` stores only **changes**, never the procedural world. It is
generic over the application's model/transform payload and keeps generated
IDs separate from user-added IDs. Removed objects are suppressed by ID;
modified objects can remain at their original cell or move to another.
Moved and added objects are indexed by their current canonical cell, so
loading a visible region does not require loading the original tile.

Applications can serialize the sorted records returned by
`generated_changes()` and `added_objects()` with their preferred format,
together with `LAYOUT_VERSION`, the world seed and layer metadata.
`WorldEdits::from_records` reconstructs the spatial index on load, rejects
duplicate IDs and validates source-cell/layer consistency when applying
generated edits. No scene-size-sized save files are required.

```bash
cargo +stable run --locked --example edits
```

### Planetary surfaces

```rust
use i_love_proland::{CubeFace, Domain, Generator, GeneratorConfig, TileKey};

let sphere = Generator::new(
    Domain::Spherical { radius_m: 600_000.0, face_cells: 1 << 18 },
    GeneratorConfig { world_seed: 1234, cells_per_tile: 16, ..Default::default() },
).unwrap();
let tile = TileKey::spherical(CubeFace::PositiveZ, 1234, 5678);
```

A sphere anchor contains a body-fixed unit direction, **not a height**. To
render the object, project that direction onto the currently displayed terrain
triangle. To simulate collisions, project the *same* anchor onto authoritative
terrain. Its identity and model choice stay the same in both systems.

**The biome/habitat filter itself must not use transient render LOD** if an
object is expected to persist: use a fixed canonical land-cover/PQS mask.
Render projection, orientation, visual mesh LOD and collision detail can
change independently of object identity.

## Stability and limitations

The reproducibility algorithm is named `LAYOUT_VERSION = 1`. Saved worlds
should store this version, world seed, domain/cell spacing, per-layer IDs and
densities, variant IDs/weights, and canonical habitat definitions. Altering
these values can intentionally regenerate positions; changing
`cells_per_tile` alone does **not**.

This first release is the **portable placement core**, not the complete forest
renderer. In particular it does not yet contain:

- GPU/WGSL compute generation, indirect draws, forest mesh LOD or impostor
  atlas baking;
- the Proland-style distance-dependent terrain canopy BRDF;
- a built-in save-file serializer or direct integration with KSP's `planetmap`;
  the mutation record schema and streaming queries are already provided;
- an exact global Poisson-disk hard-core rejection between arbitrary
  neighbourhood candidates (stratified mode is a practical, not mathematically
  identical, alternative).

These rendering and engine-specific cache layers will compose around the
canonical IDs rather than own them. No source assets, C++ or GL dependencies
are part of this crate.

## Development

```bash
cargo +stable fmt --all -- --check
cargo +stable clippy --all-targets -- -D warnings
cargo +stable test --all-targets
```

All stateful behaviour is isolated to caller-owned caches; generation is
synchronous and pure. It is safe to distribute independent tiles among worker
threads or machines. The same seeds and configuration produce the same
objects without any inter-worker coordination.

## License

MIT OR Apache-2.0 for the new Rust implementation. Proland is an independent
INRIA project; its official code and data have their own licensing and are
not bundled here.
