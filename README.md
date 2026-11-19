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

### GPU visibility and mesh/impostor compaction (opt-in)

The optional `gpu` feature enables a standalone **wgpu 30** compute pipeline.
The CPU generator still owns object existence, stable 128-bit IDs, habitat
decisions, model variants and saved-world changes. After the host projects
those objects onto the current terrain LOD and converts their positions to
camera-relative floats, `gpu::PackedInstance::from_instance` produces a fixed
64-byte shader record. It retains all four object-ID words, two model-ID words
and two layer-ID words through GPU filtering.

`gpu::GpuCompactor` accepts an arbitrary candidate batch and a camera
frustum, then dispatches a WGSL compute pass that culls bounding spheres,
chooses detailed mesh vs impostor by **projected canopy diameter in pixels**,
and atomically compacts survivors into two separate storage buffers. Each has
its own GPU-written indexed-indirect draw arguments; the host can bind
`mesh_instances()/mesh_indirect()` or
`impostor_instances()/impostor_indirect()` directly in its render passes.

Very small trees are handled by the future aggregate canopy shading path;
they are excluded from mesh/impostor lists without deleting canonical world
objects. The GPU's parallel append order is deliberately unspecified: neither
save files nor gameplay should depend on draw-buffer ordering. Group calls by
mesh/material batch when using different source geometries; variant IDs
survive in the GPU record for shader variation and batching.

```bash
cargo +stable test --locked --all-features --all-targets
cargo +stable run --release --features gpu --example gpu_compaction
```

The headless example uses a real Vulkan adapter, copies both GPU result
buffers back, and compares **every 128-bit ID and complete instance record**
against the CPU reference classifier, independent of GPU workgroup ordering.
It also reports end-to-end upload, compaction and GPU-synchronization timing.
A Vulkan-capable device is required for that example; CPU-only users never
enable wgpu. See the [GPU architecture, ABI and integration guide](docs/gpu.md)
for shader bindings, indirect-draw layouts, reproducibility boundaries and
resource limits.

### Actual near-tree meshes and 32-angle impostor draws

The GPU feature also provides `impostor::TreeMesh` and `TreeRenderer`, which
consume the two compacted instance buffers and **indexed-indirect draw counts**
inside an existing wgpu render pass. It draws multiple independently textured original source-material meshes nearby
with **one GPU visibility-compaction pass** and **one** middle-distance
camera-facing impostor draw. Each near-field material retains its own
alpha-cutoff/UVs and indexed mesh; shared near instance counts are copied
GPU-to-GPU into material-specific indirect arguments. The fixed
**8 azimuth × 4 elevation** atlas supports overhead aircraft views and blends
adjacent view angles without changing canonical object identities.

A generic Rust reference baker (`atlas_bake::bake_scene`) generates one 32-view
atlas from **multiple original UV-textured, alpha-masked source materials**
(e.g., separately textured trunk and leaves with shared depth). The
single-material `bake_mesh` convenience function uses the same pipeline.
Both operate on source triangles with bilinear sRGB
source sampling, per-supersample alpha cutout, depth testing, 2× supersampling,
linear-light resolve and independently padded tiles. A headless Vulkan example creates a deterministic
forest and actually renders both stages at **1280×720, 90° FOV**:

```bash
cargo +stable run --release --locked --features gpu --example forest_render -- /path/to/output-prefix
```

The example also accepts host-decoded raw RGBA8 texture data, including
textures decoded from the original Parallax DDS assets. Production hosts
can bind existing compressed BC3/BC7 source texture views directly,
without duplicating the source image into an RGBA GPU allocation. Source polygons are
still synthetic; integration of transformed KSP `.mu` meshes and their
separate materials, production coverage-preserving mip chains and the distant
terrain canopy are follow-up tasks. Single-sample rendering uses stable ordered
alpha coverage, while multisample targets can use hardware alpha-to-coverage. See [impostor and rendering documentation](docs/impostors.md)
for the GPU drawing contract and atlas convention.

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

- GPU-side *canonical world generation*, habitat texture filtering, model
  geometry selection or impostor atlas baking (the current GPU feature covers
  frustum/LOD filtering, compaction and indirect draw arguments);
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
