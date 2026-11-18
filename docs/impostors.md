# Instanced source meshes and multi-angle tree impostors

The optional `gpu` feature now includes two drawing pipelines that consume
the output of `GpuCompactor`. It is a real wgpu/Vulkan render path rather than
a CPU-side visibility simulation.

## Draw contract

- The source tree is a triangle mesh of `TreeVertex` records (position and
  RGBA color). Positions are normalized to the unit sphere around the object
  center. A distinct renderer/compactor is used for incompatible source
  materials or index buffers.
- `PackedInstance::position_radius.xyz` is the **camera-relative sphere
  center**. The fourth component is the physical radius, already inclusive of
  object scale. `attributes.x` gives stable yaw turns.
- Construct `TreeMesh`, `GpuCompactor::new(device, capacity,
  mesh.index_count(), 6)`, `AtlasImage::upload`, then `TreeRenderer::new`.
- Before each render pass, encode `compactor.encode()` into the **same
  command encoder**, and call `renderer.set_camera()` before submitting the
  command buffer. In a render pass using the configured color/depth formats,
  `renderer.draw()` issues two actual indexed-indirect draws. No instance
  readback, CPU draw count, or CPU render-visibility list is needed.
- The host owns its color/depth render attachments, pass load operations and
  source material grouping.

A static source tree with transparent leaf cards needs texture-aware source
shading. The current `TreeVertex` path intentionally supports arbitrary
**colored/alpha-cutout triangles** and the baker below. An importer for KSP's
original texture/UV/leaf-card meshes is a further integration task. No
original KSP/Parallax or Proland assets are shipped with this crate.

## Impostor atlas layout

Atlas tiles are row-major with **8 azimuth columns × 4 elevation rows**:

| Axis | Samples |
|---|---|
| Azimuth | 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° |
| Elevation | 0°, 30°, 60°, 90° (side to directly overhead) |

Every atlas image cell is `tile_size × tile_size`; `gutter` pixels on
each side are filled from *the same tile's* content. The texture sampler
is bilinear with no mipmaps. Fragment shading blends the two neighbouring
azimuths and two neighbouring elevations for smooth camera movement.
The quad faces the camera but remains dimensioned to the source bounding
sphere. Right/up axes are constructed from the view direction and world-up,
including a stable vertical fallback.

`atlas_bake::bake_colored_mesh()` is a deterministic, pure Rust reference
baker for arbitrary color-bearing triangulated meshes. It performs
orthographic rendering for 32 view directions, per-view depth testing,
alpha-cutout, 2× raster supersampling and alpha-weighted downsampling. The
host can supply any custom `AtlasImage`, including a texture-correct atlas
baked using the source engine's own material shader.

The 32-view atlas is an **appearance/LOD cache**, not persisted simulation
data. Atlas views can be recreated or replaced without changing canonical
object IDs.

## Reproduce the GPU picture

```bash
cargo +stable run --release --locked --features gpu --example forest_render -- \
  /home/jan/builds/i-love-proland/render/forest
magick /home/jan/builds/i-love-proland/render/forest-scene.ppm \
  /home/jan/builds/i-love-proland/render/forest-scene.png
magick /home/jan/builds/i-love-proland/render/forest-atlas.ppm \
  /home/jan/builds/i-love-proland/render/forest-atlas.png
```

The Vulkan example generates its own asymmetric broadleaf source mesh,
builds an area-seeded deterministic population and 32-angle atlas, and renders
a **1280×720, vertical 90° FOV** scene from the same GPU compacted near/mid
buffers. PPM images are recorded only in the chosen output folder. The
reference includes a separate sky/ground pass to make silhouettes legible.
Inspect alpha boundaries, dark edge fringes, view changes and top-down tree
crowns before selecting optimal per-LOD limits.

## Constraints and follow-up

- Nearest image is one mip level; a production atlas needs alpha-coverage-
  preserving mipmap generation, guard borders at each mip and likely
  alpha-to-coverage or temporally stable foliage coverage.
- A multi-material source mesh should use appropriate material batches and
  preserve its exact UV/alpha masks. The current sample tree is solid colored
  geometry, not actual KSP foliage.
- The quad geometry is camera-facing; a dense forest needs proper
  transparency/occlusion, preferably MSAA alpha-to-coverage and vegetation-
  specific horizon blending before switching to the terrain canopy BRDF.
- Below the GPU impostor pixel threshold, objects remain canonical but are
  represented by a **future terrain-attached density/canopy shader**. No
  large-range individual-tree enumeration is intended.
- GPU indirect append order is intentionally arbitrary. Never persist
  instance slots; persist canonical object IDs and sparse mutations.

Further GPU compaction ABI and buffer details:
[docs/gpu.md](gpu.md).
