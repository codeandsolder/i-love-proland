# Instanced source meshes and multi-angle tree impostors

The optional `gpu` feature now includes two drawing pipelines that consume
the output of `GpuCompactor`. It is a real wgpu/Vulkan render path rather than
a CPU-side visibility simulation.

## Draw contract

- Source geometry consists of `TreeVertex` records (position, linear RGBA
  tint and source UV). Positions are normalized to a unit sphere around the
  object center. Textures are `SourceImage` records containing row-major
  sRGB RGBA8 bytes decoded by the host from its own PNG/DDS pipeline.
  A distinct renderer/compactor is used per source material/index buffer.
- `PackedInstance::position_radius.xyz` is the **camera-relative sphere
  center**. The fourth component is the physical radius, already inclusive of
  object scale. `attributes.x` gives stable yaw turns.
- Construct `TreeMesh`, `GpuCompactor::new(device, capacity,
  mesh.index_count(), 6)`, `AtlasImage::upload`, `SourceImage::upload`, then
  `TreeRenderer::new(device, compactor, mesh, atlas, source,
  TreeRenderConfig { color_format, depth_format, sample_count, alpha_cutoff })`.
- Before each render pass, encode `compactor.encode()` into the **same
  command encoder**, and call `renderer.set_camera()` before submitting the
  command buffer. In a render pass using the configured color/depth formats,
  `renderer.draw()` issues two actual indexed-indirect draws. No instance
  readback, CPU draw count, or CPU render-visibility list is needed.
- The host owns its color/depth render attachments, pass load operations and
  source material grouping.

The near-mesh fragment shader samples the original sRGB source albedo using
bilinear **repeat-wrapped UVs**, combines it with the per-vertex tint, and
rejects fragments below the configured alpha cutoff. This preserves the
transparent holes within leaf cards. The source image and CPU reference
baker use the same UV origin (first RGBA row) and cutoff semantics.

The impostor shader interpolates neighboring views and preserves fractional
alpha coverage. For a single-sample target it uses stable 4×4 ordered
coverage dithering with ordinary hardware depth writes, avoiding unsorted
alpha blending. On a multisample render target, `sample_count > 1` enables
**alpha-to-coverage** in the impostor pipeline and uses fragment alpha as
hardware sample coverage. The host owns a matching sample-count color/depth
pass and resolve attachments. No original KSP/Parallax or Proland assets are
shipped with this crate.

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

`atlas_bake::bake_mesh()` is a deterministic Rust reference baker for
**UV-textured triangles and per-vertex tints**, with 32 orthographic view
cameras, bilinear source sampling and sRGB-to-linear decoding. It performs
per-supersample alpha cutout, per-view depth testing, 2× raster
supersampling, premultiplied linear-light resolve, alpha coverage preservation
and sRGB encoding. The host may supply its own `AtlasImage` for materials
requiring special GPU shaders (wind animation, normal maps, etc.).

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

## Real source texture validation

The example accepts an optional host-decoded raw RGBA8 texture, without
adding DDS or PNG dependencies to the engine-independent crate:

```bash
magick /path/to/oakLeaves.dds -depth 8 \
  RGBA:/home/jan/builds/i-love-proland/textured-vegetation/oak-leaves.rgba
cargo +stable run --release --locked --features gpu --example forest_render -- \
  /home/jan/builds/i-love-proland/textured-vegetation/oak-source \
  /home/jan/builds/i-love-proland/textured-vegetation/oak-leaves.rgba 1024 1024
```

This exercises **the actual source texture**, including its DXT5-derived
transparent leaf cutouts, over the sample's synthetic asymmetric tree geometry.
The example remains independent of KSP `.mu` scene-node decoding; an engine
adapter should pass original transformed source positions, mesh UVs and
separate trunk/leaf material groups. Never treat the synthetic tree image as
a screenshot of the original `.mu` asset.

## Constraints and follow-up

- Each uploaded source and angular atlas texture currently has one mip level.
  Production needs per-level alpha-coverage preservation and isolated atlas
  tile borders at every mip; the distant terrain canopy shader is separate.
- Multi-material source geometry must be grouped by source texture and alpha
  cutoff. The sample test tree is synthetic geometry, even when its texture
  data is supplied from the user's actual Parallax assets.
- The quad geometry is camera-facing with either stable screen-space coverage
  dithering (1 sample) or MSAA alpha-to-coverage. Evaluate temporal stability,
  vegetation occlusion and horizon blending before transitioning to the
  terrain canopy BRDF.
- Below the GPU impostor pixel threshold, objects remain canonical but are
  represented by a **future terrain-attached density/canopy shader**. No
  large-range individual-tree enumeration is intended.
- GPU indirect append order is intentionally arbitrary. Never persist
  instance slots; persist canonical object IDs and sparse mutations.

Further GPU compaction ABI and buffer details:
[docs/gpu.md](gpu.md).
