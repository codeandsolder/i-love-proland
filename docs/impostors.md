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
  center**, with a conservative maximum-axis radius in the fourth component.
  `rotation` carries the full normalized source quaternion in XYZW order;
  `axis_radii.xyz` holds the physical scaled source half-axes. The near-mesh
  vertex shader applies per-axis scale **before** source rotation, matching
  general KSP/Parallax affine placements including nonuniform scale and tilt.
  The impostor shader evaluates source-local azimuth/elevation through the
  inverse quaternion and projects conservative view-dependent half-extents.
- Construct one `TreeMesh` for each source material, a single
  `GpuCompactor::new(device, capacity, first_mesh.index_count(), 6)`, the
  combined `AtlasImage::upload`, and source albedo views (either
  `SourceImage::upload` or original GPU-compressed DDS views). Create a
  `TreeRenderer::new(device, &compactor, vec![TreeMaterial {
  mesh, source_view, alpha_cutoff }, ...], &atlas,
  TreeRenderConfig { color_format, depth_format, sample_count })`.
  The renderer takes ownership of the individual GPU meshes.
- Before each render pass, call `compactor.encode(queue, encoder, ...)`,
  then `renderer.encode_mesh_draws(encoder, &compactor)` in the **same
  encoder**. The latter copies the computed mesh instance count into every
  material's own indexed-indirect command, preserving its source index count.
  Call `renderer.set_camera(queue, camera)` before submission. Inside a
  matching render pass, `renderer.draw(pass, &compactor)` draws all original
  near-field materials and exactly **one** combined 32-view impostor.
  Neither mesh material count nor candidate density multiplies visibility
  classification; no CPU instance-count readback is required.
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
The quad faces the camera; its dimensions are derived from the projected
source ellipsoid, conservatively covering nonuniform XYZ scale. Its right/up
axes follow the source orientation, including a stable overhead fallback.

`atlas_bake::bake_scene()` accepts a slice of `BakeSurface` records containing
original UV-textured triangles, source texture, and material-specific alpha
cutoff. All surfaces share one per-supersample depth buffer so leaves, branches
and trunks occlude each other correctly even through alpha cutouts.
`bake_mesh()` is a single-material convenience wrapper. The baker uses 32 orthographic view
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

- `SourceImage::upload` creates one uncompressed RGBA8 mip level for the
  reference/example; a host-provided compressed GPU texture view retains its
  original compressed mip chain. The generated angular atlas currently has
  only one mip level; production needs per-level coverage preservation and
  isolated atlas tile borders at every mip. The far canopy shader is separate.
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

## HDR material vertex tint

Source mesh `TreeVertex.color` carries floating linear RGB multipliers,
not normalized RGBA8 vertex colors. RGB above 1.0 is legitimate in
original asset materials (e.g., Kerbin grass `_Color` 1.3/1.4/1.0).
Alpha remains fractional coverage in [0,1]. The near-field shader
multiplies this source material tint before per-instance terrain tint.
The CPU atlas baker also multiplies in linear space before final
sRGB RGBA8 quantization. Do not pre-clamp source material RGB, which
changes muted texture colors even when the eventual framebuffer is
limited to [0,1].
