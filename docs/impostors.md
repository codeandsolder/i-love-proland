# Instanced source meshes and multi-angle tree impostors

The optional `gpu` feature now includes two drawing pipelines that consume
the output of `GpuCompactor`. It is a real wgpu/Vulkan render path rather than
a CPU-side visibility simulation.

## Draw contract

- Source geometry consists of `TreeVertex` records (position, linear RGBA
  tint and source UV). Positions are normalized to a unit sphere around the
  object center. Textures are `SourceImage` records containing row-major
  sRGB RGBA8 bytes decoded by the host from its own PNG/DDS pipeline.
  A renderer owns its compactor for an entire material family.
- `PackedInstance::position_radius.xyz` is the **camera-relative sphere
  center**, with a conservative maximum-axis radius in the fourth component.
  `rotation` carries the full normalized source quaternion in XYZW order;
  `axis_radii.xyz` holds the physical scaled source half-axes. The near-mesh
  vertex shader applies per-axis scale **before** source rotation, matching
  general KSP/Parallax affine placements including nonuniform scale and tilt.
  The impostor shader evaluates source-local azimuth/elevation through the
  inverse quaternion and projects conservative view-dependent half-extents.
- Construct one device-scoped `CompactionPipeline` and one
  `TreePipelines` per compatible render-target configuration. All
  families share these actual compute/render pipelines, bind layouts,
  samplers, placeholder texture and quad index buffer. Construct each
  unique immutable `TreeMesh` once (cloning shares GPU buffers).
  Upload the combined `AtlasImage` and source albedo views, then create
  `TreeRenderer::new(&pipelines, capacity, lods, sources, &atlas)` with
  `sources: Vec<Vec<TreeMaterial>>`, one material vector per mesh tier.
  All tiers must have nonempty source materials and their first index
  count must match the corresponding compactor metadata. The family owns an independent
  compactor, material uniforms, source texture bindings and indirect counts;
  immutable resource sharing never merges family populations.
- Before each render pass, call `renderer.reserve(candidate_count)`
  if needed, then `renderer.encode(queue, encoder, candidates, view)`.
  The renderer owns its compactor and resizes just candidate/visible buffers
  and per-tier dynamic bind groups; all immutable mesh/material and pipeline
  resources survive capacity changes.
  Each material receives the count from its own compacted tier, keeping
  its source index count. One compacted index binding is shared by all materials in a tier. Vertex
  shaders dereference those frame-local u32 slots into the single original
  candidate buffer. No copied 96-byte records are needed per bin. Call `renderer.set_camera(queue, camera)` before
  submission; inside the pass `renderer.draw(pass)` emits every tier's
  source materials and **one** composite 32-view impostor. One compute
  dispatch classifies all candidate objects regardless of mesh-tier or
  material counts; no CPU count readback is necessary.

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
each side are filled from *the same tile's* content. GPU upload crops
those duplicated gutters into 32 independent texture-array layers.
Each mip is the exact area average of the original view content in
linear premultiplied RGBA, including odd dimensions. Trilinear
sampling cannot leak adjacent view colors or distort transparent
coverage. Fragment shading blends the two neighbouring azimuths and two
neighbouring elevations in premultiplied space, then unpremultiplies
before alpha-to-coverage or ordered dithering.
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
  original compressed mip chain. The generated angular atlas has a complete
  isolated, area-averaged premultiplied mip chain. The far canopy shader is
  separate.
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

## Authored texture-driven wind

Mesh materials can supply a MeshWind (original wind image view, validated
WindParameters, unnormalized source bounds center Y and radius). A
TreeCamera with a WindFrame activates animation; omitting the frame
keeps the source geometry unchanged. Each frame contains the camera's f64
position in a stable Cartesian field, an orthonormal render-to-field basis,
and elapsed seconds. Periodic texture origins and time are reduced in f64
before converting to GPU floats. Source height is reconstructed before
instance scaling from the normalized mesh vertex.

The vertex shader smooths each repeating planar texture sample with four
half-texel taps, blends the three projections by the absolute radial normal
components, and applies height-powered RGB displacement along the local
up and projected field direction. Animation is render-only. GpuView can
supply the maximum displacement in metres (see WindParameters::max_displacement)
to pad the visibility sphere without changing distance or projected-size
LOD thresholds. The caller supplies the original texture's color-space
interpretation and source-space height bounds. Composite impostors and
terrain canopy remain static; wind is a source-mesh vertex deformation,
not an identity or terrain edit.

Real Vulkan regression: set PROLAND_TEST_WIND=1 for the forest render
example. It renders a deterministic varying RGB fixture at two animation
times and repeats the first time; motion must exceed repeated-frame noise.

## Authored source billboards

A source mesh (often an original far LOD) can supply a Billboard descriptor
to TreeMaterial and BakeSurface, separately from the composite impostor.
The descriptor contains the original bounding-sphere center divided by
the radius, and the original radius in source units. TreeCamera::new now
requires an explicit unit camera-forward vector in the render frame.
The source vertex is restored relative to its placement anchor, then
transformed using the original cylindrical rule: local up remains Y,
local back is the camera-back direction inverse-rotated and multiplied
componentwise by the original instance XYZ scales, and local right is
the normalized cross product of back and Y. Back is intentionally not
normalized. Apply the original instance scale and rotation after this
vertex transform. At the vertical singularity the real-time shader uses
a deterministic local X fallback; the atlas uses analytic azimuth.
The wind height uses the transformed source Y, as in the source shader.
The caller must include the source billboard's motion in its visibility
padding; this is not another placement or a duplicate canonical ID.

The CPU baker also transforms authored billboard surfaces for each atlas
view at unit source scale, resolving inter-material occlusion in the same
depth buffer. It validates that all the transformed vertices fit the
normalized sphere, rather than clipping an offset card. The actual
nonuniform instance geometry remains more accurate than the static atlas.
The albedo-only Proland renderer does not distinguish lighting normal
variants of a billboard. Set PROLAND_TEST_BILLBOARD=1 for the Vulkan
forest example's rotated, nonuniformly scaled source-card regression.
