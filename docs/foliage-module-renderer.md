# Source-preserving procedural foliage modules

Proland now accepts immutable [Lorax](https://github.com/codeandsolder/lorax-rs)
branch and canopy module tiers as GPU-resident visual geometry, sharing the
existing tree family compactor, indirect draws, source wind and angular
impostor handoff. The modules are an additional *geometry source*, not a new
scatter system or an automatic replacement for the current KSP assets.

## Data contract

- Lorax owns the immutable source plant, tapering stems and spatial canopy
  group membership. Create each tier using Plant::module_tier with ScreenLod.
  Original source foliage IDs are kept for later texture/coverage baking.
- Proland procedural::compile_modules collects independently generated
  source-variant forms, normalizes them to **one explicit shared source
  frame**, and pads shorter tiers with inert records. Multiple seeded forms
  can share one persistent u64 source variant ID. The dense index order is
  explicit; use CompiledModuleBank::model_slot_for_seed(source_id, stable_seed)
  to select a visual form using persistent object/biome context.
  Source colors and geometric bounds survive. Each selected visual slot has
  a separate conservative radius in \`CompiledModuleBank.variant_culling_radii\`,
  measured about the **shared** source-frame center; the host applies that
  radius to its GPU culling/LOD sphere without modifying source-frame
  normalization, physical transform, source anchor, or canonical object ID.
  A shared maximum culling expansion would incorrectly make tiny shrubs
  project as large mature trees, causing avoidable near/middle geometry.
- Upload to TreeModules once, not each frame. Each source-space descriptor is
  64 bytes: a typed tapered stem or spatial canopy ellipsoid plus source ID
  and original membership count. GPU expansion emits two crossed cards per
  module, with a stable original-ID azimuth and source vertex wind field.
- PackedInstance now has render_model, a local visual-model slot **separate
  from canonical metadata variant_id** and 128-bit stable object ID. Assign
  it with PackedInstance::with_render_model. TreeRenderer validates all
  full-batch and sparse-patch slots against the canonical variant catalog.
  The packed instance ABI is now 128 bytes, up from 112.
- TreeMaterial contains TreeGeometry::Mesh or TreeGeometry::Modules. This is
  an intentional API change, not a compatibility wrapper. Normal source
  materials continue to draw exactly as before; the module variant uses its
  own pipeline layout with a third read-only storage/stride binding. The
  original two-group mesh and impostor pipeline layouts remain unchanged.
- Module shader respects the original per-instance quaternion/anisotropic
  radii, source tint, source alpha textures and optional existing wind frame.
  The canopy cards have an ellipse coverage envelope with anti-aliasing and
  multisample alpha-to-coverage, or stable dither when single sampled.

## Projected-diameter LOD selection

Procedural foliage can choose mesh tiers by each tree's actual screen
footprint rather than the authored metre boundaries. Construct the usual
MeshLods tiers, then explicitly select
MeshLods::with_projected_diameters(vec![80.0, 24.0, 0.0]):
near at >=80 px, clustered leaves at 24-80 px, sparse cards below 24 px.
The GPU smoke's source-module tiers use 64/24/1 spatial foliage groups:
near and mid retain a branch hierarchy, while the far representation
keeps the original root stem and a single conservative canopy aggregate.
This limits sub-24px crowns to two crossed silhouettes plus trunk rather
than submitting ~40+ triangles for every distant tiny tree.

These are experiment thresholds, not a calibrated art acceptance. Both
CPU and GPU compare projected crown times depth against threshold times
depth without a division; source identities stay unchanged.

Authored distance LOD remains the default. Per-tier prefer_mesh continues
to control the existing mesh/impostor crossover. Subpixel population and
terrain canopy handoff remain independent.

## Test with the real GPU compactor

Use the existing headless Proland scene with an *optional* module pass:

~~~sh
cargo +nightly run --locked -j 2 --features gpu --example forest_render \
  --target-dir /home/jan/build/i-love-proland-modules-20261123/target -- \
  /home/jan/build/i-love-proland-modules-20261123/capture/forest --module-smoke
~~~

The normal source renderer and all its wind/capacity tests run first. A
separate capture then draws Lorax-generated 64-, 24-, and 8-cluster tiers
from the same resident instance batch, using the standard GPU compactor and
indirect draws, comparing against an empty frame. It asserts a nontrivial
rasterized difference, checks invalid visual-model index rejection, and
writes forest-modules.ppm. The current source atlas used for fallback distant
impostors is a synthetic test texture, **not** a matching new species atlas;
this smoke test checks rendering, not visual equivalence across LOD handoff.

Two pipeline layouts are deliberately required: sharing the expanded
three-binding-group layout with ordinary source meshes makes wgpu reject a
regular source draw with an unset third bind group.

## Performance / art status

The prior [four-way GPU benchmark](foliage-lab.md) demonstrated competitive
GPU-generated and shared-module approaches at 8/24/64 foliage clumps.
These richer 64-byte branch/canopy modules are *not* the prior experiment's
32-byte proxy patches; do not directly substitute its memory or GPU times
for this renderer without fresh end-to-end benchmarks.

Do not enable this automatically for KSP until a material and wind-matched
reference source is selected, source frame/radius and conservative culling
expansion are verified against all variants, and near/mid/far appearance,
temporal fade, TAA, shadows and terrain canopy handoff are benchmarked in
a real KSP scene. Production distance LOD bins do not yet implement the
benchmark's complementary crossfade. Retain existing authored source trees
until that visual acceptance.

## Per-form angular impostor atlas (seed-coherent LOD)

The angular impostor atlas must correspond to the selected visual model.
A single authored source atlas cannot depict three distinct source plants:
using it after a module LOD switch changes the canopy silhouette, color and
species regardless of correct instance persistence.

For module families, compile the required near/mid/far `CompiledModuleBank`
tiers in the same dense source-form order. Bake the tier adjoining the
impostor handoff using `CompiledModuleBank::bake_variant_atlases` and the
same `SourceImage` used in the GPU near module material. The source's tapered
branch quads, original-ID-stable two-sided foliage cards, colors, normalized
geometry, and radial canopy coverage are rasterized in 8x4 source-relative
view directions. Upload the results using
`AtlasImage::upload_variants(device, queue, &[(canonical_id, &image), ...])`.
Preserve catalog order even when a canonical ID occurs more than once due to
distinct procedural seeds. The `TreeRenderer` checks that the atlas catalog
matches the module model bank.

The GPU atlas texture stores 32 independently filtered layers per visual
model and selects its view group from `PackedInstance.render_model`.
Ordinary single-model authored source atlases retain their original 32 layers.
No per-frame atlas generation, device readback, camera-dependent model
reseed or change to the original stable world ID is necessary.

The CPU atlas is an approximation of the shaded GPU module: its alpha mask
is baked at a fixed source-sampling resolution, then area filtered per view,
whereas GPU near foliage uses projected-derivative edges and sample coverage.
Matching silhouette at the actual KSP LOD crossover remains a visual
acceptance gate; the source must have enough normalization radius to
conservatively contain all baked card corners.
