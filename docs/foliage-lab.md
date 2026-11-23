# GPU foliage morphology, geometry-reuse and LOD experiment

An **independent renderer benchmark**, not a new KSP art asset or renderer
replacement. Compares geometry implementations at identical proxy triangle
budgets, mid-LOD transitions, and headless Vulkan captures. Source:
examples/foliage_lab.rs and examples/foliage_lab.wgsl.

Use the managed Cargo/sccache toolchain:

~~~sh
cargo +nightly run --locked -j 2 --features gpu --example foliage_lab \
  --target-dir /home/jan/build/i-love-proland-foliage-lab-20261123/target \
  -- /home/jan/build/i-love-proland-foliage-lab-20261123/results
~~~

Results: timings.csv, four 960×600 PPM images and six transition PPM
captures plus transitions.csv.

## Methods

Four modes use the same triangle counts, view/lighting, colors, placements
and fragment shader, with 1,024 and 4,096 trees and 8/24/64 foliage clumps:

- gpu_unique: generate vertices in the vertex shader from unique per-tree seed.
  No source-variant geometry buffers.
- gpu_24_variants: generate geometry from 24 species/seed variants in the vertex
  shader. Shares the exact source structure with the two following modes.
- shared_modules: fetch a compact, 32-byte per-card descriptor per vertex,
  reconstruct each foliage card in the vertex shader.
- expanded_24_variants: fetch six expanded 32-byte vertices for each card.

Three species morphologies: broadleaf, conifer and shrub; eight variants
per type. Every tree instance uses 48 bytes. Each mesh has two crossed
foliage cards per clump plus one trunk card. Shared module bytes for 24
variants: 24 * (2 * clumps + 1) * 32; expanded vertices occupy 6× that.
The last three methods should produce nearly identical images (floating-point
roundoff aside). The unique-seed mode deliberately has more distinct crowns.

GPU render-pass timestamp measurements are recorded as minimum, median and
p90; if GPU timestamp queries are unsupported, the fallback uses CPU elapsed
time including submission and waits and should **not** be compared as GPU
timing. The executable is a geometry experiment: it currently uses procedural
alpha card shapes rather than real leaves, PBR bark, shadowing, or volumetric
foliage. It does not alter the live Proland renderer.

## Three-tier morphology hierarchy

Eight deterministic parent clumps are preserved across LODs; child clumps
split around each source parent rather than regenerate unrelated locations.
Based on CPU projected-silhouette sweeps at intermediate and holdout camera
views (results and scripts documented in Notion):

| Tier | Children per parent | Radius factor multiplied by base/sqrt(children) |
|---|---:|---:|
| 8 clumps | 1 | 1.00 |
| 24 clumps | 3 | 1.45 |
| 64 clumps | 8 | 2.20 |

Across broadleaf/conifer/shrub held-out seeds, mean silhouette IoU was
approximately 0.84–0.87 for 8→24 and 0.89–0.91 for 24→64, at 128×128
rasterization. These are **CPU projected silhouettes, not rendered KSP
screenshots**. The same hierarchy is reconstructed in CPU variant modules
and WGSL generation; image comparisons test equivalence.

Near and far LODs are both rendered during the transition, with a
complementary per-object/pixel deterministic alpha hash to preserve average
screen coverage. An actual renderer integration must combine this with TAA,
wind, per-tree identity, projected-size selection, shadowing, and existing
far-canopy handoff. Impostors may outperform even the eight-clump model at
the distant end; foliage area and apparent density matter more than
polygon-count reduction alone.

See the technical [Notion foliage lab](https://app.notion.com/p/421f74be902081a689fcced26594a4bc).
