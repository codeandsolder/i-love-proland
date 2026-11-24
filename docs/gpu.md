# GPU scatter compaction (wgpu 30 / WGSL)

This module is an optional rendering adapter for the deterministic,
camera-independent i-love-proland core. Enable it with `--features gpu`. The
default crate has **zero runtime dependencies**.

## Lifetime and ownership

1. Choose the canonical surface cells within the active view or physics
   bubble via `Generator::tiles_covering_cells`.
2. Generate stable object instances with canonical land-cover/habitat filtering
   and apply `WorldEdits<T>` to include user-added and relocated objects.
3. Project these exact objects onto the currently visible terrain LOD.
   Accurate physics height can use a different terrain source; neither
   influences canonical object ID, variant, or saved changes.
4. Project positions into a stable local frame and build 112-byte
   `gpu::PackedInstance` records. The record stores the full source orientation quaternion and
   physical per-axis XYZ radii. A conservative sphere radius (the
   maximum axis radius) is derived for GPU visibility and LOD.
5. Encode `GpuCompactor` to cull spheres, choose mesh vs impostor by
   pixel footprint, and select the mesh tier using authored physical-anchor
   distances. Compact each retained object exactly once.
6. Bind the appropriate storage buffer to a host render vertex shader, use
   `@builtin(instance_index)` to fetch its record, then call
   `draw_indexed_indirect` with the matching GPU indirect argument buffer.
   All indexed draw fields except `instance_count` are supplied by the host
   at initialization.
7. Distant forest canopy and its lighting should be generated as **coverage
   on the terrain itself**. The compactor drops tiny per-object geometry;
   it does not create or modify the canonical object population.

### GPU record format

All fields are 16-byte aligned WGSL `vec4` and are explicitly serialized,
with `#[repr(C)]` + bytemuck POD in Rust.

| Offset | WGSL field | Meaning |
| ---: | --- | --- |
| 0 | `position_radius: vec4<f32>` | Camera-relative XYZ, scaled physical crown radius |
| 16 | `rotation: vec4<f32>` | Normalized XYZW quaternion; arbitrary orientation and terrain alignment |
| 32 | `axis_radii: vec4<f32>` | Physical XYZ half-axes in metres and appearance variation |
| 48 | `object_id: vec4<u32>` | Low-to-high limbs of the **entire** 128-bit stable ID |
| 64 | `metadata: vec4<u32>` | Low/high limbs of 64-bit variant and layer IDs |
| 80 | `tint: vec4<f32>` | Independent source-space per-object linear RGB and alpha multiplier, separate from material tint and identity |

Runtime uniforms: 16-byte input count/bin stride/mesh tier count/host-population flag,
16-byte projection thresholds, 16-byte normalized forward, 16-byte
normalized source bounds center, and six 16-byte frustum planes = 160 bytes. The frustum plane equation is
`dot(plane.xyz, camera_relative_position) + plane.w >= -radius`.
Zero planes are accepted as explicitly disabled planes for broad visibility
and reproducibility tests. `GpuView::from_projection` extracts normalized
planes directly from the actual column-major camera-relative clip matrix
using WebGPU's 0..w depth convention, including off-axis and orthographic
projections and a disabled infinite far plane. Culling uses the larger
of the animation-expanded physical sphere and the circumscribed sphere
of the impostor square (sqrt(2) times the physical radius), plus f32
plane-evaluation rounding allowance. Projected visual LOD uses the
original physical radius, not the inflated culling bound.

### LOD and buffer semantics

`GpuView::new` requires a positive focal length (in pixels), strictly ordered
positive mesh and impostor crown-diameter thresholds, normalized camera
forward, and six inward-facing normalized frustum planes.

The WGSL shader uses comparisons of projected diameter without division:

`(2 * radius * focal_pixels) >= (minimum_diameter_pixels * positive_depth)`

Cross-multiplication avoids rounding an exact 1-pixel diameter across the
boundary on some GPU drivers. This was discovered on a real AMD/RADV Vulkan
run and fixed before the GPU implementation was committed.

A validated `MeshLods` owns strictly increasing finite mesh range starts
(in metres) beginning at zero, with the first material's nonzero index
count per tier. Mesh selection uses the final start distance at or below
the squared physical-anchor distance; the compute shader performs a binary
search over those squared boundaries. `GpuView::with_source_center`
accepts the original source bounds center in normalized mesh coordinates.
It is multiplied by the packed physical axis radii and rotated by the
instance quaternion, then subtracted from the camera-relative sphere
center to recover the exact placement anchor. Off-axis objects therefore
use true Euclidean distance, not forward depth or source crown distance.

Each object is emitted to **at most one** of N+1 bins: one of the N
authored source mesh tiers or the single composite impostor. Indirect
draws and storage are shared contiguous buffers, with each bin's storage
offset aligned both to the record size and device storage-binding offset.
`compactor.bin(index)` exposes its `wgpu::BufferBinding`, indirect buffer
and byte offset. All indirect `first_instance` values are zero relative
to the independently bound slice. Only the instance count is atomically
updated; no object is copied to multiple bins. Choose mesh ranges
independently of the final projected-size impostor threshold. Mark a tier
`prefer_mesh: true` if the original mesh is already a cheap authored
billboard/low-cost representation preferable to a generated impostor.
The storage table contains a squared start distance and a preference flag
for each tier; the same classifier runs on CPU and GPU.

Below the impostor threshold, the ordinary intended output is the terrain
canopy appearance field, not a microscopic mesh. A host using a separately
defined world-space aggregate handoff may instead opt into
`GpuView::with_explicit_population()`: the supplied candidate population
already complements its aggregate field, so frustum-visible objects are
never discarded a second time based only on projected size. Very small
non-preferred meshes still choose the generated impostor. The compactor
does not generate the aggregate field or alter canonical existence.

GPU atomics allocate contiguous output slots in **nondeterministic order**.
Every output is a frame-local u32 candidate slot. The vertex shader uses
that slot to read the original immutable 112-byte candidate record, so ID,
variant and attributes are not duplicated or modified. Render slot order
is irrelevant to game state and saved worlds. The index buffer is not a
stable indexable database. Sort by IDs outside the GPU draw
phase if a particular external workflow requires a canonical order.

The compactor processes one compatible physical source family. A tier can
have multiple material draws; copy its GPU-computed count to each material
draw while keeping the material-specific index count. Separate source
families should use their own compactors. Call `set_candidates(queue, candidates)` whenever the population input
batch changes; it grows automatically and stores the new active count.
An empty replacement clears the active population. Call `encode(queue,
encoder, view)` once per compactor per submission, and consume the bins
in later passes in that submission. Candidate bytes stay resident across
arbitrary changes to classification/view uniforms, avoiding repeated
uploads. Replacing candidates and encoding two distinct frames into a
single unsubmitted encoder is not supported (queue writes precede it).

### Resource bounds

Let B be the number of mesh tiers plus one impostor bin, C the candidate
capacity, A the device storage buffer offset alignment and
S = align_up(C, A/gcd(A,4)) * 4 the byte stride across output index bins.
The allocation is bounded by the actual GPU storage buffer and dispatch
limits: one C*112-byte input, one B*S-byte candidate-index output buffer, B*20-byte
indirect records, N*8-byte immutable squared mesh boundaries and preferences and a
160-byte camera uniform. The shader uses four storage bindings
(input, output, indirect, thresholds) and one uniform. Slots inside a
bin never overlap another bin, even when all candidates select one tier.
Using u32 indices rather than copying 112-byte records reduces per-bin
logical storage and writes by 24×. The fixed worst-case per-bin allocation trades memory for a single
pass without per-bin prefix scans, extra dispatches or per-object CPU
work. Padding is never drawn. Empty frames reset all indirect counts
before returning.

Construct one `CompactionPipeline::new(device)` and pass its reference
to each `GpuCompactor::new(&pipeline, capacity, lods, impostor_count)`.
The pipeline and its device/layout are shared, while mutable buffers are
independent. Use distinct compactors for different material
batches. For moving aircraft, prefetch canonical cells ahead of the craft
and maintain the near physics bubble separately from camera-local visual
projection.

### Correctness test

```bash
cargo +stable test --locked --all-targets --all-features
cargo +stable run --locked --features gpu --example gpu_compaction
```

The Vulkan example constructs more than sixty thousand deterministic tree
and rock instances, reads back four mesh tiers plus the impostor bin, every
indexed indirect command, source GPU bytes, and every visible candidate
index resolved to its original record against the Rust reference in ID-keyed maps. The probe exercises full anisotropy,
rotation, non-neutral tint, nonzero source-center offset, frustum
rejection, and empty-dispatch resetting. This is stronger than count parity. It also
reports per-frame elapsed time including CPU uploads and GPU queue drain.
The example is intentionally independent of KSP source art.

For reproducible performance comparisons, use a release build and the same
GPU driver/power settings. The example's basic timing is an end-to-end latency
probe, not an isolated compute shader GPU timestamp benchmark.

## Integration and planned extensions

- Add canonical density/land-cover texture input if GPU habitat acceptance is
  desired. Preserve the CPU canonical rules and deterministic ID test vectors;
  **visibility filtering must not redefine persistence**.
- Build octahedral impostor atlases from source tree meshes; test leaf
  coverage, mipmapping, alpha-to-coverage and motion against original meshes.
- Add a terrain-attached forest canopy BRDF and optional clump height/parallax
  field for the entire visible region, without enumerating far individual
  trees.
- Render motion tests at 50–150m AGL and aerial scale references from
  1–100km around KSC, verifying actual pixel/silhouette coverage and temporal
  stability.

## Resident moving camera and distance handoff

The input pose is now in the caller's stable local frame rather than rewritten
relative to every eye. Set the camera-relative projection and eye on both the
classification view and draw camera. Each shader subtracts the eye before
frustum, distance and vertex transforms. Large absolute planetary coordinates
still require caller-side high-precision local projection.

For complementary source-range canopy, call
`GpuView::with_distance_handoff(begin, end)` and supply the physical placement
anchor and an identity-stable independent rank in [0,1) for each candidate
(`PackedInstance::with_handoff_anchor`). Classification rejects candidates
at or beyond end and uses the original smoothstep fraction in the final band.
Retained objects are not also culled by the projected canopy cutoff. The
anchor is independent of source-mesh sphere center or visual slot.

Actual AMD Renoir Vulkan: 63,333 input records; all candidate bytes and
compacted IDs/bins matched the CPU reference on three translated eyes with
distance handoff, without reuploading the population. The forest example's
480m overhead eye uses the same resident input; wind, billboard and
capacity-growth checks pass. Measured full upload+classify/sync 3.016ms,
resident classify/sync 0.844ms.


## Sparse candidate replacement

Keep a dense slot range `[0, count)` and use `CandidatePatch` to replace
individual records through `patch_candidates(queue, encoder, count, patches)`.
Patches must be sorted by slot and unique; all newly exposed slots must be
initialized. A shrink can use an empty patch list. Growth beyond capacity
uses `set_candidates` instead, which replaces the backing buffer and resets
patch storage. The patch dispatch must precede classification and drawing
in the same submitted encoder. Do not encode another candidate update
before submitting this encoder. Each patch occupies 128 bytes; use full
replacement whenever the patch payload would exceed the full input size.
The caller owns canonical ID-to-slot mapping; slot order is not semantic.

The GPU example checks actual storage bytes and classified IDs across
replacement, shrink, zero count and regrowth on AMD Vulkan.

## Staggered projected LOD transitions

A host can call `GpuView::with_projected_lod_spread(0.25)` to spread
each projected visual threshold across a deterministic ±25% band.
At a nominal 32px crossover, individual trees switch at 24–40px;
the near/mid threshold is varied by the same multiplier, preserving
tier order. The multiplier hashes all four canonical ID words, not
the GPU slot, model, camera frame or terrain handoff rank. It is
computed directly during CPU/GPU classification from the existing
instance record; each tree still contributes to exactly one draw bin.

Zero spread (default) retains the previous exact thresholds.
The terrain canopy cutoff, distance-based canopy ownership,
frustum tests, and authored distance tier boundaries are unaffected.
Staggering distributes population-level changes across distance and
zoom but does not crossfade an individual silhouette.
