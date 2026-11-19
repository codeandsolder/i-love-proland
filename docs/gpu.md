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
4. Rebase positions to the active camera and build 80-byte
   `gpu::PackedInstance` records. The record stores the full source orientation quaternion and
   physical per-axis XYZ radii. A conservative sphere radius (the
   maximum axis radius) is derived for GPU visibility and LOD.
5. Encode `GpuCompactor` to cull spheres, classify near meshes vs middle
   impostors by pixel footprint, and compact instances into storage buffers.
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

Runtime uniforms: 16-byte count header, 16-byte projection thresholds,
16-byte normalized forward, and six 16-byte inward-facing frustum planes =
144 bytes. The frustum plane equation is
`dot(plane.xyz, camera_relative_position) + plane.w >= -radius`.
Zero planes are accepted as explicitly disabled planes for broad visibility
and reproducibility tests.

### LOD and buffer semantics

`GpuView::new` requires a positive focal length (in pixels), strictly ordered
positive mesh and impostor crown-diameter thresholds, normalized camera
forward, and six inward-facing normalized frustum planes.

The WGSL shader uses comparisons of projected diameter without division:

`(2 * radius * focal_pixels) >= (minimum_diameter_pixels * positive_depth)`

Cross-multiplication avoids rounding an exact 1-pixel diameter across the
boundary on some GPU drivers. This was discovered on a real AMD/RADV Vulkan
run and fixed before the GPU implementation was committed.

Each object is emitted to **at most one** of two outputs:

- mesh instances with the first indexed-indirect draw record;
- octahedral/crossed-billboard impostor instances with the second.

Below the impostor threshold, the intended output is the terrain canopy
appearance field, not a microscopic mesh. The compactor does not bake
impostors, provide mesh/material shaders, or generate that canopy field.

GPU atomics allocate contiguous output slots in **nondeterministic order**.
All records retain their ID, variant, and attributes bit-for-bit, so render
storage order is irrelevant to game state and saved worlds. The instance
buffer is not a stable indexable database. Sort by IDs outside the GPU draw
phase if a particular external workflow requires a canonical order.

The compactor processes one compatible geometry/material batch, with its
own index count for the near mesh and the impostor. Do not blend unrelated
mesh geometries into the same indirect draw unless the host vertex renderer
knows how to decode them. Call `encode` once per compactor per submission
and consume the resulting buffers in a subsequent render pass in that
submission.

### Resource bounds

Allocation checks device storage-buffer and dispatch workgroup limits.
The instance capacity is fixed at construction. The pass has no per-object
GPU→CPU synchronization and resets both indirect counts through queue writes
before dispatch. Buffer memory is approximately
`3 * 80 * candidate_capacity + 2 * 20 + 144` bytes, excluding driver
allocation granularity and render-side geometry.

Use distinct compactors or a higher-level pool for different material
batches. For moving aircraft, prefetch canonical cells ahead of the craft
and maintain the near physics bubble separately from camera-local visual
projection.

### Correctness test

```bash
cargo +stable test --locked --all-targets --all-features
cargo +stable run --locked --features gpu --example gpu_compaction
```

The Vulkan example constructs more than sixty thousand deterministic tree
and rock instances, reads back the **whole** mesh/impostor buffers and both
indexed indirect commands, then compares every packed record against the
Rust reference in ID-keyed maps. This is stronger than count parity. It also
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
