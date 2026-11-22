// i-love-proland: renderer-independent GPU instance classification/compaction.
// Canonical existence, IDs, variants, and mutation overlays are resolved on the
// CPU. GPU atomics affect *storage order only*, never the object population.

struct PackedInstance {
    position_radius: vec4<f32>,    // local-frame xyz, physical crown radius
    rotation: vec4<f32>,           // normalized source quaternion XYZW
    axis_radii: vec4<f32>,         // physical XYZ scale, appearance variation
    object_id: vec4<u32>,          // full stable u128, low word first
    metadata: vec4<u32>,           // variant u64 + layer u64
    tint: vec4<f32>,               // per-instance linear RGB, alpha multiplier
    handoff_anchor_rank: vec4<f32>, // physical anchor xyz and stable handoff rank
}

struct IndexedDraw {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct View {
    counts: vec4<u32>,             // input count, aligned bin stride (u32 indices), mesh bin count, host-owned population flag
    lod: vec4<f32>,                // focal pixels, mesh threshold, impostor threshold, animation displacement bound
    forward: vec4<f32>,            // normalized camera forward
    source_center: vec4<f32>,      // normalized original mesh bounds centre
    planes: array<vec4<f32>, 6>,   // xyz normal, w offset; camera-relative
    eye: vec4<f32>,               // eye in resident local frame
    handoff_range: vec4<f32>,     // begin/end, zero pair disables
}

@group(0) @binding(0) var<storage, read> input: array<PackedInstance>;
@group(0) @binding(1) var<storage, read_write> visible: array<u32>;
@group(0) @binding(2) var<storage, read_write> draws: array<IndexedDraw>;
@group(0) @binding(3) var<storage, read> mesh_lods: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> camera: View;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@compute @workgroup_size(64)
fn classify_and_compact(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_id = global_id.x;
    if input_id >= camera.counts.x {
        return;
    }

    let instance = input[input_id];
    if camera.handoff_range.y > camera.handoff_range.x {
        let distance = length(instance.handoff_anchor_rank.xyz - camera.eye.xyz);
        if distance >= camera.handoff_range.y {
            return;
        }
        if distance > camera.handoff_range.x {
            let t = (distance - camera.handoff_range.x) / (camera.handoff_range.y - camera.handoff_range.x);
            let fraction = t * t * fma(-2.0, t, 3.0);
            if instance.handoff_anchor_rank.w < fraction {
                return;
            }
        }
    }
    let position = instance.position_radius.xyz - camera.eye.xyz;
    let radius = instance.position_radius.w;
    // Mesh sphere plus animation, or the circumscribed sphere of a
    // conservative impostor square. Keep visual LOD diameter unchanged.
    let visibility_radius = max(radius + camera.lod.w, radius * sqrt(2.0));

    // Conservatively keep every sphere intersecting the camera frustum.
    for (var p: u32 = 0u; p < 6u; p += 1u) {
        let plane = camera.planes[p];
        // Eight f32 epsilons conservatively cover normalized-plane
        // evaluation error near a tangent, including large offsets.
        let margin = (dot(abs(plane.xyz), abs(position)) + abs(plane.w) + visibility_radius) * (8.0 * 1.1920928955078125e-7);
        if dot(plane.xyz, position) + plane.w < -visibility_radius - margin {
            return;
        }
    }

    let depth = dot(position, camera.forward.xyz);
    if depth + visibility_radius < 0.0 {
        return;
    }
    // Cross-multiply the projected diameter test. This avoids a costly
    // division and preserves the exact threshold at 1px, where some Vulkan
    // implementations otherwise round a quotient across the comparison.
    let projected_crown = (2.0 * radius) * camera.lod.x;
    let safe_depth = max(depth, 0.001);
    // With a host-owned distance handoff, the submitted population
    // already complements aggregate coverage. Do not drop it twice.
    if camera.counts.w == 0u && projected_crown < camera.lod.z * safe_depth {
        return;
    }
    let anchor = position - rotate(instance.rotation, camera.source_center.xyz * instance.axis_radii.xyz);
    let distance_squared = dot(anchor, anchor);
    // Upper bound search for the final inclusive authored start range.
    var low = 0u;
    var high = camera.counts.z;
    while low + 1u < high {
        let middle = low + (high - low) / 2u;
        if mesh_lods[middle].x <= distance_squared {
            low = middle;
        } else {
            high = middle;
        }
    }
    // Cheap original cards can be preferable to generated impostors,
    // independently of the family's projected-size crossover.
    var bin = camera.counts.z;
    if mesh_lods[low].y > 0.5 || projected_crown >= camera.lod.y * safe_depth {
        bin = low;
    }
    let slot = atomicAdd(&draws[bin].instance_count, 1u);
    visible[bin * camera.counts.y + slot] = input_id;
}
