// i-love-proland: renderer-independent GPU instance classification/compaction.
// Canonical existence, IDs, variants, and mutation overlays are resolved on the
// CPU. GPU atomics affect *storage order only*, never the object population.

struct PackedInstance {
    position_radius: vec4<f32>,    // camera-relative xyz, physical crown radius
    rotation: vec4<f32>,           // normalized source quaternion XYZW
    axis_radii: vec4<f32>,         // physical XYZ scale, appearance variation
    object_id: vec4<u32>,          // full stable u128, low word first
    metadata: vec4<u32>,           // variant u64 + layer u64
    tint: vec4<f32>,               // per-instance linear RGB, alpha multiplier
}

struct IndexedDraw {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct View {
    counts: vec4<u32>,             // input count, aligned bin stride (records), mesh bin count
    lod: vec4<f32>,                // focal pixels, mesh threshold, impostor threshold, animation displacement bound
    forward: vec4<f32>,            // normalized camera forward
    source_center: vec4<f32>,      // normalized original mesh bounds centre
    planes: array<vec4<f32>, 6>,   // xyz normal, w offset; camera-relative
}

@group(0) @binding(0) var<storage, read> input: array<PackedInstance>;
@group(0) @binding(1) var<storage, read_write> visible: array<PackedInstance>;
@group(0) @binding(2) var<storage, read_write> draws: array<IndexedDraw>;
@group(0) @binding(3) var<storage, read> mesh_starts_squared: array<f32>;
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
    let position = instance.position_radius.xyz;
    let radius = instance.position_radius.w;
    let visibility_radius = radius + camera.lod.w;

    // Conservatively keep every sphere intersecting the camera frustum.
    for (var p: u32 = 0u; p < 6u; p += 1u) {
        let plane = camera.planes[p];
        if dot(plane.xyz, position) + plane.w < -visibility_radius {
            return;
        }
    }

    let depth = dot(position, camera.forward.xyz);
    if depth + visibility_radius <= 0.0 {
        return;
    }
    // Cross-multiply the projected diameter test. This avoids a costly
    // division and preserves the exact threshold at 1px, where some Vulkan
    // implementations otherwise round a quotient across the comparison.
    let projected_crown = (2.0 * radius) * camera.lod.x;
    let safe_depth = max(depth, 0.001);
    // Smaller foliage is the responsibility of the distant canopy field.
    if projected_crown < camera.lod.z * safe_depth {
        return;
    }

    // The source ID, variant, and appearance parameters are passed unchanged.
    // Append order across GPU workgroups is intentionally unspecified.
    var bin = camera.counts.z; // final bin is the composite impostor
    if projected_crown >= camera.lod.y * safe_depth {
        let anchor = position - rotate(instance.rotation, camera.source_center.xyz * instance.axis_radii.xyz);
        let distance_squared = dot(anchor, anchor);
        // Upper bound search for the final inclusive authored start range.
        var low = 0u;
        var high = camera.counts.z;
        while low + 1u < high {
            let middle = low + (high - low) / 2u;
            if mesh_starts_squared[middle] <= distance_squared {
                low = middle;
            } else {
                high = middle;
            }
        }
        bin = low;
    }
    let slot = atomicAdd(&draws[bin].instance_count, 1u);
    visible[bin * camera.counts.y + slot] = instance;
}
