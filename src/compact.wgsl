// i-love-proland: renderer-independent GPU instance classification/compaction.
// Canonical existence, IDs, variants, and mutation overlays are resolved on the
// CPU. GPU atomics affect *storage order only*, never the object population.

struct PackedInstance {
    position_radius: vec4<f32>,    // camera-relative xyz, physical crown radius
    attributes: vec4<f32>,         // yaw in turns, scale, appearance, reserved
    object_id: vec4<u32>,          // full stable u128, low word first
    metadata: vec4<u32>,           // variant u64 + layer u64
}

struct IndexedDraw {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct View {
    counts: vec4<u32>,             // x = input candidate count
    lod: vec4<f32>,                // focal pixels, mesh threshold, impostor threshold, unused
    forward: vec4<f32>,            // normalized camera forward
    planes: array<vec4<f32>, 6>,   // xyz normal, w offset; camera-relative
}

@group(0) @binding(0) var<storage, read> input: array<PackedInstance>;
@group(0) @binding(1) var<storage, read_write> mesh_instances: array<PackedInstance>;
@group(0) @binding(2) var<storage, read_write> impostor_instances: array<PackedInstance>;
@group(0) @binding(3) var<storage, read_write> mesh_draw: IndexedDraw;
@group(0) @binding(4) var<storage, read_write> impostor_draw: IndexedDraw;
@group(0) @binding(5) var<uniform> camera: View;

@compute @workgroup_size(64)
fn classify_and_compact(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_id = global_id.x;
    if input_id >= camera.counts.x {
        return;
    }

    let instance = input[input_id];
    let position = instance.position_radius.xyz;
    let radius = instance.position_radius.w;

    // Conservatively keep every sphere intersecting the camera frustum.
    for (var p: u32 = 0u; p < 6u; p += 1u) {
        let plane = camera.planes[p];
        if dot(plane.xyz, position) + plane.w < -radius {
            return;
        }
    }

    let depth = dot(position, camera.forward.xyz);
    if depth + radius <= 0.0 {
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
    if projected_crown >= camera.lod.y * safe_depth {
        let slot = atomicAdd(&mesh_draw.instance_count, 1u);
        mesh_instances[slot] = instance;
    } else {
        let slot = atomicAdd(&impostor_draw.instance_count, 1u);
        impostor_instances[slot] = instance;
    }
}
