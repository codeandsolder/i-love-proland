// Compact source-stem / clustered-foliage expansion. Reuses RenderUniform,
// canonical PackedInstance and the exact same GPU compaction bins as mesh_vs.

struct ModuleData {
    origin_radius: vec4<f32>,
    extent_radius: vec4<f32>,
    color: vec4<f32>,
    metadata: vec4<u32>, // 0 padding, 1 stem, 2 foliage; source ID; count
}
struct ModuleSettings {
    modules_per_variant: u32,
    variant_count: u32,
    unused_a: u32,
    unused_b: u32,
}
@group(2) @binding(0) var<storage,read> module_bank:array<ModuleData>;
@group(2) @binding(1) var<uniform> module_settings:ModuleSettings;

fn module_corner(vertex:u32)->vec2<f32> {
    switch (vertex) {
        case 0u: { return vec2<f32>(-1.0, 0.0); }
        case 1u, 4u: { return vec2<f32>(1.0, 0.0); }
        case 2u, 3u: { return vec2<f32>(-1.0, 1.0); }
        default: { return vec2<f32>(1.0, 1.0); }
    }
}

fn module_hash(value:u32)->u32 {
    var x=value;
    x=(x ^ (x >> 16u)) * 0x7feb352du;
    x=(x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

@vertex
fn module_vs(
    @builtin(vertex_index) vertex_id:u32,
    @builtin(instance_index) instance_id:u32,
) -> MeshOutput {
    let object = objects[visible_indices[instance_id]];
    let model = object.render_model.x;
    let module_index = vertex_id / 12u;
    let side = (vertex_id / 6u) % 2u;
    let corner = module_corner(vertex_id % 6u);
    let module_data = module_bank[model * module_settings.modules_per_variant + module_index];

    var out:MeshOutput;
    if module_data.metadata.x == 0u {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.uv = vec2<f32>(0.0);
        out.kind = 0u;
        out.stable_id = 0u;
        return out;
    }

    var position:vec3<f32>;
    if module_data.metadata.x == 1u {
        let start = module_data.origin_radius.xyz;
        let end = module_data.extent_radius.xyz;
        let direction = normalize(end - start);
        let reference = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(direction.y) > 0.9);
        let normal_a = normalize(cross(direction, reference));
        let normal_b = cross(direction, normal_a);
        let normal = select(normal_a, normal_b, side != 0u);
        let r = mix(module_data.origin_radius.w, module_data.extent_radius.w, corner.y);
        position = mix(start, end, corner.y) + normal * (corner.x * r);
    } else {
        // One source cluster expands to two orthogonal lobes. Its original
        // material membership, conservative shape envelope and source ID
        // were prepared by Lorax, rather than regenerated per camera frame.
        let theta = f32(module_hash(module_data.metadata.y) & 0xffffu)
            * (TAU / 65536.0) + f32(side) * HALF_PI;
        let s = sin(theta);
        let c = cos(theta);
        let x = corner.x * module_data.origin_radius.w;
        let y = (2.0 * corner.y - 1.0) * module_data.extent_radius.x;
        position = module_data.origin_radius.xyz
            + vec3<f32>(c * x, y, s * x * module_data.extent_radius.y / module_data.origin_radius.w);
    }
    let world = (object.position_radius.xyz - camera.eye.xyz)
        + rotate(object.rotation, position * object.axis_radii.xyz);
    let source_y = position.y * camera.wind_source.y + camera.wind_source.x;
    out.clip = camera.projection * vec4<f32>(
        world + wind_offset(world, source_y, object.rotation), 1.0
    );
    out.color = module_data.color * object.tint;
    out.uv = vec2<f32>(0.5 + 0.5 * corner.x, corner.y);
    out.kind = module_data.metadata.x;
    out.stable_id = module_hash(object.object_id.x ^ module_data.metadata.y);
    return out;
}

@fragment
fn module_fs(input:MeshOutput)->@location(0) vec4<f32> {
    let texel = textureSample(source_texture,source_sampler,input.uv);
    if input.kind == 0u {
        discard;
    }
    if input.kind == 1u {
        if texel.a * input.color.a < camera.material.x {discard;}
        return vec4<f32>(texel.rgb * input.color.rgb, 1.0);
    }
    let xy = 2.0 * input.uv - vec2<f32>(1.0);
    let radius = length(xy);
    // An anti-aliased envelope is more stable in the intermediate LOD than
    // dropping a polygon exactly when its screen-space edge gets subpixel.
    let edge_width = max(fwidth(radius), 0.025);
    let shape = 1.0 - smoothstep(1.0 - edge_width, 1.0 + edge_width, radius);
    let alpha = shape * texel.a * input.color.a;
    if alpha <= 0.0 {discard;}
    if camera.material.y < 0.5 {
        // Stable per-pixel cutout on non-MSAA targets; on multisample targets,
        // the module pipeline enables hardware alpha-to-coverage instead.
        let pixel = vec2<u32>(input.clip.xy);
        let value = module_hash(
            input.stable_id ^ (pixel.x * 0x9e3779b9u) ^ (pixel.y * 0x85ebca6bu)
        );
        let threshold = (f32(value & 65535u) + 0.5) / 65536.0;
        if alpha < threshold {discard;}
    }
    return vec4<f32>(texel.rgb * input.color.rgb, alpha);
}
