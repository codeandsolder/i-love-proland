// Generic upright-object mesh and multi-angle tree impostor renderer.
// Camera-relative records are produced by the compactor. The source object
// identity, material variant, and placement remain independent of visual LOD.

const TAU: f32 = 6.283185307179586;
const HALF_PI: f32 = 1.5707963267948966;
const AZIMUTH_VIEWS: f32 = 8.0;
const ELEVATION_VIEWS: f32 = 4.0;

struct Instance {
    position_radius: vec4<f32>,
    attributes: vec4<f32>,
    object_id: vec4<u32>,
    metadata: vec4<u32>,
}

struct RenderUniform {
    projection: mat4x4<f32>,
    atlas: vec4<f32>, // (8,4, gutter fraction, reserved)
    material: vec4<f32>, // source alpha cutoff, 1 for multisample alpha-to-coverage
}

@group(0) @binding(0) var<uniform> camera: RenderUniform;
@group(0) @binding(1) var<storage, read> objects: array<Instance>;
@group(0) @binding(2) var atlas_texture: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;
@group(0) @binding(4) var source_texture: texture_2d<f32>;
@group(0) @binding(5) var source_sampler: sampler;

struct MeshVertex {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
}

struct MeshOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn mesh_vs(input: MeshVertex, @builtin(instance_index) instance_id: u32) -> MeshOutput {
    let object = objects[instance_id];
    let angle = object.attributes.x * TAU;
    let cs = cos(angle);
    let sn = sin(angle);
    let local = input.position.xyz;
    let rotated = vec3<f32>(
        cs * local.x + sn * local.z,
        local.y,
        -sn * local.x + cs * local.z
    );
    let world = object.position_radius.xyz + rotated * object.position_radius.w;
    var out: MeshOutput;
    out.clip = camera.projection * vec4<f32>(world, 1.0);
    out.color = input.color;
    out.uv = input.uv;
    return out;
}

@fragment
fn mesh_fs(input: MeshOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(source_texture, source_sampler, input.uv);
    if (texel.a * input.color.a < camera.material.x) {
        discard;
    }
    return vec4<f32>(texel.rgb * input.color.rgb, 1.0);
}

struct ImpostorOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_indices: vec4<f32>, // azimuth0, azimuth1, elevation0, elevation1
    @location(2) blend: vec2<f32>,
    @location(3) tint: vec3<f32>,
}

@vertex
fn impostor_vs(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> ImpostorOutput {
    let object = objects[instance_id];
    let center = object.position_radius.xyz;
    let radius = object.position_radius.w;
    let to_eye = select(normalize(-center), vec3<f32>(0.0, 0.0, 1.0), dot(center, center) < 0.000001);

    // Analytic billboard frame: remains continuous into directly overhead
    // views and shares the same azimuth convention as the atlas baker.
    let world_azimuth = atan2(to_eye.x, to_eye.z);
    let right = vec3<f32>(-cos(world_azimuth), 0.0, sin(world_azimuth));
    let up = cross(right, to_eye);

    // The source atlas uses a 2-radii orthographic footprint for all views.
    let uv = vec2<f32>(f32(vertex_id & 1u), f32(vertex_id >> 1u));
    let corners = uv * 2.0 - vec2<f32>(1.0);
    let world = center + radius * (right * corners.x + up * corners.y);

    let azimuth = fract(world_azimuth / TAU - object.attributes.x)
        * AZIMUTH_VIEWS;
    let a0 = floor(azimuth);
    let a1 = (a0 + 1.0) % AZIMUTH_VIEWS;
    // Four elevation rows from the horizon to straight overhead.
    let elevation = asin(clamp(to_eye.y, 0.0, 1.0))
        / HALF_PI * (ELEVATION_VIEWS - 1.0);
    let e0 = floor(elevation);
    let e1 = min(e0 + 1.0, ELEVATION_VIEWS - 1.0);

    var out: ImpostorOutput;
    out.clip = camera.projection * vec4<f32>(world, 1.0);
    // Bitmap atlas coordinates run from top to bottom; geometry UVs do not.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    out.view_indices = vec4<f32>(a0, a1, e0, e1);
    out.blend = vec2<f32>(fract(azimuth), fract(elevation));
    // Tiny deterministic visual variation without redefining the world ID.
    let variation = 0.90 + object.attributes.z * 0.10;
    out.tint = vec3<f32>(variation);
    return out;
}

fn atlas_sample(tile: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
    let gutter = camera.atlas.z;
    let local = gutter + clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (1.0 - 2.0 * gutter);
    return textureSample(
        atlas_texture,
        atlas_sampler,
        (tile + local) / camera.atlas.xy
    );
}

@fragment
fn impostor_fs(input: ImpostorOutput) -> @location(0) vec4<f32> {
    let a0 = input.view_indices.x;
    let a1 = input.view_indices.y;
    let e0 = input.view_indices.z;
    let e1 = input.view_indices.w;
    let lower = mix(
        atlas_sample(vec2<f32>(a0, e0), input.uv),
        atlas_sample(vec2<f32>(a1, e0), input.uv),
        input.blend.x
    );
    let upper = mix(
        atlas_sample(vec2<f32>(a0, e1), input.uv),
        atlas_sample(vec2<f32>(a1, e1), input.uv),
        input.blend.x
    );
    let color = mix(lower, upper, input.blend.y);
    if (color.a <= 0.0) {
        discard;
    }
    if (camera.material.y < 0.5) {
        // Stable 4x4 ordered alpha coverage: no sorting or camera-dependent
        // RNG, and each accepted fragment writes normal hardware depth.
        let x = u32(input.clip.x) & 3u;
        let y = u32(input.clip.y) & 3u;
        let bayer = array<u32, 16>(
            0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u,
            3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u
        );
        let threshold = (f32(bayer[y * 4u + x]) + 0.5) / 16.0;
        if (color.a < threshold) {
            discard;
        }
    }
    // With multisample targets, hardware alpha-to-coverage uses this alpha.
    return vec4<f32>(color.rgb * input.tint, color.a);
}
