// Generic fully rotated source-mesh and multi-angle tree impostor renderer.
// Camera-relative records are produced by the compactor. The source object
// identity, material variant, and placement remain independent of visual LOD.

const TAU: f32 = 6.283185307179586;
const HALF_PI: f32 = 1.5707963267948966;
const AZIMUTH_VIEWS: f32 = 8.0;
const ELEVATION_VIEWS: f32 = 4.0;

struct Instance {
    position_radius: vec4<f32>,
    rotation: vec4<f32>,
    axis_radii: vec4<f32>,
    object_id: vec4<u32>,
    metadata: vec4<u32>,
    tint: vec4<f32>,
}

struct RenderUniform {
    projection: mat4x4<f32>,
    material: vec4<f32>, // source alpha cutoff, 1 for multisample alpha-to-coverage
    wind_parameters: vec4<f32>, // spatial scale, height start, exponent, intensity
    wind_source: vec4<f32>, // original source center Y, radius
    field_axes: array<vec4<f32>, 3>,
    field_origin: vec4<f32>,
    field_phase: vec4<f32>, // camera origin modulo texture repetition, time scroll
    camera_forward: vec4<f32>,
    billboard_source: vec4<f32>, // normalized source center, original radius (zero = static)
}

@group(0) @binding(0) var<uniform> camera: RenderUniform;
@group(1) @binding(0) var<storage, read> objects: array<Instance>;
@group(1) @binding(1) var<storage, read> visible_indices: array<u32>;
@group(0) @binding(2) var atlas_texture: texture_2d_array<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;
@group(0) @binding(4) var source_texture: texture_2d<f32>;
@group(0) @binding(5) var source_sampler: sampler;
@group(0) @binding(6) var wind_texture: texture_2d<f32>;

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

// Quaternion XYZW rotation without constructing a matrix per vertex.
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn inverse_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return rotate(vec4<f32>(-q.xyz, q.w), v);
}

// Four half-texel-offset bilinear taps reproduce the authored smooth wind
// field; UVs repeat and use explicit LOD zero in the vertex stage.
fn smooth_wind(uv: vec2<f32>) -> vec3<f32> {
    let half_texel = 0.5 / vec2<f32>(textureDimensions(wind_texture, 0));
    let p = fract(uv);
    return 0.25 * (
        textureSampleLevel(wind_texture, source_sampler, p - half_texel, 0.0).rgb
        + textureSampleLevel(wind_texture, source_sampler, p + vec2<f32>(half_texel.x, -half_texel.y), 0.0).rgb
        + textureSampleLevel(wind_texture, source_sampler, p + vec2<f32>(-half_texel.x, half_texel.y), 0.0).rgb
        + textureSampleLevel(wind_texture, source_sampler, p + half_texel, 0.0).rgb);
}

fn wind_offset(world: vec3<f32>, source_y: f32, rotation: vec4<f32>) -> vec3<f32> {
    let p = camera.wind_parameters;
    if p.w <= 0.0 || source_y <= p.y || source_y <= 0.0 {
        return vec3<f32>(0.0);
    }
    let basis = mat3x3<f32>(camera.field_axes[0].xyz, camera.field_axes[1].xyz, camera.field_axes[2].xyz);
    let relative = basis * world;
    let field_position = camera.field_origin.xyz + relative;
    let normal = abs(field_position);
    let weights = normal / max(normal.x + normal.y + normal.z, 0.000001);
    let uv = relative * p.x + camera.field_phase.xyz + vec3<f32>(camera.field_phase.w);
    let wind = smooth_wind(uv.yz) * weights.x
        + smooth_wind(uv.zx) * weights.y
        + smooth_wind(uv.xy) * weights.z;
    let up = basis * rotate(rotation, vec3<f32>(0.0, 1.0, 0.0));
    let direction = vec3<f32>(1.0) - up * dot(up, vec3<f32>(1.0));
    // A plane perpendicular to (1,1,1) has no preferred lateral direction.
    let lateral = -direction * inverseSqrt(max(dot(direction, direction), 0.00000001));
    let amplitude = pow(source_y, p.z) * p.w;
    let offset = (lateral + 0.4 * up) * wind * amplitude;
    return transpose(basis) * offset;
}

@vertex
fn mesh_vs(input: MeshVertex, @builtin(instance_index) instance_id: u32) -> MeshOutput {
    let object = objects[visible_indices[instance_id]];
    var vertex = input.position.xyz;
    let source = camera.billboard_source;
    if source.w > 0.0 {
        // Unity's inverse-transpose view row points back toward the camera.
        // Original forward is multiplied by the complete scaled instance
        // basis; normalize only the horizontal right, not forward.
        let back = inverse_rotate(object.rotation, -camera.camera_forward.xyz)
            * (object.axis_radii.xyz / source.w);
        let horizontal = length(back.xz);
        var right = vec3<f32>(1.0, 0.0, 0.0);
        if horizontal > 0.000001 {
            right = vec3<f32>(-back.z, 0.0, back.x) / horizontal;
        }
        let local = vertex + source.xyz;
        vertex = local.x * right + vec3<f32>(0.0, local.y, 0.0)
            + local.z * back - source.xyz;
    }
    let world = object.position_radius.xyz
        + rotate(object.rotation, vertex * object.axis_radii.xyz);
    var out: MeshOutput;
    let source_y = vertex.y * camera.wind_source.y + camera.wind_source.x;
    let displaced = world + wind_offset(world, source_y, object.rotation);
    out.clip = camera.projection * vec4<f32>(displaced, 1.0);
    out.color = input.color * object.tint;
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
    @location(3) tint: vec4<f32>,
}

@vertex
fn impostor_vs(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> ImpostorOutput {
    let object = objects[visible_indices[instance_id]];
    let center = object.position_radius.xyz;
    let radius = object.position_radius.w;
    let to_eye = select(normalize(-center), vec3<f32>(0.0, 0.0, 1.0), dot(center, center) < 0.000001);

    // Camera azimuth and elevation in the original *source* frame.
    // Quaternion inverse makes terrain-aligned/tilted source plants sample
    // the same local atlas angles as upright plants.
    let local_eye = inverse_rotate(object.rotation, to_eye);
    let azimuth_angle = atan2(local_eye.x, local_eye.z);
    // Build the continuous local-azimuth billboard basis, then rotate it
    // into the world with the actual source orientation.
    let right = rotate(object.rotation, vec3<f32>(
        -cos(azimuth_angle), 0.0, sin(azimuth_angle)
    ));
    let up = cross(right, to_eye);
    // Exact near geometry is anisotropic. Impostors preserve its projected
    // conservative extents, while angular atlas shading remains approximate.
    let local_right = inverse_rotate(object.rotation, right);
    let local_up = inverse_rotate(object.rotation, up);
    let half_width = length(local_right * object.axis_radii.xyz);
    let half_height = length(local_up * object.axis_radii.xyz);
    let uv = vec2<f32>(f32(vertex_id & 1u), f32(vertex_id >> 1u));
    let corners = uv * 2.0 - vec2<f32>(1.0);
    let world = center + right * (corners.x * half_width)
        + up * (corners.y * half_height);

    let azimuth = fract(azimuth_angle / TAU) * AZIMUTH_VIEWS;
    let a0 = floor(azimuth);
    let a1 = (a0 + 1.0) % AZIMUTH_VIEWS;
    let elevation = asin(clamp(local_eye.y, 0.0, 1.0))
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
    let variation = 0.90 + object.axis_radii.w * 0.10;
    out.tint = vec4<f32>(object.tint.rgb * variation, object.tint.a);
    return out;
}

fn atlas_sample(tile: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
    let layer = i32(tile.y * AZIMUTH_VIEWS + tile.x);
    return textureSample(atlas_texture, atlas_sampler, uv, layer);
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
    let alpha = color.a * input.tint.a;
    if (alpha <= 0.0) {
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
        if (alpha < threshold) {
            discard;
        }
    }
    // With multisample targets, hardware alpha-to-coverage uses this alpha.
    // Unpremultiply after both spatial mip filtering and angular blending.
    return vec4<f32>(color.rgb / color.a * input.tint.rgb, alpha);
}
