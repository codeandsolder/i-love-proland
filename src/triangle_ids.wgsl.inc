// Diagnostic-only fragment entry points, never linked into production shaders.
// Every accepted pixel encodes a deterministic 32-bit primitive identity.
fn triangle_hash(value: u32) -> u32 {
    var hash = value;
    hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
    hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
    return hash ^ (hash >> 16u);
}
fn srgb_to_linear_id(value:f32)->f32 {
    return select(value/12.92, pow((value+0.055)/1.055,2.4), value>0.04045);
}
fn triangle_id_color(seed:u32, primitive:u32, category:u32)->vec4<f32> {
    let id = (triangle_hash(seed ^ (primitive * 0x9e3779b9u) ^ 0x457e1987u) & 0x1fffffffu)
        | ((category+1u)<<29u);
    let rgba = vec4<f32>(f32(id&255u), f32((id>>8u)&255u),
                         f32((id>>16u)&255u), f32((id>>24u)&255u)) / 255.0;
    return vec4<f32>(srgb_to_linear_id(rgba.r),srgb_to_linear_id(rgba.g),
                     srgb_to_linear_id(rgba.b),rgba.a);
}
@fragment
fn mesh_ids_fs(input: MeshOutput, @builtin(primitive_index) primitive:u32) -> @location(0) vec4<f32> {
    let texel = textureSample(source_texture, source_sampler, input.uv);
    let alpha=texel.a*input.color.a;
    if (camera.material.x>0.0 && alpha<camera.material.x){discard;}
    return triangle_id_color(input.triangle_seed ^
        (bitcast<u32>(camera.material.z) * 0xd8163841u),primitive,3u);
}
@fragment
fn module_ids_fs(input:MeshOutput,@builtin(primitive_index) primitive:u32)->@location(0) vec4<f32>{
    let material=module_material(input);
    if material.a<=0.0{discard;}
    return triangle_id_color(input.triangle_seed ^
        (bitcast<u32>(camera.material.z) * 0xd8163841u),primitive,3u);
}
@fragment
fn impostor_ids_fs(input: ImpostorOutput, @builtin(primitive_index) primitive:u32) -> @location(0) vec4<f32> {
    let alpha=angular_sample(atlas_texture,input).a * input.tint.a;
    impostor_cutout(input,alpha);
    return triangle_id_color(input.triangle_seed ^
        (bitcast<u32>(camera.material.z) * 0xd8163841u),primitive,4u);
}
