struct Pair { meshes:vec2<u32>, transforms:vec2<u32> }
struct Config { margin:f32, mode:u32, count:u32, transform_count:u32 }
@group(0) @binding(3) var<storage,read> transforms:array<mat4x4<f32>>;
@group(0) @binding(4) var<storage,read> pairs:array<Pair>;
@group(0) @binding(5) var<storage,read_write> contacts:array<Contact>;
@group(0) @binding(6) var<uniform> config:Config;


@compute @workgroup_size(64)
fn query(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let pair=pairs[group.x];
    let local=scan(pair.meshes,transforms[pair.transforms.x],transforms[pair.transforms.y],config.margin,config.mode,lane);
    var result=reduce_contact(local,lane);
    if (config.mode&4u)!=0u && result.features.z==0u && result.features.w==0u {
        let inside=reduce_contact(containment(pair.meshes,transforms[pair.transforms.x],transforms[pair.transforms.y],lane),lane);
        result.features.w=inside.features.w;
        result.statistics.w=inside.statistics.w;
    }
    if lane==0u{contacts[group.x]=result;}
}
