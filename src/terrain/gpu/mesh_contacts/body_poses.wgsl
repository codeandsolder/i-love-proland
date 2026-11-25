struct MeshInstance { metadata:vec4<u32>, local:mat4x4<f32> }
@group(0) @binding(0) var<storage,read> bodies:array<Body>;
@group(0) @binding(1) var<storage,read> instances:array<MeshInstance>;
@group(0) @binding(2) var<storage,read_write> matrices:array<mat4x4<f32>>;
@compute @workgroup_size(64)
fn compose(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&instances){return;}
    let instance=instances[i];
    if instance.metadata.x>=arrayLength(&bodies){
        matrices[i]=mat4x4<f32>();
        return;
    }
    let body=bodies[instance.metadata.x];
    let q=body.rotation;
    let world=mat4x4<f32>(
        vec4<f32>(quat_rotate(q,vec3<f32>(1.0,0.0,0.0)),0.0),
        vec4<f32>(quat_rotate(q,vec3<f32>(0.0,1.0,0.0)),0.0),
        vec4<f32>(quat_rotate(q,vec3<f32>(0.0,0.0,1.0)),0.0),
        vec4<f32>(body.position.xyz,1.0));
    matrices[i]=world*instance.local;
}
