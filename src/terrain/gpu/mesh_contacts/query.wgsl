struct Pair { meshes:vec2<u32>, transforms:vec2<u32> }
struct Config { margin:f32, mode:u32, count:u32, transform_count:u32 }
struct Dispatch { x:atomic<u32>, y:u32, z:u32 }
@group(0) @binding(3) var<storage,read> transforms:array<mat4x4<f32>>;
@group(0) @binding(4) var<storage,read> pairs:array<Pair>;
@group(0) @binding(5) var<storage,read_write> contacts:array<Contact>;
@group(0) @binding(6) var<uniform> config:Config;
@group(0) @binding(7) var<storage,read_write> candidates:array<u32>;
@group(0) @binding(8) var<storage,read_write> dispatch:Dispatch;

// Conservatively transformed root bounds in a shared relative frame.
// False positives are allowed. The narrow phase repeats its tighter test
// in B-local space, and invalid poses remain explicit unresolved results.
fn root_bounds(mesh:u32,m:mat4x4<f32>,origin:vec3<f32>)->Bounds{
    let root=nodes[meshes[mesh].nodes.x];
    let l=linear(m);
    let t=m[3].xyz-origin;
    let center=l*((root.lo+root.hi)*0.5)+t;
    let half=(root.hi-root.lo)*0.5;
    let radius=abs(l[0])*half.x+abs(l[1])*half.y+abs(l[2])*half.z;
    let size=max(abs(root.lo),abs(root.hi));
    let error=(abs(l[0])*size.x+abs(l[1])*size.y+abs(l[2])*size.z+abs(t)+vec3<f32>(1.0))*0.000008;
    return Bounds(center-radius-error,center+radius+error);
}

@compute @workgroup_size(64)
fn prequalify(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=config.count{return;}
    let pair=pairs[i];
    let a=transforms[pair.transforms.x];
    let b=transforms[pair.transforms.y];
    contacts[i]=empty();
    if valid_matrix(a)&&valid_matrix(b){
        // Preserve the narrow phase invalid/singular-transform semantics.
        let la=linear(a);
        let lb=linear(b);
        let det_a=dot(la[0],cross(la[1],la[2]));
        let det_b=dot(lb[0],cross(lb[1],lb[2]));
        if abs(det_a)>1e-20 && abs(det_b)>1e-20 {
            let ab=root_bounds(pair.meshes.x,a,b[3].xyz);
            let bb=root_bounds(pair.meshes.y,b,b[3].xyz);
            if any(ab.lo-bb.hi>vec3<f32>(config.margin))||
               any(bb.lo-ab.hi>vec3<f32>(config.margin)){return;}
        }
    }
    let slot=atomicAdd(&dispatch.x,1u);
    candidates[slot]=i;
}

fn query_pair(i:u32,lane:u32){
    let pair=pairs[i];
    let local=scan(pair.meshes,transforms[pair.transforms.x],transforms[pair.transforms.y],config.margin,lane);
    var result=orient_contact(pair.meshes,reduce_contact(local,lane));
    if (config.mode&4u)!=0u && result.features.z==0u && result.features.w==0u {
        let inside=reduce_contact(containment(pair.meshes,transforms[pair.transforms.x],transforms[pair.transforms.y],lane),lane);
        result.features.w=inside.features.w;
        result.statistics.w=inside.statistics.w;
    }
    if lane==0u{contacts[i]=result;}
}
@compute @workgroup_size(64)
fn query(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    query_pair(group.x,lane);
}
@compute @workgroup_size(64)
fn query_compact(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    query_pair(candidates[group.x],lane);
}
