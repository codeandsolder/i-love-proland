// A complete small contact island fits one workgroup. Every lane takes
// part in the barriers (including unused lanes), so both Jacobi sweeps
// update the original authoritative buffer without scratch dispatches.
@group(0) @binding(0) var<storage,read_write> island:array<Body>;
@group(0) @binding(1) var<storage,read> shapes:array<Collider>;
var<workgroup> broad:array<vec4<f32>,64>;
var<workgroup> degree:array<u32,64>;

fn overlaps(a:vec4<f32>,b:vec4<f32>)->bool{
    let d=a.xyz-b.xyz;
    let r=a.w+b.w;
    return a.w>0.0 && b.w>0.0 && dot(d,d)<r*r;
}
@compute @workgroup_size(64)
fn solve_local(@builtin(local_invocation_index) i:u32){
    let n=arrayLength(&island);
    for(var sweep=0u;sweep<2u;sweep+=1u){
        broad[i]=vec4<f32>(0.0);
        if i<n {
            let a=island[i];
            if a.metadata.z==0u{
                broad[i]=vec4<f32>(a.position.xyz,shapes[a.metadata.x].axes.x);
            }
        }
        workgroupBarrier();
        var neighbors=0u;
        if i<n{
            for(var j=0u;j<n;j+=1u){
                if i!=j && overlaps(broad[i],broad[j]){neighbors+=1u;}
            }
        }
        degree[i]=neighbors;
        workgroupBarrier();
        var output:Body;
        if i<n{
            let a=island[i];
            var impulse=vec3<f32>(0.0);
            var torque=vec3<f32>(0.0);
            var correction=vec3<f32>(0.0);
            if neighbors!=0u{
                for(var j=0u;j<n;j+=1u){
                    if i==j || !overlaps(broad[i],broad[j]){continue;}
                    let effect=pair_effect(a,island[j],i,j,broad[i].w,broad[j].w,degree[i],degree[j],sweep==0u);
                    impulse+=effect.impulse;
                    torque+=effect.torque;
                    correction+=effect.correction;
                }
            }
            output=apply_pair_effect(a,impulse,torque,correction);
        }
        // All storage reads for this sweep precede all storage writes.
        storageBarrier();
        if i<n{island[i]=output;}
        storageBarrier();
    }
}
