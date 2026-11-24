// Deterministic GPU Jacobi contact sweeps. Each invocation owns one output
// body and reads an immutable input snapshot. Symmetric pair weights
// preserve linear and angular impulse reciprocity without float atomics.
struct PairConfig { restitution:u32, padding0:u32, padding1:u32, padding2:u32 }
@group(0) @binding(0) var<storage,read> input_bodies:array<Body>;
@group(0) @binding(1) var<storage,read> pair_shapes:array<Collider>;
@group(0) @binding(2) var<storage,read_write> degrees:array<u32>;
@group(0) @binding(3) var<storage,read_write> output_bodies:array<Body>;
@group(0) @binding(4) var<uniform> pair_config:PairConfig;

fn touching(a:Body,b:Body,ra:f32,rb:f32)->bool{
    let d=a.position.xyz-b.position.xyz;
    let extent=ra+rb;
    return b.metadata.z==0u && dot(d,d)<extent*extent;
}

@compute @workgroup_size(64)
fn count_neighbors(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&input_bodies){return;}
    let a=input_bodies[i];
    var degree=0u;
    if a.metadata.z==0u{
        let ra=pair_shapes[a.metadata.x].axes.x;
        for(var j=0u;j<arrayLength(&input_bodies);j+=1u){
            if i==j{continue;}
            let b=input_bodies[j];
            degree+=u32(touching(a,b,ra,pair_shapes[b.metadata.x].axes.x));
        }
    }
    degrees[i]=degree;
}

@compute @workgroup_size(64)
fn resolve_pairs(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&input_bodies){return;}
    let a=input_bodies[i];
    var output=a;
    if a.metadata.z!=0u || degrees[i]==0u {
        output_bodies[i]=output;
        return;
    }
    let ra=pair_shapes[a.metadata.x].axes.x;
    var impulse_sum=vec3<f32>(0.0);
    var torque_sum=vec3<f32>(0.0);
    var position_sum=vec3<f32>(0.0);
    for(var j=0u;j<arrayLength(&input_bodies);j+=1u){
        if i==j{continue;}
        let b=input_bodies[j];
        let rb=pair_shapes[b.metadata.x].axes.x;
        if !touching(a,b,ra,rb){continue;}
        let effect=pair_effect(a,b,i,j,ra,rb,degrees[i],degrees[j],pair_config.restitution!=0u);
        impulse_sum+=effect.impulse;
        torque_sum+=effect.torque;
        position_sum+=effect.correction;
    }
    output_bodies[i]=apply_pair_effect(a,impulse_sum,torque_sum,position_sum);
}
