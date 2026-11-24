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

fn pair_normal(a:Body,b:Body,i:u32,j:u32)->vec3<f32>{
    let delta=a.position.xyz-b.position.xyz;
    let length2=dot(delta,delta);
    if length2>1e-12{return delta*inverseSqrt(length2);}
    // Exactly coincident initial centers still separate symmetrically.
    return vec3<f32>(select(-1.0,1.0,i<j),0.0,0.0);
}
fn spin_mass(b:Body,r:vec3<f32>,axis:vec3<f32>)->f32{
    return dot(cross(inverse_world_inertia(b,cross(r,axis)),r),axis);
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
        let n=pair_normal(a,b,i,j);
        // Use the same world-space contact point for both bodies even
        // during penetration, preserving the pair angular impulse.
        let distance=length(a.position.xyz-b.position.xyz);
        let depth=ra+rb-distance;
        let arm_a=-n*(ra-0.5*depth);
        let arm_b=n*(rb-0.5*depth);
        let relative=a.velocity.xyz+cross(a.angular.xyz,arm_a)
                    -b.velocity.xyz-cross(b.angular.xyz,arm_b);
        let inv_mass=a.position.w+b.position.w;
        let vn=dot(relative,n);
        let restitution=select(0.0,min(a.angular.w,b.angular.w),pair_config.restitution!=0u&&vn < -0.8);
        let normal_impulse=max(0.0,-vn*(1.0+restitution)/inv_mass);
        let tangent=relative-vn*n;
        let speed=length(tangent);
        var friction=vec3<f32>(0.0);
        if speed>1e-6{
            let axis=tangent/speed;
            let k=inv_mass+spin_mass(a,arm_a,axis)+spin_mass(b,arm_b,axis);
            friction=-axis*min(speed/max(k,1e-8),min(a.velocity.w,b.velocity.w)*normal_impulse);
        }
        // Symmetric relaxation for bodies with several simultaneous contacts.
        let weight=1.0/f32(max(1u,max(degrees[i],degrees[j])));
        let impulse=(n*normal_impulse+friction)*weight;
        impulse_sum+=impulse;
        torque_sum+=cross(arm_a,impulse);
        // Split positional correction does not inject kinetic energy.
        position_sum+=n*(max(depth-0.001,0.0)*0.7*a.position.w/inv_mass*weight);
    }
    output.velocity=vec4<f32>(a.velocity.xyz+impulse_sum*a.position.w,a.velocity.w);
    output.angular=vec4<f32>(a.angular.xyz+inverse_world_inertia(a,torque_sum),a.angular.w);
    output.position=vec4<f32>(a.position.xyz+position_sum,a.position.w);
    output_bodies[i]=output;
}
