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

struct PairEffect {
    impulse: vec3<f32>,
    torque: vec3<f32>,
    correction: vec3<f32>,
}
fn pair_effect(a:Body,b:Body,i:u32,j:u32,ra:f32,rb:f32,degree_a:u32,degree_b:u32,bounce:bool)->PairEffect{
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
        let restitution=select(0.0,min(a.angular.w,b.angular.w),bounce&&vn < -0.8);
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
        let weight=1.0/f32(max(1u,max(degree_a,degree_b)));
        let impulse=(n*normal_impulse+friction)*weight;
        // Split positional correction does not inject kinetic energy.
        let correction=n*(max(depth-0.001,0.0)*0.7*a.position.w/inv_mass*weight);
    return PairEffect(impulse,cross(arm_a,impulse),correction);
}
fn apply_pair_effect(a:Body,impulse:vec3<f32>,torque:vec3<f32>,correction:vec3<f32>)->Body{
    var output=a;
    output.velocity=vec4<f32>(a.velocity.xyz+impulse*a.position.w,a.velocity.w);
    output.angular=vec4<f32>(a.angular.xyz+inverse_world_inertia(a,torque),a.angular.w);
    output.position=vec4<f32>(a.position.xyz+correction,a.position.w);
    return output;
}
