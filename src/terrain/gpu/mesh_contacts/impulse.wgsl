// A predictive surface contact has a nonzero separation normal. The
// caller must refine a crossing/contained pair instead of manufacturing
// a penetration direction from coincident closest points.
struct Contact {
    point_a:vec4<f32>, point_b:vec4<f32>, normal:vec4<f32>,
    features:vec4<u32>, statistics:vec4<u32>,
}
struct Impact {
    point_speed:vec4<f32>,
    impulse_normal:vec4<f32>,
    metadata:vec4<u32>,
}
@group(0) @binding(0) var<storage,read> bodies:array<Body>;
@group(0) @binding(1) var<storage,read> contacts:array<Contact>;
@group(0) @binding(2) var<storage,read> links:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> raw:array<Impact>;
@group(0) @binding(4) var<storage,read> offsets:array<u32>;
@group(0) @binding(5) var<storage,read> adjacency:array<u32>;
@group(0) @binding(6) var<storage,read_write> output_bodies:array<Body>;
@group(0) @binding(7) var<storage,read_write> applied:array<Impact>;

@compute @workgroup_size(64)
fn predict_impulses(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&links){return;}
    let c=contacts[i];
    var impact=Impact(vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(0u,c.features.xy,0u));
    if c.features.w!=0u || (c.features.z>0u && dot(c.normal.xyz,c.normal.xyz)<0.9){
        impact.metadata.x=3u;
    }else if c.features.z>0u{
        let pair=links[i];
        let a=bodies[pair.x];
        let b=bodies[pair.y];
        if a.metadata.z!=0u || b.metadata.z!=0u || any(a.clock.xy!=b.clock.xy){
            impact.metadata.x=3u;
        }else{
            let point=(c.point_a.xyz+c.point_b.xyz)*0.5;
            let ra=point-a.position.xyz;
            let rb=point-b.position.xyz;
            let n=c.normal.xyz;
            let relative=a.velocity.xyz+cross(a.angular.xyz,ra)-b.velocity.xyz-cross(b.angular.xyz,rb);
            let vn=dot(relative,n);
            let inverse_mass=a.position.w+b.position.w;
            let normal_mass=inverse_mass+spin_mass(a,ra,n)+spin_mass(b,rb,n);
            let restitution=select(0.0,min(a.angular.w,b.angular.w),vn< -0.8);
            let normal_impulse=max(0.0,-vn*(1.0+restitution)/max(normal_mass,1e-12));
            let tangent=relative-vn*n;
            let speed=length(tangent);
            var friction=vec3<f32>(0.0);
            if speed>1e-6{
                let axis=tangent/speed;
                let k=inverse_mass+spin_mass(a,ra,axis)+spin_mass(b,rb,axis);
                friction=-axis*min(speed/max(k,1e-12),min(a.velocity.w,b.velocity.w)*normal_impulse);
            }
            impact.point_speed=vec4<f32>(point,max(-vn,0.0));
            impact.impulse_normal=vec4<f32>(n*normal_impulse+friction,normal_impulse);
            impact.metadata.x=select(2u,1u,normal_impulse>0.0);
        }
    }
    raw[i]=impact;
}

fn degree(body:u32)->u32{
    var count=0u;
    for(var k=offsets[body];k<offsets[body+1u];k+=1u){
        count+=u32(raw[adjacency[k]].metadata.x==1u);
    }
    return count;
}
@compute @workgroup_size(64)
fn gather_impulses(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&bodies){return;}
    let body=bodies[i];
    var impulse=vec3<f32>(0.0);
    var torque=vec3<f32>(0.0);
    let own_degree=degree(i);
    for(var k=offsets[i];k<offsets[i+1u];k+=1u){
        let j=adjacency[k];
        let pair=links[j];
        var event=raw[j];
        if event.metadata.x==1u{
            let other=select(pair.x,pair.y,i==pair.x);
            let weight=1.0/f32(max(1u,max(own_degree,degree(other))));
            event.impulse_normal*=weight;
            let direction=select(-1.0,1.0,i==pair.x);
            let effect=event.impulse_normal.xyz*direction;
            impulse+=effect;
            torque+=cross(event.point_speed.xyz-body.position.xyz,effect);
        }
        // One writer per event. The immutable raw array is never mutated.
        if i==pair.x{applied[j]=event;}
    }
    output_bodies[i]=apply_pair_effect(body,impulse,torque,vec3<f32>(0.0));
}
