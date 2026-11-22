// Local chart-space rigid bodies over the exact resident GPU height grid.
// One invocation owns each independent body. Fixed dt, arbitrary batch size.
// Coordinates are tile-relative west/north/up metres. No world-sized f32
// coordinates, no CPU collision contacts, no water collision surface.
struct Config {
    side:u32, slot:u32, count:u32, steps:u32,
    spacing:f32, dt:f32, gravity:f32, slop:f32,
    iterations:u32, stop_at_contact:u32, padding1:u32, padding2:u32,
}
struct Contact { normal:vec3<f32>, depth:f32 }
struct Constraint {
    normal:vec3<f32>, desired_speed:f32,
    arm:vec3<f32>, inv_effective_mass:f32,
    friction_impulse:vec3<f32>, normal_impulse:f32,
}
@group(0) @binding(0) var<storage,read> terrain:array<TerrainSample>;
@group(0) @binding(1) var<storage,read_write> bodies:array<Body>;
@group(0) @binding(2) var<storage,read> spheres:array<vec4<f32>>;
@group(0) @binding(3) var<uniform> config:Config;

fn quat_rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32>{
    let t=2.0*cross(q.xyz,v);
    return v+q.w*t+cross(q.xyz,t);
}
fn quat_mul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{
    return vec4<f32>(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));
}
fn inverse_world_inertia(b:Body,v:vec3<f32>)->vec3<f32>{
    let inverse=vec4<f32>(-b.rotation.xyz,b.rotation.w);
    return quat_rotate(b.rotation,quat_rotate(inverse,v)*b.inverse_inertia.xyz);
}
fn vertex(x:u32,y:u32)->vec3<f32>{
    let i=config.slot*config.side*config.side+y*config.side+x;
    return vec3<f32>(vec2<f32>(f32(x),f32(y))*config.spacing,terrain[i].bed);
}
fn segment_point(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>)->vec3<f32>{
    let edge=b-a;
    return a+edge*clamp(dot(p-a,edge)/max(dot(edge,edge),1e-12),0.0,1.0);
}
// Closest point on the same southwest/northeast triangles as the renderer.
// The projected plane point is accepted only if inside all oriented edges.
fn closest_triangle(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->vec3<f32>{
    let n=normalize(cross(b-a,c-a));
    let plane=p-n*dot(p-a,n);
    if dot(cross(b-a,plane-a),n)>=0.0 && dot(cross(c-b,plane-b),n)>=0.0 && dot(cross(a-c,plane-c),n)>=0.0{return plane;}
    let ab=segment_point(p,a,b);
    let bc=segment_point(p,b,c);
    let ca=segment_point(p,c,a);
    let dab=dot(p-ab,p-ab);let dbc=dot(p-bc,p-bc);let dca=dot(p-ca,p-ca);
    if dab<=dbc&&dab<=dca{return ab;}
    if dbc<=dca{return bc;}
    return ca;
}
fn terrain_contact(center:vec3<f32>,radius:f32)->Contact{
    let lo=vec2<i32>(floor((center.xy-vec2<f32>(radius))/config.spacing));
    let hi=vec2<i32>(floor((center.xy+vec2<f32>(radius))/config.spacing));
    let limit=i32(config.side)-2;
    var best=Contact(vec3<f32>(0.0,0.0,1.0),0.0);
    for(var y=max(lo.y,0);y<=min(hi.y,limit);y+=1){
        for(var x=max(lo.x,0);x<=min(hi.x,limit);x+=1){
            let a=vertex(u32(x),u32(y));let b=vertex(u32(x+1),u32(y));
            let c=vertex(u32(x),u32(y+1));let d=vertex(u32(x+1),u32(y+1));
            // Reject cells entirely below the sphere before narrow phase.
            if center.z-radius>max(max(a.z,b.z),max(c.z,d.z)){continue;}
            for(var face=0u;face<2u;face+=1u){
                let v1=select(d,b,face==0u);
                let v2=select(c,d,face==0u);
                let n=normalize(cross(v1-a,v2-a));
                let point=closest_triangle(center,a,v1,v2);
                let delta=center-point;
                let distance=length(delta);
                let signed=dot(delta,n);
                // Heightfield is solid below the surface. Detect a deeply
                // embedded center if its vertical projection is in the face.
                let penetration=radius-distance;
                if penetration>best.depth && signed>=0.0{
                    best=Contact(select(n,delta/max(distance,1e-8),distance>1e-6),penetration);
                }else if signed<0.0{
                    let planar=center.xy;
                    let edge0=v1.xy-a.xy;let edge1=v2.xy-a.xy;
                    let q=planar-a.xy;
                    let denom=edge0.x*edge1.y-edge0.y*edge1.x;
                    let u=(q.x*edge1.y-q.y*edge1.x)/denom;
                    let v=(edge0.x*q.y-edge0.y*q.x)/denom;
                    if u>=0.0&&v>=0.0&&u+v<=1.0{
                        let depth=radius-dot(center-a,n);
                        if depth>best.depth{best=Contact(n,depth);}
                    }
                }
            }
        }
    }
    return best;
}
fn effective_mass(b:Body,r:vec3<f32>,axis:vec3<f32>)->f32{
    return b.position.w+dot(cross(inverse_world_inertia(b,cross(r,axis)),r),axis);
}
fn apply_impulse(b:ptr<function,Body>,r:vec3<f32>,impulse:vec3<f32>){
    (*b).velocity=vec4<f32>((*b).velocity.xyz+impulse*(*b).position.w,(*b).velocity.w);
    (*b).angular=vec4<f32>((*b).angular.xyz+inverse_world_inertia(*b,cross(r,impulse)),(*b).angular.w);
}
// Generate contact geometry once at the integrated pose. Solver
// iterations change velocities only; there is no reason to traverse the
// terrain repeatedly for precisely the same pose.
fn prepare(b:Body,sphere:vec4<f32>)->Constraint{
    let center=b.position.xyz+quat_rotate(b.rotation,sphere.xyz);
    let contact=terrain_contact(center,sphere.w);
    if contact.depth<=0.0{
        return Constraint(vec3<f32>(0.0),0.0,vec3<f32>(0.0),0.0,vec3<f32>(0.0),0.0);
    }
    let n=contact.normal;
    let arm=center-n*sphere.w-b.position.xyz;
    let vn=dot(b.velocity.xyz+cross(b.angular.xyz,arm),n);
    let bias=min(2.0,max(contact.depth-config.slop,0.0)*0.22/config.dt);
    let bounce=select(0.0,-vn*b.angular.w,vn < -0.8);
    return Constraint(n,max(bias,bounce),arm,1.0/max(effective_mass(b,arm,n),1e-8),vec3<f32>(0.0),0.0);
}
fn solve(b:ptr<function,Body>,constraint:Constraint)->Constraint{
    var c=constraint;
    let n=c.normal;
    let r=c.arm;
    let relative=(*b).velocity.xyz+cross((*b).angular.xyz,r);
    let vn=dot(relative,n);
    // Project the accumulated normal impulse onto the unilateral cone.
    // Subsequent iterations can undo an excessive prior impulse.
    let next_normal=max(0.0,c.normal_impulse+(c.desired_speed-vn)*c.inv_effective_mass);
    apply_impulse(b,r,n*(next_normal-c.normal_impulse));
    c.normal_impulse=next_normal;
    let after=(*b).velocity.xyz+cross((*b).angular.xyz,r);
    let tangent=after-n*dot(after,n);
    let speed=length(tangent);
    var next_friction=c.friction_impulse;
    if speed>1e-6{
        let axis=tangent/speed;
        next_friction-=axis*speed/max(effective_mass(*b,r,axis),1e-8);
    }
    // The complete accumulated tangential impulse is constrained by the
    // current normal impulse, not an independent friction budget per pass.
    let limit=(*b).velocity.w*c.normal_impulse;
    next_friction*=min(1.0,limit/max(length(next_friction),1e-8));
    apply_impulse(b,r,next_friction-c.friction_impulse);
    c.friction_impulse=next_friction;
    return c;
}
fn covered(b:Body)->bool{
    let extent=f32(config.side-1u)*config.spacing;
    for(var j=0u;j<b.metadata.y;j+=1u){
        let s=spheres[b.metadata.x+j];
        let center=b.position.xyz+quat_rotate(b.rotation,s.xyz);
        if any(center.xy-vec2<f32>(s.w)<vec2<f32>(0.0))||any(center.xy+vec2<f32>(s.w)>=vec2<f32>(extent)){return false;}
    }
    return true;
}
@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=config.count{return;}
    var b=bodies[id.x];
    if b.metadata.z!=0u{return;}
    var min_radius=1e20;var max_arm=0.0;
    for(var j=0u;j<b.metadata.y;j+=1u){
        let s=spheres[b.metadata.x+j];min_radius=min(min_radius,s.w);max_arm=max(max_arm,length(s.xyz)+s.w);
    }
    for(var tick=0u;tick<config.steps;tick+=1u){
        let previous=b;
        b.velocity.z-=config.gravity*config.dt;
        // Bounded discrete integration: refuse steps that could traverse
        // more than half a sphere radius. A future swept solver can remove
        // this restriction; never silently tunnel through a terrain tile.
        if (length(b.velocity.xyz)+length(b.angular.xyz)*max_arm)*config.dt>min_radius*0.5{
            b=previous;b.metadata.z=2u;break;
        }
        b.position=vec4<f32>(b.position.xyz+b.velocity.xyz*config.dt,b.position.w);
        b.rotation=normalize(b.rotation+0.5*config.dt*quat_mul(vec4<f32>(b.angular.xyz,0.0),b.rotation));
        if !covered(b){b=previous;b.metadata.z=1u;break;}
        var constraints:array<Constraint,64>;
        var count=0u;
        for(var j=0u;j<b.metadata.y;j+=1u){
            let c=prepare(b,spheres[b.metadata.x+j]);
            if c.inv_effective_mass>0.0{
                constraints[count]=c;
                count+=1u;
            }
        }
        for(var iteration=0u;iteration<config.iterations;iteration+=1u){
            for(var j=0u;j<count;j+=1u){constraints[j]=solve(&b,constraints[j]);}
        }
        b.clock.x+=1u;
        if b.clock.x==0u{b.clock.y+=1u;}
        if count>0u && b.metadata.w==0u {
            b.metadata.w=1u;
            b.clock.z=b.clock.x;
            b.clock.w=b.clock.y;
            if config.stop_at_contact!=0u{b.metadata.z=3u;break;}
        }
    }
    bodies[id.x]=b;
}
