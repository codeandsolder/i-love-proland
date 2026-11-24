// Local chart-space rigid bodies over the exact resident GPU height grid.
// One invocation owns each independent body. Fixed dt, arbitrary batch size.
// Coordinates are tile-relative west/north/up metres. No world-sized f32
// coordinates, no CPU collision contacts, no water collision surface.
struct Config {
    side:u32, slot:u32, count:u32, steps:u32,
    spacing:f32, dt:f32, gravity:f32, slop:f32,
    iterations:u32, stop_at_contact:u32, record_contacts:u32, contact_stride:u32,
    projection:GridProjection,
    first_tick_low:u32, pad0:u32, pad1:u32, pad2:u32,
}
struct Contact { normal:vec3<f32>, depth:f32, point:vec3<f32>, material:u32 }
struct Constraint {
    normal:vec3<f32>, desired_speed:f32,
    arm:vec3<f32>, inv_effective_mass:f32,
    friction_impulse:vec3<f32>, normal_impulse:f32,
    point:vec3<f32>, depth:f32,
    incident_slip:vec3<f32>, reserved:f32,
    area:f32, shape:u32, bounce:f32, material:u32,
}
struct ContactEvent{
    position_force:vec4<f32>,
    normal_depth:vec4<f32>,
    rotation:vec4<f32>,
    slip_area:vec4<f32>,
    metadata:vec4<u32>,
}
@group(0) @binding(0) var<storage,read> terrain:array<TerrainSample>;
@group(0) @binding(1) var<storage,read_write> bodies:array<Body>;
@group(0) @binding(2) var<storage,read> shapes:array<Collider>;
@group(0) @binding(3) var<uniform> config:Config;
@group(0) @binding(4) var<storage,read_write> contact_events:array<ContactEvent>;
@group(0) @binding(5) var<storage,read> materials:array<vec2<f32>>;
@group(0) @binding(6) var<storage,read> material_cells:array<u32>;
fn cell_material(x:u32,y:u32)->u32{
    if arrayLength(&material_cells)==1u{return material_cells[0];}
    return material_cells[y*(config.side-1u)+x];
}

fn vertex(x:u32,y:u32)->vec3<f32>{
    let i=config.slot*config.side*config.side+y*config.side+x;
    return grid_position(config.projection,vec2<u32>(x,y),terrain[i].bed);
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
struct ShapePose{
    center:vec3<f32>, radius:f32,
    rotation:vec4<f32>,
    axes:vec3<f32>, min_axis:f32,
}
fn shape_pose(b:Body,shape:Collider)->ShapePose{
    let axes=shape.axes.xyz;
    let radius=max(axes.x,max(axes.y,axes.z));
    let min_axis=min(axes.x,min(axes.y,axes.z));
    // Preserve the original sphere metric without unnecessary rotations.
    let rotation=select(quat_mul(b.rotation,shape.rotation),vec4<f32>(0.0,0.0,0.0,1.0),radius==min_axis);
    return ShapePose(b.position.xyz+quat_rotate(b.rotation,shape.center.xyz),radius,rotation,axes,min_axis);
}
fn inverse_rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32>{
    return quat_rotate(vec4<f32>(-q.xyz,q.w),v);
}
fn metric(s:ShapePose,v:vec3<f32>)->vec3<f32>{
    return inverse_rotate(s.rotation,v-s.center)*(s.min_axis/s.axes);
}
fn metric_normal(s:ShapePose,n:vec3<f32>)->vec3<f32>{
    return normalize(quat_rotate(s.rotation,n*(s.min_axis/s.axes)));
}
fn world_point(s:ShapePose,v:vec3<f32>)->vec3<f32>{
    return s.center+quat_rotate(s.rotation,v*(s.axes/s.min_axis));
}
fn support_height(s:ShapePose,n:vec3<f32>)->f32{
    return length(s.axes*inverse_rotate(s.rotation,n));
}
fn support_arm(s:ShapePose,n:vec3<f32>)->vec3<f32>{
    let local=inverse_rotate(s.rotation,n);
    return quat_rotate(s.rotation,s.axes*s.axes*local/max(length(s.axes*local),1e-12));
}
fn loaded_area(s:ShapePose,n:vec3<f32>,depth:f32)->f32{
    // Exact area of the ellipsoid intersected by its supporting plane:
    // pi*a*b*c/h * (1 - d²/h²). Use depth instead of subtracting
    // nearly equal squares for shallow contacts.
    let h=support_height(s,n);
    let indentation=clamp(depth,0.0001,h);
    let fraction=indentation/h;
    return 3.14159265*s.axes.x*s.axes.y*s.axes.z/h*fraction*(2.0-fraction);
}
fn terrain_contact(s:ShapePose)->Contact{
    // Transform the triangle into the ellipsoid's isotropic metric.
    // Triangle-sphere intersection there is exact even on edges/slopes.
    // Convert the contact normal back with the inverse transpose.
    let margin=vec2<f32>(s.radius)+config.projection.grid.zw;
    let lo=vec2<i32>(floor((s.center.xy-margin)/config.spacing));
    let hi=vec2<i32>(floor((s.center.xy+margin)/config.spacing));
    let limit=i32(config.side)-2;
    var best=Contact(vec3<f32>(0.0,0.0,1.0),0.0,vec3<f32>(0.0),0u);
    let bottom=s.center.z-support_height(s,vec3<f32>(0.0,0.0,1.0));
    for(var y=max(lo.y,0);y<=min(hi.y,limit);y+=1){
        for(var x=max(lo.x,0);x<=min(hi.x,limit);x+=1){
            let a=vertex(u32(x),u32(y));let b=vertex(u32(x+1),u32(y));
            let c=vertex(u32(x),u32(y+1));let d=vertex(u32(x+1),u32(y+1));
            if bottom>max(max(a.z,b.z),max(c.z,d.z)){continue;}
            let ma=metric(s,a);
            for(var face=0u;face<2u;face+=1u){
                let v1=select(d,b,face==0u);
                let v2=select(c,d,face==0u);
                let mb=metric(s,v1);let mc=metric(s,v2);
                let mn=normalize(cross(mb-ma,mc-ma));
                let point=closest_triangle(vec3<f32>(0.0),ma,mb,mc);
                let distance=length(point);
                let signed=dot(-point,mn);
                let material=cell_material(u32(x),u32(y));
                if distance<s.min_axis && signed>=0.0{
                    let normal=metric_normal(s,select(mn,-point/max(distance,1e-8),distance>1e-6));
                    let support=world_point(s,point);
                    let depth=support_height(s,normal)-dot(s.center-support,normal);
                    if depth>best.depth{best=Contact(normal,depth,support,material);}
                }else if signed<0.0{
                    // Solid heightfield below the top surface. Recover
                    // embedded centers via the actual triangle plane.
                    let normal=normalize(cross(v1-a,v2-a));
                    let edge0=v1.xy-a.xy;let edge1=v2.xy-a.xy;
                    let q=s.center.xy-a.xy;
                    let denom=edge0.x*edge1.y-edge0.y*edge1.x;
                    if abs(denom)<1e-12 {continue;}
                    let u=(q.x*edge1.y-q.y*edge1.x)/denom;
                    let v=(edge0.x*q.y-edge0.y*q.x)/denom;
                    if u>=0.0&&v>=0.0&&u+v<=1.0{
                        let signed_depth=dot(s.center-a,normal);
                        let depth=support_height(s,normal)-signed_depth;
                        if depth>best.depth{best=Contact(normal,depth,s.center-normal*signed_depth,material);}
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
fn prepare(b:Body,collider:Collider,shape:u32)->Constraint{
    let s=shape_pose(b,collider);
    var contact=terrain_contact(s);
    let obstacle=static_contact(s);
    if obstacle.depth>contact.depth {contact=obstacle;}
    if contact.depth<=0.0{
        return Constraint(vec3<f32>(0.0),0.0,vec3<f32>(0.0),0.0,vec3<f32>(0.0),0.0,vec3<f32>(0.0),0.0,vec3<f32>(0.0),0.0,0.0,shape,0.0,0u);
    }
    let n=contact.normal;
    let arm=s.center-support_arm(s,n)-b.position.xyz;
    let velocity=b.velocity.xyz+cross(b.angular.xyz,arm);
    let vn=dot(velocity,n);
    let slip=velocity-n*vn;
    let bias=min(2.0,max(contact.depth-config.slop,0.0)*0.22/config.dt);
    let bounce=select(0.0,-vn*b.angular.w,vn < -0.8);
    let area=loaded_area(s,n,contact.depth);
    return Constraint(n,max(bias,bounce),arm,1.0/max(effective_mass(b,arm,n),1e-8),vec3<f32>(0.0),0.0,contact.point,contact.depth,slip,0.0,area,shape,bounce,contact.material);
}
// The normal impulse enters both velocity and the pressure-dependent
// equilibrium depth. Solve the monotone rational equation exactly rather
// than iterating a stiff explicit target (which can oscillate at high mass).
fn compliant_impulse(c:Constraint,vn:f32,material:vec2<f32>)->f32{
    let k=1.0/c.inv_effective_mass;
    let base=vn-k*c.normal_impulse;
    let beta=0.22/config.dt;
    let h=max(material.y*c.area*config.dt,1e-12);
    let raw=beta*(c.depth-config.slop);
    var root=0.0;
    if raw>base{
        let b=base+k*h-raw+beta*material.x;
        let q=(raw-base)*h;
        let discriminant=sqrt(max(0.0,b*b+4.0*k*q));
        if b>=0.0{
            root=2.0*q/max(b+discriminant,1e-20);
        }else{
            root=(discriminant-b)/(2.0*k);
        }
    }
    let minimum=select(-2.0,c.bounce,c.bounce>0.0);
    let maximum=max(2.0,c.bounce);
    return clamp(root,max(0.0,(minimum-base)/k),max(0.0,(maximum-base)/k));
}
fn solve(b:ptr<function,Body>,constraint:Constraint)->Constraint{
    var c=constraint;
    let n=c.normal;
    let r=c.arm;
    let relative=(*b).velocity.xyz+cross((*b).angular.xyz,r);
    let vn=dot(relative,n);
    // Project the accumulated normal impulse onto the unilateral cone.
    // Subsequent iterations can undo an excessive prior impulse.
    var next_normal=max(0.0,c.normal_impulse+(c.desired_speed-vn)*c.inv_effective_mass);
    let material=materials[c.material];
    if material.x>0.0{next_normal=compliant_impulse(c,vn,material);}
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
        let s=shape_pose(b,shapes[b.metadata.x+j]);
        let margin=vec2<f32>(support_height(s,vec3<f32>(1.0,0.0,0.0)),support_height(s,vec3<f32>(0.0,1.0,0.0)))+config.projection.grid.zw;
        if any(s.center.xy-margin<vec2<f32>(0.0))||any(s.center.xy+margin>=vec2<f32>(extent)){return false;}
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
        let s=shapes[b.metadata.x+j];
        let axes=s.axes.xyz;
        min_radius=min(min_radius,min(axes.x,min(axes.y,axes.z)));
        max_arm=max(max_arm,length(s.center.xyz)+max(axes.x,max(axes.y,axes.z)));
    }
    for(var tick=0u;tick<config.steps;tick+=1u){
        let previous=b;
        let radial=vec3<f32>(b.position.xy-vec2<f32>(config.projection.grid.y),
                             b.position.z+config.projection.sphere.x);
        b.velocity=vec4<f32>(b.velocity.xyz-normalize(radial)*config.gravity*config.dt,b.velocity.w);
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
            let c=prepare(b,shapes[b.metadata.x+j],j);
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
        if config.record_contacts!=0u{
            for(var j=0u;j<count;j+=1u){
                let c=constraints[j];
                let batch_tick=b.clock.x-config.first_tick_low-1u;
                let index=(batch_tick*config.count+id.x)*config.contact_stride+c.shape;
                contact_events[index]=ContactEvent(vec4<f32>(c.point,c.normal_impulse/config.dt),
                    vec4<f32>(c.normal,c.depth),b.rotation,vec4<f32>(c.incident_slip,c.area),
                    vec4<u32>(b.clock.xy,b.metadata.x+c.shape,select(0u,c.material+1u,c.normal_impulse>0.0)));
            }
        }
        if count>0u && b.metadata.w==0u {
            b.metadata.w=1u;
            b.clock.z=b.clock.x;
            b.clock.w=b.clock.y;
            if config.stop_at_contact!=0u{b.metadata.z=3u;break;}
        }
    }
    bodies[id.x]=b;
}
