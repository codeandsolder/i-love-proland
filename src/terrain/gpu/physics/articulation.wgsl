// Each invocation exclusively owns one connected island. Constraint
// scratch is sized by the real topology and lives in GPU memory.
struct JointRecord {
    anchor_a:vec4<f32>,anchor_b:vec4<f32>,
    frame_a:vec4<f32>,frame_b:vec4<f32>,
    limits:vec4<f32>,drive:vec4<f32>,meta:vec4<u32>,
}
struct ArticulationScratch{
    contact:Constraint,
    impulses:vec4<f32>,
}
@group(0) @binding(7) var<storage,read> topology:array<JointRecord>;
@group(0) @binding(8) var<storage,read_write> scratch:array<ArticulationScratch>;

fn inverse_quat(q:vec4<f32>)->vec4<f32>{return vec4<f32>(-q.xyz,q.w);}
fn rotation_vector(q:vec4<f32>)->vec3<f32>{
    let shortest=select(q,-q,q.w<0.0);
    let sine=length(shortest.xyz);
    if sine<1e-6{return 2.0*shortest.xyz;}
    return shortest.xyz*(2.0*atan2(sine,shortest.w)/sine);
}
fn angular_mass(a:Body,b:Body,axis:vec3<f32>)->f32{
    return dot(axis,inverse_world_inertia(a,axis)+inverse_world_inertia(b,axis));
}
fn apply_angular(a:ptr<function,Body>,b:ptr<function,Body>,impulse:vec3<f32>){
    (*a).angular=vec4<f32>((*a).angular.xyz-inverse_world_inertia(*a,impulse),(*a).angular.w);
    (*b).angular=vec4<f32>((*b).angular.xyz+inverse_world_inertia(*b,impulse),(*b).angular.w);
}
struct LinearAxis{axis:vec3<f32>,ra:vec3<f32>,rb:vec3<f32>}
fn linear_speed(a:Body,b:Body,axis:LinearAxis)->f32{
    return dot(axis.axis,b.velocity.xyz+cross(b.angular.xyz,axis.rb)-a.velocity.xyz-cross(a.angular.xyz,axis.ra));
}
fn linear_mass(a:Body,b:Body,axis:LinearAxis)->f32{
    return effective_mass(a,axis.ra,axis.axis)+effective_mass(b,axis.rb,axis.axis);
}
fn apply_linear(a:ptr<function,Body>,b:ptr<function,Body>,axis:LinearAxis,impulse:f32){
    apply_impulse(a,axis.ra,-axis.axis*impulse);
    apply_impulse(b,axis.rb,axis.axis*impulse);
}
fn joint_bias(error:f32)->f32{return clamp(-0.2*error/config.dt,-4.0,4.0);}
fn lock_linear(a:ptr<function,Body>,b:ptr<function,Body>,axis:LinearAxis,error:f32){
    let mass=max(linear_mass(*a,*b,axis),1e-12);
    apply_linear(a,b,axis,(joint_bias(error)-linear_speed(*a,*b,axis))/mass);
}
fn lock_angular(a:ptr<function,Body>,b:ptr<function,Body>,axis:vec3<f32>,error:f32){
    let speed=dot((*b).angular.xyz-(*a).angular.xyz,axis);
    let mass=max(angular_mass(*a,*b,axis),1e-12);
    apply_angular(a,b,axis*(joint_bias(error)-speed)/mass);
}
// Implicit force-based spring-damper motor. The accumulated impulse is
// capped once per physical step, independent of iteration count.
fn motor_impulse(joint:JointRecord,position:f32,velocity:f32,mass:f32,old:f32)->f32{
    let stiffness=joint.limits.z;
    let damping=joint.limits.w;
    let denominator=damping+config.dt*stiffness;
    if denominator<=0.0||joint.drive.z<=0.0{return 0.0;}
    let softness=1.0/(config.dt*denominator);
    let desired=joint.drive.y+stiffness*(joint.drive.x-position)/denominator;
    let maximum=joint.drive.z*config.dt;
    return clamp(old+(desired-velocity-softness*old)/(mass+softness),-maximum,maximum);
}
fn limit_impulse(position:f32,velocity:f32,mass:f32,bound:f32,old:f32,sign:f32)->f32{
    // If already outside travel, recover gradually. Inside, limit the
    // next integrated step only when it would cross the hard stop.
    let error=(position-bound)*sign;
    let desired=select(-error/config.dt,joint_bias(error),error<0.0);
    return max(0.0,old+(desired-sign*velocity)/mass);
}
fn free_linear(a:ptr<function,Body>,b:ptr<function,Body>,axis:LinearAxis,joint:JointRecord,position:f32,old:vec4<f32>)->vec4<f32>{
    let mass=max(linear_mass(*a,*b,axis),1e-12);
    var next=old;
    next.x=motor_impulse(joint,position,linear_speed(*a,*b,axis),mass,old.x);
    apply_linear(a,b,axis,next.x-old.x);
    if joint.meta.w!=0u{
        next.y=limit_impulse(position,linear_speed(*a,*b,axis),mass,joint.limits.x,old.y,1.0);
        apply_linear(a,b,axis,next.y-old.y);
        next.z=limit_impulse(position,linear_speed(*a,*b,axis),mass,joint.limits.y,old.z,-1.0);
        apply_linear(a,b,axis,old.z-next.z);
    }
    return next;
}
fn free_angular(a:ptr<function,Body>,b:ptr<function,Body>,axis:vec3<f32>,joint:JointRecord,position:f32,old:vec4<f32>)->vec4<f32>{
    let mass=max(angular_mass(*a,*b,axis),1e-12);
    var next=old;
    // Position targets use shortest angular error for unlimited hinges.
    var adjusted=joint;
    if joint.meta.w==0u{
        let delta=joint.drive.x-position;
        adjusted.drive.x=position+atan2(sin(delta),cos(delta));
    }
    next.x=motor_impulse(adjusted,position,dot((*b).angular.xyz-(*a).angular.xyz,axis),mass,old.x);
    apply_angular(a,b,axis*(next.x-old.x));
    if joint.meta.w!=0u{
        next.y=limit_impulse(position,dot((*b).angular.xyz-(*a).angular.xyz,axis),mass,joint.limits.x,old.y,1.0);
        apply_angular(a,b,axis*(next.y-old.y));
        next.z=limit_impulse(position,dot((*b).angular.xyz-(*a).angular.xyz,axis),mass,joint.limits.y,old.z,-1.0);
        apply_angular(a,b,axis*(old.z-next.z));
    }
    return next;
}
fn solve_joint(index:u32,joint:JointRecord){
    var a=bodies[joint.meta.x];
    var b=bodies[joint.meta.y];
    let qa=quat_mul(a.rotation,joint.frame_a);
    let qb=quat_mul(b.rotation,joint.frame_b);
    let rb=quat_rotate(b.rotation,joint.anchor_b.xyz);
    let pb=b.position.xyz+rb;
    let pa=a.position.xyz+quat_rotate(a.rotation,joint.anchor_a.xyz);
    let delta=pb-pa;
    // The translation axes rotate with A. Its Jacobian includes the
    // moving-axis derivative, i.e. apply the reaction at B's anchor.
    let ra=pb-a.position.xyz;
    let axes=array<vec3<f32>,3>(quat_rotate(qa,vec3<f32>(1.0,0.0,0.0)),quat_rotate(qa,vec3<f32>(0.0,1.0,0.0)),quat_rotate(qa,vec3<f32>(0.0,0.0,1.0)));
    let kind=joint.meta.z;
    let first=select(0u,1u,kind==1u);
    for(var axis=first;axis<3u;axis+=1u){
        lock_linear(&a,&b,LinearAxis(axes[axis],ra,rb),dot(delta,axes[axis]));
    }
    if kind==0u||kind==1u{
        let error=rotation_vector(quat_mul(qb,inverse_quat(qa)));
        for(var axis=0u;axis<3u;axis+=1u){lock_angular(&a,&b,axes[axis],dot(error,axes[axis]));}
    }
    if kind==1u{
        scratch[index].impulses=free_linear(&a,&b,LinearAxis(axes[0],ra,rb),joint,dot(delta,axes[0]),scratch[index].impulses);
    }
    if kind==2u{
        let bx=quat_rotate(qb,vec3<f32>(1.0,0.0,0.0));
        let swing=cross(axes[0],bx);
        let sine=length(swing);
        var error=vec3<f32>(0.0);
        if sine>1e-6{error=swing*(atan2(sine,dot(axes[0],bx))/sine);}
        else if dot(axes[0],bx)<0.0{error=axes[1]*3.14159265;}
        lock_angular(&a,&b,axes[1],dot(error,axes[1]));
        lock_angular(&a,&b,axes[2],dot(error,axes[2]));
        let by=quat_rotate(qb,vec3<f32>(0.0,1.0,0.0));
        let angle=atan2(dot(by,axes[2]),dot(by,axes[1]));
        scratch[index].impulses=free_angular(&a,&b,axes[0],joint,angle,scratch[index].impulses);
    }
    bodies[joint.meta.x]=a;
    bodies[joint.meta.y]=b;
}
struct Integrated{body:Body,status:u32}
fn propose(b:Body)->Integrated{
    var next=b;
    let radial=vec3<f32>(b.position.xy-vec2<f32>(config.projection.grid.y),b.position.z+config.projection.sphere.x);
    next.velocity=vec4<f32>(b.velocity.xyz-normalize(radial)*config.gravity*config.dt,b.velocity.w);
    var min_radius=1e20;
    var max_arm=0.0;
    for(var j=0u;j<b.metadata.y;j+=1u){
        let shape=shapes[b.metadata.x+j];
        let axes=shape.axes.xyz;
        min_radius=min(min_radius,min(axes.x,min(axes.y,axes.z)));
        max_arm=max(max_arm,length(shape.center.xyz)+max(axes.x,max(axes.y,axes.z)));
    }
    if (length(next.velocity.xyz)+length(b.angular.xyz)*max_arm)*config.dt>min_radius*0.5{return Integrated(b,2u);}
    next.position=vec4<f32>(b.position.xyz+next.velocity.xyz*config.dt,b.position.w);
    next.rotation=normalize(b.rotation+0.5*config.dt*quat_mul(vec4<f32>(b.angular.xyz,0.0),b.rotation));
    if !covered(next){return Integrated(b,1u);}
    return Integrated(next,0u);
}

@compute @workgroup_size(1)
fn advance_island(@builtin(global_invocation_id) invocation:vec3<u32>){
    let header=topology[0].meta;
    if invocation.x>=header.x{return;}
    let island=topology[1u+invocation.x].meta;
    let body_end=island.x+island.y;
    let joint_end=island.z+island.w;
    let joint_base=1u+header.x;
    let scratch_base=config.count*config.contact_stride;
    if bodies[island.x].metadata.z!=0u{return;}
    for(var tick=0u;tick<config.steps;tick+=1u){
        var status=0u;
        // An island commits a whole tick or none. Preflight all bodies
        // before publishing any integrated positions.
        for(var i=island.x;i<body_end;i+=1u){status=max(status,propose(bodies[i]).status);}
        if status!=0u{
            for(var i=island.x;i<body_end;i+=1u){bodies[i].metadata.z=status;}
            break;
        }
        for(var i=island.x;i<body_end;i+=1u){
            let b=propose(bodies[i]).body;
            bodies[i]=b;
            for(var j=0u;j<b.metadata.y;j+=1u){
                scratch[i*config.contact_stride+j].contact=prepare(b,shapes[b.metadata.x+j],j);
            }
        }
        for(var j=island.z;j<joint_end;j+=1u){scratch[scratch_base+j].impulses=vec4<f32>(0.0);}
        for(var iteration=0u;iteration<config.iterations;iteration+=1u){
            for(var i=island.x;i<body_end;i+=1u){
                var b=bodies[i];
                for(var j=0u;j<b.metadata.y;j+=1u){
                    let index=i*config.contact_stride+j;
                    let c=scratch[index].contact;
                    if c.inv_effective_mass>0.0{scratch[index].contact=solve(&b,c);}
                }
                bodies[i]=b;
            }
            for(var j=island.z;j<joint_end;j+=1u){solve_joint(scratch_base+j,topology[joint_base+j]);}
        }
        var stop=false;
        for(var i=island.x;i<body_end;i+=1u){
            var b=bodies[i];
            b.clock.x+=1u;
            if b.clock.x==0u{b.clock.y+=1u;}
            var contact=false;
            for(var j=0u;j<b.metadata.y;j+=1u){
                let c=scratch[i*config.contact_stride+j].contact;
                if c.inv_effective_mass<=0.0{continue;}
                contact=true;
                if config.record_contacts!=0u{
                    let index=(tick*config.count+i)*config.contact_stride+j;
                    contact_events[index]=ContactEvent(vec4<f32>(c.point,c.normal_impulse/config.dt),vec4<f32>(c.normal,c.depth),
                        b.rotation,vec4<f32>(c.incident_slip,c.area),vec4<u32>(b.clock.xy,b.metadata.x+j,select(0u,c.material+1u,c.normal_impulse>0.0)));
                }
            }
            if contact&&b.metadata.w==0u{
                b.metadata.w=1u;
                b.clock.zw=b.clock.xy;
                stop=stop||config.stop_at_contact!=0u;
            }
            bodies[i]=b;
        }
        if stop{
            for(var i=island.x;i<body_end;i+=1u){bodies[i].metadata.z=3u;}
            break;
        }
    }
}
