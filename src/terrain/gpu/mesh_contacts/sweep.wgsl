// Earliest conservative interval for linear translation + shortest-arc
// rigid rotation, with an explicit extra trajectory deviation allowance.
// No collision can disappear solely because both endpoint poses are clear.
struct RigidPose { position:vec4<f32>, rotation:vec4<f32> }
struct SweepPair { mesh_ids:vec2<u32>, padding:vec2<u32>, poses:array<RigidPose,4> }
struct SweepConfig { margin:f32, tolerance:f32, deviation:f32, budget:u32 }
struct SweepResult {
    interval:vec4<f32>, // clear-before, possible-until, sampled time, sample margin
    status:vec4<u32>,  // clear=0, candidate=1, invalid=2, budget=3
    contact:Contact,
}
@group(0) @binding(3) var<storage,read> sweeps:array<SweepPair>;
@group(0) @binding(4) var<storage,read_write> results:array<SweepResult>;
@group(0) @binding(5) var<uniform> config:SweepConfig;

fn valid_pose(p:RigidPose)->bool{
    return all(abs(p.position.xyz)<=vec3<f32>(1.0e7)) &&
        all(abs(p.rotation)<=vec4<f32>(1.001)) && abs(dot(p.rotation,p.rotation)-1.0)<0.001;
}
fn rotate(q:vec4<f32>,p:vec3<f32>)->vec3<f32>{
    return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);
}
fn interpolate(a:RigidPose,b:RigidPose,t:f32)->mat4x4<f32>{
    let qa=normalize(a.rotation);
    let qb=normalize(b.rotation);
    let signed=dot(qa,qb);
    let end=select(-qb,qb,signed>=0.0);
    let cosine=clamp(abs(signed),0.0,1.0);
    var q=normalize(mix(qa,end,t));
    if cosine<0.9995{
        let theta=acos(cosine);
        q=normalize((sin((1.0-t)*theta)*qa+sin(t*theta)*end)/sin(theta));
    }
    return mat4x4<f32>(vec4<f32>(rotate(q,vec3<f32>(1.0,0.0,0.0)),0.0),
                       vec4<f32>(rotate(q,vec3<f32>(0.0,1.0,0.0)),0.0),
                       vec4<f32>(rotate(q,vec3<f32>(0.0,0.0,1.0)),0.0),
                       vec4<f32>(mix(a.position.xyz,b.position.xyz,t),1.0));
}
fn mesh_radius(id:u32)->f32{
    let root=nodes[meshes[id].nodes.x];
    return length(max(abs(root.lo),abs(root.hi)));
}
fn travel(a:RigidPose,b:RigidPose,radius:f32)->f32{
    let qa=normalize(a.rotation);
    let qb=normalize(b.rotation);
    // Chord form avoids acos loss of significance for small rotations.
    let end=select(-qb,qb,dot(qa,qb)>=0.0);
    let angle=4.0*asin(clamp(length(qa-end)*0.5,0.0,1.0));
    // Near-identical quaternions use normalized lerp; its peak angular
    // velocity is bounded by 1.001 times the shortest-arc average.
    return length(a.position.xyz-b.position.xyz)+radius*angle*1.001;
}
struct Cursor { depth:u32, index:u32, done:u32, tests:u32 }
var<workgroup> cursor:Cursor;
@compute @workgroup_size(64)
fn sweep(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let pair=sweeps[group.x];
    if lane==0u{
        cursor=Cursor(0u,0u,0u,0u);
        if any(pair.mesh_ids>=vec2<u32>(arrayLength(&meshes))) ||
            !valid_pose(pair.poses[0])||!valid_pose(pair.poses[1])||
            !valid_pose(pair.poses[2])||!valid_pose(pair.poses[3]){
            cursor.done=2u;
            results[group.x]=SweepResult(vec4<f32>(0.0,1.0,0.0,0.0),vec4<u32>(2u,0u,0u,0u),empty());
        }
    }
    var state=workgroupUniformLoad(&cursor);
    if state.done!=0u{return;}
    let ra=mesh_radius(pair.mesh_ids.x);
    let rb=mesh_radius(pair.mesh_ids.y);
    let displacement=travel(pair.poses[0],pair.poses[1],ra)+travel(pair.poses[2],pair.poses[3],rb);
    // Include f32 transform and feature arithmetic error in the
    // conservative radius, independent of the temporal subdivision.
    let coordinate_scale=max(length(pair.poses[0].position.xyz),max(length(pair.poses[1].position.xyz),
        max(length(pair.poses[2].position.xyz),length(pair.poses[3].position.xyz))));
    let rounding=(ra+rb+coordinate_scale+1.0)*0.000008;
    loop{
        let width=1.0/f32(1u<<state.depth);
        let lo=f32(state.index)*width;
        let hi=lo+width;
        let mid=(lo+hi)*0.5;
        let margin=config.margin+displacement*width*0.5+config.deviation+rounding;
        let a=interpolate(pair.poses[0],pair.poses[1],mid);
        let b=interpolate(pair.poses[2],pair.poses[3],mid);
        var result=orient_contact(pair.mesh_ids,reduce_contact(scan(pair.mesh_ids,a,b,margin,lane),lane));
        if result.features.z==0u && result.features.w==0u {
            let inside=reduce_contact(containment(pair.mesh_ids,a,b,lane),lane);
            result.features.w=inside.features.w;
            result.statistics.w=inside.statistics.w;
        }
        if lane==0u{
            cursor.tests+=1u;
            if (result.features.w&3u)!=0u{
                cursor.done=2u;
                results[group.x]=SweepResult(vec4<f32>(lo,hi,mid,margin),vec4<u32>(2u,cursor.tests,0u,0u),result);
            }else if result.features.z!=0u||result.features.w!=0u{
                if width<=config.tolerance || cursor.tests>=config.budget{
                    cursor.done=1u;
                    let status=select(1u,3u,width>config.tolerance);
                    results[group.x]=SweepResult(vec4<f32>(lo,hi,mid,margin),vec4<u32>(status,cursor.tests,0u,0u),result);
                }else{
                    cursor.depth+=1u;
                    cursor.index*=2u;
                }
            }else{
                // Clear interval. Ascend completed right children and
                // move to the next chronological interval.
                while cursor.depth>0u&&(cursor.index&1u)!=0u{
                    cursor.depth-=1u;
                    cursor.index/=2u;
                }
                if cursor.depth==0u{
                    cursor.done=1u;
                    results[group.x]=SweepResult(vec4<f32>(1.0,1.0,1.0,0.0),vec4<u32>(0u,cursor.tests,0u,0u),empty());
                }else if cursor.tests>=config.budget{
                    cursor.done=1u;
                    results[group.x]=SweepResult(vec4<f32>(hi,1.0,mid,margin),vec4<u32>(3u,cursor.tests,0u,0u),empty());
                }else{
                    cursor.index+=1u;
                }
            }
        }
        state=workgroupUniformLoad(&cursor);
        if state.done!=0u{break;}
    }
}
