// General compound contacts: immutable input state, per-body output,
// symmetric contact weights and canonical narrow-phase ordering.
struct PairConfig { restitution:u32, padding0:u32, padding1:u32, padding2:u32 }
@group(0) @binding(0) var<storage,read> input_bodies:array<Body>;
@group(0) @binding(1) var<storage,read> pair_shapes:array<Collider>;
@group(0) @binding(2) var<storage,read_write> degrees:array<u32>;
@group(0) @binding(3) var<storage,read> collision_groups:array<u32>;
@group(0) @binding(4) var<storage,read_write> output_bodies:array<Body>;
@group(0) @binding(5) var<uniform> pair_config:PairConfig;

fn shape_contact(a:Body,b:Body,sa:u32,sb:u32,forward:bool)->PairContact{
    let pa=pair_shape(a,pair_shapes[sa]);
    let pb=pair_shape(b,pair_shapes[sb]);
    if forward{return ellipsoid_contact(pa,pb);}
    let c=ellipsoid_contact(pb,pa);
    return PairContact(-c.normal,c.point,c.depth);
}
fn active_pair(a:Body,b:Body)->bool{
    return a.metadata.z==0u && b.metadata.z==0u && all(a.clock.xy==b.clock.xy);
}
@compute @workgroup_size(64)
fn count_neighbors(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&input_bodies){return;}
    let a=input_bodies[i];
    var degree=0u;
    if a.metadata.z==0u{
        for(var j=0u;j<arrayLength(&input_bodies);j+=1u){
            let b=input_bodies[j];
            if collision_groups[i]==collision_groups[j] || !active_pair(a,b){continue;}
            for(var sa=a.metadata.x;sa<a.metadata.x+a.metadata.y;sa+=1u){
                for(var sb=b.metadata.x;sb<b.metadata.x+b.metadata.y;sb+=1u){
                    let c=shape_contact(a,b,sa,sb,i<j);
                    degree+=u32(c.depth>0.0);
                }
            }
        }
    }
    degrees[i]=degree;
}
@compute @workgroup_size(64)
fn resolve_pairs(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;
    if i>=arrayLength(&input_bodies){return;}
    let a=input_bodies[i];
    if a.metadata.z!=0u || degrees[i]==0u{
        output_bodies[i]=a;
        return;
    }
    var impulse=vec3<f32>(0.0);
    var torque=vec3<f32>(0.0);
    var correction=vec3<f32>(0.0);
    for(var j=0u;j<arrayLength(&input_bodies);j+=1u){
        let b=input_bodies[j];
        if collision_groups[i]==collision_groups[j] || degrees[j]==0u || !active_pair(a,b){continue;}
        let weight=1.0/f32(max(degrees[i],degrees[j]));
        for(var sa=a.metadata.x;sa<a.metadata.x+a.metadata.y;sa+=1u){
            for(var sb=b.metadata.x;sb<b.metadata.x+b.metadata.y;sb+=1u){
                let c=shape_contact(a,b,sa,sb,i<j);
                if c.depth<=0.0{continue;}
                let effect=compound_effect(a,b,c,weight,pair_config.restitution!=0u);
                impulse+=effect.impulse;
                torque+=effect.torque;
                correction+=effect.correction;
            }
        }
    }
    output_bodies[i]=apply_pair_effect(a,impulse,torque,correction);
}
