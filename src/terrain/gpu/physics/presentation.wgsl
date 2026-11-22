// Independent presentation time samples two immutable GPU physics states.
// Absolute ticks are two u32 words. Subtract before converting to float,
// preserving fine interpolation even after years of simulation time.
struct Query { time:vec2<u32>, fraction:f32, count:u32 }
@group(0) @binding(0) var<storage,read> older:array<Body>;
@group(0) @binding(1) var<storage,read> newer:array<Body>;
@group(0) @binding(2) var<storage,read_write> poses:array<Pose>;
@group(0) @binding(3) var<uniform> query:Query;
fn before(a:vec2<u32>,b:vec2<u32>)->bool{
    return a.y<b.y || (a.y==b.y && a.x<b.x);
}
fn relative_ticks(a:vec2<u32>,b:vec2<u32>)->f32{
    let borrow=select(0u,1u,a.x<b.x);
    return f32(a.y-b.y-borrow)*4294967296.0+f32(a.x-b.x);
}
@compute @workgroup_size(64)
fn interpolate(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=query.count{return;}
    let a=older[id.x];
    let b=newer[id.x];
    var t=1.0;
    if before(a.clock.xy,b.clock.xy){
        if before(query.time,a.clock.xy){t=0.0;}
        else if before(query.time,b.clock.xy){
            t=clamp((relative_ticks(query.time,a.clock.xy)+query.fraction)/relative_ticks(b.clock.xy,a.clock.xy),0.0,1.0);
        }
    }
    // Shortest path interpolation; no extrapolation through uncomputed
    // physics, and no generation of new physical states in this pass.
    let q=select(b.rotation,-b.rotation,dot(a.rotation,b.rotation)<0.0);
    poses[id.x]=Pose(vec4<f32>(mix(a.position.xyz,b.position.xyz,t),1.0),normalize(mix(a.rotation,q,t)));
}
