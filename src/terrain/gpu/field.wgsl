// Regional terrain evaluator. Integer chart addresses are converted to
// metres exactly once, independent of region_patch size, slot, and request order.
struct Config {
    extent_ticks: u32, control_side: u32, bin_side: u32, grid_side: u32,
    spacing_m: f32, detail: f32, seed: u32, padding: u32,
}
struct Reach {
    ends: vec4<f32>,       // ax, ay, bx, by
    stage_width: vec4<f32>,// stage a/b, width a/b
    bend_valley: vec4<f32>,
}
struct Job { origin: vec2<u32>, spacing: u32, slot: u32 }
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage,read> controls: array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> reaches: array<Reach>;
@group(0) @binding(3) var<storage,read> bins: array<u32>;
@group(0) @binding(4) var<storage,read> jobs: array<Job>;
@group(0) @binding(5) var<storage,read_write> output: array<TerrainSample>;
@group(0) @binding(6) var<storage,read> points: array<vec2<u32>>;

fn smooth_unit(t:f32)->f32 {let q=clamp(t,0.0,1.0);return q*q*(3.0-2.0*q);}
fn mix_hash(v:u32)->u32 {
    var h=v;
    h=(h^(h>>16u))*0x7feb352du;
    h=(h^(h>>15u))*0x846ca68bu;
    return h^(h>>16u);
}
fn lattice(p:vec2<i32>,seed:u32)->f32 {
    return f32(mix_hash(bitcast<u32>(p.x)*0x9e3779b9u ^ bitcast<u32>(p.y)*0x85ebca6bu ^ seed)>>8u)*(2.0/16777215.0)-1.0;
}
fn noise(p:vec2<f32>,seed:u32)->f32{
    let i=vec2<i32>(floor(p));
    let f=fract(p);
    let w=f*f*(3.0-2.0*f);
    return mix(mix(lattice(i,seed),lattice(i+vec2<i32>(1,0),seed),w.x),
               mix(lattice(i+vec2<i32>(0,1),seed),lattice(i+vec2<i32>(1,1),seed),w.x),w.y);
}
fn unpack_cell(cell:vec2<u32>)->vec4<f32>{
    let bed=f32(bitcast<i32>(cell.x<<16u)>>16u)*0.25;
    let lake=f32(bitcast<i32>(cell.x)>>16u)*0.25;
    return vec4<f32>(bed,max(max(lake,0.0),bed),0.0,0.0);
}
fn evaluate_controls(p:vec2<f32>)->vec4<f32>{
    let q=clamp(p/config.spacing_m,vec2<f32>(0.0),vec2<f32>(f32(config.control_side-1u)));
    let ij=min(vec2<u32>(floor(q)),vec2<u32>(config.control_side-2u));
    let f=q-vec2<f32>(ij);
    let idx=ij.y*config.control_side+ij.x;
    let a=controls[idx];let b=controls[idx+1u];
    let c=controls[idx+config.control_side];let d=controls[idx+config.control_side+1u];
    let aa=unpack_cell(a);let bb=unpack_cell(b);let cc=unpack_cell(c);let dd=unpack_cell(d);
    let v=mix(mix(aa.xy,bb.xy,f.x),mix(cc.xy,dd.xy,f.x),f.y);
    let grad=vec2<f32>(mix(bb.x-aa.x,dd.x-cc.x,f.y),mix(cc.x-aa.x,dd.x-bb.x,f.x))/config.spacing_m;
    return vec4<f32>(v,grad);
}
fn color_at(p:vec2<f32>)->vec3<f32>{
    let q=clamp(p/config.spacing_m,vec2<f32>(0.0),vec2<f32>(f32(config.control_side-1u)));
    let ij=min(vec2<u32>(floor(q)),vec2<u32>(config.control_side-2u));
    let f=q-vec2<f32>(ij);let i=ij.y*config.control_side+ij.x;
    return mix(mix(unpack4x8unorm(controls[i].y),unpack4x8unorm(controls[i+1u].y),f.x),
        mix(unpack4x8unorm(controls[i+config.control_side].y),unpack4x8unorm(controls[i+config.control_side+1u].y),f.x),f.y).rgb;
}
fn projection(r:Reach,p:vec2<f32>)->vec2<f32>{
    let axis=r.ends.zw-r.ends.xy;
    let delta=p-r.ends.xy;
    let t=clamp(dot(delta,axis)/max(dot(axis,axis),1e-9),0.0,1.0);
    let residual=delta-t*axis;
    return vec2<f32>(dot(residual,residual),t);
}
// Oriented ridge detail, progressively reduced on flat, wet and eroded
// surfaces. The control gradients are used only to rotate the ridges;
// no tile-local random seed or finite-difference across tile boundaries.
fn relief(p:vec2<f32>,g:vec2<f32>)->f32 {
    let slope=length(g);
    let axis=g/max(slope,0.0001);

    // Continuous world-coordinate noise (the slope-dependent amplitude
    // fades to zero on flats). Large bands remain anchored to the world.
    let broad=noise(p/96.0,config.seed);
    let ridge=1.0-abs(noise(p/23.0+vec2<f32>(broad*0.6),config.seed^17u));
    let small=noise(p/5.5,config.seed^31u);
    // Oriented relief is modulated rather than using a rotating coordinate
    // system as the sole field, which would make slopes unstable.
    let transverse=vec2<f32>(-axis.y,axis.x);
    let ribs=noise((p+transverse*18.0)/17.0,config.seed^73u)-noise((p-transverse*18.0)/17.0,config.seed^73u);
    return smooth_unit(slope/0.5)*(broad*7.0+(ridge-0.5)*3.0+small*0.65+ribs*0.4)*config.detail;
}
fn evaluate(p:vec2<f32>)->TerrainSample{
    let control=evaluate_controls(p);
    var bed=control.x;
    var water=bed;
    var kind=0u;
    var color=color_at(p);
    if control.y>bed+0.02 {
        return TerrainSample(bed,control.y,pack4x8unorm(vec4<f32>(color,1.0)),1u);
    }
    let delta=config.spacing_m*0.5;
    let g=vec2<f32>(
        evaluate_controls(p+vec2<f32>(delta,0.0)).x-evaluate_controls(p-vec2<f32>(delta,0.0)).x,
        evaluate_controls(p+vec2<f32>(0.0,delta)).x-evaluate_controls(p-vec2<f32>(0.0,delta)).x
    )/(2.0*delta);
    bed+=relief(p,g)*smooth_unit(bed/30.0);
    water=bed;
    if bed<=0.0 {return TerrainSample(bed,0.0,pack4x8unorm(vec4<f32>(color,1.0)),1u);}
    let bin=min(vec2<u32>(p/250.0),vec2<u32>(config.bin_side-1u));
    let bucket=bin.y*config.bin_side+bin.x;
    let begin=bins[bucket];let end=bins[bucket+1u];
    var nearest=670.0*670.0;
    for(var i=begin;i<end;i+=1u){
        nearest=min(nearest,projection(reaches[bins[i]],p).x);
    }
    if nearest>=670.0*670.0 {return TerrainSample(bed,water,pack4x8unorm(vec4<f32>(color,1.0)),kind);}
    let cross=sqrt(nearest);
    var center=vec2<f32>(0.0);var tangent=vec2<f32>(0.0);
    var sum=0.0;var stage=0.0;var width=0.0;var valley=0.0;var bend=0.0;
    for(var i=begin;i<end;i+=1u){
        let r=reaches[bins[i]];let proj=projection(r,p);
        if proj.x>(cross+18.0)*(cross+18.0){continue;}
        let t=proj.y;
        let w=mix(r.stage_width.z,r.stage_width.w,t);
        let direction=r.ends.zw-r.ends.xy;
        let len=length(direction);
        let weight=(1.0-smooth_unit((sqrt(proj.x)-cross)/clamp(w*0.16,4.0,18.0)))*max(len,0.01);
        sum+=weight;center+=weight*mix(r.ends.xy,r.ends.zw,t);
        tangent+=weight*direction/max(len,1e-6);
        stage+=weight*mix(r.stage_width.x,r.stage_width.y,t);
        width+=weight*w;
        valley+=weight*r.bend_valley.z;
        bend+=weight*mix(r.bend_valley.x,r.bend_valley.y,t);
    }
    if sum<=0.0 {return TerrainSample(bed,water,pack4x8unorm(vec4<f32>(color,1.0)),kind);}
    stage/=sum;valley/=sum;center/=sum;
    var radius=width/(2.0*sum);
    tangent/=max(length(tangent),1e-9);
    let offset=p-center;
    bend=bend/sum*clamp((tangent.x*offset.y-tangent.y*offset.x)/max(radius,1.0),-1.0,1.0)*valley;
    let inner=max(bend,0.0);let outer=max(-bend,0.0);
    let sediment=noise(p/65.0,0x534544u);
    radius*=1.0-0.27*inner+0.13*outer+0.06*sediment;
    let bank=(10.0+0.48*radius+24.0*valley)*(1.0+0.8*inner-0.3*outer);
    let terrace=25.0+radius*(0.6+2.0*valley)+100.0*valley;
    if cross>=radius+bank+terrace {return TerrainSample(bed,water,pack4x8unorm(vec4<f32>(color,1.0)),kind);}
    let depth=(0.65+0.068*radius)*(1.0+0.35*outer-0.35*inner);
    let rise=(2.0+0.08*radius)*(1.0+0.65*outer-0.45*inner);
    let u=cross/radius;
    let shelf=inner*smooth_unit((u-0.38)/0.47)*(1.0-smooth_unit((u-0.94)/0.06));
    let fine=noise(p/13.0,0x524950u);
    var desired_bed=stage-depth*(1.0-pow(min(u,1.0),3.5+2.0*outer-2.2*inner))+depth*0.78*shelf*(1.0+0.22*sediment)+0.32*shelf*fine;
    if cross>radius {
        let lateral=(cross-radius)/bank;
        let bar=inner*(1.0-smooth_unit(abs(lateral-0.3)/0.8));
        desired_bed=stage+rise*smooth_unit(lateral)+0.018*max(cross-radius-bank,0.0)
            +0.9*bar*smooth_unit(lateral*3.0)
            +(0.3+0.65*valley)*sediment*smooth_unit(lateral)*(1.0-smooth_unit(lateral/4.0))
            +(0.35+0.5*outer)*fine*smooth_unit(lateral*3.0)*(1.0-smooth_unit(lateral/3.0));
    }
    let support=1.0-smooth_unit((cross-radius-bank)/terrace);
    let base=bed;
    bed=clamp(mix(bed,desired_bed,support),bed-42.0*support,bed+6.0*support);
    let damp=(1.0-smooth_unit(abs(cross-radius)/bank))*support;
    let gravel=max(shelf,(1.0-smooth_unit(abs(cross-radius-bank*0.45)/(bank*1.4)))*support);
    color=mix(color,vec3<f32>(0.19,0.31,0.125),support*valley*0.30);
    color=mix(color,vec3<f32>(0.50,0.465,0.385),clamp(0.25*gravel+0.65*max(inner*gravel,shelf),0.0,1.0));
    color=mix(color,vec3<f32>(0.25,0.245,0.20),clamp(damp*(0.22+0.28*outer),0.0,1.0));
    water=bed;
    if cross<=radius && stage>bed+0.02 && stage<=base+6.0 {water=stage;kind=2u;}
    return TerrainSample(bed,water,pack4x8unorm(vec4<f32>(color,1.0)),kind);
}

@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) id:vec3<u32>){
    let job=jobs[id.y];
    let count=config.grid_side*config.grid_side;
    if id.x>=count {return;}
    let address=job.origin+vec2<u32>(id.x%config.grid_side,id.x/config.grid_side)*job.spacing;
    let p=vec2<f32>(address)*0.0625;
    output[job.slot*count+id.x]=evaluate(p);
}

// Planetmap's spherical grid can supply its chart-projected addresses in
// arbitrary order. The canonical evaluator is identical to regular grids.
@compute @workgroup_size(64)
fn generate_points(@builtin(global_invocation_id) id:vec3<u32>){
    let count=config.grid_side*config.grid_side;
    if id.x>=count {return;}
    let address=points[id.y*count+id.x];
    output[jobs[id.y].slot*count+id.x]=evaluate(vec2<f32>(address)*0.0625);
}
