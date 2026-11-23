// GPU foliage experiment: procedural, shared modules and expanded mesh.
// Same source patches and fragment shader across all draw modes.
struct Scene {
  view: vec4<f32>,       // focal_y, aspect, camera_y, far
  counts: vec4<u32>,     // clumps, model count, lod-side, trees
  switches: vec4<f32>,   // crossfade, width, height, camera_offset_z
};
struct Tree { position_scale:vec4<f32>, ids:vec4<u32>, tint:vec4<f32> };
struct Patch { center_radius:vec4<f32>, orient:vec4<f32> };
struct BakedVertex { position_uvx:vec4<f32>, uvy_brightness_material:vec4<f32> };
@group(0) @binding(0) var<uniform> scene:Scene;
@group(0) @binding(1) var<storage, read> trees:array<Tree>;
@group(0) @binding(2) var<storage, read> patches:array<Patch>;
@group(0) @binding(3) var<storage, read> expanded:array<BakedVertex>;
fn hash(value:u32)->u32 {
 var x=value;
 x=(x^(x>>16u))*0x7feb352du;
 x=(x^(x>>15u))*0x846ca68bu;
 return x^(x>>16u);
}
fn random(seed:u32, salt:u32)->f32 {
 return f32(hash(seed^salt)&0xffffu)/65536.0;
}
fn describe(seed:u32,part:u32,kind:u32,clumps:u32)->Patch {
 var module_data:Patch;
 if part==0u {
   module_data.center_radius=vec4f(0.,0.45,0.,0.065);
   module_data.orient=vec4f(0.,0.90,1.,1.);
   return module_data;
 }
 let cluster=(part-1u)/2u;
 let side=(part-1u)%2u;
 let parent=cluster%8u;
 let children=(clumps+7u-parent)/8u;
 let a=random(seed,parent*0x9e3779b9u+21u);
 let b=random(seed,parent*0x85ebca6bu+77u);
 let c=random(seed,parent*0xc2b2ae35u+135u);
 let d=random(seed,parent*0x27d4eb2fu+389u);
 let angle=6.283185307*a;
 let cone=select(0.,1.,kind==1u);
 let shrub=select(0.,1.,kind==2u);
 let radial=mix(0.48*sqrt(b),0.48*(1.0-c),cone);
 let altitude=mix(0.64+0.5*(c-0.5),0.19+0.68*c,cone);
 let r=mix(radial,0.44*sqrt(b),shrub);
 let y_base=mix(altitude,0.27+0.36*c,shrub);
 var pos=vec3f(r*cos(angle),y_base,r*sin(angle));
 let base_radius=select(0.36,0.235,kind==1u);
 if children>1u {
    let a2=6.283185307*random(seed,cluster*0xbf58476du+531u);
    let step=base_radius*0.5*sqrt(random(seed,cluster*0x94d049bbu+227u));
    pos.x+=step*cos(a2);
    pos.z+=step*sin(a2);
    pos.y+=step*0.35*(random(seed,cluster*0x87f53u+155u)-0.5);
 }
 let coverage_scale=select(select(1.,1.45,children>1u),2.20,children>3u);
 let radius=base_radius*inverseSqrt(f32(children))*coverage_scale;
 module_data.center_radius=vec4f(pos,radius);
 module_data.orient=vec4f(angle+f32(side)*1.5707963,radius*(1.5+0.45*d),0.78+0.34*b,0.);
 return module_data;
}
struct Output {
 @builtin(position) position:vec4<f32>,
 @location(0) uv:vec2<f32>,
 @location(1) color:vec3<f32>,
 @location(2) @interpolate(flat) id:u32,
 @location(3) @interpolate(flat) trunk:u32,
};
fn corner(i:u32)->vec2<f32> {
 switch(i) {
  case 0u:{return vec2f(-1.,-1.);}
  case 1u,4u:{return vec2f(1.,-1.);}
  case 2u,3u:{return vec2f(-1.,1.);}
  default:{return vec2f(1.,1.);}
 }
}
fn write_vertex(local:vec3<f32>,uv:vec2<f32>,brightness:f32,trunk:bool,tree:Tree)->Output {
 var out:Output;
 let world=tree.position_scale.xyz+local*tree.position_scale.w;
 let z=world.z+scene.switches.w;
 out.position=vec4f(world.x*scene.view.x/scene.view.y,(world.y-scene.view.z)*scene.view.x,z-0.1,z);
 let bark=vec3f(0.41,0.27,0.14);
 let foliage=select(vec3f(0.18,0.38,0.13),vec3f(0.15,0.30,0.18),tree.ids.z==1u);
 out.color=select(foliage*tree.tint.xyz*brightness,bark*brightness,trunk);
 out.uv=uv;
 out.id=tree.ids.x;
 out.trunk=select(0u,1u,trunk);
 return out;
}
fn expand(module_data:Patch,i:u32,tree:Tree)->Output {
 let uv=corner(i);
 let angle=module_data.orient.x;
 let r=module_data.center_radius.w;
 let local=module_data.center_radius.xyz+vec3f(uv.x*cos(angle)*r,uv.y*module_data.orient.y*0.5,uv.x*sin(angle)*r);
 return write_vertex(local,uv,module_data.orient.z,module_data.orient.w>0.5,tree);
}
@vertex fn vs_generated(@builtin(vertex_index) vertex:u32,@builtin(instance_index) id:u32)->Output {
 let tree=trees[id];
 return expand(describe(tree.ids.x,vertex/6u,tree.ids.z,scene.counts.x),vertex%6u,tree);
}
@vertex fn vs_limited(@builtin(vertex_index) vertex:u32,@builtin(instance_index) id:u32)->Output {
 let tree=trees[id];
 return expand(describe(tree.ids.y^0x56cb1a93u,vertex/6u,tree.ids.z,scene.counts.x),vertex%6u,tree);
}
@vertex fn vs_modules(@builtin(vertex_index) vertex:u32,@builtin(instance_index) id:u32)->Output {
 let tree=trees[id];
 let patch_count=scene.counts.x*2u+1u;
 let module_data=patches[tree.ids.y*patch_count+vertex/6u];
 return expand(module_data,vertex%6u,tree);
}
@vertex fn vs_expanded(@builtin(vertex_index) vertex:u32,@builtin(instance_index) id:u32)->Output {
 let tree=trees[id];
 let count=(scene.counts.x*2u+1u)*6u;
 let vertex_data=expanded[tree.ids.y*count+vertex];
 return write_vertex(vertex_data.position_uvx.xyz,
  vec2f(vertex_data.position_uvx.w,vertex_data.uvy_brightness_material.x),
  vertex_data.uvy_brightness_material.y,
  vertex_data.uvy_brightness_material.z>0.5,tree);
}
@fragment fn fs_main(v:Output)->@location(0) vec4<f32> {
 if v.trunk==0u {
  let r=length(v.uv*vec2f(0.95,1.05));
  let teeth=sin(v.uv.x*11.0+f32(v.id%23u))*sin(v.uv.y*14.0+f32(v.id%31u));
  if r>0.85+teeth*0.09 {discard;}
 }
 if scene.counts.z!=0u {
   let pixel=vec2u(v.position.xy);
   let threshold=random(v.id^(pixel.x*0x9e3779b9u),pixel.y*0x85ebca6bu);
   if scene.counts.z==1u && threshold<scene.switches.x {discard;}
   if scene.counts.z==2u && threshold>=scene.switches.x {discard;}
 }
 return vec4f(v.color*(0.78+0.14*v.uv.y),1.);
}
