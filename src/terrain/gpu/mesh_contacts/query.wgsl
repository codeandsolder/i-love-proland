struct Triangle { a:vec3<f32>, id:u32, b:vec4<f32>, c:vec4<f32> }
struct Node {
    lo:vec3<f32>, first:u32,
    hi:vec3<f32>, count:u32,
    escape:u32, mask_low:u32, mask_high:u32, end:u32,
}
struct Mesh { nodes:vec2<u32>, triangles:vec2<u32> }
struct Pair { meshes:vec2<u32>, transforms:vec2<u32> }
struct Contact {
    point_a:vec4<f32>,
    point_b:vec4<f32>,
    features:vec4<u32>,
    statistics:vec4<u32>,
}
struct Config { margin:f32, mode:u32, count:u32, transform_count:u32 }
@group(0) @binding(0) var<storage,read> nodes:array<Node>;
@group(0) @binding(1) var<storage,read> triangles:array<Triangle>;
@group(0) @binding(2) var<storage,read> meshes:array<Mesh>;
@group(0) @binding(3) var<storage,read> transforms:array<mat4x4<f32>>;
@group(0) @binding(4) var<storage,read> pairs:array<Pair>;
@group(0) @binding(5) var<storage,read_write> contacts:array<Contact>;
@group(0) @binding(6) var<uniform> config:Config;

const FAR=3.0e38;
struct Witness { a:vec3<f32>, b:vec3<f32>, distance2:f32 }
fn witness(a:vec3<f32>,b:vec3<f32>)->Witness{
    let d=a-b;
    return Witness(a,b,dot(d,d));
}
fn nearer(a:Witness,b:Witness)->Witness{
    if b.distance2<a.distance2{return b;}
    return a;
}
fn point_segment(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>)->vec3<f32>{
    let edge=b-a;
    let len2=dot(edge,edge);
    if len2==0.0{return a;}
    return a+edge*clamp(dot(p-a,edge)/len2,0.0,1.0);
}
// Independent feature tests, also defined for zero-area triangles.
fn inside(p:vec3<f32>,t:mat3x3<f32>,n:vec3<f32>)->bool{
    for(var i=0u;i<3u;i+=1u){
        let a=t[i];
        let b=t[(i+1u)%3u];
        if dot(cross(b-a,p-a),n)<0.0{return false;}
    }
    return true;
}
fn point_triangle(p:vec3<f32>,t:mat3x3<f32>)->Witness{
    let n=cross(t[1]-t[0],t[2]-t[0]);
    let n2=dot(n,n);
    if n2>0.0{
        let q=p-n*(dot(p-t[0],n)/n2);
        if inside(q,t,n){return witness(p,q);}
    }
    var best=witness(p,point_segment(p,t[0],t[1]));
    for(var i=1u;i<3u;i+=1u){
        best=nearer(best,witness(p,point_segment(p,t[i],t[(i+1u)%3u])));
    }
    return best;
}
fn edge_triangle(a:vec3<f32>,b:vec3<f32>,t:mat3x3<f32>)->Witness{
    let n=cross(t[1]-t[0],t[2]-t[0]);
    let da=dot(a-t[0],n);
    let db=dot(b-t[0],n);
    let denominator=da-db;
    if denominator!=0.0 && ((da>=0.0 && db<=0.0)||(da<=0.0 && db>=0.0)){
        let p=a+(b-a)*clamp(da/denominator,0.0,1.0);
        if inside(p,t,n){return Witness(p,p,0.0);}
    }
    return Witness(a,b,FAR);
}
fn edges(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:vec3<f32>)->Witness{
    var best=witness(a,point_segment(a,c,d));
    best=nearer(best,witness(b,point_segment(b,c,d)));
    best=nearer(best,witness(point_segment(c,a,b),c));
    best=nearer(best,witness(point_segment(d,a,b),d));
    let u=b-a;
    let v=d-c;
    let w=c-a;
    let n=cross(u,v);
    let denominator=dot(n,n);
    if denominator>0.0{
        let s=dot(cross(w,v),n)/denominator;
        let t=dot(cross(w,u),n)/denominator;
        if s>=0.0 && s<=1.0 && t>=0.0 && t<=1.0{
            best=nearer(best,witness(a+u*s,c+v*t));
        }
    }
    return best;
}
fn triangle_distance(a:mat3x3<f32>,b:mat3x3<f32>)->Witness{
    var best=Witness(a[0],b[0],FAR);
    for(var i=0u;i<3u;i+=1u){
        let j=(i+1u)%3u;
        let ab=edge_triangle(a[i],a[j],b);
        if ab.distance2==0.0{return ab;}
        let ba=edge_triangle(b[i],b[j],a);
        if ba.distance2==0.0{return Witness(ba.b,ba.a,0.0);}
        best=nearer(best,point_triangle(a[i],b));
        let p=point_triangle(b[i],a);
        best=nearer(best,Witness(p.b,p.a,p.distance2));
        for(var k=0u;k<3u;k+=1u){
            best=nearer(best,edges(a[i],a[j],b[k],b[(k+1u)%3u]));
        }
    }
    return best;
}
fn valid_matrix(m:mat4x4<f32>)->bool{
    for(var i=0u;i<4u;i+=1u){
        if !all(abs(m[i])<=vec4<f32>(1.0e8)){return false;}
    }
    return m[0].w==0.0 && m[1].w==0.0 && m[2].w==0.0 && m[3].w==1.0;
}
fn linear(m:mat4x4<f32>)->mat3x3<f32>{
    return mat3x3<f32>(m[0].xyz,m[1].xyz,m[2].xyz);
}
fn transformed(t:Triangle,m:mat3x3<f32>,offset:vec3<f32>)->mat3x3<f32>{
    return mat3x3<f32>(m*t.a+offset,m*t.b.xyz+offset,m*t.c.xyz+offset);
}
fn voxel_overlap(node:Node,lo:vec3<f32>,hi:vec3<f32>)->bool{
    if node.mask_low==0xffffffffu && node.mask_high==0xffffffffu{return true;}
    // The query is already in the local space of this brick. Inflate
    // cell-index arithmetic to retain both sides of touching boundaries.
    let extent=node.hi-node.lo;
    let a=vec3<u32>(clamp(floor((lo-node.lo)/extent*4.0-0.0001),vec3<f32>(0.0),vec3<f32>(3.0)));
    let b=vec3<u32>(clamp(floor((hi-node.lo)/extent*4.0+0.0001),vec3<f32>(0.0),vec3<f32>(3.0)));
    let bits=((1u<<(b.x-a.x+1u))-1u)<<a.x;
    for(var z=a.z;z<=b.z;z+=1u){
        for(var y=a.y;y<=b.y;y+=1u){
            let row=(z%2u)*16u+y*4u;
            if (select(node.mask_low,node.mask_high,z>=2u)&(bits<<row))!=0u{return true;}
        }
    }
    return false;
}
fn choose(a:Contact,b:Contact)->Contact{
    var result=a;
    if b.point_a.w<a.point_a.w ||
       (b.point_a.w==a.point_a.w &&
        (b.features.x<a.features.x||(b.features.x==a.features.x&&b.features.y<a.features.y))){
        result=b;
    }
    result.features.z=a.features.z+b.features.z;
    result.features.w=max(a.features.w,b.features.w);
    result.statistics=a.statistics+b.statistics;
    return result;
}
fn empty()->Contact{
    return Contact(vec4<f32>(0.0,0.0,0.0,FAR),vec4<f32>(0.0),vec4<u32>(0xffffffffu,0xffffffffu,0u,0u),vec4<u32>(0u));
}
struct QueryFrame {
    la:mat3x3<f32>, lb:mat3x3<f32>,
    relative:vec3<f32>,
    local_linear:mat3x3<f32>,
    local_offset:vec3<f32>,
    padding:vec3<f32>,
    origin:vec3<f32>,
    margin2:f32,
}
struct Bounds { lo:vec3<f32>, hi:vec3<f32> }
fn node_bounds(n:Node,f:QueryFrame)->Bounds{
    let center=(n.lo+n.hi)*0.5;
    let half=(n.hi-n.lo)*0.5;
    let c=f.local_linear*center+f.local_offset;
    let r=abs(f.local_linear[0])*half.x+abs(f.local_linear[1])*half.y+abs(f.local_linear[2])*half.z+f.padding;
    return Bounds(c-r,c+r);
}
fn tri_bounds(t:Triangle,f:QueryFrame)->Bounds{
    let local=transformed(t,f.local_linear,f.local_offset);
    return Bounds(min(local[0],min(local[1],local[2]))-f.padding,
                  max(local[0],max(local[1],local[2]))+f.padding);
}
fn accepts(n:Node,b:Bounds,r:ptr<function,Contact>)->bool{
    (*r).statistics.x+=1u;
    if any(n.lo>b.hi)||any(n.hi<b.lo){return false;}
    if (config.mode&1u)!=0u && !voxel_overlap(n,b.lo,b.hi){
        (*r).statistics.z+=1u;
        return false;
    }
    return true;
}
fn test_triangle(ta:Triangle,wa:mat3x3<f32>,tb:Triangle,f:QueryFrame,r:ptr<function,Contact>){
    let wb=transformed(tb,f.lb,vec3<f32>(0.0));
    (*r).statistics.y+=1u;
    let w=triangle_distance(wa,wb);
    if w.distance2<=f.margin2{
        (*r).features.z+=1u;
        if w.distance2<(*r).point_a.w ||
            (w.distance2==(*r).point_a.w && (ta.id<(*r).features.x||(ta.id==(*r).features.x&&tb.id<(*r).features.y))){
            (*r).point_a=vec4<f32>(w.a+f.origin,w.distance2);
            (*r).point_b=vec4<f32>(w.b+f.origin,0.0);
            (*r).features.x=ta.id;
            (*r).features.y=tb.id;
        }
    }
}
fn scan_triangles(mesh_a:Mesh,mesh_b:Mesh,f:QueryFrame,lane:u32)->Contact{
    var result=empty();
    for(var ai=mesh_a.triangles.x+lane;ai<mesh_a.triangles.y;ai+=64u){
        let ta=triangles[ai];
        let bounds=tri_bounds(ta,f);
        let wa=transformed(ta,f.la,f.relative);
        var node_id=mesh_b.nodes.x;
        while node_id<mesh_b.nodes.y{
            let node=nodes[node_id];
            if !accepts(node,bounds,&result){node_id=node.escape;continue;}
            if node.count==0u{node_id+=1u;continue;}
            for(var bi=node.first;bi<node.end;bi+=1u){
                test_triangle(ta,wa,triangles[bi],f,&result);
            }
            node_id=node.escape;
        }
    }
    return result;
}
// Cut A's balanced hierarchy into at most 64 disjoint subtrees. Each
// invocation walks pairs of nodes depth-first, splitting the larger
// remaining triangle range. The median-built tree has depth <32 even
// for u32-sized triangle arrays; the combined stack cannot exceed 64.
fn scan_dual(mesh_a:Mesh,mesh_b:Mesh,f:QueryFrame,lane:u32)->Contact{
    var result=empty();
    var root=mesh_a.nodes.x;
    for(var depth=0u;depth<6u;depth+=1u){
        if nodes[root].count!=0u{
            if (lane>>depth)!=0u{return result;}
            break;
        }
        let left=root+1u;
        root=select(left,nodes[left].escape,((lane>>depth)&1u)!=0u);
    }
    var stack:array<vec2<u32>,64>;
    stack[0]=vec2<u32>(root,mesh_b.nodes.x);
    var top=1u;
    while top>0u{
        top-=1u;
        let ids=stack[top];
        let a=nodes[ids.x];
        let b=nodes[ids.y];
        if !accepts(b,node_bounds(a,f),&result){continue;}
        if a.count!=0u&&b.count!=0u{
            for(var ai=a.first;ai<a.end;ai+=1u){
                let ta=triangles[ai];
                if !accepts(b,tri_bounds(ta,f),&result){continue;}
                let wa=transformed(ta,f.la,f.relative);
                for(var bi=b.first;bi<b.end;bi+=1u){
                    test_triangle(ta,wa,triangles[bi],f,&result);
                }
            }
            continue;
        }
        if top+2u>64u{
            result.features.w=2u;
            return result;
        }
        if a.count==0u&&(b.count!=0u||a.end-a.first>=b.end-b.first){
            let left=ids.x+1u;
            stack[top]=vec2<u32>(nodes[left].escape,ids.y);
            stack[top+1u]=vec2<u32>(left,ids.y);
        }else{
            let left=ids.y+1u;
            stack[top]=vec2<u32>(ids.x,nodes[left].escape);
            stack[top+1u]=vec2<u32>(ids.x,left);
        }
        top+=2u;
    }
    return result;
}
fn scan(pair:Pair,lane:u32)->Contact{
    var result=empty();
    let a=transforms[pair.transforms.x];
    let b=transforms[pair.transforms.y];
    let la=linear(a);
    let lb=linear(b);
    let r0=cross(lb[1],lb[2]);
    let r1=cross(lb[2],lb[0]);
    let r2=cross(lb[0],lb[1]);
    let det=dot(lb[0],r0);
    if !valid_matrix(a)||!valid_matrix(b)||!(abs(det)>1e-20){
        result.features.w=1u;
        return result;
    }
    let inverse=transpose(mat3x3<f32>(r0,r1,r2))*(1.0/det);
    let relative=a[3].xyz-b[3].xyz;
    let row_norm=vec3<f32>(length(r0),length(r1),length(r2))/abs(det);
    let pad=config.margin*row_norm;
    let local_linear=inverse*la;
    let local_offset=inverse*relative;
    let mesh_a=meshes[pair.meshes.x];
    let mesh_b=meshes[pair.meshes.y];
    // Reject entire transformed A root before walking any triangles.
    let root_a=nodes[mesh_a.nodes.x];
    let root_b=nodes[mesh_b.nodes.x];
    let center=(root_a.lo+root_a.hi)*0.5;
    let half=(root_a.hi-root_a.lo)*0.5;
    let transformed_center=local_linear*center+local_offset;
    let transformed_half=abs(local_linear[0])*half.x+abs(local_linear[1])*half.y+abs(local_linear[2])*half.z;
    // Bound roundoff of both the relative transform and inverse. For
    // ill-conditioned transforms this increases candidates rather than
    // shrinking their conservative query envelope.
    let condition=max(length(lb[0]),max(length(lb[1]),length(lb[2])))*max(row_norm.x,max(row_norm.y,row_norm.z));
    let scale=abs(local_offset)+abs(local_linear[0])*max(abs(root_a.lo.x),abs(root_a.hi.x))
        +abs(local_linear[1])*max(abs(root_a.lo.y),abs(root_a.hi.y))
        +abs(local_linear[2])*max(abs(root_a.lo.z),abs(root_a.hi.z));
    let error=(scale+vec3<f32>(1.0))*0.000008*max(condition,1.0);
    if any(transformed_center+transformed_half+pad+error<root_b.lo)||any(transformed_center-transformed_half-pad-error>root_b.hi){
        return result;
    }
    let frame=QueryFrame(la,lb,relative,local_linear,local_offset,pad+error,b[3].xyz,config.margin*config.margin);
    if (config.mode&2u)!=0u{return scan_dual(mesh_a,mesh_b,frame,lane);}
    return scan_triangles(mesh_a,mesh_b,frame,lane);
}
var<workgroup> reduction:array<Contact,64>;
@compute @workgroup_size(64)
fn query(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    reduction[lane]=scan(pairs[group.x],lane);
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if lane<stride{reduction[lane]=choose(reduction[lane],reduction[lane+stride]);}
        workgroupBarrier();
    }
    if lane==0u{contacts[group.x]=reduction[0];}
}
