override DUAL:bool=false;
override VOXELS:bool=true;
override CLUSTERS:bool=false;
override PLANES:bool=true;
override LANES:u32=64u;
struct Triangle { a:vec3<f32>, id:u32, b:vec4<f32>, c:vec3<f32>, flags:u32 }
struct Node {
    lo:vec3<f32>, first:u32,
    hi:vec3<f32>, count:u32,
    escape:u32, mask_low:u32, mask_high:u32, end:u32,
}
struct Mesh { nodes:vec2<u32>, triangles:vec2<u32>, closed:u32, pad0:u32, solid_mask:vec2<u32>, seeds:vec2<u32> }
struct Contact {
    point_a:vec4<f32>,
    point_b:vec4<f32>,
    normal:vec4<f32>,
    features:vec4<u32>,
    statistics:vec4<u32>,
}
@group(0) @binding(0) var<storage,read> nodes:array<Node>;
@group(0) @binding(1) var<storage,read> triangles:array<Triangle>;
@group(0) @binding(2) var<storage,read> meshes:array<Mesh>;
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
fn point_triangle(p:vec3<f32>,t:mat3x3<f32>,n:vec3<f32>,n2:f32)->Witness{
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
fn edge_triangle(a:vec3<f32>,b:vec3<f32>,t:mat3x3<f32>,n:vec3<f32>)->Witness{
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
    let u=b-a;
    let v=d-c;
    let w=a-c;
    let uu=dot(u,u);
    let vv=dot(v,v);
    if uu==0.0{return witness(a,point_segment(a,c,d));}
    if vv==0.0{return witness(point_segment(c,a,b),c);}
    let uv=dot(u,v);
    let uw=dot(u,w);
    let vw=dot(v,w);
    let n=cross(u,v);
    let denominator=dot(n,n);
    var s=0.0;
    if denominator>0.0{
        s=clamp(dot(cross(-w,v),n)/denominator,0.0,1.0);
    }
    var t=(uv*s+vw)/vv;
    if t<0.0{
        t=0.0;
        s=clamp(-uw/uu,0.0,1.0);
    }else if t>1.0{
        t=1.0;
        s=clamp((uv-uw)/uu,0.0,1.0);
    }
    return witness(a+u*s,c+v*t);
}
fn triangle_distance(a:mat3x3<f32>,b:mat3x3<f32>,na:vec3<f32>,nb:vec3<f32>)->Witness{
    let na2=dot(na,na);
    let nb2=dot(nb,nb);
    var best=Witness(a[0],b[0],FAR);
    for(var i=0u;i<3u;i+=1u){
        let j=(i+1u)%3u;
        let ab=edge_triangle(a[i],a[j],b,nb);
        if ab.distance2==0.0{return ab;}
        let ba=edge_triangle(b[i],b[j],a,na);
        if ba.distance2==0.0{return Witness(ba.b,ba.a,0.0);}
        best=nearer(best,point_triangle(a[i],b,nb,nb2));
        let p=point_triangle(b[i],a,na,na2);
        best=nearer(best,Witness(p.b,p.a,p.distance2));
        if best.distance2==0.0{return best;}
        for(var k=0u;k<3u;k+=1u){
            best=nearer(best,edges(a[i],a[j],b[k],b[(k+1u)%3u]));
            if best.distance2==0.0{return best;}
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
    return mat3x3<f32>(m*t.a+offset,m*t.b.xyz+offset,m*t.c+offset);
}
fn voxel_overlap(node:Node,lo:vec3<f32>,hi:vec3<f32>)->bool{
    if node.mask_low==0xffffffffu && node.mask_high==0xffffffffu{return true;}
    // The query is already in the local space of this brick. Inflate
    // cell-index arithmetic to retain both sides of touching boundaries.
    let extent=node.hi-node.lo;
    let a=vec3<u32>(clamp(floor((lo-node.lo)/extent*4.0-0.0001),vec3<f32>(0.0),vec3<f32>(3.0)));
    let b=vec3<u32>(clamp(floor((hi-node.lo)/extent*4.0+0.0001),vec3<f32>(0.0),vec3<f32>(3.0)));
    // Generate the entire rectangular 4^3 mask with bit arithmetic.
    // X repeats one nibble in every row, Y repeats one 16-bit slab,
    // and Z selects the two slabs in each u32. No per-row loops.
    let x=(((1u<<(b.x-a.x+1u))-1u)<<a.x)*0x11111111u;
    let y_row=((1u<<(4u*(b.y-a.y+1u)))-1u)<<(4u*a.y);
    let y=y_row|(y_row<<16u);
    let z=((1u<<(b.z-a.z+1u))-1u)<<a.z;
    let low=select(0u,0xffffu,(z&1u)!=0u)|select(0u,0xffff0000u,(z&2u)!=0u);
    let high=select(0u,0xffffu,(z&4u)!=0u)|select(0u,0xffff0000u,(z&8u)!=0u);
    return (node.mask_low&x&y&low)!=0u||(node.mask_high&x&y&high)!=0u;
}
fn choose(a:Contact,b:Contact)->Contact{
    var result=a;
    if b.point_a.w<a.point_a.w ||
       (b.point_a.w==a.point_a.w &&
        (b.features.x<a.features.x||(b.features.x==a.features.x&&b.features.y<a.features.y))){
        result=b;
    }
    result.features.z=a.features.z+b.features.z;
    result.features.w=a.features.w|b.features.w;
    result.statistics=a.statistics+b.statistics;
    return result;
}
fn empty()->Contact{
    return Contact(vec4<f32>(0.0,0.0,0.0,FAR),vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(0xffffffffu,0xffffffffu,0u,0u),vec4<u32>(0u));
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
    return box_bounds(n.lo,n.hi,f);
}
fn tri_bounds(t:Triangle,f:QueryFrame)->Bounds{
    let local=transformed(t,f.local_linear,f.local_offset);
    return Bounds(min(local[0],min(local[1],local[2]))-f.padding,
                  max(local[0],max(local[1],local[2]))+f.padding);
}
fn accepts(n:Node,b:Bounds,r:ptr<function,Contact>)->bool{
    (*r).statistics.x+=1u;
    if any(n.lo>b.hi)||any(n.hi<b.lo){return false;}
    if VOXELS && !voxel_overlap(n,b.lo,b.hi){
        (*r).statistics.z+=1u;
        return false;
    }
    return true;
}
// Reject only if the complete projected intervals are separated by
// more than the physical query margin. No assumption about a normalized
// or accurate face normal is needed: any finite nonzero axis is valid.
// Use the L1 axis norm as an upper bound on its Euclidean norm and a
// generous f32 projection error envelope. Degenerate axes cannot reject.
fn plane_separates(a:mat3x3<f32>,b:mat3x3<f32>,axis:vec3<f32>,margin:f32)->bool{
    let scale=max(abs(axis.x),max(abs(axis.y),abs(axis.z)));
    if !(scale>0.0&&scale<1e30){return false;}
    let n=axis/scale;
    let abs_n=abs(n);
    let ap=vec3<f32>(dot(a[0],n),dot(a[1],n),dot(a[2],n));
    let bp=vec3<f32>(dot(b[0],n),dot(b[1],n),dot(b[2],n));
    let am=max(abs(a[0]),max(abs(a[1]),abs(a[2])));
    let bm=max(abs(b[0]),max(abs(b[1]),abs(b[2])));
    let error=0.000032*(dot(am+bm,abs_n)+margin+1.0);
    let limit=margin*(abs_n.x+abs_n.y+abs_n.z)+error;
    return min(ap.x,min(ap.y,ap.z))-max(bp.x,max(bp.y,bp.z))>limit ||
           min(bp.x,min(bp.y,bp.z))-max(ap.x,max(ap.y,ap.z))>limit;
}
fn test_triangle(ta:Triangle,wa:mat3x3<f32>,tb:Triangle,bounds:Bounds,f:QueryFrame,r:ptr<function,Contact>){
    let lo=min(tb.a,min(tb.b.xyz,tb.c));
    let hi=max(tb.a,max(tb.b.xyz,tb.c));
    if any(lo>bounds.hi)||any(hi<bounds.lo){return;}
    let wb=transformed(tb,f.lb,vec3<f32>(0.0));
    let na=cross(wa[1]-wa[0],wa[2]-wa[0]);
    let nb=cross(wb[1]-wb[0],wb[2]-wb[0]);
    if PLANES {
        let margin=sqrt(f.margin2);
        if plane_separates(wa,wb,na,margin)||plane_separates(wa,wb,nb,margin){
            return;
        }
    }
    (*r).statistics.y+=1u;
    let w=triangle_distance(wa,wb,na,nb);
    if w.distance2<=f.margin2{
        (*r).features.z+=1u;
        if w.distance2<(*r).point_a.w ||
            (w.distance2==(*r).point_a.w && (ta.id<(*r).features.x||(ta.id==(*r).features.x&&tb.id<(*r).features.y))){
            (*r).point_a=vec4<f32>(w.a+f.origin,w.distance2);
            (*r).point_b=vec4<f32>(w.b+f.origin,0.0);
            let distance=sqrt(w.distance2);
            (*r).normal=vec4<f32>((w.a-w.b)/max(distance,1e-20),distance);
            (*r).features.x=ta.id;
            (*r).features.y=tb.id;
        }
    }
}
fn scan_triangle_range(mesh_b:Mesh,f:QueryFrame,first:u32,end:u32,stride:u32)->Contact{
    var result=empty();
    for(var ai=first;ai<end;ai+=stride){
        let ta=triangles[ai];
        let bounds=tri_bounds(ta,f);
        let wa=transformed(ta,f.la,f.relative);
        var node_id=mesh_b.nodes.x;
        while node_id<mesh_b.nodes.y{
            let node=nodes[node_id];
            if !accepts(node,bounds,&result){node_id=node.escape;continue;}
            if node.count==0u{node_id+=1u;continue;}
            for(var bi=node.first;bi<node.end;bi+=1u){
                test_triangle(ta,wa,triangles[bi],bounds,f,&result);
            }
            node_id=node.escape;
        }
    }
    return result;
}
fn scan_triangles(mesh_a:Mesh,mesh_b:Mesh,f:QueryFrame,lane:u32)->Contact{
    return scan_triangle_range(mesh_b,f,mesh_a.triangles.x+lane,mesh_a.triangles.y,LANES);
}
// Divide the source into disjoint existing BVH subtrees. Entire
// clusters outside the target's conservative voxel envelope are
// discarded before reading even one source triangle.
fn cluster_root(mesh:Mesh,lane:u32)->u32{
    var root=mesh.nodes.x;
    for(var depth=0u;(1u<<depth)<LANES;depth+=1u){
        if nodes[root].count!=0u{
            if (lane>>depth)!=0u{return 0xffffffffu;}
            break;
        }
        let left=root+1u;
        root=select(left,nodes[left].escape,((lane>>depth)&1u)!=0u);
    }
    return root;
}
fn scan_clusters(mesh_a:Mesh,mesh_b:Mesh,f:QueryFrame,lane:u32)->Contact{
    if mesh_a.triangles.y-mesh_a.triangles.x<512u{return scan_triangles(mesh_a,mesh_b,f,lane);}
    var result=empty();
    let root=cluster_root(mesh_a,lane);
    if root==0xffffffffu{return result;}
    let cluster=nodes[root];
    if !accepts(nodes[mesh_b.nodes.x],node_bounds(cluster,f),&result){return result;}
    return choose(result,scan_triangle_range(mesh_b,f,cluster.first,cluster.end,1u));
}
// Cut A's balanced hierarchy into at most 64 disjoint subtrees. Each
// invocation walks pairs of nodes depth-first, splitting the larger
// remaining triangle range. The median-built tree has depth <32 even
// for u32-sized triangle arrays; the combined stack cannot exceed 64.
fn scan_dual(mesh_a:Mesh,mesh_b:Mesh,f:QueryFrame,lane:u32)->Contact{
    var result=empty();
    let root=cluster_root(mesh_a,lane);
    if root==0xffffffffu{return result;}
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
                let bounds=tri_bounds(ta,f);
                if !accepts(b,bounds,&result){continue;}
                let wa=transformed(ta,f.la,f.relative);
                for(var bi=b.first;bi<b.end;bi+=1u){
                    test_triangle(ta,wa,triangles[bi],bounds,f,&result);
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
fn query_frame(a:mat4x4<f32>,b:mat4x4<f32>,root_a:Node,margin:f32)->QueryFrame{
    let la=linear(a);
    let lb=linear(b);
    let r0=cross(lb[1],lb[2]);
    let r1=cross(lb[2],lb[0]);
    let r2=cross(lb[0],lb[1]);
    let det=dot(lb[0],r0);
    let inverse=transpose(mat3x3<f32>(r0,r1,r2))*(1.0/det);
    let relative=a[3].xyz-b[3].xyz;
    let row_norm=vec3<f32>(length(r0),length(r1),length(r2))/abs(det);
    let pad=margin*row_norm;
    let local_linear=inverse*la;
    let local_offset=inverse*relative;
    // Bound roundoff of both the relative transform and inverse. For
    // ill-conditioned transforms this increases candidates rather than
    // shrinking their conservative query envelope.
    let condition=max(length(lb[0]),max(length(lb[1]),length(lb[2])))*max(row_norm.x,max(row_norm.y,row_norm.z));
    let scale=abs(local_offset)+abs(local_linear[0])*max(abs(root_a.lo.x),abs(root_a.hi.x))
        +abs(local_linear[1])*max(abs(root_a.lo.y),abs(root_a.hi.y))
        +abs(local_linear[2])*max(abs(root_a.lo.z),abs(root_a.hi.z));
    let error=(scale+vec3<f32>(1.0))*0.000008*max(condition,1.0);
    return QueryFrame(la,lb,relative,local_linear,local_offset,pad+error,b[3].xyz,margin*margin);
}
fn box_bounds(lo:vec3<f32>,hi:vec3<f32>,f:QueryFrame)->Bounds{
    let center=(lo+hi)*0.5;
    let half=(hi-lo)*0.5;
    let c=f.local_linear*center+f.local_offset;
    let r=abs(f.local_linear[0])*half.x+abs(f.local_linear[1])*half.y+abs(f.local_linear[2])*half.z+f.padding;
    return Bounds(c-r,c+r);
}
fn scan_ordered(mesh_ids:vec2<u32>,a:mat4x4<f32>,b:mat4x4<f32>,margin:f32,lane:u32)->Contact{
    let mesh_a=meshes[mesh_ids.x];
    let mesh_b=meshes[mesh_ids.y];
    let root_a=nodes[mesh_a.nodes.x];
    let root_b=nodes[mesh_b.nodes.x];
    let frame=query_frame(a,b,root_a,margin);
    let bounds=box_bounds(root_a.lo,root_a.hi,frame);
    if any(bounds.hi<root_b.lo)||any(bounds.lo>root_b.hi){return empty();}

    if DUAL{return scan_dual(mesh_a,mesh_b,frame,lane);}
    if CLUSTERS{return scan_clusters(mesh_a,mesh_b,frame,lane);}
    return scan_triangles(mesh_a,mesh_b,frame,lane);
}
// Use the smaller authored triangle range as the query side. Each
// triangle on that side starts one BVH traversal; reversing the pair
// can therefore avoid thousands of root visits without changing the
// actual geometric predicate. Preserve the public pair's witness order.
fn scan(mesh_ids:vec2<u32>,a:mat4x4<f32>,b:mat4x4<f32>,margin:f32,lane:u32)->Contact{
    // A singular input must not become valid merely because the pair
    // order changed. Degenerate triangles are still allowed in meshes.
    let la=linear(a);
    let lb=linear(b);
    if !valid_matrix(a)||!valid_matrix(b)||
       !(abs(dot(la[0],cross(la[1],la[2])))>1e-20)||
       !(abs(dot(lb[0],cross(lb[1],lb[2])))>1e-20){
        var result=empty();
        result.features.w=1u;
        return result;
    }
    if !reverse_order(mesh_ids){return scan_ordered(mesh_ids,a,b,margin,lane);}
    return scan_ordered(mesh_ids.yx,b,a,margin,lane);
}
fn reverse_order(mesh_ids:vec2<u32>)->bool{
    let ma=meshes[mesh_ids.x].triangles;
    let mb=meshes[mesh_ids.y].triangles;
    return ma.y-ma.x>mb.y-mb.x;
}
fn orient_contact(mesh_ids:vec2<u32>,contact:Contact)->Contact{
    if !reverse_order(mesh_ids)||contact.features.z==0u{return contact;}
    var result=contact;
    let pa=result.point_a.xyz;
    result.point_a=vec4<f32>(result.point_b.xyz,result.point_a.w);
    result.point_b=vec4<f32>(pa,0.0);
    result.normal=vec4<f32>(-result.normal.xyz,result.normal.w);
    result.features=vec4<u32>(result.features.yx,result.features.zw);
    return result;
}
// Eight conservative occupied octants of the existing root brick.
// A set bit means some original triangle intersects that voxel. The
// union covers every source triangle, including planar and degenerate
// faces. No approximation of the original collision surface is used.
fn occupied_octant(root:Node,octant:u32)->bool{
    let bits=select(root.mask_low,root.mask_high,(octant&4u)!=0u);
    let shift=(octant&1u)*2u+((octant>>1u)&1u)*8u;
    return (bits&(0x00330033u<<shift))!=0u;
}
fn envelope_overlap(mesh_a:Mesh,mesh_b:Mesh,a:mat4x4<f32>,b:mat4x4<f32>,margin:f32,solid:bool)->bool{
    var root_a=nodes[mesh_a.nodes.x];
    var root_b=nodes[mesh_b.nodes.x];
    if solid{
        root_a.mask_low=mesh_a.solid_mask.x;
        root_a.mask_high=mesh_a.solid_mask.y;
        root_b.mask_low=mesh_b.solid_mask.x;
        root_b.mask_high=mesh_b.solid_mask.y;
    }
    let f=query_frame(a,b,root_a,margin);
    let half=(root_a.hi-root_a.lo)*0.5;
    // Nonfinite transforms/envelopes must continue to the exact path.
    if !all(abs(f.padding)<=vec3<f32>(1e30)){return true;}
    for(var octant=0u;octant<8u;octant+=1u){
        if !occupied_octant(root_a,octant){continue;}
        let corner=vec3<f32>(f32(octant&1u),f32((octant>>1u)&1u),f32(octant>>2u));
        let lo=root_a.lo+corner*half;
        let hi=lo+half;
        let bounds=box_bounds(lo,hi,f);
        if any(bounds.lo>root_b.hi)||any(bounds.hi<root_b.lo){continue;}
        if voxel_overlap(root_b,bounds.lo,bounds.hi){return true;}
    }
    return false;
}

fn select_envelope(ids:vec2<u32>,a:mat4x4<f32>,b:mat4x4<f32>,margin:f32,solid:bool)->bool{
    if reverse_order(ids){return envelope_overlap(meshes[ids.y],meshes[ids.x],b,a,margin,solid);}
    return envelope_overlap(meshes[ids.x],meshes[ids.y],a,b,margin,solid);
}

var<workgroup> reduction:array<Contact,64>;
fn reduce_lanes(local:Contact,lane:u32,width:u32){
    reduction[lane]=local;
    workgroupBarrier();
    for(var stride=width/2u;stride>0u;stride/=2u){
        if (lane&(width-1u))<stride{reduction[lane]=choose(reduction[lane],reduction[lane+stride]);}
        workgroupBarrier();
    }
}
fn reduce_contact(local:Contact,lane:u32)->Contact{
    reduce_lanes(local,lane,64u);
    return workgroupUniformLoad(&reduction[0]);
}
fn reduce_packed(local:Contact,lane:u32)->Contact{
    if LANES==1u{return local;}
    reduce_lanes(local,lane,LANES);
    let result=reduction[lane&~(LANES-1u)];
    // All lanes read before a following containment reduction can
    // overwrite the same workgroup memory.
    workgroupBarrier();
    return result;
}
