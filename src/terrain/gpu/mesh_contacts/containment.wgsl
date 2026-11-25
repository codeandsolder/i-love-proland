// A watertight consistently wound shell can contain a complete disconnected
// component without intersecting any of its triangles. Test one authored
// vertex per welded component, using BVH-accelerated parity rays.
fn cross2(a:vec2<f32>,b:vec2<f32>)->f32{return a.x*b.y-a.y*b.x;}
fn half_open(a:vec2<f32>,b:vec2<f32>,e:f32)->bool{
    let d=b-a;
    return e>0.0||(e==0.0&&(d.y>0.0||(d.y==0.0&&d.x<0.0)));
}
fn ray_triangle(p:vec3<f32>,axis:u32,t:Triangle)->u32{
    let u=(axis+1u)%3u;
    let v=(axis+2u)%3u;
    let a=t.a-p;
    var b=t.b.xyz-p;
    var c=t.c-p;
    let pa=vec2<f32>(a[u],a[v]);
    var pb=vec2<f32>(b[u],b[v]);
    var pc=vec2<f32>(c[u],c[v]);
    var area=cross2(pb-pa,pc-pa);
    if area==0.0{return 0u;}
    if area<0.0{
        let temp=pb;pb=pc;pc=temp;
        let vertex=b;b=c;c=vertex;
        area=-area;
    }
    let e0=cross2(pb,pc);
    let e1=cross2(pc,pa);
    let e2=cross2(pa,pb);
    // Shared edge ownership uses an orientation-independent half-open
    // rule, as in rasterization. Near-but-not-exact ties are ambiguous.
    let error=0.000002*(abs(e0)+abs(e1)+abs(e2)+area);
    if min(e0,min(e1,e2)) < -error{return 0u;}
    if (e0!=0.0&&abs(e0)<=error)||(e1!=0.0&&abs(e1)<=error)||(e2!=0.0&&abs(e2)<=error){return 2u;}
    if !half_open(pb,pc,e0)||!half_open(pc,pa,e1)||!half_open(pa,pb,e2){return 0u;}
    let x=e0*a[axis]+e1*b[axis]+e2*c[axis];
    let xerror=0.000004*(abs(e0*a[axis])+abs(e1*b[axis])+abs(e2*c[axis]));
    if abs(x)<=xerror{return 2u;}
    return u32(x>0.0);
}
fn parity(p:vec3<f32>,mesh:Mesh,axis:u32,visits:ptr<function,u32>)->u32{
    let u=(axis+1u)%3u;
    let v=(axis+2u)%3u;
    var id=mesh.nodes.x;
    var count=0u;
    while id<mesh.nodes.y{
        let n=nodes[id];
        (*visits)+=1u;
        if p[axis]>n.hi[axis]||p[u]<n.lo[u]||p[u]>n.hi[u]||p[v]<n.lo[v]||p[v]>n.hi[v]{
            id=n.escape;continue;
        }
        if n.count==0u{id+=1u;continue;}
        for(var i=n.first;i<n.end;i+=1u){
            let hit=ray_triangle(p,axis,triangles[i]);
            if hit==2u{return 2u;}
            count^=hit;
        }
        id=n.escape;
    }
    return count;
}
fn point_inside(p:vec3<f32>,mesh:Mesh,visits:ptr<function,u32>)->u32{
    let root=nodes[mesh.nodes.x];
    if any(p<root.lo)||any(p>root.hi){return 0u;}
    var votes=vec2<u32>(0u);
    for(var axis=0u;axis<3u;axis+=1u){
        let vote=parity(p,mesh,axis,visits);
        if vote<2u{votes[vote]+=1u;}
    }
    if votes.x>0u&&votes.y>0u{return 8u;}
    if votes.y>=2u{return 4u;}
    if votes.x>=2u{return 0u;}
    return 8u;
}
fn contains_components(source:Mesh,destination:Mesh,a:mat4x4<f32>,b:mat4x4<f32>,lane:u32,result:ptr<function,Contact>){
    if destination.closed==0u{return;}
    let lb=linear(b);
    let r0=cross(lb[1],lb[2]);
    let r1=cross(lb[2],lb[0]);
    let r2=cross(lb[0],lb[1]);
    let det=dot(lb[0],r0);
    if !(abs(det)>1e-20){(*result).features.w|=1u;return;}
    let inverse=transpose(mat3x3<f32>(r0,r1,r2))*(1.0/det);
    let relative=inverse*linear(a);
    let offset=inverse*(a[3].xyz-b[3].xyz);
    for(var i=source.triangles.x+lane;i<source.triangles.y;i+=64u){
        let triangle=triangles[i];
        if triangle.representative==0u{continue;}
        let p=relative*triangle.a+offset;
        var visits=0u;
        (*result).features.w|=point_inside(p,destination,&visits);
        (*result).statistics.w+=visits;
    }
}
fn containment(mesh_ids:vec2<u32>,a:mat4x4<f32>,b:mat4x4<f32>,lane:u32)->Contact{
    var result=empty();
    let ma=meshes[mesh_ids.x];
    let mb=meshes[mesh_ids.y];
    contains_components(ma,mb,a,b,lane,&result);
    contains_components(mb,ma,b,a,lane,&result);
    return result;
}
