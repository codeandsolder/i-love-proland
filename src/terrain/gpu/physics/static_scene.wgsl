// Immutable pre-order BVH, stackless traversal via escape links.
struct StaticNode {
    lower:vec3<f32>, first:u32,
    upper:vec3<f32>, count:u32,
    escape:u32, pad0:u32, pad1:u32, pad2:u32,
}
struct StaticTriangle { a:vec3<f32>, material:u32, b:vec4<f32>, c:vec4<f32> }
@group(0) @binding(7) var<storage,read> static_nodes:array<StaticNode>;
@group(0) @binding(8) var<storage,read> static_triangles:array<StaticTriangle>;

fn static_contact(s:ShapePose)->Contact {
    var best=Contact(vec3<f32>(0.0,0.0,1.0),0.0,vec3<f32>(0.0),0u);
    let lower=s.center-vec3<f32>(s.radius);
    let upper=s.center+vec3<f32>(s.radius);
    var i=0u;
    let end=arrayLength(&static_nodes);
    while i<end {
        let node=static_nodes[i];
        if any(node.lower>upper)||any(node.upper<lower)||any(node.lower>node.upper) {
            i=node.escape;
            continue;
        }
        if node.count==0u {
            i+=1u;
            continue;
        }
        for(var t=node.first;t<node.first+node.count;t+=1u) {
            let tri=static_triangles[t];
            let a=metric(s,tri.a.xyz);
            let b=metric(s,tri.b.xyz);
            let c=metric(s,tri.c.xyz);
            let cross_product=cross(b-a,c-a);
            if dot(cross_product,cross_product)<1e-20 {continue;}
            let point=closest_triangle(vec3<f32>(0.0),a,b,c);
            let d=length(point);
            if d>=s.min_axis {continue;}
            // Double-sided surfaces: normals always oppose approach.
            let plane=normalize(cross_product);
            let direction=select(select(-plane,plane,dot(-a,plane)>=0.0),-point/max(d,1e-8),d>1e-6);
            let normal=metric_normal(s,direction);
            let support=world_point(s,point);
            let depth=support_height(s,normal)-dot(s.center-support,normal);
            if depth>best.depth {
                best=Contact(normal,depth,support,tri.material);
            }
        }
        i=node.escape;
    }
    return best;
}
