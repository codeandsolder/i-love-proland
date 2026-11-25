// Perram-Wertheim ellipsoid overlap. The maximum of
// lambda*(1-lambda)*d^T*((1-lambda)*A+lambda*B)^-1*d is <1
// precisely when the closed ellipsoids overlap. A and B are squared
// semi-axis shape matrices. Evaluate the inverse with sums of squared
// minors of six weighted basis columns, avoiding cancellation of the
// determinant for thin, obliquely rotated ellipsoids.
struct PairShape {
    center:vec3<f32>,
    axes:vec3<f32>,
    rotation:vec4<f32>,
}
fn pair_shape(b:Body,s:Collider)->PairShape{
    return PairShape(b.position.xyz+quat_rotate(b.rotation,s.center.xyz),
        s.axes.xyz,quat_mul(b.rotation,s.rotation));
}
fn bound_radius(s:PairShape)->f32{
    return max(s.axes.x,max(s.axes.y,s.axes.z));
}
fn sphere_overlap(a:PairShape,b:PairShape)->bool{
    let d=a.center-b.center;
    let r=bound_radius(a)+bound_radius(b);
    return dot(d,d)<r*r;
}
fn basis(s:PairShape,scale:f32)->mat3x3<f32>{
    let a=s.axes*scale;
    return mat3x3<f32>(
        quat_rotate(s.rotation,vec3<f32>(a.x,0.0,0.0)),
        quat_rotate(s.rotation,vec3<f32>(0.0,a.y,0.0)),
        quat_rotate(s.rotation,vec3<f32>(0.0,0.0,a.z)));
}
struct ContactFunction {
    value:f32,
    gradient:f32,
    normal:vec3<f32>,
}
fn contact_function(a:mat3x3<f32>,b:mat3x3<f32>,d:vec3<f32>,t:f32)->ContactFunction{
    let u=sqrt(1.0-t);
    let v=sqrt(t);
    let columns=array<vec3<f32>,6>(a[0]*u,a[1]*u,a[2]*u,b[0]*v,b[1]*v,b[2]*v);
    var adj_d=vec3<f32>(0.0);
    var det=0.0;
    for(var i=0u;i<5u;i+=1u){
        for(var j=i+1u;j<6u;j+=1u){
            let c=cross(columns[i],columns[j]);
            adj_d+=c*dot(c,d);
            for(var k=j+1u;k<6u;k+=1u){
                let triple=dot(c,columns[k]);
                det+=triple*triple;
            }
        }
    }
    let x=adj_d/det;
    let quad=dot(d,x);
    let ax=transpose(a)*x;
    let bx=transpose(b)*x;
    let derivative=(1.0-2.0*t)*quad-t*(1.0-t)*(dot(bx,bx)-dot(ax,ax));
    return ContactFunction(t*(1.0-t)*quad,derivative,x);
}
fn ellipsoid_support(s:PairShape,n:vec3<f32>)->vec3<f32>{
    let local=quat_rotate(vec4<f32>(-s.rotation.xyz,s.rotation.w),n);
    let weighted=s.axes*local;
    return s.center+quat_rotate(s.rotation,s.axes*weighted/max(length(weighted),1e-20));
}
struct PairContact {
    normal:vec3<f32>,
    point:vec3<f32>,
    depth:f32,
}
fn ellipsoid_contact(a:PairShape,b:PairShape)->PairContact{
    let miss=PairContact(vec3<f32>(0.0),vec3<f32>(0.0),0.0);
    if !sphere_overlap(a,b){return miss;}
    let d=a.center-b.center;
    let scale=1.0/max(bound_radius(a),bound_radius(b));
    let delta=d*scale;
    var n=vec3<f32>(1.0,0.0,0.0);
    if dot(delta,delta)>1e-20{
        let ma=basis(a,scale);
        let mb=basis(b,scale);
        // The contact function is concave, with a unique interior maximum
        // for noncoincident centers. A positive derivative moves right.
        var lo=0.0;
        var hi=1.0;
        for(var k=0u;k<24u;k+=1u){
            let t=(lo+hi)*0.5;
            let f=contact_function(ma,mb,delta,t);
            // Any evaluated separating certificate is sufficient.
            if f.value>=1.0{return miss;}
            if f.gradient>0.0{lo=t;}else{hi=t;}
        }
        let f=contact_function(ma,mb,delta,(lo+hi)*0.5);
        if f.value>=1.0{return miss;}
        n=normalize(f.normal);
    }else{
        // Coincident centers: use the thinnest of the six authored axes
        // as a deterministic separating direction, rather than inventing
        // a fixed world-space axis for all orientations.
        var best=1e20;
        let axes=array<vec3<f32>,6>(
            quat_rotate(a.rotation,vec3<f32>(1.0,0.0,0.0)),
            quat_rotate(a.rotation,vec3<f32>(0.0,1.0,0.0)),
            quat_rotate(a.rotation,vec3<f32>(0.0,0.0,1.0)),
            quat_rotate(b.rotation,vec3<f32>(1.0,0.0,0.0)),
            quat_rotate(b.rotation,vec3<f32>(0.0,1.0,0.0)),
            quat_rotate(b.rotation,vec3<f32>(0.0,0.0,1.0)));
        for(var i=0u;i<6u;i+=1u){
            let axis=axes[i];
            let depth=dot(ellipsoid_support(b,axis)-ellipsoid_support(a,-axis),axis);
            if depth<best{best=depth;n=axis;}
        }
    }
    let pa=ellipsoid_support(a,-n);
    let pb=ellipsoid_support(b,n);
    let depth=dot(pb-pa,n);
    return PairContact(n,(pa+pb)*0.5,max(depth,0.0));
}

// Contact is evaluated in canonical body/collider order on both lanes,
// and the very same point is used for the equal and opposite impulses.
fn compound_effect(a:Body,b:Body,c:PairContact,weight:f32,bounce:bool)->PairEffect{
    let n=c.normal;
    let ra=c.point-a.position.xyz;
    let rb=c.point-b.position.xyz;
    let relative=a.velocity.xyz+cross(a.angular.xyz,ra)-b.velocity.xyz-cross(b.angular.xyz,rb);
    let vn=dot(relative,n);
    let inverse_mass=a.position.w+b.position.w;
    let normal_mass=inverse_mass+spin_mass(a,ra,n)+spin_mass(b,rb,n);
    let restitution=select(0.0,min(a.angular.w,b.angular.w),bounce&&vn < -0.8);
    let normal_impulse=max(0.0,-vn*(1.0+restitution)/max(normal_mass,1e-12));
    let tangent=relative-vn*n;
    let speed=length(tangent);
    var friction=vec3<f32>(0.0);
    if speed>1e-6{
        let axis=tangent/speed;
        let k=inverse_mass+spin_mass(a,ra,axis)+spin_mass(b,rb,axis);
        friction=-axis*min(speed/max(k,1e-12),min(a.velocity.w,b.velocity.w)*normal_impulse);
    }
    let impulse=(n*normal_impulse+friction)*weight;
    let correction=n*(max(c.depth-0.001,0.0)*0.7*a.position.w/inverse_mass*weight);
    return PairEffect(impulse,cross(ra,impulse),correction);
}
