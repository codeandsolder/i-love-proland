fn quat_rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32>{
    let t=2.0*cross(q.xyz,v);
    return v+q.w*t+cross(q.xyz,t);
}
fn quat_mul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{
    return vec4<f32>(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));
}
fn inverse_world_inertia(b:Body,v:vec3<f32>)->vec3<f32>{
    let inverse=vec4<f32>(-b.rotation.xyz,b.rotation.w);
    return quat_rotate(b.rotation,quat_rotate(inverse,v)*b.inverse_inertia.xyz);
}
