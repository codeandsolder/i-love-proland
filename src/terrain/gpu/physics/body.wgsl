struct Body {
    position:vec4<f32>, // xyz, inverse mass
    rotation:vec4<f32>, // xyzw
    velocity:vec4<f32>, // xyz, friction
    angular:vec4<f32>, // xyz, restitution
    inverse_inertia:vec4<f32>, // principal body-axis inverse inertia
    metadata:vec4<u32>, // first sphere, sphere count, status, contact recorded
    clock:vec4<u32>, // low/high completed tick, low/high first-contact tick
}
