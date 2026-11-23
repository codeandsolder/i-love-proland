struct PackedInstance {
    position_radius: vec4<f32>,    // local-frame xyz, physical crown radius
    rotation: vec4<f32>,           // normalized source quaternion XYZW
    axis_radii: vec4<f32>,         // physical XYZ scale, appearance variation
    object_id: vec4<u32>,          // full stable u128, low word first
    metadata: vec4<u32>,           // variant u64 + layer u64
    render_model: vec4<u32>,       // visual model index, independent of variant ID
    tint: vec4<f32>,               // per-instance linear RGB, alpha multiplier
    handoff_anchor_rank: vec4<f32>, // physical anchor xyz and stable handoff rank
}

