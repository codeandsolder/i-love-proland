use super::{Result, body, config, sample};
use i_love_proland::terrain::gpu::{
    TerrainGpu,
    physics::{Collider, PhysicsGpu, RigidBody},
};

fn pair(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    bodies: &[RigidBody; 2],
    shapes: &[Collider],
    steps: u32,
) -> Result<Vec<RigidBody>> {
    let mut physics = PhysicsGpu::new(device, terrain, 0, config(0.0), bodies, shapes)?;
    physics.enable_body_pairs(device)?;
    let result = sample(device, queue, terrain, &mut physics, 0, steps)?;
    assert!(
        result
            .iter()
            .all(|b| b.metadata[2] == 0 && b.completed_tick() == u64::from(steps))
    );
    Ok(result)
}

fn head_on(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    let shapes = [Collider::ellipsoid(
        [0.0; 3],
        [0.8, 0.3, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    )];
    let initial = [
        body([30.0, 32.0, 108.0], [2.0, 0.0, 0.0], 1.0),
        body([32.0, 32.0, 108.0], [-2.0, 0.0, 0.0], 1.0),
    ];
    let out = pair(device, queue, terrain, &initial, &shapes, 70)?;
    assert!(
        (out[0].velocity[0] + 2.0).abs() < 0.01 && (out[1].velocity[0] - 2.0).abs() < 0.01,
        "{out:?}"
    );
    println!(
        "GPU_COMPOUND head_on velocities={:?}",
        out.iter().map(|b| b.velocity[0]).collect::<Vec<_>>()
    );
    Ok(())
}

fn separated(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    // Circumscribed spheres overlap for the whole pass, but the authored
    // thin ellipsoids are separated in Y. Test false positives and rotation.
    let rotation = [0.0, 0.0, 0.38268343, 0.9238795];
    let shapes = [Collider::ellipsoid([0.0; 3], [1.5, 0.08, 0.25], rotation)];
    let initial = [
        body([30.0, 32.0, 108.0], [0.0; 3], 1.0),
        body([30.0, 32.5, 108.0], [0.0; 3], 1.0),
    ];
    let out = pair(device, queue, terrain, &initial, &shapes, 1)?;
    assert_eq!(
        bytemuck::bytes_of(&out[0].position),
        bytemuck::bytes_of(&initial[0].position)
    );
    assert_eq!(
        bytemuck::bytes_of(&out[1].position),
        bytemuck::bytes_of(&initial[1].position)
    );
    println!("GPU_COMPOUND rotated_thin_separation=true");
    Ok(())
}

fn off_center(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    let shapes = [
        Collider::sphere([0.0, 0.5, 0.0], 0.5),
        Collider::sphere([0.0, -0.8, 0.0], 0.2),
        Collider::sphere([0.0; 3], 0.5),
    ];
    let mut a = body([30.0, 31.5, 108.0], [1.0, 0.0, 0.0], 1.0);
    a.metadata[1] = 2;
    a.inverse_inertia = [1.0, 1.0, 1.0, 0.0];
    a.velocity[3] = 0.0;
    let mut b = body([30.99, 32.0, 108.0], [-1.0, 0.0, 0.0], 1.0);
    b.metadata[0] = 2;
    b.inverse_inertia = a.inverse_inertia;
    b.velocity[3] = 0.0;
    let out = pair(device, queue, terrain, &[a, b], &shapes, 1)?;
    // Normal impulse = (1+e)*2/(1+1+0.5^2)=16/9.
    let expected = 16.0 / 9.0;
    assert!(
        (out[0].velocity[0] - (1.0 - expected)).abs() < 0.001,
        "{out:?}"
    );
    assert!((out[1].velocity[0] - (-1.0 + expected)).abs() < 0.001);
    assert!((out[0].angular[2] - 0.5 * expected).abs() < 0.001);
    assert!(out[1].angular[2].abs() < 0.001);
    println!(
        "GPU_COMPOUND off_center impulse={:.6} spin={:.6}",
        1.0 - out[0].velocity[0],
        out[0].angular[2]
    );
    Ok(())
}

pub(super) fn run(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    head_on(device, queue, terrain)?;
    separated(device, queue, terrain)?;
    off_center(device, queue, terrain)?;
    Ok(())
}
