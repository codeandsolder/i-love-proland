//! Actual device regressions for interacting spheres and static triangles.
use super::{Result, encoder};
use i_love_proland::terrain::gpu::{
    TerrainGpu,
    physics::{
        Collider, ContactMaterial, ContactMaterialField, PhysicsConfig, PhysicsGpu, RigidBody,
        StaticScene, StaticTriangle, StepBatch,
    },
};
use std::{io, time::Instant};

const HZ: u32 = 240;

const fn body(position: [f32; 3], velocity: [f32; 3], restitution: f32) -> RigidBody {
    RigidBody {
        position: [position[0], position[1], position[2], 1.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        velocity: [velocity[0], velocity[1], velocity[2], 0.45],
        angular: [0.0, 0.0, 0.0, restitution],
        inverse_inertia: [10.0, 10.0, 10.0, 0.0],
        metadata: [0, 1, 0, 0],
        clock: [0; 4],
    }
}

const fn config(gravity: f32) -> PhysicsConfig<'static> {
    PhysicsConfig {
        initial_tick: 0,
        hz: HZ,
        gravity_m_s2: gravity,
        contact_slop_m: 0.001,
        solver_iterations: 5,
        material: ContactMaterialField::Uniform(ContactMaterial::RIGID),
        stop_at_contact: false,
    }
}

fn sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    physics: &mut PhysicsGpu,
    start_tick: u64,
    steps: u32,
) -> Result<Vec<RigidBody>> {
    let mut commands = encoder(device);
    physics.encode(
        device,
        &mut commands,
        terrain,
        StepBatch {
            start_tick,
            steps,
            hz: HZ,
        },
    )?;
    let snapshot = physics.snapshot(device, &mut commands);
    queue.submit([commands.finish()]);
    let mut pending = snapshot.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(pending
        .try_take()?
        .ok_or_else(|| io::Error::other("marble readback not complete"))?)
}

fn wall() -> [StaticTriangle; 2] {
    let [a, b, c, d] = [
        [36.0, 8.0, 99.0],
        [36.0, 56.0, 99.0],
        [36.0, 56.0, 118.0],
        [36.0, 8.0, 118.0],
    ];
    [[a, b, c], [a, c, d]].map(|vertices| StaticTriangle {
        vertices,
        material: 0,
    })
}

pub fn run(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    let shapes = [Collider::sphere([0.0; 3], 0.5)];
    let initial = [
        body([30.0, 32.0, 108.0], [2.0, 0.0, 0.0], 1.0),
        body([32.0, 32.0, 108.0], [-2.0, 0.0, 0.0], 1.0),
    ];
    let mut physics = PhysicsGpu::new(device, terrain, 0, config(0.0), &initial, &shapes)?;
    physics.enable_sphere_pairs(device)?;
    let bounced = sample(device, queue, terrain, &mut physics, 0, 100)?;
    assert!((bounced[0].velocity[0] + 2.0).abs() < 0.01, "{bounced:?}");
    assert!((bounced[1].velocity[0] - 2.0).abs() < 0.01, "{bounced:?}");
    assert!((bounced[0].position[0] + bounced[1].position[0] - 62.0).abs() < 0.001);
    println!(
        "GPU_MARBLES elastic pair velocities={:?}",
        bounced.iter().map(|b| b.velocity[0]).collect::<Vec<_>>()
    );

    let initial = [body([32.0, 32.0, 108.0], [8.0, 0.0, 0.0], 0.8)];
    let mut physics = PhysicsGpu::new(device, terrain, 0, config(0.0), &initial, &shapes)?;
    physics.set_static_scene(StaticScene::new(device, &wall())?)?;
    let bounced = sample(device, queue, terrain, &mut physics, 0, 180)?;
    assert!(
        bounced[0].velocity[0] < -6.3 && bounced[0].position[0] < 35.5,
        "{bounced:?}"
    );
    assert_eq!(bounced[0].completed_tick(), 180);
    println!("GPU_MARBLES static wall velocity={:?}", bounced[0].velocity);

    let initial = (0..50_u16)
        .map(|i| {
            body(
                [
                    f32::from(i % 5).mul_add(1.15, 24.0),
                    f32::from((i / 5) % 5).mul_add(1.15, 20.0),
                    f32::from(i / 25).mul_add(1.15, 104.0),
                ],
                [
                    f32::from(i % 3).mul_add(0.25, 4.0),
                    0.2_f32.mul_add(f32::from(i % 7), -0.6),
                    0.0,
                ],
                0.65,
            )
        })
        .collect::<Vec<_>>();
    let mut physics = PhysicsGpu::new(device, terrain, 0, config(9.81), &initial, &shapes)?;
    physics.set_static_scene(StaticScene::new(device, &wall())?)?;
    physics.enable_sphere_pairs(device)?;
    let mut contacts = 0;
    let mut bounced = 0;
    let mut peak_penetration = 0.0_f32;
    let start = Instant::now();
    for frame in 0..120_u64 {
        let result = sample(device, queue, terrain, &mut physics, frame * 4, 4)?;
        assert!(
            result
                .iter()
                .all(|b| b.metadata[2] == 0 && b.completed_tick() == (frame + 1) * 4)
        );
        assert!(result.iter().all(|b| {
            b.position
                .iter()
                .chain(&b.velocity)
                .chain(&b.angular)
                .all(|x| x.is_finite())
        }));
        contacts = result
            .iter()
            .filter(|b| b.first_contact_tick().is_some())
            .count();
        bounced += result
            .iter()
            .filter(|b| b.velocity[2] > 0.5 || b.velocity[0] < -0.5)
            .count();
        for (i, a) in result.iter().enumerate() {
            for b in &result[i + 1..] {
                let distance = (0..3)
                    .map(|j| (a.position[j] - b.position[j]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                peak_penetration = peak_penetration.max(1.0 - distance);
            }
        }
    }
    assert!(contacts >= 40, "only {contacts} contacts");
    assert!(bounced > 50, "no realistic bounces");
    assert!(
        peak_penetration < 0.25,
        "peak penetration {peak_penetration}"
    );
    println!(
        "GPU_MARBLES bodies=50 ticks=480 frames=120 contacts={contacts} bounce_observations={bounced} peak_pair_penetration_m={peak_penetration:.6} wall_ms={:.3} frame_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0,
        start.elapsed().as_secs_f64() * 1000.0 / 120.0
    );
    Ok(())
}
