//! End-to-end measurements and contact regressions on the actual GPU.
use super::{Result, encoder};
use i_love_proland::terrain::{
    Region, RegionCell, RegionChart,
    gpu::{
        TerrainGpu, TerrainRequest,
        physics::{PhysicsConfig, PhysicsGpu, RigidBody, SimulationClock, Sphere},
    },
};
use std::{
    io,
    time::{Duration, Instant},
};

fn slope_region(slope: f64) -> io::Result<Region> {
    let chart = RegionChart {
        radius_m: 600_000.0,
        latitude_rad: 0.0,
        longitude_rad: 0.0,
        spacing_m: 16.0,
        cells: 8,
    };
    let cells = (0..=chart.cells)
        .flat_map(|_| {
            (0..=chart.cells).map(move |x| {
                RegionCell::new(
                    (f64::from(x) * chart.spacing_m).mul_add(slope, 100.0),
                    None,
                    [0.3, 0.5, 0.2],
                )
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    Region::new(chart, cells, vec![])
}
fn body(position: [f32; 3], velocity: [f32; 3], count: u32) -> RigidBody {
    RigidBody {
        position: [position[0], position[1], position[2], 1.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        velocity: [velocity[0], velocity[1], velocity[2], 0.8],
        angular: [0.0, 0.0, 0.0, 0.0],
        inverse_inertia: [2.0, 2.0, 2.0, 0.0],
        metadata: [0, count, 0, 0],
        clock: [0; 4],
    }
}
const CONFIG: PhysicsConfig = PhysicsConfig {
    initial_tick: 0,
    hz: 240,
    gravity_m_s2: 9.81,
    contact_slop_m: 0.001,
    solver_iterations: 5,
};
fn simulate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    initial: &[RigidBody],
    spheres: &[Sphere],
    batch_size: u32,
) -> Result<(Vec<RigidBody>, f64)> {
    let initial_tick = initial[0].completed_tick();
    let mut physics = PhysicsGpu::new(
        device,
        terrain,
        0,
        PhysicsConfig {
            initial_tick,
            ..CONFIG
        },
        initial,
        spheres,
    )?;
    let mut clock = SimulationClock::at_tick(CONFIG.hz, initial_tick)?;
    // 60x warp, 100 ms host interval, 6 seconds of fixed simulation time.
    clock.advance(Duration::from_millis(100), 60_000_000)?;
    let start = Instant::now();
    let mut commands = encoder(device);
    while let Some(batch) = clock.plan(batch_size, 1440) {
        physics.encode(device, &mut commands, terrain, batch)?;
        clock.submitted(batch)?;
    }
    let snapshot = physics.snapshot(device, &mut commands);
    queue.submit([commands.finish()]);
    let mut pending = snapshot.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let output = pending
        .try_take()?
        .ok_or_else(|| io::Error::other("physics readback unfinished"))?;
    // Completion is the committed simulation time, not simply a submitted
    // dispatch fence. A stopped body's tick counter must be considered.
    let actual = output
        .iter()
        .map(RigidBody::completed_tick)
        .min()
        .ok_or_else(|| io::Error::other("no physics state"))?;
    clock.completed(actual)?;
    println!(
        "GPU_PHYSICS bodies={} batch_steps={batch_size} requested={} actual={} duration_ms={:.3} bytes={}",
        initial.len(),
        clock.target_tick(),
        clock.completed_tick(),
        start.elapsed().as_secs_f64() * 1000.0,
        output.len() * size_of::<RigidBody>()
    );
    Ok((output, start.elapsed().as_secs_f64()))
}
fn check_presentation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
) -> Result<()> {
    use i_love_proland::terrain::gpu::physics::{PresentationHistory, PresentationTime, StepBatch};
    let sphere = [Sphere {
        center: [0.0; 3],
        radius: 0.5,
    }];
    let mut initial = body([32.0, 32.0, 110.0], [1.0, 0.0, 0.0], 1);
    initial.clock = [u32::MAX - 100, 15, 0, 0];
    let epoch = initial.completed_tick();
    let mut physics = PhysicsGpu::new(
        device,
        terrain,
        0,
        PhysicsConfig {
            initial_tick: epoch,
            gravity_m_s2: 0.0,
            ..CONFIG
        },
        &[initial],
        &sphere,
    )?;
    let mut history = PresentationHistory::new(device, &physics)?;
    let mut commands = encoder(device);
    let midpoint = PresentationTime {
        tick: epoch + 120,
        fraction: 0.5,
    };
    assert!(
        history
            .interpolate(device, &mut commands, midpoint)
            .is_err()
    );
    history.capture(&mut commands);
    physics.encode(
        device,
        &mut commands,
        terrain,
        StepBatch {
            start_tick: epoch,
            steps: 240,
            hz: CONFIG.hz,
        },
    )?;
    history.capture(&mut commands);
    let mut snapshots = Vec::new();
    for time in [
        PresentationTime {
            tick: epoch - 10,
            fraction: 0.0,
        },
        midpoint,
        PresentationTime {
            tick: epoch + 500,
            fraction: 0.0,
        },
    ] {
        history.interpolate(device, &mut commands, time)?;
        snapshots.push(history.snapshot(device, &mut commands));
    }
    let final_state = physics.snapshot(device, &mut commands);
    queue.submit([commands.finish()]);
    let mut pending: Vec<_> = snapshots.into_iter().map(|s| s.begin()).collect();
    let mut final_state = final_state.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let poses = pending
        .iter_mut()
        .map(|p| {
            p.try_take()?
                .ok_or_else(|| io::Error::other("missing render pose"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let final_state = final_state
        .try_take()?
        .ok_or_else(|| io::Error::other("missing final physics state"))?;
    assert!((poses[0][0].position[0] - 32.0).abs() < 1e-5);
    assert!((poses[1][0].position[0] - (32.0 + 120.5 / 240.0)).abs() < 0.0005);
    assert!((poses[2][0].position[0] - 33.0).abs() < 0.0005);
    assert_eq!(final_state[0].completed_tick(), epoch + 240);
    println!(
        "GPU_PRESENTATION epoch={epoch} samples=[{:.6},{:.6},{:.6}] physics_ticks=240",
        poses[0][0].position[0], poses[1][0].position[0], poses[2][0].position[0]
    );
    Ok(())
}

pub(super) fn run(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let region = slope_region(0.0)?;
    let mut terrain = TerrainGpu::new(device, &region, 65, 1, 0, 0.0)?;
    let mut commands = encoder(device);
    terrain.encode(
        device,
        &mut commands,
        &[TerrainRequest {
            origin_ticks: [0, 0],
            spacing_ticks: 16,
            slot: 0,
        }],
    )?;
    queue.submit([commands.finish()]);
    check_presentation(device, queue, &terrain)?;
    let sphere = [Sphere {
        center: [0.0; 3],
        radius: 0.5,
    }];
    let initial = [body([32.0, 32.0, 103.0], [0.0; 3], 1)];
    let (a, _) = simulate(device, queue, &terrain, &initial, &sphere, 64)?;
    let (b, _) = simulate(device, queue, &terrain, &initial, &sphere, 7)?;
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&a),
        bytemuck::cast_slice::<_, u8>(&b)
    );
    assert_eq!(a[0].metadata, [0, 1, 0, 0]);
    assert_eq!(a[0].completed_tick(), 1440);
    assert!(
        (a[0].position[2] - 100.5).abs() < 0.015,
        "sphere settled at {:?}",
        a[0]
    );
    assert!(a[0].velocity[..3].iter().all(|v| v.abs() < 0.1));
    let feet = [
        Sphere {
            center: [-0.6, -0.6, -0.4],
            radius: 0.25,
        },
        Sphere {
            center: [0.6, -0.6, -0.4],
            radius: 0.25,
        },
        Sphere {
            center: [-0.6, 0.6, -0.4],
            radius: 0.25,
        },
        Sphere {
            center: [0.6, 0.6, -0.4],
            radius: 0.25,
        },
    ];
    let mut lander = body([32.0, 32.0, 103.0], [1.0, 0.0, 0.0], 4);
    lander.rotation = [0.09983342, 0.0, 0.0, 0.9950042];
    let (compound, _) = simulate(device, queue, &terrain, &[lander], &feet, 64)?;
    println!("GPU_PHYSICS landed={:?}", compound[0]);
    assert_eq!(compound[0].metadata, [0, 4, 0, 0]);
    assert_eq!(compound[0].completed_tick(), 1440);
    assert!((compound[0].position[2] - 100.65).abs() < 0.04);
    assert!(compound[0].rotation[0].abs() < 0.06);
    assert!(compound[0].velocity[..3].iter().all(|v| v.abs() < 0.1));
    // A batch cannot move an unsupported collider off the resident grid
    // or hide that its integration step exceeded the prototype's bound.
    let initial = [
        body([0.55, 32.0, 102.0], [-3.0, 0.0, 0.0], 1),
        body([32.0, 32.0, 105.0], [200.0, 0.0, 0.0], 1),
    ];
    let (stopped, _) = simulate(device, queue, &terrain, &initial, &sphere, 64)?;
    assert_eq!(stopped[0].metadata[2], 1);
    assert_eq!(stopped[1].metadata[2], 2);
    assert_eq!(stopped[1].completed_tick(), 0);
    let many = vec![lander; 128];
    simulate(device, queue, &terrain, &many, &feet, 128)?;
    let region = slope_region(0.1)?;
    let mut sloped = TerrainGpu::new(device, &region, 65, 1, 0, 0.0)?;
    let mut commands = encoder(device);
    sloped.encode(
        device,
        &mut commands,
        &[TerrainRequest {
            origin_ticks: [0, 0],
            spacing_ticks: 16,
            slot: 0,
        }],
    )?;
    queue.submit([commands.finish()]);
    let mut rolling = body([32.0, 32.0, 106.0], [0.0; 3], 1);
    rolling.clock = [u32::MAX - 100, 15, 0, 0];
    let start = rolling.completed_tick();
    let (result, _) = simulate(device, queue, &sloped, &[rolling], &sphere, 128)?;
    let result = result[0];
    assert_eq!(result.completed_tick(), start + 1440);
    assert_eq!(result.metadata[2], 0);
    let distance = (result.position[2] - 100.0 - 0.1 * result.position[0]) / 1.01_f32.sqrt();
    assert!(
        (distance - 0.5).abs() < 0.035,
        "slope contact distance {distance}"
    );
    assert!(result.position[0] < 30.0);
    println!("GPU_PHYSICS slope={result:?} tick_rollover=true");
    Ok(())
}
