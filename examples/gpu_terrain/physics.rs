//! End-to-end measurements and contact regressions on the actual GPU.
use super::{Result, encoder};
use i_love_proland::terrain::{
    Region, RegionCell, RegionChart,
    gpu::{
        TerrainGpu, TerrainRequest,
        physics::{
            Collider, ContactMaterial, ContactMaterialField, PhysicsConfig, PhysicsGpu, RigidBody,
            SimulationClock, StepBatch,
        },
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
const fn body(position: [f32; 3], velocity: [f32; 3], count: u32) -> RigidBody {
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
const CONFIG: PhysicsConfig<'static> = PhysicsConfig {
    initial_tick: 0,
    hz: 240,
    gravity_m_s2: 9.81,
    contact_slop_m: 0.001,
    solver_iterations: 5,
    material: ContactMaterialField::Uniform(ContactMaterial::RIGID),
    stop_at_contact: false,
};
fn simulate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    initial: &[RigidBody],
    spheres: &[Collider],
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
    let sphere = [Collider::sphere([0.0; 3], 0.5)];
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
    let mut pending: Vec<_> = snapshots
        .into_iter()
        .map(i_love_proland::terrain::gpu::RecordedReadback::begin)
        .collect();
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

fn check_prediction(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    initial: &[RigidBody],
    sphere: &[Collider],
    expected: Option<u64>,
) -> Result<()> {
    let mut predictor = PhysicsGpu::new(
        device,
        terrain,
        0,
        PhysicsConfig {
            stop_at_contact: true,
            ..CONFIG
        },
        initial,
        sphere,
    )?;
    let mut prediction_commands = encoder(device);
    predictor.encode(
        device,
        &mut prediction_commands,
        terrain,
        StepBatch {
            start_tick: 0,
            steps: 960,
            hz: CONFIG.hz,
        },
    )?;
    let predicted = predictor.snapshot(device, &mut prediction_commands);
    queue.submit([prediction_commands.finish()]);
    let mut pending = predicted.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let predicted = pending
        .try_take()?
        .ok_or_else(|| io::Error::other("prediction unfinished"))?;
    assert_eq!(predicted[0].metadata[2], 3);
    assert_eq!(predicted[0].first_contact_tick(), expected);
    assert_eq!(
        predicted[0].first_contact_tick(),
        Some(predicted[0].completed_tick())
    );
    println!(
        "GPU_PHYSICS predicted_first_contact={:?}",
        predicted[0].first_contact_tick()
    );
    Ok(())
}

// Real solver regression: same initial state and fixed triangles, with
// rigid and compliant material; logged support/force must match equilibrium.
fn check_compliance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    collider: Collider,
    axes: [f32; 3],
    expected_height: f32,
) -> Result<(f32, f32)> {
    let sphere = [collider];
    let mut initial = body([32.0, 32.0, 103.0], [0.0; 3], 1);
    initial.position[3] = 0.001; // 1000 kg
    initial.inverse_inertia = [0.01, 0.01, 0.01, 0.0];
    let material = ContactMaterial {
        max_sinkage_m: 0.06,
        half_sinkage_pressure_pa: 35_000.0,
    };
    let (rigid, _) = simulate(device, queue, terrain, &[initial], &sphere, 64)?;
    let mut physics = PhysicsGpu::new(
        device,
        terrain,
        0,
        PhysicsConfig {
            material: ContactMaterialField::Uniform(material),
            ..CONFIG
        },
        &[initial],
        &sphere,
    )?;
    let mut commands = encoder(device);
    physics.encode(
        device,
        &mut commands,
        terrain,
        StepBatch {
            start_tick: 0,
            steps: 1024,
            hz: CONFIG.hz,
        },
    )?;
    let events = physics.encode_contacts(
        device,
        &mut commands,
        terrain,
        StepBatch {
            start_tick: 1024,
            steps: 416,
            hz: CONFIG.hz,
        },
    )?;
    let state = physics.snapshot(device, &mut commands);
    queue.submit([commands.finish()]);
    let mut events = events.begin();
    let mut state = state.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let events = events
        .try_take()?
        .ok_or_else(|| io::Error::other("contact observations unfinished"))?;
    let state = state
        .try_take()?
        .ok_or_else(|| io::Error::other("compliant state unfinished"))?;
    assert_eq!(events.len(), 416);
    let last = events
        .last()
        .ok_or_else(|| io::Error::other("no final contact"))?;
    assert_eq!(
        last.metadata[3], 1,
        "final state={:?}, event={last:?}",
        state[0]
    );
    assert_eq!(last.tick(), 1440);
    let force = last.position_force[3];
    let area = last.slip_area[3];
    let depth = last.normal_depth[3];
    let expected = CONFIG.contact_slop_m
        + material.max_sinkage_m * force / material.half_sinkage_pressure_pa.mul_add(area, force);
    let actual = rigid[0].position[2] - state[0].position[2];
    println!(
        "GPU_COMPLIANT_CONTACT rigid_z={:.6} compliant_z={:.6} delta={actual:.6} force_n={force:.3} area_m2={area:.6} depth_m={depth:.6} predicted_m={expected:.6} events={}",
        rigid[0].position[2],
        state[0].position[2],
        events.len()
    );
    assert!(actual > 0.0 && actual < material.max_sinkage_m + 0.005);
    assert!((rigid[0].position[2] - (100.0 + expected_height)).abs() < 0.004);
    // Independent ellipsoid section formula for axis-aligned resting
    // plane (the fixture rotates a principal axis exactly onto world Z).
    let normalized = (1.0 - depth.min(expected_height) / expected_height).powi(2);
    let predicted_area =
        std::f32::consts::PI * axes.iter().product::<f32>() / expected_height * (1.0 - normalized);
    assert!(
        (area - predicted_area).abs() < 0.001,
        "area {area} predicted {predicted_area}"
    );
    assert!((force - 9810.0).abs() < 100.0);
    assert!((depth - expected).abs() < 0.002);
    assert!(state[0].velocity[..3].iter().all(|v| v.abs() < 0.1));
    Ok((actual, area))
}

pub fn run(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let region = slope_region(0.0)?;
    let mut terrain = TerrainGpu::new(device, &region, 65, 1)?;
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
    let (sphere_sinkage, _) = check_compliance(
        device,
        queue,
        &terrain,
        Collider::sphere([0.0; 3], 0.5),
        [0.5; 3],
        0.5,
    )?;
    let axes = [0.9, 0.6, 0.2];
    let (wide_sinkage, wide_area) = check_compliance(
        device,
        queue,
        &terrain,
        Collider::ellipsoid([0.0; 3], axes, [0.0, 0.0, 0.0, 1.0]),
        axes,
        0.2,
    )?;
    let axes = [0.5, 0.5, 0.13];
    let rotation = [
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let (wheel_sinkage, wheel_area) = check_compliance(
        device,
        queue,
        &terrain,
        Collider::ellipsoid([0.0; 3], axes, rotation),
        axes,
        0.5,
    )?;
    assert!(wide_sinkage < sphere_sinkage && sphere_sinkage < wheel_sinkage);
    assert!(wide_area > wheel_area);
    println!(
        "GPU_ELLIPSOID_COMPLIANCE wide_sinkage_m={wide_sinkage:.6} wheel_sinkage_m={wheel_sinkage:.6} wide_area_m2={wide_area:.6} wheel_area_m2={wheel_area:.6}"
    );
    let sphere = [Collider::sphere([0.0; 3], 0.5)];
    let initial = [body([32.0, 32.0, 103.0], [0.0; 3], 1)];
    let (a, _) = simulate(device, queue, &terrain, &initial, &sphere, 64)?;
    check_prediction(
        device,
        queue,
        &terrain,
        &initial,
        &sphere,
        a[0].first_contact_tick(),
    )?;
    let (b, _) = simulate(device, queue, &terrain, &initial, &sphere, 7)?;
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&a),
        bytemuck::cast_slice::<_, u8>(&b)
    );
    assert_eq!(a[0].metadata, [0, 1, 0, 1]);
    assert_eq!(a[0].completed_tick(), 1440);
    assert!(
        (a[0].position[2] - 100.5).abs() < 0.015,
        "sphere settled at {:?}",
        a[0]
    );
    assert!(a[0].velocity[..3].iter().all(|v| v.abs() < 0.1));
    let feet = [
        Collider::sphere([-0.6, -0.6, -0.4], 0.25),
        Collider::sphere([0.6, -0.6, -0.4], 0.25),
        Collider::sphere([-0.6, 0.6, -0.4], 0.25),
        Collider::sphere([0.6, 0.6, -0.4], 0.25),
    ];
    let mut lander = body([32.0, 32.0, 103.0], [1.0, 0.0, 0.0], 4);
    lander.rotation = [0.099_833_42, 0.0, 0.0, 0.995_004_2];
    let (compound, _) = simulate(device, queue, &terrain, &[lander], &feet, 64)?;
    println!("GPU_PHYSICS landed={:?}", compound[0]);
    assert_eq!(compound[0].metadata, [0, 4, 0, 1]);
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
    check_slope(device, queue, &sphere)
}

fn check_slope(device: &wgpu::Device, queue: &wgpu::Queue, sphere: &[Collider]) -> Result<()> {
    let region = slope_region(0.1)?;
    let mut sloped = TerrainGpu::new(device, &region, 65, 1)?;
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
    let snapshot = sloped.snapshot(device, &mut commands, 0)?;
    queue.submit([commands.finish()]);
    let mut pending = snapshot.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let grid = pending
        .try_take()?
        .ok_or_else(|| io::Error::other("missing slope terrain samples"))?;
    let mut rolling = body([32.0, 32.0, 106.0], [0.0; 3], 1);
    rolling.clock = [u32::MAX - 100, 15, 0, 0];
    let start = rolling.completed_tick();
    let (result, _) = simulate(device, queue, &sloped, &[rolling], sphere, 128)?;
    let result = result[0];
    assert_eq!(result.completed_tick(), start + 1440);
    assert_eq!(result.metadata[2], 0);
    assert!(
        result
            .first_contact_tick()
            .is_some_and(|tick| tick > start && tick <= result.completed_tick())
    );
    let distance = geometry::sphere_clearance(&grid, region.chart(), result.position)?;
    assert!(
        (distance - 0.5).abs() < 0.035,
        "slope contact distance {distance}"
    );
    assert!(result.position[0] < 32.0);
    println!("GPU_PHYSICS slope={result:?} contact_distance_m={distance:.6} tick_rollover=true");
    Ok(())
}

#[path = "physics_geometry.rs"]
mod geometry;
