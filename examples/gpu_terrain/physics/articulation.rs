use super::{
    CONFIG, Collider, PhysicsConfig, PhysicsGpu, Result, RigidBody, StepBatch, TerrainGpu, body,
    encoder, io,
};
use i_love_proland::terrain::gpu::physics::{
    ArticulationLayout, Island, Joint, JointDrive, JointFrame, JointKind,
};

const HZ: u32 = 240;
const SLIDER_FRAME: [f32; 4] = [
    0.0,
    -std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,
];

fn read_state(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    physics: &mut PhysicsGpu,
    terrain: &TerrainGpu,
    first: u64,
    steps: u32,
) -> Result<Vec<RigidBody>> {
    let mut commands = encoder(device);
    physics.encode(
        device,
        &mut commands,
        terrain,
        StepBatch {
            start_tick: first,
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
        .ok_or_else(|| io::Error::other("articulated snapshot unfinished"))?)
}

fn suspension(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    let shapes = [
        Collider::sphere([0.0; 3], 0.4),
        Collider::ellipsoid([0.0; 3], [0.35, 0.22, 0.2], [0.0, 0.0, 0.0, 1.0]),
    ];
    let mut hull = body([32.0, 32.0, 103.0], [0.0; 3], 1);
    hull.position[3] = 0.001;
    hull.inverse_inertia = [0.01, 0.01, 0.01, 0.0];
    let mut foot = body([32.0, 32.0, 102.0], [0.0; 3], 1);
    foot.position[3] = 0.025;
    foot.inverse_inertia = [0.2, 0.2, 0.2, 0.0];
    foot.metadata[0] = 1;
    let spring = Joint {
        bodies: [0, 1],
        kind: JointKind::Prismatic,
        frames: [
            JointFrame {
                anchor: [0.0, 0.0, -1.0],
                rotation: SLIDER_FRAME,
            },
            JointFrame {
                anchor: [0.0; 3],
                rotation: SLIDER_FRAME,
            },
        ],
        limits: Some([0.0, 0.5]),
        drive: JointDrive {
            target_position: 0.0,
            target_velocity: 0.0,
            stiffness: 65_400.0,
            damping: 9_000.0,
            max_force: 1e6,
        },
    };
    let islands = [Island {
        first_body: 0,
        body_count: 2,
        first_joint: 0,
        joint_count: 1,
    }];
    let mut physics = PhysicsGpu::articulated(
        device,
        terrain,
        0,
        PhysicsConfig {
            solver_iterations: 24,
            ..CONFIG
        },
        &[hull, foot],
        &shapes,
        ArticulationLayout {
            islands: &islands,
            joints: &[spring],
        },
    )?;
    let _ = read_state(device, queue, &mut physics, terrain, 0, 1024)?;
    let mut commands = encoder(device);
    let batch = StepBatch {
        start_tick: 1024,
        steps: 416,
        hz: HZ,
    };
    let contacts = physics.encode_contacts(device, &mut commands, terrain, batch)?;
    let snapshot = physics.snapshot(device, &mut commands);
    queue.submit([commands.finish()]);
    let mut pending = snapshot.begin();
    let mut events = contacts.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let states = pending
        .try_take()?
        .ok_or_else(|| io::Error::other("missing joint states"))?;
    let events = events
        .try_take()?
        .ok_or_else(|| io::Error::other("missing contact forces"))?;
    let compression = 1.0 + states[1].position[2] - states[0].position[2];
    let force = events
        .last()
        .ok_or_else(|| io::Error::other("missing foot contact"))?
        .position_force[3];
    println!(
        "GPU_SUSPENSION compression_m={compression:.6} expected_m=0.15 force_n={force:.3} hull={:?} foot={:?}",
        states[0].position, states[1].position
    );
    assert!(
        states
            .iter()
            .all(|state| state.completed_tick() == 1440 && state.metadata[2] == 0)
    );
    assert!((compression - 0.15).abs() < 0.012);
    assert!((force - 10202.4).abs() < 100.0);
    assert!((states[1].position[2] - 100.2).abs() < 0.004);
    assert!(
        states
            .iter()
            .all(|state| state.velocity[..3].iter().all(|v| v.abs() < 0.1))
    );
    Ok(())
}

fn hinges(device: &wgpu::Device, queue: &wgpu::Queue, terrain: &TerrainGpu) -> Result<()> {
    let shapes = [Collider::sphere([0.0; 3], 0.2)];
    let mut initial = Vec::new();
    for x in [20.0, 40.0, 50.0] {
        initial.extend([
            body([x, 32.0, 110.0], [0.0; 3], 1),
            body([x, 32.0, 110.0], [0.0; 3], 1),
        ]);
    }
    initial[4].angular[0] = 1.0;
    initial[5].angular[0] = -1.0;
    let islands = [
        Island {
            first_body: 0,
            body_count: 2,
            first_joint: 0,
            joint_count: 1,
        },
        Island {
            first_body: 2,
            body_count: 2,
            first_joint: 1,
            joint_count: 1,
        },
        Island {
            first_body: 4,
            body_count: 2,
            first_joint: 2,
            joint_count: 1,
        },
    ];
    let drive = JointDrive {
        target_position: 0.0,
        target_velocity: 1.0,
        stiffness: 0.0,
        damping: 20.0,
        max_force: 10.0,
    };
    let joints = [
        Joint {
            bodies: [0, 1],
            kind: JointKind::Revolute,
            frames: [JointFrame::IDENTITY; 2],
            limits: Some([-0.4, 0.4]),
            drive,
        },
        Joint {
            bodies: [2, 3],
            kind: JointKind::Fixed,
            frames: [JointFrame::IDENTITY; 2],
            limits: None,
            drive: JointDrive::PASSIVE,
        },
        Joint {
            bodies: [4, 5],
            kind: JointKind::Spherical,
            frames: [JointFrame::IDENTITY; 2],
            limits: None,
            drive: JointDrive::PASSIVE,
        },
    ];
    let mut physics = PhysicsGpu::articulated(
        device,
        terrain,
        0,
        PhysicsConfig {
            gravity_m_s2: 0.0,
            solver_iterations: 12,
            ..CONFIG
        },
        &initial,
        &shapes,
        ArticulationLayout {
            islands: &islands,
            joints: &joints,
        },
    )?;
    let states = read_state(device, queue, &mut physics, terrain, 0, 240)?;
    let angle = |state: RigidBody| 2.0 * state.rotation[0].atan2(state.rotation[3]);
    let hinge = angle(states[1]) - angle(states[0]);
    println!(
        "GPU_JOINT hinge_at_limit_rad={hinge:.6} opposing_omega={:?} ball_free_rad={:.6}",
        [states[0].angular[0], states[1].angular[0]],
        angle(states[5]) - angle(states[4])
    );
    assert!((hinge - 0.4).abs() < 0.01);
    assert!((states[0].angular[0] + states[1].angular[0]).abs() < 1e-4);
    assert!((angle(states[3]) - angle(states[2])).abs() < 1e-4);
    assert!((angle(states[5]) - angle(states[4]) + 2.0).abs() < 0.01);
    reverse_hinge(device, queue, terrain, &mut physics, drive)
}

fn reverse_hinge(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
    physics: &mut PhysicsGpu,
    drive: JointDrive,
) -> Result<()> {
    let angle = |state: RigidBody| 2.0 * state.rotation[0].atan2(state.rotation[3]);
    let mut commands = encoder(device);
    physics.encode_joint_drive(
        device,
        &mut commands,
        0,
        JointDrive {
            target_velocity: -1.0,
            ..drive
        },
    )?;
    physics.encode(
        device,
        &mut commands,
        terrain,
        StepBatch {
            start_tick: 240,
            steps: 240,
            hz: HZ,
        },
    )?;
    let snapshot = physics.snapshot(device, &mut commands);
    queue.submit([commands.finish()]);
    let mut pending = snapshot.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let states = pending
        .try_take()?
        .ok_or_else(|| io::Error::other("missing reversed hinge"))?;
    let reversed = angle(states[1]) - angle(states[0]);
    println!("GPU_JOINT reversed_rad={reversed:.6}");
    assert!((reversed + 0.4).abs() < 0.01);
    Ok(())
}

// A third body hits one end of a fixed assembly. The impulse must
// propagate into the other linked body, but overlapping colliders inside
// the same island must never generate internal pair corrections.
fn external_contact(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
) -> Result<()> {
    let shape = Collider::sphere([0.0; 3], 0.5);
    let mut bodies = [
        body([30.0, 32.0, 108.0], [0.0; 3], 1),
        body([30.7, 32.0, 108.0], [0.0; 3], 1),
        body([32.0, 32.0, 108.0], [-2.0, 0.0, 0.0], 1),
    ];
    for b in &mut bodies {
        b.velocity[3] = 0.0;
    }
    let joints = [Joint {
        bodies: [0, 1],
        kind: JointKind::Fixed,
        frames: [
            JointFrame {
                anchor: [0.7, 0.0, 0.0],
                ..JointFrame::IDENTITY
            },
            JointFrame::IDENTITY,
        ],
        limits: None,
        drive: JointDrive::PASSIVE,
    }];
    let islands = [
        Island {
            first_body: 0,
            body_count: 2,
            first_joint: 0,
            joint_count: 1,
        },
        Island {
            first_body: 2,
            body_count: 1,
            first_joint: 1,
            joint_count: 0,
        },
    ];
    let mut physics = PhysicsGpu::articulated(
        device,
        terrain,
        0,
        PhysicsConfig {
            gravity_m_s2: 0.0,
            solver_iterations: 12,
            ..CONFIG
        },
        &bodies,
        &[shape],
        ArticulationLayout {
            islands: &islands,
            joints: &joints,
        },
    )?;
    physics.enable_body_pairs(device)?;
    let before = read_state(device, queue, &mut physics, terrain, 0, 24)?;
    assert!((before[1].position[0] - before[0].position[0] - 0.7).abs() < 0.001);
    assert!(before[0].velocity[0].abs() < 0.001 && before[1].velocity[0].abs() < 0.001);
    let after = read_state(device, queue, &mut physics, terrain, 24, 216)?;
    let momentum = after.iter().map(|b| b.velocity[0]).sum::<f32>();
    assert!((momentum + 2.0).abs() < 0.001, "{after:?}");
    assert!(
        after[0].velocity[0] < -0.2 && after[1].velocity[0] < -0.2,
        "{after:?}"
    );
    assert!((after[1].position[0] - after[0].position[0] - 0.7).abs() < 0.01);
    assert!((after[1].velocity[0] - after[0].velocity[0]).abs() < 0.01);
    println!(
        "GPU_ARTICULATED_PAIR velocities={:?} momentum={momentum:.6} spacing={:.6}",
        after.iter().map(|b| b.velocity[0]).collect::<Vec<_>>(),
        after[1].position[0] - after[0].position[0]
    );
    Ok(())
}

pub(super) fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain: &TerrainGpu,
) -> Result<()> {
    suspension(device, queue, terrain)?;
    hinges(device, queue, terrain)?;
    external_contact(device, queue, terrain)
}
