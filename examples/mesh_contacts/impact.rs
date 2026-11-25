use super::{Geometry, IDENTITY, Result};
use i_love_proland::terrain::gpu::{
    mesh_contacts::{
        BodyMeshInstance, BodyMeshPoses, CandidateDispatch, Domain, MeshCatalog, MeshImpact,
        MeshImpactLink, MeshImpulses, MeshPair, Midphase, QuerySettings,
    },
    physics::RigidBody,
};
use std::io;
use wgpu::util::DeviceExt;

const fn body(position: [f32; 3], velocity: [f32; 3], angular: [f32; 3]) -> RigidBody {
    RigidBody {
        position: [position[0], position[1], position[2], 1.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        velocity: [velocity[0], velocity[1], velocity[2], 0.0],
        angular: [angular[0], angular[1], angular[2], 1.0],
        inverse_inertia: [1.0, 1.0, 1.0, 0.0],
        metadata: [0, 1, 0, 0],
        clock: [0; 4],
    }
}

fn catalog(device: &wgpu::Device) -> Result<MeshCatalog> {
    let horizontal = Geometry {
        positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        indices: vec![0, 1, 2],
    };
    let vertical = Geometry {
        positions: vec![[0., 0., 0.], [0.1, 0., 0.], [0., 0., 0.1]],
        indices: vec![0, 1, 2],
    };
    let offset = Geometry {
        positions: vertical
            .positions
            .iter()
            .map(|p| [p[0] + 2.0, p[1], p[2]])
            .collect(),
        indices: vertical.indices.clone(),
    };
    Ok(MeshCatalog::new(
        device,
        &[horizontal.source(), offset.source(), vertical.source()],
    )?)
}

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let catalog = catalog(device)?;
    let states = [
        body([0.; 3], [0., 0., 1.], [0.; 3]),
        body([0., 0., 0.005], [0., 0., -1.], [0.; 3]),
        body([0., 10., 0.], [0.; 3], [0., 0., 1.]),
        body([2., 10.005, 0.], [0.; 3], [0.; 3]),
        body([0., 20., 0.], [0., 0., 1.], [0.; 3]),
        body([0., 20., 0.], [0., 0., -1.], [0.; 3]),
        body([0., 30., 0.], [0.; 3], [0.; 3]),
        body([2., 30.005, 0.], [0.; 3], [0.; 3]),
    ];
    let bodies = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("authoritative test rigid bodies"),
        contents: bytemuck::cast_slice(&states),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let instances = (0..8_u32)
        .map(|i| BodyMeshInstance {
            metadata: [i, 0, 0, 0],
            local: IDENTITY,
            local_rate: if i == 6 {
                // Unit-speed prescribed rotation around the local Z axis.
                [[0., 1., 0., 0.], [-1., 0., 0., 0.], [0.; 4], [0.; 4]]
            } else {
                [[0.; 4]; 4]
            },
        })
        .collect::<Vec<_>>();
    let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh body instance mappings"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let poses = BodyMeshPoses::new(device, &bodies, &instances)?;
    let links = [[0, 1], [2, 3], [4, 5], [6, 7]].map(|pair| MeshImpactLink {
        bodies: pair,
        instances: pair,
    });
    let pairs = [
        MeshPair {
            meshes: [0, 0],
            transforms: links[0].instances,
        },
        MeshPair {
            meshes: [1, 2],
            transforms: links[1].instances,
        },
        MeshPair {
            meshes: [0, 0],
            transforms: links[2].instances,
        },
        MeshPair {
            meshes: [1, 2],
            transforms: links[3].instances,
        },
    ];
    let queries = catalog.queries(
        device,
        poses.matrices(),
        &pairs,
        QuerySettings {
            margin: 0.01,
            mode: Midphase::VoxelBvh,
            domain: Domain::ClosedSolid,
            dispatch: CandidateDispatch::Direct,
        },
    )?;
    let solver = MeshImpulses::new(device, &bodies, &instances, queries.results(), &links)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    // Three GPU stages in queue order, no CPU transform or witness readback.
    poses.encode(&mut encoder);
    queries.encode(&mut encoder);
    solver.encode(&mut encoder);
    let output = solver.snapshot_states(device, &mut encoder);
    let impacts = solver.snapshot(device, &mut encoder);
    queue.submit([encoder.finish()]);
    let mut output = output.begin();
    let mut impacts = impacts.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let output = output
        .try_take()?
        .ok_or_else(|| io::Error::other("missing mesh body snapshot"))?;
    let impacts = impacts
        .try_take()?
        .ok_or_else(|| io::Error::other("missing mesh impulse snapshot"))?;
    validate(&states, &output, &impacts);
    Ok(())
}

fn validate(states: &[RigidBody], output: &[RigidBody], impacts: &[MeshImpact]) {
    assert_eq!(
        impacts.iter().map(|e| e.metadata[0]).collect::<Vec<_>>(),
        [1, 1, 3, 1],
        "{impacts:?}"
    );
    assert!((output[0].velocity[2] + 1.0).abs() < 1e-5);
    assert!((output[1].velocity[2] - 1.0).abs() < 1e-5);
    let expected = 2.0 / 3.0;
    assert!((impacts[1].point_speed[3] - 2.0).abs() < 1e-5);
    assert!((impacts[1].impulse_normal[3] - expected).abs() < 1e-5);
    assert!((output[2].velocity[1] + expected).abs() < 1e-5);
    assert!((output[3].velocity[1] - expected).abs() < 1e-5);
    assert!((output[2].angular[2] + 1.0 / 3.0).abs() < 1e-5);
    assert_eq!(
        bytemuck::bytes_of(&output[4]),
        bytemuck::bytes_of(&states[4])
    );
    assert!((impacts[3].point_speed[3] - 2.0).abs() < 1e-5);
    assert!((impacts[3].impulse_normal[3] - expected).abs() < 1e-5);
    assert!((output[6].angular[2] + 4.0 / 3.0).abs() < 1e-5);
    println!(
        "MESH_MOTOR_IMPACT stationary_owner=true local_panel_speed={:.6} impulse={:.6} owner_spin={:.6}",
        impacts[3].point_speed[3], impacts[3].impulse_normal[3], output[6].angular[2]
    );
    println!(
        "MESH_IMPACT elastic_exchange=true panel_speed={:.6} impulse={:.6} panel_spin={:.6} crossing_requires_refinement=true",
        impacts[1].point_speed[3], impacts[1].impulse_normal[3], output[2].angular[2]
    );
}
