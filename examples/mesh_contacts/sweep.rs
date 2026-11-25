use super::{Geometry, Result};
use i_love_proland::terrain::gpu::mesh_contacts::{
    MeshCatalog, RigidPose, SweepPair, SweepResult, SweepSettings,
};
use std::io;
use wgpu::util::DeviceExt;

const fn at(position: [f32; 3], rotation: [f32; 4]) -> RigidPose {
    RigidPose {
        position: [position[0], position[1], position[2], 0.0],
        rotation,
    }
}
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    pairs: &wgpu::Buffer,
    settings: SweepSettings,
) -> Result<Vec<SweepResult>> {
    let sweeps = catalog.sweeps(device, pairs, settings)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    sweeps.encode(&mut encoder);
    let snapshot = sweeps.snapshot(device, &mut encoder);
    queue.submit([encoder.finish()]);
    let mut pending = snapshot.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(pending
        .try_take()?
        .ok_or_else(|| io::Error::other("missing mesh sweep results"))?)
}
pub(super) fn check(device: &wgpu::Device, queue: &wgpu::Queue, panel: &Geometry) -> Result<()> {
    let point = Geometry {
        positions: vec![[0.0; 3]],
        indices: vec![0, 0, 0],
    };
    let catalog = MeshCatalog::new(device, &[panel.source(), point.source()])?;
    let identity = [0.0, 0.0, 0.0, 1.0];
    let start = at([0.0; 3], identity);
    let end = at([0.0; 3], [0.0, 0.0, 1.0, 0.0]);
    let obstacle = |y| at([0.0, y, 0.0], identity);
    let pair = |y| SweepPair {
        meshes: [0, 1],
        padding: [0; 2],
        poses: [start, end, obstacle(y), obstacle(y)],
    };
    let pairs = [pair(3.0), pair(5.0)];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rotating panel sweep"),
        contents: bytemuck::cast_slice(&pairs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let settings = SweepSettings {
        margin: 0.001,
        time_tolerance: 1.0 / 4096.0,
        trajectory_deviation_m: 0.0,
        query_budget: 4096,
    };
    let result = run(device, queue, &catalog, &buffer, settings)?;
    let expected = ((0.2 + f64::from(settings.margin)) / 3.0).acos() / std::f64::consts::PI;
    assert_eq!(result[0].status[0], 1, "{result:?}");
    assert!(
        f64::from(result[0].interval[0]) <= expected,
        "missed first contact {result:?}"
    );
    assert!(
        expected - f64::from(result[0].interval[0]) < 0.001,
        "excessive forecast conservatism"
    );
    assert!(result[0].interval[1] - result[0].interval[0] <= settings.time_tolerance);
    assert_eq!(result[1].status[0], 0);
    let limited = run(
        device,
        queue,
        &catalog,
        &buffer,
        SweepSettings {
            query_budget: 1,
            ..settings
        },
    )?;
    assert_eq!(limited[0].status[0], 3);
    assert_eq!(limited[0].interval[..2], [0.0, 1.0]);
    println!(
        "MESH_SWEEP rotating_panel analytic={expected:.9} interval={:?} queries={} clear_queries={} budget_explicit=true",
        result[0].interval, result[0].status[1], result[1].status[1]
    );
    Ok(())
}
