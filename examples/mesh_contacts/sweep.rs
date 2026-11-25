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
fn translate(mut pair: SweepPair, start: [f32; 3], end: [f32; 3]) -> SweepPair {
    for (i, pose) in pair.poses.iter_mut().enumerate() {
        let delta = if i % 2 == 0 { start } else { end };
        for (x, dx) in pose.position[..3].iter_mut().zip(delta) {
            *x += dx;
        }
    }
    pair
}

fn check_comoving(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    settings: SweepSettings,
    pairs: &[SweepPair; 2],
    reference: &[SweepResult],
) -> Result<()> {
    let mut clear = pairs[1];
    clear.poses[1].rotation = clear.poses[0].rotation;
    let translated = [
        translate(
            pairs[0],
            [100_000.0, -10_000.0, 20_000.0],
            [101_000.0, 20_000.0, -30_000.0],
        ),
        translate(
            pairs[1],
            [100_000.0, -10_000.0, 20_000.0],
            [101_000.0, 20_000.0, -30_000.0],
        ),
        translate(clear, [0.0; 3], [1000.0, 2000.0, 3000.0]),
    ];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fast common translation sweep"),
        contents: bytemuck::cast_slice(&translated),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let actual = run(device, queue, catalog, &buffer, settings)?;
    for (a, b) in actual.iter().zip(reference) {
        assert_eq!(a.status, b.status, "common motion changed status");
        assert_eq!(a.interval, b.interval, "common motion changed interval");
        assert_eq!(
            a.contact.normal, b.contact.normal,
            "common motion changed normal"
        );
        assert_eq!(
            a.contact.features, b.contact.features,
            "common motion changed features"
        );
    }
    assert_eq!(
        actual[2].status[..2],
        [0, 1],
        "co-moving clear pair did not reject in one query"
    );
    println!(
        "MESH_SWEEP_COMOVING rotated_same_bracket=true rotated_same_queries={} clear_queries={} translation_m=3741.657",
        actual[0].status[1], actual[2].status[1]
    );
    Ok(())
}
fn check_linear_crossing(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    settings: SweepSettings,
) -> Result<()> {
    let identity = [0.0, 0.0, 0.0, 1.0];
    let panel = at([0.0; 3], identity);
    let pair = SweepPair {
        meshes: [0, 1],
        padding: [0; 2],
        poses: [
            panel,
            panel,
            at([1.0, 0.0, 1.0], identity),
            at([1.0, 0.0, -1.0], identity),
        ],
    };
    let shift = [1000.0, 2000.0, 3000.0];
    let pairs = [pair, translate(pair, [0.0; 3], shift)];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("relative linear crossing"),
        contents: bytemuck::cast_slice(&pairs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result = run(device, queue, catalog, &buffer, settings)?;
    let expected = (1.0 - f64::from(settings.margin)) * 0.5;
    let a = &result[0];
    let b = &result[1];
    assert_eq!(a.status[0], 1);
    assert!(f64::from(a.interval[0]) <= expected);
    assert!(expected - f64::from(a.interval[0]) < 0.001);
    assert_eq!(a.interval, b.interval);
    assert_eq!(a.status, b.status);
    assert_eq!(a.contact.features, b.contact.features);
    assert_eq!(a.contact.normal, b.contact.normal);
    for (a_point, b_point) in [
        (a.contact.point_a, b.contact.point_a),
        (a.contact.point_b, b.contact.point_b),
    ] {
        for axis in 0..3 {
            let expected = shift[axis].mul_add(a.interval[2], a_point[axis]);
            assert!(
                (expected - b_point[axis]).abs() < 0.000_5,
                "world witness was not restored"
            );
        }
    }
    println!(
        "MESH_SWEEP_LINEAR analytic={expected:.9} interval={:?} common_translation_preserved=true world_witness_restored=true",
        a.interval
    );
    Ok(())
}

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, panel: &Geometry) -> Result<()> {
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
    check_comoving(device, queue, &catalog, settings, &pairs, &result)?;
    check_linear_crossing(device, queue, &catalog, settings)?;
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
