//! Actual GPU benchmark for arbitrary relative mesh poses.
#[path = "mesh_contacts/sweep.rs"]
mod sweep;
use i_love_proland::terrain::gpu::mesh_contacts::{
    Domain, MeshCatalog, MeshContact, MeshPair, MeshQueries, MeshSource, Midphase,
};
use std::{error::Error, io, time::Instant};
use wgpu::util::DeviceExt;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Matrix = [[f32; 4]; 4];
const IDENTITY: Matrix = [
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
];

fn pose(angle: f32, x: f32, y: f32, z: f32) -> Matrix {
    let (sin, cos) = angle.sin_cos();
    [
        [cos, sin, 0., 0.],
        [-sin, cos, 0., 0.],
        [0., 0., 1., 0.],
        [x, y, z, 1.],
    ]
}
fn sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    queries: &MeshQueries,
) -> Result<Vec<MeshContact>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    catalog.encode(&mut encoder, queries);
    let snapshot = queries.snapshot(device, &mut encoder);
    queue.submit([encoder.finish()]);
    let mut pending = snapshot.begin();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(pending
        .try_take()?
        .ok_or_else(|| io::Error::other("missing mesh contact results"))?)
}

struct Geometry {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
}
impl Geometry {
    fn source(&self) -> MeshSource<'_> {
        MeshSource {
            positions: &self.positions,
            indices: &self.indices,
        }
    }
    fn tile(&mut self, x: f32, y: f32, size: f32) -> Result<()> {
        let i = u32::try_from(self.positions.len())?;
        self.positions.extend([
            [x, y, 0.],
            [x + size, y, 0.],
            [x + size, y + size, 0.],
            [x, y + size, 0.],
        ]);
        self.indices.extend([i, i + 1, i + 2, i, i + 2, i + 3]);
        Ok(())
    }
}

fn geometry() -> Result<[Geometry; 3]> {
    let triangle = Geometry {
        positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        indices: vec![0, 1, 2],
    };
    let mut frame = Geometry {
        positions: Vec::new(),
        indices: Vec::new(),
    };
    for y in 0..16_u16 {
        for x in 0..16_u16 {
            if !(3..13).contains(&x) || !(3..13).contains(&y) {
                frame.tile(f32::from(x) - 8., f32::from(y) - 8., 1.)?;
            }
        }
    }
    let panel = Geometry {
        positions: vec![[0., -0.2, 0.], [4., -0.2, 0.], [4., 0.2, 0.], [0., 0.2, 0.]],
        indices: vec![0, 1, 2, 0, 2, 3],
    };
    Ok([triangle, frame, panel])
}

struct Cases {
    poses: Vec<Matrix>,
    pairs: Vec<MeshPair>,
    rotations: std::ops::Range<usize>,
}
fn cases() -> Result<Cases> {
    let mut poses = vec![
        IDENTITY,
        pose(0., 0., 0., 0.05),
        pose(0., 0., 0., 0.25),
        pose(0., 0.7, 0.7, 0.),
        pose(0., 0.2, 0.2, 0.),
        pose(0., 0., 0., 0.),
    ];
    // Crossing vertical triangle; endpoints lie on opposite sides.
    poses.push([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., -1., 0., 0.],
        [0.2, 0.2, -0.5, 1.],
    ]);
    poses.push([
        [2., 0., 0., 0.],
        [0.4, 0.5, 0., 0.],
        [0., 0., 2., 0.],
        [0.1, 0.1, 0.02, 1.],
    ]);
    let mut pairs = (1..=4)
        .map(|i| MeshPair {
            meshes: [0, 0],
            transforms: [0, i],
        })
        .collect::<Vec<_>>();
    pairs.extend([
        MeshPair {
            meshes: [0, 0],
            transforms: [0, 6],
        },
        MeshPair {
            meshes: [0, 0],
            transforms: [0, 7],
        },
        MeshPair {
            meshes: [0, 1],
            transforms: [0, 0],
        },
        MeshPair {
            meshes: [2, 1],
            transforms: [5, 0],
        },
    ]);
    let test_count = pairs.len();
    // The rotating panel is translated toward the square frame. Its
    // corner may touch at some orientations and stay in the hole at others.
    for i in 0..360_u16 {
        let angle = f32::from(i).to_radians();
        let index = u32::try_from(poses.len())?;
        poses.push(pose(angle, 2.0, 0.0, 0.025));
        pairs.push(MeshPair {
            meshes: [2, 1],
            transforms: [index, 0],
        });
    }
    let rotation_end = pairs.len();
    for i in 0..36_u16 {
        let angle = f32::from(i) * std::f32::consts::TAU / 36.0;
        let index = u32::try_from(poses.len())?;
        poses.push(pose(angle, 12.0, 3.0 * angle.sin(), 0.05));
        pairs.push(MeshPair {
            meshes: [1, 1],
            transforms: [index, 0],
        });
    }
    Ok(Cases {
        poses,
        pairs,
        rotations: test_count..rotation_end,
    })
}

fn check(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let geometry = geometry()?;
    sweep::check(device, queue, &geometry[2])?;
    let catalog = MeshCatalog::new(
        device,
        &geometry.iter().map(Geometry::source).collect::<Vec<_>>(),
    )?;
    let Cases {
        poses,
        pairs,
        rotations,
    } = cases()?;
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test resident animated matrices"),
        contents: bytemuck::cast_slice(&poses),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let plain = catalog.queries(
        device,
        &matrices,
        &pairs,
        0.1,
        Midphase::Bvh,
        Domain::Surface,
    )?;
    let voxels = catalog.queries(
        device,
        &matrices,
        &pairs,
        0.1,
        Midphase::VoxelBvh,
        Domain::Surface,
    )?;
    let a = sample(device, queue, &catalog, &plain)?;
    let b = sample(device, queue, &catalog, &voxels)?;
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_eq!(a.features, b.features, "feature mismatch {i}: {a:?} {b:?}");
        assert_eq!(a.point_a, b.point_a, "point mismatch {i}");
        assert_eq!(a.point_b, b.point_b, "point mismatch {i}");
        assert_eq!(a.features[3], 0);
    }
    for (i, expected) in [true, false, false, true, true, true, false, false]
        .into_iter()
        .enumerate()
    {
        assert_eq!(a[i].features[2] > 0, expected, "fixture {i}: {:?}", a[i]);
    }
    assert!((a[0].point_a[3] - 0.0025).abs() < 1e-6);
    assert!(a[4].point_a[3] < 1e-10);
    assert!(a[5].point_a[3] < 0.000_401);
    let rotating = &a[rotations];
    let hits = rotating.iter().filter(|c| c.features[2] > 0).count();
    assert!(hits > 0 && hits < rotating.len());
    // Modify only the resident pose, not the mesh or bind group.
    queue.write_buffer(&matrices, 5 * 64, bytemuck::bytes_of(&pose(0., 3., 0., 0.)));
    let updated = sample(device, queue, &catalog, &voxels)?;
    assert!(updated[7].features[2] > 0);
    queue.write_buffer(&matrices, 5 * 64, bytemuck::bytes_of(&IDENTITY));
    println!(
        "MESH_ROTATION cases={} hits={hits} resident_bytes={} exact_mode_agreement=true",
        rotating.len(),
        catalog.resident_bytes()
    );
    let dual = catalog.queries(
        device,
        &matrices,
        &pairs,
        0.1,
        Midphase::DualBvh,
        Domain::Surface,
    )?;
    let dual_voxel = catalog.queries(
        device,
        &matrices,
        &pairs,
        0.1,
        Midphase::DualVoxelBvh,
        Domain::Surface,
    )?;
    let c = sample(device, queue, &catalog, &dual)?;
    let d = sample(device, queue, &catalog, &dual_voxel)?;
    for other in [&c, &d] {
        for (i, (a, b)) in a.iter().zip(other).enumerate() {
            assert_eq!(
                a.features, b.features,
                "dual feature mismatch {i}: {a:?} {b:?}"
            );
            assert_eq!(a.point_a, b.point_a, "dual point A {i}");
            assert_eq!(a.point_b, b.point_b, "dual point B {i}");
        }
    }
    for (name, q, results) in [
        ("bvh", &plain, &a),
        ("voxel_bvh", &voxels, &b),
        ("dual_bvh", &dual, &c),
        ("dual_voxel", &dual_voxel, &d),
    ] {
        benchmark(device, queue, &catalog, name, q, results)?;
    }
    Ok(())
}

fn benchmark(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    name: &str,
    queries: &MeshQueries,
    results: &[MeshContact],
) -> Result<()> {
    let mut times = Vec::new();
    for _ in 0..12 {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for _ in 0..16 {
            catalog.encode(&mut encoder, queries);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely())?;
        times.push(start.elapsed().as_secs_f64() * 1000. / 16.);
    }
    times.sort_by(f64::total_cmp);
    let sums = |i: usize| {
        results
            .iter()
            .map(|r| u64::from(r.statistics[i]))
            .sum::<u64>()
    };
    println!(
        "MESH_BENCH mode={name} pairs={} median_ms={:.6} visits={} triangle_tests={} mask_rejections={}",
        results.len(),
        times[6],
        sums(0),
        sums(1),
        sums(2)
    );

    Ok(())
}

fn main() -> Result<()> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    println!("MESH_GPU adapter={}", adapter.get_info().name);
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    check(&device, &queue)
}
