//! Root voxel envelope: reject holes without visiting original triangles.
use super::{Geometry, Matrix, Result, benchmark, cases, direct_settings, pose, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{
    CandidateDispatch, Domain, MeshCatalog, MeshPair, Midphase, QuerySettings,
};
use wgpu::util::DeviceExt;

fn hole_cases() -> Result<(Vec<Matrix>, Vec<MeshPair>)> {
    let mut fixture = cases()?;
    for i in 0..4097_u32 {
        let j = u16::try_from(i % 256)?;
        let angle = f32::from(j).to_radians();
        let x = if i % 16 == 0 {
            4.8
        } else {
            f32::from(j % 7).mul_add(0.5, -1.5)
        };
        let id = u32::try_from(fixture.poses.len())?;
        let mut m = pose(angle, x, 0.0, 0.025);
        if i % 3 == 0 {
            m[0] = m[0].map(|v| -v);
        }
        if i % 5 == 0 {
            m[1] = [0.2, 1.2, 0.0, 0.0];
        }
        fixture.poses.push(m);
        fixture.pairs.push(if i % 2 == 0 {
            MeshPair {
                meshes: [0, 1],
                transforms: [id, 0],
            }
        } else {
            MeshPair {
                meshes: [1, 0],
                transforms: [0, id],
            }
        });
    }
    Ok((fixture.poses, fixture.pairs))
}

fn cluster_fixture() -> Result<Geometry> {
    let mut mesh = Geometry {
        positions: Vec::new(),
        indices: Vec::new(),
    };
    for y in 0..24_u16 {
        for x in 0..24_u16 {
            mesh.tile(f32::from(x), f32::from(y), 1.0)?;
        }
    }
    Ok(mesh)
}
fn clusters(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let mesh = cluster_fixture()?;
    let catalog = MeshCatalog::new(device, &[mesh.source()])?;
    let poses = [
        super::IDENTITY,
        pose(0.1, 23.8, 4.0, 0.025),
        pose(0.4, 22.0, 10.0, 0.025),
        pose(0.0, 23.95, 23.95, 0.0),
    ];
    let pairs = (1..4)
        .map(|i| MeshPair {
            meshes: [0, 0],
            transforms: [0, i],
        })
        .collect::<Vec<_>>();
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cluster-local partial mesh overlap"),
        contents: bytemuck::cast_slice(&poses),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let settings = direct_settings(Midphase::VoxelBvh);
    let plain = catalog.queries(device, &matrices, &pairs, settings)?;
    let clustered = catalog.queries(
        device,
        &matrices,
        &pairs,
        QuerySettings {
            mode: Midphase::ClusteredVoxelBvh,
            ..settings
        },
    )?;
    let a = sample(device, queue, &plain)?;
    let b = sample(device, queue, &clustered)?;
    compare(&a, &b);
    assert!(
        b.iter().map(|r| r.statistics[0]).sum::<u32>()
            < a.iter().map(|r| r.statistics[0]).sum::<u32>()
    );
    benchmark(device, queue, "triangle_source", &plain, &a)?;
    benchmark(device, queue, "cluster_source", &clustered, &b)?;
    println!(
        "MESH_CLUSTERS triangles_per_asset={} cases={} exact=true",
        mesh.indices.len() / 3,
        pairs.len()
    );
    Ok(())
}

pub fn compare(expected: &[super::MeshContact], actual: &[super::MeshContact]) {
    for (i, (a, b)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(a.features, b.features, "envelope features {i}");
        assert_eq!(a.point_a, b.point_a, "envelope point A {i}");
        assert_eq!(a.point_b, b.point_b, "envelope point B {i}");
        assert_eq!(a.normal, b.normal, "envelope normal {i}");
    }
}
pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, geometry: &[Geometry]) -> Result<()> {
    let catalog = MeshCatalog::new(
        device,
        &geometry.iter().map(Geometry::source).collect::<Vec<_>>(),
    )?;
    let (poses, pairs) = hole_cases()?;
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sparse root envelope poses"),
        contents: bytemuck::cast_slice(&poses),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let settings = QuerySettings {
        domain: Domain::Surface,
        dispatch: CandidateDispatch::Compact,
        ..direct_settings(Midphase::VoxelBvh)
    };
    let plain = catalog.queries(device, &matrices, &pairs, settings)?;
    let envelope = catalog.queries(
        device,
        &matrices,
        &pairs,
        QuerySettings {
            dispatch: CandidateDispatch::CompactEnvelope,
            ..settings
        },
    )?;
    let expected = sample(device, queue, &plain)?;
    let actual = sample(device, queue, &envelope)?;
    compare(&expected, &actual);
    let count =
        |results: &[super::MeshContact]| results.iter().filter(|r| r.statistics[0] > 0).count();
    assert!(
        count(&actual) < count(&expected),
        "coarse envelope rejected no traversals"
    );
    benchmark(device, queue, "root_aabb", &plain, &expected)?;
    benchmark(device, queue, "root_envelope", &envelope, &actual)?;
    println!(
        "MESH_ENVELOPE pairs={} traversed_before={} traversed_after={} exact=true",
        pairs.len(),
        count(&expected),
        count(&actual)
    );
    clusters(device, queue)?;
    Ok(())
}
