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
            f32::from(j % 7) * 0.5 - 1.5
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
    for (i, (a, b)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(a.features, b.features, "envelope features {i}");
        assert_eq!(a.point_a, b.point_a, "envelope point A {i}");
        assert_eq!(a.point_b, b.point_b, "envelope point B {i}");
        assert_eq!(a.normal, b.normal, "envelope normal {i}");
    }
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
    Ok(())
}
