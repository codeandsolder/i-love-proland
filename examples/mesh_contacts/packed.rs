//! Exact comparisons of independent pair widths, including tail lanes.
use super::{Geometry, Result, benchmark, cases, direct_settings, pose, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{
    CandidateDispatch, Domain, MeshCatalog, MeshPair, Midphase, PairWidth, QuerySettings,
};
use wgpu::util::DeviceExt;

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, geometry: &[Geometry]) -> Result<()> {
    let cube = super::containment::cube(0.5);
    let mut sources = geometry.iter().map(Geometry::source).collect::<Vec<_>>();
    sources.push(cube.source());
    let catalog = MeshCatalog::new(device, &sources)?;
    let mut fixture = cases()?;
    let mixed_count = fixture.pairs.len();
    // 4097 simple stock-like box/triangle candidate pairs. Reuse a small
    // resident transform set, varying contact status and distance.
    for i in 0..4097_u32 {
        let j = u16::try_from(i % 256)?;
        let id = u32::try_from(fixture.poses.len())?;
        fixture.poses.push(pose(
            f32::from(j) * 0.025,
            f32::from(j % 7) * 0.4,
            0.1,
            f32::from(j % 3) * 0.05,
        ));
        fixture.pairs.push(MeshPair {
            meshes: [0, 0],
            transforms: [0, id],
        });
    }
    let triangle_end = fixture.pairs.len();
    for i in 0..257_u32 {
        let j = u16::try_from(i)?;
        let id = u32::try_from(fixture.poses.len())?;
        fixture.poses.push(pose(
            f32::from(j) * 0.025,
            f32::from(j % 7) * 0.4,
            0.1,
            f32::from(j % 3) * 0.05,
        ));
        fixture.pairs.push(MeshPair {
            meshes: if i % 3 == 0 { [0, 3] } else { [3, 3] },
            transforms: [0, id],
        });
    }
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("packed query fixtures"),
        contents: bytemuck::cast_slice(&fixture.poses),
        usage: wgpu::BufferUsages::STORAGE,
    });
    for (name, pairs) in [
        ("mixed", &fixture.pairs[..mixed_count]),
        ("small", &fixture.pairs[mixed_count..triangle_end]),
        ("box", &fixture.pairs[triangle_end..]),
    ] {
        compare(device, queue, &catalog, &matrices, pairs, name)?;
    }
    compare_dual(
        device,
        queue,
        &catalog,
        &matrices,
        &fixture.pairs[..mixed_count],
    )?;
    Ok(())
}

fn compare_dual(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    matrices: &wgpu::Buffer,
    pairs: &[MeshPair],
) -> Result<()> {
    let settings = QuerySettings {
        domain: Domain::ClosedSolid,
        ..direct_settings(Midphase::DualVoxelBvh)
    };
    let reference = catalog.queries(device, matrices, pairs, settings)?;
    let expected = sample(device, queue, &reference)?;
    for width in [
        PairWidth::One,
        PairWidth::Four,
        PairWidth::Eight,
        PairWidth::Sixteen,
        PairWidth::ThirtyTwo,
    ] {
        let queries = catalog.queries(
            device,
            matrices,
            pairs,
            QuerySettings {
                dispatch: CandidateDispatch::Packed(width),
                ..settings
            },
        )?;
        let actual = sample(device, queue, &queries)?;
        assert_same(&expected, &actual, "dual", width, false);
    }
    println!(
        "MESH_PACKED_DUAL cases={} exact_all_widths=true",
        pairs.len()
    );
    Ok(())
}

fn assert_same(
    expected: &[super::MeshContact],
    actual: &[super::MeshContact],
    name: &str,
    width: PairWidth,
    statistics: bool,
) {
    for (i, (a, b)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(a.features, b.features, "{name} {width:?} feature {i}");
        assert_eq!(a.point_a, b.point_a, "{name} {width:?} point A {i}");
        assert_eq!(a.point_b, b.point_b, "{name} {width:?} point B {i}");
        assert_eq!(a.normal, b.normal, "{name} {width:?} normal {i}");
        if statistics {
            assert_eq!(a.statistics, b.statistics, "{name} {width:?} stats {i}");
        }
    }
}

fn compare(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    catalog: &MeshCatalog,
    matrices: &wgpu::Buffer,
    pairs: &[MeshPair],
    name: &str,
) -> Result<()> {
    let settings = QuerySettings {
        domain: Domain::ClosedSolid,
        ..direct_settings(Midphase::VoxelBvh)
    };
    let reference = catalog.queries(device, matrices, pairs, settings)?;
    let expected = sample(device, queue, &reference)?;
    let mut measurements = vec![(format!("{name}_wide64"), reference, expected.clone())];
    for width in [
        PairWidth::One,
        PairWidth::Four,
        PairWidth::Eight,
        PairWidth::Sixteen,
        PairWidth::ThirtyTwo,
    ] {
        let queries = catalog.queries(
            device,
            matrices,
            pairs,
            QuerySettings {
                dispatch: CandidateDispatch::Packed(width),
                ..settings
            },
        )?;
        let actual = sample(device, queue, &queries)?;
        assert_same(&expected, &actual, name, width, true);
        measurements.push((format!("{name}_packed_{width:?}"), queries, actual));
    }
    if device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    {
        super::timing::compare(device, queue, &measurements)?;
    } else {
        for (label, queries, results) in &measurements {
            benchmark(device, queue, label, queries, results)?;
        }
    }
    println!(
        "MESH_PACKED workload={name} cases={} exact_all_widths=true",
        pairs.len()
    );
    Ok(())
}
