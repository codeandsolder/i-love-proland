use super::{Geometry, IDENTITY, Result, benchmark, cases, pose, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{
    CandidateDispatch, Domain, MeshCatalog, MeshPair, Midphase, QuerySettings,
};
use wgpu::util::DeviceExt;

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, geometry: &[Geometry]) -> Result<()> {
    let catalog = MeshCatalog::new(
        device,
        &geometry.iter().map(Geometry::source).collect::<Vec<_>>(),
    )?;
    let mut cases = cases()?;
    let distant = u32::try_from(cases.poses.len())?;
    cases.poses.push(pose(0.3, 500.0, 500.0, 500.0));
    let singular = u32::try_from(cases.poses.len())?;
    cases.poses.push([[0.; 4]; 4]);
    cases.pairs.extend((0..4096).map(|_| MeshPair {
        meshes: [1, 1],
        transforms: [0, distant],
    }));
    cases.pairs.push(MeshPair {
        meshes: [1, 1],
        transforms: [0, singular],
    });
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("broadphase fixture poses"),
        contents: bytemuck::cast_slice(&cases.poses),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let settings = QuerySettings {
        margin: 0.1,
        mode: Midphase::VoxelBvh,
        domain: Domain::ClosedSolid,
        dispatch: CandidateDispatch::Direct,
    };
    let direct = catalog.queries(device, &matrices, &cases.pairs, settings)?;
    let compact = catalog.queries(
        device,
        &matrices,
        &cases.pairs,
        QuerySettings {
            dispatch: CandidateDispatch::Compact,
            ..settings
        },
    )?;
    let expected = sample(device, queue, &direct)?;
    let actual = sample(device, queue, &compact)?;
    for (a, b) in expected.iter().zip(&actual) {
        assert_eq!(a.features, b.features);
        assert_eq!(a.point_a, b.point_a);
        assert_eq!(a.point_b, b.point_b);
        assert_eq!(a.normal, b.normal);
    }
    assert_eq!(actual.last().map(|a| a.features[3]), Some(1));
    benchmark(device, queue, "direct_sparse", &direct, &expected)?;
    benchmark(device, queue, "compact_sparse", &compact, &actual)?;
    // The same batch must reset its atomic counter on every dispatch and
    // clear old results when the resident poses move away.
    queue.write_buffer(
        &matrices,
        0,
        bytemuck::bytes_of(&pose(0.0, 900.0, 900.0, 900.0)),
    );
    let expected = sample(device, queue, &direct)?;
    let actual = sample(device, queue, &compact)?;
    for (a, b) in expected.iter().zip(&actual) {
        assert_eq!(a.features, b.features);
        assert_eq!(a.point_a, b.point_a);
    }
    queue.write_buffer(&matrices, 0, bytemuck::bytes_of(&IDENTITY));
    println!(
        "MESH_COMPACT pairs={} exact=true invalid_pose_preserved=true repeated_pose_updates=true",
        actual.len()
    );
    Ok(())
}
