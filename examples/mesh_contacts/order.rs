use super::{Geometry, Result, benchmark, cases, direct_settings, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{MeshCatalog, MeshPair, Midphase};
use wgpu::util::DeviceExt;

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, geometry: &[Geometry]) -> Result<()> {
    let catalog = MeshCatalog::new(
        device,
        &geometry.iter().map(Geometry::source).collect::<Vec<_>>(),
    )?;
    let fixture = cases()?;
    let original = &fixture.pairs[fixture.rotations];
    let reversed = original
        .iter()
        .map(|p| MeshPair {
            meshes: [p.meshes[1], p.meshes[0]],
            transforms: [p.transforms[1], p.transforms[0]],
        })
        .collect::<Vec<_>>();
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("asymmetric query order fixture"),
        contents: bytemuck::cast_slice(&fixture.poses),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let settings = direct_settings(Midphase::VoxelBvh);
    let forward = catalog.queries(device, &matrices, original, settings)?;
    let reverse = catalog.queries(device, &matrices, &reversed, settings)?;
    let a = sample(device, queue, &forward)?;
    let b = sample(device, queue, &reverse)?;
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(
            a.features,
            [b.features[1], b.features[0], b.features[2], b.features[3]]
        );
        assert_eq!(a.point_a[..3], b.point_b[..3]);
        assert_eq!(a.point_b[..3], b.point_a[..3]);
        assert_eq!(a.point_a[3].to_bits(), b.point_a[3].to_bits());
        assert_eq!(a.statistics, b.statistics);
        assert!(
            a.normal[..3]
                .iter()
                .zip(&b.normal[..3])
                .all(|(x, y)| (*x + *y).abs() < 1e-7)
        );
    }
    benchmark(device, queue, "asymmetric_forward", &forward, &a)?;
    benchmark(device, queue, "asymmetric_reverse", &reverse, &b)?;
    println!(
        "MESH_ORDER asymmetric_pairs={} same_features_and_cost=true",
        a.len()
    );
    Ok(())
}
