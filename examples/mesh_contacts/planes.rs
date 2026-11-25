//! Differential broad and narrow phase test with the same feature solver.
use super::{Geometry, Matrix, Result, benchmark, cases, direct_settings, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{MeshCatalog, MeshPair, Midphase};
use wgpu::util::DeviceExt;

fn tilted(angle: f32, offset: f32) -> Matrix {
    let (s, c) = angle.sin_cos();
    [
        [c, 0., s, 0.],
        [0., 1., 0., 0.],
        [-s, 0., c, 0.],
        [-s * offset, 0., c * offset, 1.],
    ]
}

fn oblique_cases() -> Result<(Vec<Matrix>, Vec<MeshPair>)> {
    let mut fixture = cases()?;
    // Rotate both parallel planes away from the world axes: their
    // AABBs can overlap even when the actual surfaces are separated.
    // Also include coplanar, crossing, marginal and reflected cases.
    for i in 0..256_u16 {
        let angle = f32::from(i) * std::f32::consts::TAU / 256.0;
        for gap in [0.0, 0.05, 0.099_99, 0.100_01, 0.25] {
            let a = u32::try_from(fixture.poses.len())?;
            let mut left = tilted(angle, 0.0);
            let mut right = tilted(angle, gap);
            if i % 3 == 0 {
                left[0] = left[0].map(|v| -v);
            }
            if i % 5 == 0 {
                right[1] = [0.2, 1.4, 0.1, 0.0];
            }
            if i % 7 == 0 {
                right[0] = tilted(angle + 0.04, gap)[0];
            }
            fixture.poses.extend([left, right]);
            fixture.pairs.push(MeshPair {
                meshes: [0, 0],
                transforms: [a, a + 1],
            });
        }
        // One dense frame pair per orientation with deliberately
        // overlapping coarse bounds but a physical gap.
        let a = u32::try_from(fixture.poses.len())?;
        fixture
            .poses
            .extend([tilted(angle, 0.0), tilted(angle, 0.25)]);
        fixture.pairs.push(MeshPair {
            meshes: [1, 1],
            transforms: [a, a + 1],
        });
    }
    Ok((fixture.poses, fixture.pairs))
}

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, geometry: &[Geometry]) -> Result<()> {
    let catalog = MeshCatalog::new(
        device,
        &geometry.iter().map(Geometry::source).collect::<Vec<_>>(),
    )?;
    let (poses, pairs) = oblique_cases()?;
    let matrices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("oblique and near threshold triangles"),
        contents: bytemuck::cast_slice(&poses),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let settings = direct_settings(Midphase::VoxelBvh);
    let reference = catalog.reference_queries(device, &matrices, &pairs, settings)?;
    let optimized = catalog.queries(device, &matrices, &pairs, settings)?;
    let a = sample(device, queue, &reference)?;
    let b = sample(device, queue, &optimized)?;
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            a.features, b.features,
            "plane rejection features {i}: {a:?} {b:?}"
        );
        assert_eq!(a.point_a, b.point_a, "plane rejection point A {i}");
        assert_eq!(a.point_b, b.point_b, "plane rejection point B {i}");
        assert_eq!(a.normal, b.normal, "plane rejection normal {i}");
        assert!(b.statistics[1] <= a.statistics[1]);
    }
    benchmark(device, queue, "plane_reference", &reference, &a)?;
    benchmark(device, queue, "plane_filtered", &optimized, &b)?;
    println!(
        "MESH_PLANES cases={} exact_features_and_witnesses=true",
        pairs.len()
    );
    Ok(())
}
