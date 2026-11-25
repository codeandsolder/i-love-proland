use super::{Geometry, IDENTITY, Result, pose, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{
    CandidateDispatch, Domain, MeshCatalog, MeshPair, Midphase, QuerySettings,
};
use wgpu::util::DeviceExt;

pub fn cube(half: f32) -> Geometry {
    let positions = (0..8_u16)
        .map(|i| {
            [
                if i & 1 == 0 { -half } else { half },
                if i & 2 == 0 { -half } else { half },
                if i & 4 == 0 { -half } else { half },
            ]
        })
        .collect();
    Geometry {
        positions,
        indices: vec![
            0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4,
            6, 1, 3, 5, 3, 7, 5,
        ],
    }
}
fn catalog(device: &wgpu::Device) -> Result<MeshCatalog> {
    let outer = cube(2.0);
    let inner = cube(0.2);
    let disconnected = Geometry {
        positions: vec![
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [10.0, 0.1, 0.0],
            [0.0; 3],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        indices: vec![0, 1, 2, 3, 4, 5],
    };
    let mut mixed = cube(2.0);
    mixed
        .positions
        .extend([[10.0, 0.0, 0.0], [10.1, 0.0, 0.0], [10.0, 0.1, 0.0]]);
    mixed.indices.extend([8, 9, 10]);
    Ok(MeshCatalog::new(
        device,
        &[
            outer.source(),
            inner.source(),
            disconnected.source(),
            mixed.source(),
        ],
    )?)
}

// Two separated closed components share one large asset AABB. The
// central hole must clear at coarse resolution, whereas a small box
// inside either component must still report exact containment.
fn separated_solids() -> Geometry {
    let mut left = cube(1.0);
    let right = cube(1.0);
    for p in &mut left.positions {
        p[0] -= 4.0;
    }
    left.positions
        .extend(right.positions.into_iter().map(|mut p| {
            p[0] += 4.0;
            p
        }));
    left.indices
        .extend(right.indices.into_iter().map(|i| i + 8));
    left
}

fn sparse_closed(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let shell = separated_solids();
    let small = cube(0.1);
    let catalog = MeshCatalog::new(device, &[shell.source(), small.source()])?;
    let poses = [
        IDENTITY,
        pose(0.0, 0.0, 0.0, 0.0),
        pose(0.0, -4.0, 0.0, 0.0),
        pose(0.0, 4.0, 0.0, 0.0),
        pose(0.0, 2.95, 0.0, 0.0),
    ];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("closed sparse hull poses"),
        contents: bytemuck::cast_slice(&poses),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pairs = (1..5)
        .flat_map(|i| {
            [
                MeshPair {
                    meshes: [0, 1],
                    transforms: [0, i],
                },
                MeshPair {
                    meshes: [1, 0],
                    transforms: [i, 0],
                },
            ]
        })
        .collect::<Vec<_>>();
    let settings = QuerySettings {
        margin: 0.001,
        mode: Midphase::VoxelBvh,
        domain: Domain::ClosedSolid,
        dispatch: CandidateDispatch::Compact,
    };
    let reference = catalog.queries(device, &buffer, &pairs, settings)?;
    let envelope = catalog.queries(
        device,
        &buffer,
        &pairs,
        QuerySettings {
            dispatch: CandidateDispatch::CompactEnvelope,
            ..settings
        },
    )?;
    let a = sample(device, queue, &reference)?;
    let b = sample(device, queue, &envelope)?;
    super::envelope::compare(&a, &b);
    assert_eq!(b[0].features[2..], [0, 0]);
    assert_eq!(b[1].features[2..], [0, 0]);
    for contact in &b[2..6] {
        assert_eq!(contact.features[2..], [0, 4], "nested closed shell");
    }
    for contact in &b[6..] {
        assert!(contact.features[2] > 0, "surface crossing");
    }
    assert!(a[0].statistics[0] > 0);
    assert_eq!(b[0].statistics, [0; 4]);
    assert_eq!(b[1].statistics, [0; 4]);
    println!("MESH_SOLID_ENVELOPE clear_hole=true nested_both_orders=true crossing=true");
    Ok(())
}

pub fn check(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let catalog = catalog(device)?;
    let matrices = [
        IDENTITY,
        pose(0.37, 0.2, -0.3, 0.15),
        pose(0.6, 5.0, 0.0, 0.0),
    ];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("closed shell poses"),
        contents: bytemuck::cast_slice(&matrices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pairs = [
        MeshPair {
            meshes: [0, 1],
            transforms: [0, 1],
        },
        MeshPair {
            meshes: [1, 0],
            transforms: [1, 0],
        },
        MeshPair {
            meshes: [2, 0],
            transforms: [0, 0],
        },
        MeshPair {
            meshes: [0, 1],
            transforms: [0, 2],
        },
        MeshPair {
            meshes: [3, 1],
            transforms: [0, 1],
        },
        MeshPair {
            meshes: [1, 3],
            transforms: [1, 0],
        },
    ];
    let surface = catalog.queries(
        device,
        &buffer,
        &pairs,
        QuerySettings {
            margin: 0.001,
            mode: Midphase::VoxelBvh,
            domain: Domain::Surface,
            dispatch: CandidateDispatch::Direct,
        },
    )?;
    let result = sample(device, queue, &surface)?;
    assert!(
        result
            .iter()
            .all(|r| r.features[2] == 0 && r.features[3] == 0)
    );
    for dispatch in [
        CandidateDispatch::Direct,
        CandidateDispatch::CompactEnvelope,
    ] {
        let solid = catalog.queries(
            device,
            &buffer,
            &pairs,
            QuerySettings {
                margin: 0.001,
                mode: Midphase::VoxelBvh,
                domain: Domain::ClosedSolid,
                dispatch,
            },
        )?;
        let result = sample(device, queue, &solid)?;
        assert_eq!(
            result.iter().map(|r| r.features[3]).collect::<Vec<_>>(),
            [4, 4, 4, 0, 4, 4],
            "{result:?}"
        );
    }
    println!(
        "MESH_CONTAINMENT nested_both_orders=true disconnected_component=true mixed_open_closed=true separated=true"
    );
    sparse_closed(device, queue)?;
    Ok(())
}
