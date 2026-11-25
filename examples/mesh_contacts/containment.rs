use super::{Geometry, IDENTITY, Result, pose, sample};
use i_love_proland::terrain::gpu::mesh_contacts::{
    CandidateDispatch, Domain, MeshCatalog, MeshPair, Midphase, QuerySettings,
};
use wgpu::util::DeviceExt;

fn cube(half: f32) -> Geometry {
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
pub fn check(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
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
    let catalog = MeshCatalog::new(
        device,
        &[
            outer.source(),
            inner.source(),
            disconnected.source(),
            mixed.source(),
        ],
    )?;
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
    let solid = catalog.queries(
        device,
        &buffer,
        &pairs,
        QuerySettings {
            margin: 0.001,
            mode: Midphase::VoxelBvh,
            domain: Domain::ClosedSolid,
            dispatch: CandidateDispatch::Direct,
        },
    )?;
    let result = sample(device, queue, &solid)?;
    assert_eq!(
        result.iter().map(|r| r.features[3]).collect::<Vec<_>>(),
        [4, 4, 4, 0, 4, 4],
        "{result:?}"
    );
    println!(
        "MESH_CONTAINMENT nested_both_orders=true disconnected_component=true mixed_open_closed=true separated=true"
    );
    Ok(())
}
