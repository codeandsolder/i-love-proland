//! Validate an actual Vulkan compute pass against the Rust reference.
//! Full readback verifies stable IDs, variants, attributes and indirect args.
//! This is an end-to-end latency probe, not a shader-only timestamp benchmark.

use i_love_proland::gpu::{
    GpuCompactor, GpuView, IndexedDraw, PackedInstance, RenderPose, VisualLod, classify_cpu,
};
use i_love_proland::{Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::time::Instant;

type Expected = [BTreeMap<u128, PackedInstance>; 2];

fn readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, Box<dyn Error>> {
    let slice = buffer.slice(..);
    let (send, recv) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = send.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    recv.recv()??;
    let bytes = slice.get_mapped_range()?.to_vec();
    buffer.unmap();
    Ok(bytes)
}

/// Generate real canonical IDs/variants for two unrelated object layers.
fn generate_sample() -> Result<(Vec<PackedInstance>, f64), Box<dyn Error>> {
    let generator = Generator::new(
        Domain::Planar { cell_size_m: 8.0 },
        GeneratorConfig {
            world_seed: 0x7654_3210_9abc_def0,
            cells_per_tile: 16,
            ..GeneratorConfig::default()
        },
    )?;
    let layers = [
        Layer::new(
            0x004f_414b,
            0.015,
            Distribution::Poisson,
            vec![
                Variant {
                    id: 1001,
                    weight: 3.0,
                },
                Variant {
                    id: 1002,
                    weight: 1.0,
                },
            ],
        )?,
        Layer::new(
            0x524f_434b,
            0.012,
            Distribution::Stratified,
            vec![Variant {
                id: 2001,
                weight: 1.0,
            }],
        )?,
    ];

    let start = Instant::now();
    let mut result = Vec::new();
    for y in -6..6 {
        for x in -6..6 {
            for layer in &layers {
                for original in generator.tile(TileKey::planar(x, y), layer)? {
                    let cell = original.candidate.cell;
                    let id_bytes = original.candidate.id.to_le_bytes();
                    let distance_seed = u32::from_le_bytes(id_bytes[4..8].try_into()?);
                    let depth = 60.0 + f32::from(u16::try_from(distance_seed % 14_000)?);
                    let radius = 4.0 + f32::from(id_bytes[10] % 18);
                    let position = [
                        f32::from(i16::try_from(cell.x)?) * 1.75,
                        f32::from(i16::try_from(cell.y)?) * 1.75,
                        depth,
                    ];
                    result.push(PackedInstance::from_instance(
                        &original,
                        RenderPose::upright(position, radius, original.candidate.yaw_turns),
                    )?);
                }
            }
        }
    }
    Ok((result, start.elapsed().as_secs_f64() * 1000.0))
}

fn reference(candidates: &[PackedInstance], view: GpuView) -> (Expected, usize) {
    let mut accepted = [BTreeMap::new(), BTreeMap::new()];
    let mut canopy = 0;
    for candidate in candidates {
        match classify_cpu(candidate, view) {
            VisualLod::Mesh => {
                assert!(
                    accepted[0]
                        .insert(candidate.stable_id(), *candidate)
                        .is_none()
                );
            }
            VisualLod::Impostor => {
                assert!(
                    accepted[1]
                        .insert(candidate.stable_id(), *candidate)
                        .is_none()
                );
            }
            VisualLod::TerrainCanopy => canopy += 1,
            VisualLod::OutsideFrustum => {}
        }
    }
    (accepted, canopy)
}

fn device() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn Error>> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;
    println!(
        "GPU_SCATTER_ADAPTER name={:?} backend={:?}",
        adapter.get_info().name,
        adapter.get_info().backend,
    );
    Ok(pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("i-love-proland GPU correctness and timing"),
            ..Default::default()
        },
    ))?)
}

/// Verify both output lists and both indirect arguments byte-for-byte
/// against the same canonical CPU objects, regardless of atomic append order.
fn gpu_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &GpuCompactor,
    candidates: &[PackedInstance],
    view: GpuView,
    expected: &Expected,
) -> Result<(IndexedDraw, IndexedDraw, f64), Box<dyn Error>> {
    let start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("i-love-proland correctness and readback"),
    });
    compactor.encode(queue, &mut encoder, candidates, view)?;
    let args = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indexed draw readback"),
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let output_bytes =
        u64::try_from(candidates.len())? * u64::try_from(core::mem::size_of::<PackedInstance>())?;
    let make_readback = |label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let mesh = make_readback("mesh instances readback");
    let impostor = make_readback("impostor instances readback");

    encoder.copy_buffer_to_buffer(compactor.mesh_indirect(), 0, &args, 0, 20);
    encoder.copy_buffer_to_buffer(compactor.impostor_indirect(), 0, &args, 32, 20);
    encoder.copy_buffer_to_buffer(compactor.mesh_instances(), 0, &mesh, 0, output_bytes);
    encoder.copy_buffer_to_buffer(
        compactor.impostor_instances(),
        0,
        &impostor,
        0,
        output_bytes,
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

    let bytes = readback(device, &args)?;
    let mesh_draw = bytemuck::pod_read_unaligned::<IndexedDraw>(&bytes[..20]);
    let impostor_draw = bytemuck::pod_read_unaligned::<IndexedDraw>(&bytes[32..52]);
    assert_eq!((mesh_draw.index_count, impostor_draw.index_count), (369, 6));
    assert_eq!(
        (mesh_draw.first_instance, impostor_draw.first_instance),
        (0, 0)
    );
    assert_eq!(
        usize::try_from(mesh_draw.instance_count)?,
        expected[0].len()
    );
    assert_eq!(
        usize::try_from(impostor_draw.instance_count)?,
        expected[1].len()
    );

    let mut unique = BTreeSet::new();
    for (lod, (buffer, count)) in [
        (&mesh, mesh_draw.instance_count),
        (&impostor, impostor_draw.instance_count),
    ]
    .into_iter()
    .enumerate()
    {
        let bytes = readback(device, buffer)?;
        let visible_bytes = usize::try_from(count)? * core::mem::size_of::<PackedInstance>();
        let mut gpu = BTreeMap::new();
        for record in bytes[..visible_bytes]
            .as_chunks::<{ core::mem::size_of::<PackedInstance>() }>()
            .0
        {
            let object = bytemuck::pod_read_unaligned::<PackedInstance>(record);
            assert!(gpu.insert(object.stable_id(), object).is_none());
            assert!(unique.insert(object.stable_id()));
        }
        assert_eq!(
            gpu, expected[lod],
            "GPU LOD={lod} differs from the CPU reference"
        );
    }
    Ok((mesh_draw, impostor_draw, gpu_ms))
}

fn benchmark(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &GpuCompactor,
    candidates: &[PackedInstance],
    view: GpuView,
) -> Result<f64, Box<dyn Error>> {
    const REPETITIONS: u32 = 30;
    let start = Instant::now();
    for _ in 0..REPETITIONS {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("i-love-proland end-to-end GPU latency probe"),
        });
        compactor.encode(queue, &mut encoder, candidates, view)?;
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely())?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(REPETITIONS))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (candidates, generation_ms) = generate_sample()?;
    let view = GpuView::unculled(540.0, 32.0, 1.0)?;
    let (expected, canopy) = reference(&candidates, view);
    let (device, queue) = device()?;
    let compactor = GpuCompactor::new(&device, u32::try_from(candidates.len())?, 369, 6)?;

    let (mesh, impostor, first_gpu_ms) =
        gpu_parity(&device, &queue, &compactor, &candidates, view, &expected)?;

    // Exercise nonzero frustum planes, not just projected-size classification.
    let frustum = GpuView::new(
        540.0,
        32.0,
        1.0,
        [0.0, 0.0, 1.0],
        [
            [1.0, 0.0, 0.0, 50.0],
            [0.0, 1.0, 0.0, 90.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
        ],
    )?;
    let (frustum_reference, _) = reference(&candidates, frustum);
    let (culled_mesh, culled_impostor, _) = gpu_parity(
        &device,
        &queue,
        &compactor,
        &candidates,
        frustum,
        &frustum_reference,
    )?;
    let frame_ms = benchmark(&device, &queue, &compactor, &candidates, view)?;

    let generated = candidates.len();
    let mesh_count = mesh.instance_count;
    let impostor_count = impostor.instance_count;
    let frustum_mesh = culled_mesh.instance_count;
    let frustum_impostor = culled_impostor.instance_count;
    println!(
        "GPU_SCATTER_PARITY PASS generated={generated} mesh={mesh_count} impostor={impostor_count} far_canopy={canopy} frustum_mesh={frustum_mesh} frustum_impostor={frustum_impostor} generation_ms={generation_ms:.2} first_gpu_ms={first_gpu_ms:.2} cpu_upload_gpu_sync_ms_per_frame={frame_ms:.4}",
    );
    Ok(())
}
