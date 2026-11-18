//! Run a real headless Vulkan instance-filtering pass with readback and an
//! order-independent CPU-reference comparison. No source assets required.
use i_love_proland::gpu::{
    GpuCompactor, GpuView, IndexedDraw, PackedInstance, RenderPose, VisualLod, classify_cpu,
};
use i_love_proland::{Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::time::Instant;

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

fn main() -> Result<(), Box<dyn Error>> {
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
            0x4f41_4b,
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
    let before_generate = Instant::now();
    let mut candidates = Vec::new();
    for y in -6..6 {
        for x in -6..6 {
            for layer in &layers {
                for instance in generator.tile(TileKey::planar(x, y), layer)? {
                    let id = instance.candidate.id;
                    let position = [
                        (instance.candidate.cell.x as f32) * 1.75,
                        (instance.candidate.cell.y as f32) * 1.75,
                        60.0 + ((id >> 32) as u32 % 14_000) as f32,
                    ];
                    let radius = 4.0 + ((id >> 80) as u8 % 18) as f32;
                    candidates.push(PackedInstance::from_instance(
                        &instance,
                        RenderPose {
                            camera_relative_position: position,
                            crown_radius_m: radius,
                            scale: 1.0,
                        },
                    )?);
                }
            }
        }
    }
    let generation_ms = before_generate.elapsed().as_secs_f64() * 1000.0;

    let view = GpuView::unculled(540.0, 32.0, 1.0)?;
    let mut expected = [BTreeMap::new(), BTreeMap::new()];
    let mut expected_canopy = 0_usize;
    for candidate in &candidates {
        match classify_cpu(candidate, view) {
            VisualLod::Mesh => {
                assert!(
                    expected[0]
                        .insert(candidate.stable_id(), *candidate)
                        .is_none()
                );
            }
            VisualLod::Impostor => {
                assert!(
                    expected[1]
                        .insert(candidate.stable_id(), *candidate)
                        .is_none()
                );
            }
            VisualLod::TerrainCanopy => expected_canopy += 1,
            VisualLod::OutsideFrustum => {}
        }
    }

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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("i-love-proland GPU correctness and timing"),
        ..Default::default()
    }))?;
    let compactor = GpuCompactor::new(&device, candidates.len() as u32, 369, 6)?;

    let before_gpu = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("i-love-proland first GPU scatter pass"),
    });
    compactor.encode(&queue, &mut encoder, &candidates, view)?;
    let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect args readback"),
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bytes = candidates.len() as u64 * 64;
    let make_readback = |label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let mesh_readback = make_readback("mesh instance readback");
    let impostor_readback = make_readback("impostor instance readback");
    encoder.copy_buffer_to_buffer(compactor.mesh_indirect(), 0, &draw_buffer, 0, 20);
    encoder.copy_buffer_to_buffer(compactor.impostor_indirect(), 0, &draw_buffer, 32, 20);
    encoder.copy_buffer_to_buffer(compactor.mesh_instances(), 0, &mesh_readback, 0, bytes);
    encoder.copy_buffer_to_buffer(
        compactor.impostor_instances(),
        0,
        &impostor_readback,
        0,
        bytes,
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let first_gpu_ms = before_gpu.elapsed().as_secs_f64() * 1000.0;

    let draw_bytes = readback(&device, &draw_buffer)?;
    let mesh_draw = bytemuck::pod_read_unaligned::<IndexedDraw>(&draw_bytes[..20]);
    let imp_draw = bytemuck::pod_read_unaligned::<IndexedDraw>(&draw_bytes[32..52]);
    assert_eq!(mesh_draw.index_count, 369);
    assert_eq!(imp_draw.index_count, 6);
    assert_eq!(mesh_draw.first_instance, 0);
    assert_eq!(imp_draw.first_instance, 0);
    assert_eq!(mesh_draw.instance_count as usize, expected[0].len());
    assert_eq!(imp_draw.instance_count as usize, expected[1].len());

    let bytes_per_instance = core::mem::size_of::<PackedInstance>();
    let mut all_ids = BTreeSet::new();
    for (i, (buffer, count)) in [
        (&mesh_readback, mesh_draw.instance_count),
        (&impostor_readback, imp_draw.instance_count),
    ]
    .into_iter()
    .enumerate()
    {
        let bytes = readback(&device, buffer)?;
        let mut actual = BTreeMap::new();
        for chunk in bytes[..count as usize * bytes_per_instance].chunks_exact(bytes_per_instance) {
            let item = bytemuck::pod_read_unaligned::<PackedInstance>(chunk);
            assert!(actual.insert(item.stable_id(), item).is_none());
            assert!(all_ids.insert(item.stable_id()));
        }
        assert_eq!(
            actual, expected[i],
            "GPU LOD={i} differs from CPU reference"
        );
    }

    // No readback or GPU→CPU sync per frame in the benchmark loop.
    let repetitions = 30_u32;
    let start = Instant::now();
    for _ in 0..repetitions {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("i-love-proland GPU compaction steady-state benchmark"),
        });
        compactor.encode(&queue, &mut encoder, &candidates, view)?;
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely())?;
    }
    let ms_per_frame = start.elapsed().as_secs_f64() * 1000.0 / f64::from(repetitions);

    println!(
        "GPU_SCATTER_PARITY PASS generated={} mesh={} impostor={} far_canopy={} generation_ms={generation_ms:.2} first_gpu_ms={first_gpu_ms:.2} cpu_upload_gpu_sync_ms_per_frame={ms_per_frame:.4}",
        candidates.len(),
        mesh_draw.instance_count,
        imp_draw.instance_count,
        expected_canopy,
    );
    Ok(())
}
