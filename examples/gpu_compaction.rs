//! Validate an actual Vulkan compute pass against the Rust reference.
//! Full readback verifies stable IDs, variants, attributes and indirect args.
//! This is an end-to-end latency probe, not a shader-only timestamp benchmark.

use i_love_proland::gpu::{
    CandidatePatch, CompactionPipeline, GpuCompactor, GpuView, IndexedDraw, MeshLod, MeshLods,
    PackedInstance, RenderPose, VisualLod, classify_cpu,
};
use i_love_proland::{Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::time::Instant;

type Expected = Vec<BTreeMap<u128, PackedInstance>>;

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
                    let mut pose =
                        RenderPose::upright(position, radius, original.candidate.yaw_turns);
                    pose.axis_radii_m = [radius * 0.6, radius, radius * 0.8];
                    result.push(
                        PackedInstance::from_instance(&original, pose)?
                            .with_tint([0.4, 1.4, 0.8, 1.0])?,
                    );
                }
            }
        }
    }
    Ok((result, start.elapsed().as_secs_f64() * 1000.0))
}

fn reference(candidates: &[PackedInstance], view: GpuView, lods: &MeshLods) -> (Expected, usize) {
    let mut accepted = vec![BTreeMap::new(); lods.impostor_bin() + 1];
    let mut canopy = 0;
    for candidate in candidates {
        let bin = match classify_cpu(candidate, view, lods) {
            VisualLod::Mesh(tier) => tier,
            VisualLod::Impostor => lods.impostor_bin(),
            VisualLod::TerrainCanopy => {
                canopy += 1;
                continue;
            }
            VisualLod::OutsideFrustum => continue,
        };
        assert!(
            accepted[bin]
                .insert(candidate.stable_id(), *candidate)
                .is_none()
        );
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

/// Verify output candidate indices, source records and indirect arguments
/// against the same canonical CPU objects, regardless of atomic append order.
fn gpu_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &GpuCompactor,
    candidates: &[PackedInstance],
    view: GpuView,
    expected: &Expected,
) -> Result<(Vec<IndexedDraw>, f64), Box<dyn Error>> {
    let start = Instant::now();
    let bins = compactor.bin_count();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("i-love-proland correctness and readback"),
    });
    compactor.encode(queue, &mut encoder, view)?;
    let args = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indexed draw readback"),
        size: u64::try_from(bins)? * 20,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let output_bytes = u64::from(compactor.capacity()) * 4;
    let outputs = (0..bins)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compacted LOD indices readback"),
                size: output_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect::<Vec<_>>();
    for (index, output) in outputs.iter().enumerate() {
        let bin = compactor.bin(index).ok_or("missing compacted bin")?;
        encoder.copy_buffer_to_buffer(
            bin.indirect,
            bin.indirect_offset,
            &args,
            u64::try_from(index)? * 20,
            20,
        );
        encoder.copy_buffer_to_buffer(
            bin.indices.buffer,
            bin.indices.offset,
            output,
            0,
            output_bytes,
        );
    }
    // The input records are the actual render-side source data. Verify
    // those GPU bytes too, not merely the host copy used for dereferencing.
    let source_size = u64::try_from(core::mem::size_of_val(candidates))?;
    let source = if source_size == 0 {
        None
    } else {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("submitted source instance readback"),
            size: source_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let binding = compactor.bin(0).ok_or("missing source binding")?.candidates;
        encoder.copy_buffer_to_buffer(binding.buffer, binding.offset, &buffer, 0, source_size);
        Some(buffer)
    };
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(source) = source {
        assert_eq!(
            readback(device, &source)?,
            bytemuck::cast_slice::<PackedInstance, u8>(candidates)
        );
    }
    let args_bytes = readback(device, &args)?;
    let draws = args_bytes
        .as_chunks::<20>()
        .0
        .iter()
        .map(|bytes| bytemuck::pod_read_unaligned::<IndexedDraw>(bytes))
        .collect::<Vec<_>>();
    assert_eq!(draws.len(), bins);
    let mut unique = BTreeSet::new();
    for (index, (draw, output)) in draws.iter().zip(&outputs).enumerate() {
        assert_eq!(
            draw.index_count,
            compactor.bin(index).ok_or("missing bin")?.index_count
        );
        assert_eq!(draw.first_instance, 0);
        assert_eq!(usize::try_from(draw.instance_count)?, expected[index].len());
        let bytes = readback(device, output)?;
        let visible_bytes = usize::try_from(draw.instance_count)? * 4;
        let mut gpu = BTreeMap::new();
        for record in bytes[..visible_bytes].as_chunks::<4>().0 {
            let slot = usize::try_from(u32::from_le_bytes(*record))?;
            let object = *candidates
                .get(slot)
                .ok_or("out-of-range compacted candidate")?;
            assert!(gpu.insert(object.stable_id(), object).is_none());
            assert!(unique.insert(object.stable_id()));
        }
        assert_eq!(
            gpu, expected[index],
            "GPU LOD bin {index} differs from CPU reference"
        );
    }
    Ok((draws, gpu_ms))
}

fn benchmark(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &mut GpuCompactor,
    candidates: Option<&[PackedInstance]>,
    view: GpuView,
) -> Result<f64, Box<dyn Error>> {
    const REPETITIONS: u32 = 30;
    let start = Instant::now();
    for _ in 0..REPETITIONS {
        if let Some(candidates) = candidates {
            compactor.set_candidates(queue, candidates)?;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("i-love-proland end-to-end GPU latency probe"),
        });
        compactor.encode(queue, &mut encoder, view)?;
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely())?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(REPETITIONS))
}

fn validate_replacement(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &mut GpuCompactor,
    candidates: &[PackedInstance],
    view: GpuView,
) -> Result<(), Box<dyn Error>> {
    // Shrinking changes the submitted count, never the already allocated capacity.
    let subset = &candidates[..37];
    let (subset_expected, _) = reference(subset, view, compactor.mesh_lods());
    assert!(!compactor.set_candidates(queue, subset)?);
    assert_eq!(compactor.candidate_count(), 37);
    gpu_parity(device, queue, compactor, subset, view, &subset_expected)?;
    // Empty frames must reset every indirect bin, including late mesh tiers.
    let empty = vec![BTreeMap::new(); compactor.bin_count()];
    assert!(!compactor.set_candidates(queue, &[])?);
    let (cleared, _) = gpu_parity(device, queue, compactor, &[], view, &empty)?;
    assert!(cleared.iter().all(|draw| draw.instance_count == 0));

    // Restore the full batch in the already grown storage.
    let (expected, _) = reference(candidates, view, compactor.mesh_lods());
    assert!(!compactor.set_candidates(queue, candidates)?);
    gpu_parity(device, queue, compactor, candidates, view, &expected)?;
    Ok(())
}

fn validate_moving_handoff(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &mut GpuCompactor,
    candidates: &[PackedInstance],
    frustum: GpuView,
) -> Result<(), Box<dyn Error>> {
    // Move the eye against one immutable resident population. Compare
    // all GPU output identities and bins with the CPU classifier.
    let local_candidates = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut c = *c;
            let offset = [1024.0, 2048.0, -4096.0];
            for (coordinate, shift) in c.position_radius[..3].iter_mut().zip(offset) {
                *coordinate += shift;
            }
            let anchor = c.position_radius[..3].try_into()?;
            let rank = f32::from(u8::try_from(i % 251)?) / 251.0;
            Ok(c.with_handoff_anchor(anchor, rank)?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    assert!(!compactor.set_candidates(queue, &local_candidates)?);
    for eye in [
        [1024.0, 2048.0, -4096.0],
        [1032.0, 2020.0, -3920.0],
        [1100.0, 2100.0, -3190.0],
    ] {
        let moving = frustum
            .with_eye(eye)?
            .with_distance_handoff(900.0, 1000.0)?;
        let (expected, _) = reference(&local_candidates, moving, compactor.mesh_lods());
        gpu_parity(
            device,
            queue,
            compactor,
            &local_candidates,
            moving,
            &expected,
        )?;
    }
    println!("GPU_RESIDENT_MOVING_HANDOFF_PARITY PASS");
    assert!(!compactor.set_candidates(queue, candidates)?);
    Ok(())
}

fn validate_sparse_patches(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compactor: &mut GpuCompactor,
    candidates: &[PackedInstance],
    view: GpuView,
) -> Result<(), Box<dyn Error>> {
    let mut expected = candidates[..37].to_vec();
    assert!(!compactor.set_candidates(queue, &expected)?);
    let patch = |compactor: &mut GpuCompactor,
                 expected: &[PackedInstance],
                 slots: &[u32]|
     -> Result<(), Box<dyn Error>> {
        let records = slots
            .iter()
            .map(|&slot| Ok(CandidatePatch::new(slot, expected[usize::try_from(slot)?])))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        compactor.patch_candidates(
            queue,
            &mut encoder,
            u32::try_from(expected.len())?,
            &records,
        )?;
        queue.submit(Some(encoder.finish()));
        let (reference, _) = reference(expected, view, compactor.mesh_lods());
        gpu_parity(device, queue, compactor, expected, view, &reference)?;
        Ok(())
    };
    expected[2].tint = [0.2, 0.3, 0.4, 1.0];
    expected[17].position_radius[2] += 120.0;
    expected.extend_from_slice(&candidates[37..42]);
    patch(compactor, &expected, &[2, 17, 37, 38, 39, 40, 41])?;
    expected.truncate(9);
    patch(compactor, &expected, &[])?;
    expected.extend_from_slice(&candidates[9..20]);
    expected[9].position_radius[1] += 9.0;
    patch(compactor, &expected, &(9..20).collect::<Vec<_>>())?;
    expected.clear();
    patch(compactor, &expected, &[])?;
    expected.extend_from_slice(&candidates[..4]);
    patch(compactor, &expected, &[0, 1, 2, 3])?;
    assert!(!compactor.set_candidates(queue, candidates)?);
    println!("GPU_SPARSE_PATCH_PARITY PASS");
    Ok(())
}

fn validate_projected_lod_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &CompactionPipeline,
    candidates: &[PackedInstance],
    original: &MeshLods,
) -> Result<(), Box<dyn Error>> {
    // Camera zoom and physical crown radii must select the same tier on
    // CPU and GPU. Distance-based authored LOD is exercised above instead.
    let tiers = original
        .tiers()
        .iter()
        .map(|tier| MeshLod {
            prefer_mesh: true,
            ..*tier
        })
        .collect::<Vec<_>>();
    let projected = MeshLods::new(tiers)?.with_projected_diameters(vec![256.0, 96.0, 24.0, 0.0])?;
    let mut projected_compactor =
        GpuCompactor::new(pipeline, u32::try_from(candidates.len())?, projected, 6)?;
    projected_compactor.set_candidates(queue, candidates)?;
    for focal in [270.0, 540.0, 1080.0] {
        let projected_view =
            GpuView::unculled(focal, 32.0, 1.0)?.with_source_center([0.12, 0.4, -0.19])?;
        let (expected, _) = reference(candidates, projected_view, projected_compactor.mesh_lods());
        let (draws, _) = gpu_parity(
            device,
            queue,
            &projected_compactor,
            candidates,
            projected_view,
            &expected,
        )?;
        println!(
            "GPU_PROJECTED_LOD_PARITY focal={focal} bins={:?}",
            draws
                .iter()
                .map(|draw| draw.instance_count)
                .collect::<Vec<_>>(),
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (candidates, generation_ms) = generate_sample()?;
    let view = GpuView::unculled(540.0, 32.0, 1.0)?.with_source_center([0.12, 0.4, -0.19])?;
    let lods = MeshLods::new(vec![
        MeshLod {
            start_m: 0.0,
            index_count: 369,
            prefer_mesh: false,
        },
        MeshLod {
            start_m: 180.0,
            index_count: 147,
            prefer_mesh: false,
        },
        MeshLod {
            start_m: 380.0,
            index_count: 27,
            prefer_mesh: false,
        },
        MeshLod {
            start_m: 600.0,
            index_count: 9,
            prefer_mesh: false,
        },
    ])?;
    let (expected, canopy) = reference(&candidates, view, &lods);
    let (device, queue) = device()?;
    let pipeline = CompactionPipeline::new(&device);
    let mut compactor = GpuCompactor::new(&pipeline, u32::try_from(candidates.len())?, lods, 6)?;

    assert!(!compactor.set_candidates(&queue, &candidates)?);
    let (draws, first_gpu_ms) =
        gpu_parity(&device, &queue, &compactor, &candidates, view, &expected)?;
    assert!(draws.iter().all(|draw| draw.instance_count > 0));

    // Exercise nonzero frustum planes and rotated source-anchor offsets.
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
    )?
    .with_source_center([0.12, 0.4, -0.19])?;
    let (frustum_reference, _) = reference(&candidates, frustum, compactor.mesh_lods());
    let (culled_draws, _) = gpu_parity(
        &device,
        &queue,
        &compactor,
        &candidates,
        frustum,
        &frustum_reference,
    )?;
    let mut preferred = compactor.mesh_lods().tiers().to_vec();
    preferred
        .last_mut()
        .ok_or("missing authored tier")?
        .prefer_mesh = true;
    let preferred = MeshLods::new(preferred)?;
    let authored_view = view.with_explicit_population();
    let (authored_reference, _) = reference(&candidates, authored_view, &preferred);
    let mut authored =
        GpuCompactor::new(&pipeline, u32::try_from(candidates.len())?, preferred, 6)?;
    assert!(!authored.set_candidates(&queue, &candidates)?);
    let (authored_draws, _) = gpu_parity(
        &device,
        &queue,
        &authored,
        &candidates,
        authored_view,
        &authored_reference,
    )?;
    assert!(authored_draws[authored.bin_count() - 2].instance_count > 0);
    println!(
        "GPU_AUTHORED_HANDOFF_PARITY bins={:?}",
        authored_draws
            .iter()
            .map(|d| d.instance_count)
            .collect::<Vec<_>>()
    );
    validate_projected_lod_parity(
        &device,
        &queue,
        &pipeline,
        &candidates,
        compactor.mesh_lods(),
    )?;
    validate_moving_handoff(&device, &queue, &mut compactor, &candidates, frustum)?;
    let frame_ms = benchmark(&device, &queue, &mut compactor, Some(&candidates), view)?;
    let resident_ms = benchmark(&device, &queue, &mut compactor, None, view)?;
    println!("GPU_RESIDENT_INPUT upload_ms={frame_ms:.4} resident_ms={resident_ms:.4}");
    validate_replacement(&device, &queue, &mut compactor, &candidates, view)?;
    validate_sparse_patches(&device, &queue, &mut compactor, &candidates, view)?;

    let generated = candidates.len();
    let counts = |draws: &[IndexedDraw]| draws.iter().map(|d| d.instance_count).collect::<Vec<_>>();
    println!(
        "GPU_SCATTER_PARITY PASS generated={generated} lod_bins={:?} far_canopy={canopy} frustum_bins={:?} generation_ms={generation_ms:.2} first_gpu_ms={first_gpu_ms:.2} cpu_upload_gpu_sync_ms_per_frame={frame_ms:.4}",
        counts(&draws),
        counts(&culled_draws),
    );
    Ok(())
}
