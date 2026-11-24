//! Matched-geometry A/B for atlas padding, with interleaved GPU queries.
use super::{
    ForestPass, ForestQueries, HEIGHT, WIDTH, background_pipeline, capture, create_device,
    generate_forest, make_frame_targets, pitched_view, reference_pipelines, render_frame,
    render_frame_with_queries, sample_tree, write_ppm,
};
use i_love_proland::{
    atlas_bake::bake_mesh,
    gpu::{GpuView, MeshLod, MeshLods, PackedInstance, VisualLod, classify_cpu},
    impostor::{AtlasImage, ModelAtlas, TreeGeometry, TreeMaterial, TreeMesh, TreeRenderer},
    material::SourceImage,
};
use std::error::Error;

const PAIRS: u32 = 24;
const FRAMES: u32 = 2 * PAIRS;
const SMALL: f32 = 0.25;

fn make_renderers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    candidates: &[PackedInstance],
    view: GpuView,
) -> Result<[TreeRenderer; 2], Box<dyn Error>> {
    let (full, indices) = sample_tree()?;
    let small = full
        .iter()
        .copied()
        .map(|mut vertex| {
            for coordinate in &mut vertex.position[..3] {
                *coordinate *= SMALL;
            }
            vertex
        })
        .collect::<Vec<_>>();
    let source = SourceImage::white();
    let texture = source.upload(device, queue)?;
    let mesh = TreeMesh::new(device, &small, &indices)?;
    let lods = MeshLods::new(vec![MeshLod {
        start_m: 0.0,
        index_count: mesh.index_count(),
        prefer_mesh: false,
    }])?;
    let visible = candidates
        .iter()
        .filter(|object| classify_cpu(object, view, &lods) == VisualLod::Impostor)
        .count();
    if visible == 0
        || candidates
            .iter()
            .any(|object| matches!(classify_cpu(object, view, &lods), VisualLod::Mesh(_)))
    {
        return Err("benchmark must select only impostors, with nonzero visible population".into());
    }
    println!(
        "IMPOSTOR_BENCH candidates={} visible_impostors={visible} predicted_quad_area_ratio={}",
        candidates.len(),
        SMALL * SMALL
    );
    let pipelines = reference_pipelines(device)?;
    let make =
        |vertices: &[super::TreeVertex], radius_scale| -> Result<TreeRenderer, Box<dyn Error>> {
            let image = bake_mesh(vertices, &indices, &source, 0.35, 64, 2)?;
            let atlas = AtlasImage::upload_variants(
                device,
                queue,
                &[ModelAtlas {
                    variant_id: 900,
                    image,
                    radius_scale,
                }],
            )?;
            let mut renderer = TreeRenderer::new(
                &pipelines,
                u32::try_from(candidates.len())?,
                lods.clone(),
                vec![vec![TreeMaterial {
                    geometry: TreeGeometry::Mesh(mesh.clone()),
                    source_view: texture.view(),
                    alpha_cutoff: 0.35,
                    wind: None,
                    billboard: None,
                }]],
                &atlas,
            )?;
            renderer.set_candidates(queue, candidates)?;
            Ok(renderer)
        };
    Ok([make(&small, 1.0)?, make(&full, SMALL)?])
}

fn read_queries(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    set: &wgpu::QuerySet,
    count: u32,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let size = u64::from(count) * 8;
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("impostor query resolve"),
        size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("impostor query readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.resolve_query_set(set, 0..count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, size);
    queue.submit(Some(encoder.finish()));
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    read.slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receive.recv()??;
    let values = read
        .slice(..)
        .get_mapped_range()?
        .as_chunks::<8>()
        .0
        .iter()
        .copied()
        .map(u64::from_le_bytes)
        .collect();
    read.unmap();
    Ok(values)
}

fn summarize(ticks: &[u64], fragments: &[u64], period: f32) -> Result<(), Box<dyn Error>> {
    let mut durations = [Vec::new(), Vec::new()];
    let mut invocations = [Vec::new(), Vec::new()];
    for (frame, (times, count)) in ticks.as_chunks::<2>().0.iter().zip(fragments).enumerate() {
        let mode = (frame % 2) ^ ((frame / 2) % 2);
        durations[mode].push(
            times[1]
                .checked_sub(times[0])
                .ok_or("non-monotonic GPU timestamps")?,
        );
        invocations[mode].push(*count);
    }
    for values in durations.iter_mut().chain(&mut invocations) {
        values.sort_unstable();
    }
    for (mode, name) in ["padded", "tight"].iter().enumerate() {
        let middle = usize::try_from(PAIRS / 2)?;
        let ticks = durations[mode][middle];
        let count = invocations[mode][middle];
        #[expect(
            clippy::cast_precision_loss,
            reason = "GPU tick durations are converted to approximate milliseconds"
        )]
        let millis = ticks as f64 * f64::from(period) / 1e6;
        println!(
            "IMPOSTOR_BENCH_RESULT mode={name} median_gpu_render_pass_ms={millis:.4} median_tree_fragment_invocations={count}"
        );
    }
    Ok(())
}

pub(super) fn run(prefix: &str) -> Result<(), Box<dyn Error>> {
    let features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PIPELINE_STATISTICS_QUERY;
    let (device, queue) = create_device(features)?;
    let (candidates, _) = generate_forest()?;
    let (camera, view) = pitched_view(0.0, 1e6)?;
    let mut renderers = make_renderers(&device, &queue, &candidates, view)?;
    for renderer in &renderers {
        renderer.set_camera(&queue, camera);
    }
    let targets = make_frame_targets(&device);
    let background = background_pipeline(&device, "fs_main");
    let timestamps = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("impostor render pass timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2 * FRAMES,
    });
    let fragments = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("tree fragment shader invocations"),
        ty: wgpu::QueryType::PipelineStatistics(
            wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
        ),
        count: FRAMES,
    });
    for frame in 0..8 + FRAMES + 2 {
        let mode = usize::try_from((frame % 2) ^ ((frame / 2) % 2))?;
        let scene = ForestPass {
            queue: &queue,
            renderer: &mut renderers[mode],
            candidates: &candidates,
            view,
            color: &targets.color_view,
            depth: &targets.depth_view,
            background: &background,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if frame >= 8 + FRAMES {
            render_frame(&mut encoder, &scene)?;
            let pixels = capture(&device, &queue, encoder, &targets.color)?;
            let name = ["padded", "tight"][mode];
            write_ppm(&format!("{prefix}-{name}.ppm"), WIDTH, HEIGHT, &pixels)?;
        } else {
            let query = frame
                .checked_sub(8)
                .map_or_else(ForestQueries::default, |i| ForestQueries {
                    timestamps: Some((&timestamps, 2 * i)),
                    fragments: Some((&fragments, i)),
                });
            render_frame_with_queries(&mut encoder, &scene, query)?;
            queue.submit(Some(encoder.finish()));
        }
    }
    summarize(
        &read_queries(&device, &queue, &timestamps, 2 * FRAMES)?,
        &read_queries(&device, &queue, &fragments, FRAMES)?,
        queue.get_timestamp_period(),
    )?;
    Ok(())
}
