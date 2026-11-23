//! Headless Vulkan comparison of procedural vertex synthesis, shared foliage
//! modules and expanded precomputed variants. This is a geometry/rendering
//! experiment, not a new live KSP foliage renderer.
use std::{error::Error, fs, path::Path, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

const W: u32 = 960;
const H: u32 = 600;
const KINDS: u32 = 3;
const VARIANTS: u32 = 8;
const MODELS: u32 = KINDS * VARIANTS;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const QUERIES: u32 = 32;
type AnyResult<T> = Result<T, Box<dyn Error>>;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Tree {
    position_scale: [f32; 4],
    ids: [u32; 4],
    tint: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Patch {
    center_radius: [f32; 4],
    orient: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BakedVertex {
    position_uvx: [f32; 4],
    uvy_brightness_material: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Scene {
    view: [f32; 4],
    counts: [u32; 4],
    switches: [f32; 4],
}

fn hash(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
#[expect(
    clippy::cast_precision_loss,
    reason = "16-bit fraction is exact in f32"
)]
fn random(seed: u32, salt: u32) -> f32 {
    (hash(seed ^ salt) & 0xffff) as f32 / 65_536.0
}
fn patch_for(seed: u32, part: u32, kind: u32) -> Patch {
    if part == 0 {
        return Patch {
            center_radius: [0.0, 0.45, 0.0, 0.065],
            orient: [0.0, 0.9, 1.0, 1.0],
        };
    }
    let cluster = (part - 1) / 2;
    let side = (part - 1) % 2;
    let a = random(seed, cluster.wrapping_mul(0x9e37_79b9).wrapping_add(21));
    let b = random(seed, cluster.wrapping_mul(0x85eb_ca6b).wrapping_add(77));
    let c = random(seed, cluster.wrapping_mul(0xc2b2_ae35).wrapping_add(135));
    let d = random(seed, cluster.wrapping_mul(0x27d4_eb2f).wrapping_add(389));
    let angle = core::f32::consts::TAU * a;
    let (radial, altitude, radius) = if kind == 1 {
        (
            0.48 * (1.0 - c),
            c.mul_add(0.68, 0.19),
            d.mul_add(0.045, 0.13),
        )
    } else {
        (
            0.48 * b.sqrt(),
            c.mul_add(0.5, 0.39),
            d.mul_add(0.075, 0.17),
        )
    };
    let (radial, altitude) = if kind == 2 {
        (0.44 * b.sqrt(), c.mul_add(0.36, 0.27))
    } else {
        (radial, altitude)
    };
    let (sin, cos) = angle.sin_cos();
    Patch {
        center_radius: [radial * cos, altitude, radial * sin, radius],
        orient: [
            f32::from(side as u16).mul_add(core::f32::consts::FRAC_PI_2, angle),
            radius * d.mul_add(0.45, 1.5),
            b.mul_add(0.34, 0.78),
            0.0,
        ],
    }
}
fn expanded(patch: Patch, i: u32) -> BakedVertex {
    let uv: [f32; 2] = match i {
        0 => [-1.0, -1.0],
        1 | 4 => [1.0, -1.0],
        2 | 3 => [-1.0, 1.0],
        _ => [1.0, 1.0],
    };
    let (sin, cos) = patch.orient[0].sin_cos();
    let r = patch.center_radius[3];
    BakedVertex {
        position_uvx: [
            uv[0].mul_add(r * cos, patch.center_radius[0]),
            (uv[1] * patch.orient[1] * 0.5).mul_add(1.0, patch.center_radius[1]),
            uv[0].mul_add(r * sin, patch.center_radius[2]),
            uv[0],
        ],
        uvy_brightness_material: [uv[1], patch.orient[2], patch.orient[3], 0.0],
    }
}
fn make_modules(clumps: u32) -> (Vec<Patch>, Vec<BakedVertex>) {
    let mut models = Vec::new();
    let mut expanded_models = Vec::new();
    for variant in 0..MODELS {
        for part in 0..(clumps * 2 + 1) {
            let data = patch_for(variant ^ 0x56cb_1a93, part, variant / VARIANTS);
            models.push(data);
            for vertex in 0..6 {
                expanded_models.push(expanded(data, vertex));
            }
        }
    }
    (models, expanded_models)
}
#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark grid has <2^16 positions"
)]
fn make_trees(count: u32) -> Vec<Tree> {
    (0..count)
        .map(|i| {
            let kind = hash(i) % KINDS;
            let model = kind * VARIANTS + hash(i ^ 0x9e37_79b9) % VARIANTS;
            let x = ((i % 88) as f32 - 44.0) * 1.65 + (random(i, 55) - 0.5) * 0.75;
            let z = 14.0 + (i / 88) as f32 * 3.4 + random(i, 72) * 1.4;
            let size = if kind == 2 {
                4.4
            } else {
                7.5 + random(i, 91) * 5.0
            };
            let tint = random(i, 999);
            Tree {
                position_scale: [x, 0.0, z, size],
                ids: [hash(i ^ 0x7f4a_7c15), model, kind, 0],
                tint: [0.80 + 0.3 * tint, 0.9 + 0.25 * tint, 0.8 + 0.25 * tint, 1.0],
            }
        })
        .collect()
}
fn gpu_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    value: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(value),
        usage: wgpu::BufferUsages::STORAGE,
    })
}
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::RenderPipeline>,
    color: wgpu::Texture,
    view: wgpu::TextureView,
    depth: wgpu::TextureView,
    timestamp_supported: bool,
    name: String,
}
fn open_gpu() -> AnyResult<Gpu> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;
    let info = adapter.get_info();
    let timestamp_supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("foliage procedural benchmark"),
        required_features: if timestamp_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        ..Default::default()
    }))?;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("foliage scene"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("modular GPU foliage"),
        source: wgpu::ShaderSource::Wgsl(include_str!("foliage_lab.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let mut pipelines = Vec::new();
    for entry in ["vs_generated", "vs_limited", "vs_modules", "vs_expanded"] {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        pipelines.push(pipeline);
    }
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage offscreen color"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("foliage offscreen depth"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Ok(Gpu {
        name: info.name,
        device,
        queue,
        layout,
        pipelines,
        view: color.create_view(&wgpu::TextureViewDescriptor::default()),
        depth: depth.create_view(&wgpu::TextureViewDescriptor::default()),
        color,
        timestamp_supported,
    })
}
struct DrawCase {
    _uniform: wgpu::Buffer,
    _trees: wgpu::Buffer,
    _modules: wgpu::Buffer,
    _expanded: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vertices_per_tree: u32,
    trees_count: u32,
    geometry_bytes: usize,
    module_bytes: usize,
}
impl DrawCase {
    fn new(gpu: &Gpu, clumps: u32, count: u32, fade: f32, side: u32) -> AnyResult<Self> {
        let models = make_modules(clumps);
        let trees = make_trees(count);
        let uniform_data = Scene {
            view: [1.73, W as f32 / H as f32, 2.5, 200.0],
            counts: [clumps, MODELS, side, count],
            switches: [fade, W as f32, H as f32, 0.0],
        };
        let uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("foliage scene params"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let trees_buf = gpu_buffer(&gpu.device, "forested instances", &trees);
        let module_buf = gpu_buffer(&gpu.device, "compressed foliage modules", &models.0);
        let expanded_buf = gpu_buffer(&gpu.device, "expanded foliage variants", &models.1);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("foliage scene binding"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: trees_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: module_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: expanded_buf.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            _uniform: uniform,
            _trees: trees_buf,
            _modules: module_buf,
            _expanded: expanded_buf,
            bind,
            vertices_per_tree: (clumps * 2 + 1) * 6,
            trees_count: count,
            geometry_bytes: models.1.len() * size_of::<BakedVertex>(),
            module_bytes: models.0.len() * size_of::<Patch>(),
        })
    }
}
fn draw(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    case: &DrawCase,
    mode: usize,
    first: bool,
    timestamp: Option<(&wgpu::QuerySet, u32)>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("foliage lab rasterization"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &gpu.view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if first {
                    wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.57,
                        g: 0.70,
                        b: 0.86,
                        a: 1.0,
                    })
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &gpu.depth,
            depth_ops: Some(wgpu::Operations {
                load: if first {
                    wgpu::LoadOp::Clear(1.0)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        timestamp_writes: timestamp.map(|(queries, start)| wgpu::RenderPassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(start),
            end_of_pass_write_index: Some(start + 1),
        }),
        multiview_mask: None,
    });
    pass.set_pipeline(&gpu.pipelines[mode]);
    pass.set_bind_group(0, &case.bind, &[]);
    pass.draw(0..case.vertices_per_tree, 0..case.trees_count);
}
fn map_read(gpu: &Gpu, buffer: &wgpu::Buffer) -> AnyResult<Vec<u8>> {
    let (tx, rx) = mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let view = buffer.slice(..).get_mapped_range()?;
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(bytes)
}
fn capture(gpu: &Gpu, pieces: &[(&DrawCase, usize)], path: &Path) -> AnyResult<Vec<u8>> {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("foliage image capture"),
        });
    for (index, &(case, mode)) in pieces.iter().enumerate() {
        draw(gpu, &mut encoder, case, mode, index == 0, None);
    }
    let stride = W * 4;
    assert!(stride.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let bytes = u64::from(stride) * u64::from(H);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("captured pixels"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let rgba = map_read(gpu, &staging)?;
    let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
    ppm.reserve(usize::try_from(W)? * usize::try_from(H)? * 3);
    for pixel in rgba.chunks_exact(4) {
        ppm.extend_from_slice(&pixel[..3]);
    }
    fs::write(path, ppm)?;
    Ok(rgba)
}
#[derive(Clone, Copy)]
struct Timing {
    min_ms: f64,
    median_ms: f64,
    p90_ms: f64,
}
fn measure(gpu: &Gpu, case: &DrawCase, mode: usize) -> AnyResult<Timing> {
    const FRAMES: usize = 16;
    const WARMUP: usize = 4;
    for _ in 0..WARMUP {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("warmup trees"),
            });
        draw(gpu, &mut encoder, case, mode, true, None);
        gpu.queue.submit([encoder.finish()]);
    }
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let start = Instant::now();
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU foliage timestamp timing"),
        });
    let qset = gpu.timestamp_supported.then(|| {
        gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("tree draw timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERIES,
        })
    });
    for frame in 0..FRAMES {
        let marker = qset.as_ref().map(|q| (q, frame as u32 * 2));
        draw(gpu, &mut encoder, case, mode, true, marker);
    }
    let mapping = qset.as_ref().map(|qs| {
        let size = u64::from(QUERIES) * 8;
        let gpu_times = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolved foliage draw timings"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let host = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback foliage draw timings"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(qs, 0..QUERIES, &gpu_times, 0);
        encoder.copy_buffer_to_buffer(&gpu_times, 0, &host, 0, size);
        (gpu_times, host)
    });
    gpu.queue.submit([encoder.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;
    let mut times = if let Some((_, host)) = mapping {
        let bytes = map_read(gpu, &host)?;
        let ticks: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|eight| u64::from_le_bytes(eight.try_into().unwrap_or([0; 8])))
            .collect();
        let period = f64::from(gpu.queue.get_timestamp_period()) * 1.0e-6;
        ticks
            .chunks_exact(2)
            .map(|pair| pair[1].saturating_sub(pair[0]) as f64 * period)
            .collect::<Vec<_>>()
    } else {
        vec![cpu_ms; FRAMES]
    };
    times.sort_unstable_by(f64::total_cmp);
    Ok(Timing {
        min_ms: times[0],
        median_ms: times[FRAMES / 2],
        p90_ms: times[FRAMES * 9 / 10],
    })
}
fn image_difference(a: &[u8], b: &[u8]) -> f64 {
    let total = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .map(|(left, right)| {
            left[..3]
                .iter()
                .zip(&right[..3])
                .map(|(l, r)| f64::from(l.abs_diff(*r)))
                .sum::<f64>()
        })
        .sum::<f64>();
    total / (f64::from(W) * f64::from(H) * 3.0)
}
fn green_pixels(data: &[u8]) -> usize {
    data.chunks_exact(4)
        .filter(|rgba| rgba[1] > rgba[0] && rgba[1] > rgba[2])
        .count()
}
fn run() -> AnyResult<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: foliage_lab OUTPUT_DIRECTORY")?;
    let output = Path::new(&output);
    fs::create_dir_all(output)?;
    let gpu = open_gpu()?;
    println!(
        "FOLIAGE_GPU name={:?} gpu_timestamps={}",
        gpu.name, gpu.timestamp_supported
    );
    let mut report = format!(
        "gpu={}\nmode,trees,clumps,triangles,geometry_bytes,patch_bytes,min_ms,p50_ms,p90_ms\n",
        gpu.name
    );
    let modes = [
        "gpu_unique",
        "gpu_24_variants",
        "shared_modules",
        "expanded_24_variants",
    ];
    for trees in [1024, 4096] {
        for clumps in [8, 24, 64] {
            let case = DrawCase::new(&gpu, clumps, trees, 0.0, 0)?;
            for (mode, name) in modes.iter().enumerate() {
                let sample = measure(&gpu, &case, mode)?;
                let bytes = match mode {
                    0 | 1 => 0,
                    2 => case.module_bytes,
                    _ => case.geometry_bytes,
                };
                println!(
                    "FOLIAGE_BENCH mode={name} trees={trees} clumps={clumps} triangles={} gpu_ms_p50={:.5} p90={:.5} geometry_bytes={bytes}",
                    (clumps * 4 + 2) * trees,
                    sample.median_ms,
                    sample.p90_ms
                );
                report += &format!(
                    "{name},{trees},{clumps},{},{bytes},{},{:.6},{:.6},{:.6}\n",
                    (clumps * 4 + 2) * trees,
                    case.module_bytes,
                    sample.min_ms,
                    sample.median_ms,
                    sample.p90_ms
                );
            }
        }
    }
    fs::write(output.join("timings.csv"), &report)?;
    let case = DrawCase::new(&gpu, 24, 2048, 0.0, 0)?;
    let mut pictures = Vec::new();
    for (mode, name) in modes.iter().enumerate() {
        let path = output.join(format!("{name}.ppm"));
        let rgba = capture(&gpu, &[(&case, mode)], &path)?;
        println!(
            "FOLIAGE_IMAGE mode={name} green_pixels={} file={}",
            green_pixels(&rgba),
            path.display()
        );
        pictures.push(rgba);
    }
    println!(
        "FOLIAGE_MATCH limited_vs_modules_rgb_mae={:.4} limited_vs_baked_rgb_mae={:.4} unique_vs_limited_rgb_mae={:.4}",
        image_difference(&pictures[1], &pictures[2]),
        image_difference(&pictures[1], &pictures[3]),
        image_difference(&pictures[0], &pictures[1])
    );
    // Identical per-pixel seeds are used in both LODs to avoid alpha spikes.
    let mut transition = String::from("fade,green_pixels,coverage_ratio_to_near\n");
    let mut near_count = 0;
    for fade in [0.0_f32, 0.20, 0.40, 0.60, 0.80, 1.0] {
        let near = DrawCase::new(&gpu, 24, 2048, fade, 1)?;
        let far = DrawCase::new(&gpu, 8, 2048, fade, 2)?;
        let path = output.join(format!("transition-{fade:.2}.ppm"));
        let rgba = capture(&gpu, &[(&near, 0), (&far, 0)], &path)?;
        let count = green_pixels(&rgba);
        if near_count == 0 {
            near_count = count;
        }
        let relative = count as f64 / near_count as f64;
        println!(
            "FOLIAGE_TRANSITION fade={fade:.2} green_pixels={count} ratio_to_near={relative:.5}"
        );
        transition += &format!("{fade:.2},{count},{relative:.6}\n");
    }
    fs::write(output.join("transitions.csv"), transition)?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("FOLIAGE_ERROR: {error}");
        std::process::exit(1);
    }
}
