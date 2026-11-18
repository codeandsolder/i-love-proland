//! Real headless Vulkan render of deterministic trees using the GPU compactor
//! and multi-angle impostors. Output PPM files can be converted to PNG with
//! ImageMagick or FFmpeg without adding runtime dependencies to the library.

use i_love_proland::atlas_bake::bake_colored_mesh;
use i_love_proland::gpu::{GpuCompactor, GpuView, PackedInstance, RenderPose};
use i_love_proland::impostor::{AtlasImage, TreeCamera, TreeMesh, TreeRenderer, TreeVertex};
use i_love_proland::{Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant};
use std::error::Error;
use std::fs;
use std::time::Instant;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn append_ellipsoid(
    vertices: &mut Vec<TreeVertex>,
    indices: &mut Vec<u32>,
    center: [f32; 3],
    radii: [f32; 3],
    color: [f32; 4],
) -> Result<(), Box<dyn Error>> {
    const SIDES: u16 = 10;
    const STACKS: u16 = 7;
    let base = u32::try_from(vertices.len())?;
    for layer in 0..=STACKS {
        let latitude = f32::from(layer) / f32::from(STACKS) * core::f32::consts::PI;
        let (sin_lat, cos_lat) = latitude.sin_cos();
        for side in 0..=SIDES {
            let azimuth = f32::from(side) / f32::from(SIDES) * core::f32::consts::TAU;
            let (sin_azi, cos_azi) = azimuth.sin_cos();
            let position = [
                center[0] + radii[0] * sin_lat * sin_azi,
                center[1] + radii[1] * cos_lat,
                center[2] + radii[2] * sin_lat * cos_azi,
                1.0,
            ];
            vertices.push(TreeVertex { position, color });
        }
    }
    for layer in 0..STACKS {
        for side in 0..SIDES {
            let a = base + u32::from(layer) * u32::from(SIDES + 1) + u32::from(side);
            let b = a + u32::from(SIDES + 1);
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    Ok(())
}

fn sample_tree() -> Result<(Vec<TreeVertex>, Vec<u32>), Box<dyn Error>> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    const SIDES: u16 = 10;
    let base = u32::try_from(vertices.len())?;
    for side in 0..SIDES {
        let angle = f32::from(side) / f32::from(SIDES) * core::f32::consts::TAU;
        let (sin_angle, cos_angle) = angle.sin_cos();
        let brown = [0.35, 0.22, 0.12, 1.0];
        vertices.push(TreeVertex {
            position: [0.085 * sin_angle, -0.89, 0.085 * cos_angle, 1.0],
            color: brown,
        });
        vertices.push(TreeVertex {
            position: [0.069 * sin_angle, 0.37, 0.069 * cos_angle, 1.0],
            color: brown,
        });
    }
    for side in 0..SIDES {
        let current = base + u32::from(side) * 2;
        let next = base + u32::from((side + 1) % SIDES) * 2;
        indices.extend_from_slice(&[current, next, current + 1, current + 1, next, next + 1]);
    }
    append_ellipsoid(
        &mut vertices,
        &mut indices,
        [0.0, 0.29, 0.0],
        [0.43, 0.49, 0.40],
        [0.18, 0.47, 0.16, 1.0],
    )?;
    append_ellipsoid(
        &mut vertices,
        &mut indices,
        [0.34, 0.13, 0.0],
        [0.24, 0.31, 0.24],
        [0.25, 0.58, 0.19, 1.0],
    )?;
    append_ellipsoid(
        &mut vertices,
        &mut indices,
        [-0.31, 0.20, -0.23],
        [0.25, 0.28, 0.22],
        [0.16, 0.42, 0.14, 1.0],
    )?;
    append_ellipsoid(
        &mut vertices,
        &mut indices,
        [0.05, 0.52, 0.08],
        [0.27, 0.22, 0.25],
        [0.31, 0.60, 0.20, 1.0],
    )?;
    Ok((vertices, indices))
}

fn generate_forest() -> Result<(Vec<PackedInstance>, f64), Box<dyn Error>> {
    let world = Generator::new(
        Domain::Planar { cell_size_m: 12.0 },
        GeneratorConfig {
            world_seed: 0x1234_abcd_5678_cdef,
            cells_per_tile: 16,
            ..GeneratorConfig::default()
        },
    )?;
    let trees = Layer::new(
        12,
        0.0045,
        Distribution::Poisson,
        vec![Variant {
            id: 900,
            weight: 1.0,
        }],
    )?;
    let start = Instant::now();
    let mut packed = Vec::new();
    for tile_y in 0..9 {
        for tile_x in -6..6 {
            for instance in world.tile(TileKey::planar(tile_x, tile_y), &trees)? {
                let cell = instance.candidate.cell;
                let xy = instance.candidate.local_uv;
                let radius = 8.0 + 5.0 * instance.candidate.size_unit as f32;
                let p = [
                    ((cell.x as f64 + xy[0]) * 12.0) as f32,
                    -6.0 + instance.candidate.size_unit as f32,
                    (45.0 + (cell.y as f64 + xy[1]) * 12.0) as f32,
                ];
                packed.push(PackedInstance::from_instance(
                    &instance,
                    RenderPose {
                        camera_relative_position: p,
                        crown_radius_m: radius,
                        scale: 1.0,
                    },
                )?);
            }
        }
    }
    Ok((packed, start.elapsed().as_secs_f64() * 1000.0))
}

fn create_device() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn Error>> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;
    println!("FOREST_GPU {:?}", adapter.get_info().name);
    Ok(pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("i-love-proland reference forest image capture"),
            ..Default::default()
        },
    ))?)
}

fn frustum_view() -> Result<GpuView, Box<dyn Error>> {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let tan_x = aspect;
    let len_x = (1.0 + tan_x * tan_x).sqrt();
    let len_y = 2.0_f32.sqrt();
    let view = GpuView::new(
        HEIGHT as f32 * 0.5,
        30.0,
        3.5,
        [0.0, 0.0, 1.0],
        [
            [1.0 / len_x, 0.0, tan_x / len_x, 0.0],
            [-1.0 / len_x, 0.0, tan_x / len_x, 0.0],
            [0.0, 1.0 / len_y, 1.0 / len_y, 0.0],
            [0.0, -1.0 / len_y, 1.0 / len_y, 0.0],
            [0.0, 0.0, 1.0, -0.1],
            [0.0, 0.0, -1.0, 50_000.0],
        ],
    )?;
    Ok(view)
}
const BACKGROUND_WGSL: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> Out {
    let xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: Out;
    out.clip = vec4<f32>(xy[idx], 0.99, 1.0);
    out.uv = (xy[idx] + 1.0) * 0.5;
    return out;
}
@fragment
fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let sky = mix(
        vec3<f32>(0.68, 0.85, 0.95),
        vec3<f32>(0.36, 0.61, 0.84),
        clamp(in.uv.y, 0.0, 1.0)
    );
    let ground = mix(
        vec3<f32>(0.28, 0.40, 0.21),
        vec3<f32>(0.43, 0.50, 0.31),
        clamp(in.uv.y * 1.8, 0.0, 1.0)
    );
    let t = smoothstep(0.495, 0.505, in.uv.y);
    return vec4<f32>(mix(ground, sky, t), 1.0);
}
";

fn background_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forest demo sky/ground"),
        source: wgpu::ShaderSource::Wgsl(BACKGROUND_WGSL.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("forest demo sky and ground"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn render_frame(
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    compactor: &GpuCompactor,
    renderer: &TreeRenderer,
    mesh: &TreeMesh,
    candidates: &[PackedInstance],
    view: GpuView,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    background: &wgpu::RenderPipeline,
) -> Result<(), Box<dyn Error>> {
    compactor.encode(queue, encoder, candidates, view)?;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("i-love-proland real forest GPU draw"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
    pass.set_pipeline(background);
    pass.draw(0..3, 0..1);
    renderer.draw(&mut pass, mesh, compactor);
    Ok(())
}

fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    color: &wgpu::Texture,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let bytes_per_row = WIDTH * 4;
    // Width=1280 => 5120 bytes, exactly 20 times the required 256B alignment.
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forest GPU framebuffer image"),
        size: u64::from(bytes_per_row) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;
    let bytes = buffer.slice(..).get_mapped_range()?.to_vec();
    buffer.unmap();
    Ok(bytes)
}

fn write_ppm(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), Box<dyn Error>> {
    let expected = usize::try_from(width)? * usize::try_from(height)? * 4;
    if rgba.len() != expected {
        return Err("invalid PPM RGBA dimensions".into());
    }
    let mut image = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in rgba.chunks_exact(4) {
        image.extend_from_slice(&pixel[..3]);
    }
    fs::write(path, image)?;
    Ok(())
}

fn write_atlas_preview(path: &str, atlas: &AtlasImage) -> Result<(), Box<dyn Error>> {
    let mut composite = atlas.rgba().to_vec();
    for px in composite.chunks_exact_mut(4) {
        let alpha = f32::from(px[3]) / 255.0;
        for channel in px.iter_mut().take(3) {
            let source = f32::from(*channel);
            let backdrop = 0.15 * 255.0;
            let mixed = source.mul_add(alpha, backdrop * (1.0 - alpha));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the RGBA blend is clamped to a valid u8 channel range"
            )]
            {
                *channel = mixed.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    write_ppm(path, atlas.width(), atlas.height(), &composite)
}

fn main() -> Result<(), Box<dyn Error>> {
    let prefix = std::env::args().nth(1).ok_or(
        "usage: cargo run --features gpu --example forest_render -- /path/to/output-prefix",
    )?;
    let (vertices, indices) = sample_tree()?;
    let start = Instant::now();
    let atlas = bake_colored_mesh(&vertices, &indices, 64, 2)?;
    let atlas_ms = start.elapsed().as_secs_f64() * 1000.0;
    write_atlas_preview(&format!("{prefix}-atlas.ppm"), &atlas)?;

    let (candidates, generation_ms) = generate_forest()?;
    let (device, queue) = create_device()?;
    let gpu_mesh = TreeMesh::new(&device, &vertices, &indices)?;
    let compactor = GpuCompactor::new(
        &device,
        u32::try_from(candidates.len())?,
        gpu_mesh.index_count(),
        6,
    )?;
    let atlas_texture = atlas.upload(&device, &queue)?;
    let renderer = TreeRenderer::new(
        &device,
        &compactor,
        &gpu_mesh,
        &atlas_texture,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    )?;
    renderer.set_camera(
        &queue,
        TreeCamera::perspective(
            core::f32::consts::FRAC_PI_2,
            WIDTH as f32 / HEIGHT as f32,
            0.5,
            50_000.0,
        )?,
    );
    let view = frustum_view()?;
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("forest GPU color capture"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("forest GPU depth capture"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let background = background_pipeline(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("first forest test scene"),
    });
    render_frame(
        &queue,
        &mut encoder,
        &compactor,
        &renderer,
        &gpu_mesh,
        &candidates,
        view,
        &color_view,
        &depth_view,
        &background,
    )?;
    let image = capture(&device, &queue, encoder, &color)?;
    let greens = image
        .chunks_exact(4)
        .filter(|p| p[1] > p[0] && p[1] > p[2])
        .count();
    if greens < 1_000 {
        return Err(format!("GPU forest capture appears empty: only {greens} green pixels").into());
    }
    write_ppm(&format!("{prefix}-scene.ppm"), WIDTH, HEIGHT, &image)?;

    // Full frame includes compaction, near mesh and billboard indirect draws.
    // No readback in the measured frames; synchronize after the batch.
    let start = Instant::now();
    let frames = 20_u32;
    for _ in 0..frames {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forest throughput probe"),
        });
        render_frame(
            &queue,
            &mut encoder,
            &compactor,
            &renderer,
            &gpu_mesh,
            &candidates,
            view,
            &color_view,
            &depth_view,
            &background,
        )?;
        queue.submit(Some(encoder.finish()));
    }
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let frame_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);
    println!(
        "FOREST_RENDER_PASS candidates={} mesh_indices={} atlas_ms={atlas_ms:.2} canonical_ms={generation_ms:.2} capture_green_pixels={greens} gpu_compact_and_indirect_render_ms={frame_ms:.3} images={prefix}-scene.ppm,{prefix}-atlas.ppm",
        candidates.len(),
        gpu_mesh.index_count(),
    );
    Ok(())
}
