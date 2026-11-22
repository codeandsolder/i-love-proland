//! Real headless Vulkan render of deterministic trees using the GPU compactor
//! and multi-angle impostors. Output PPM files can be converted to PNG with
//! external conversion tools without adding runtime dependencies to the library.

use i_love_proland::atlas_bake::{BakeSurface, bake_scene};
use i_love_proland::gpu::{
    GpuCompactor, GpuView, MeshLod, MeshLods, PackedInstance, RenderPose, VisualLod, classify_cpu,
};
use i_love_proland::impostor::{
    AtlasImage, AtlasTexture, TreeCamera, TreeMaterial, TreeMesh, TreeRenderConfig, TreeRenderer,
    TreeVertex,
};
use i_love_proland::material::SourceImage;
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
                (radii[0] * sin_lat).mul_add(sin_azi, center[0]),
                radii[1].mul_add(cos_lat, center[1]),
                (radii[2] * sin_lat).mul_add(cos_azi, center[2]),
                1.0,
            ];
            vertices.push(TreeVertex {
                position,
                color,
                uv: [
                    f32::from(side) / f32::from(SIDES),
                    f32::from(layer) / f32::from(STACKS),
                ],
            });
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
    const SIDES: u16 = 10;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let base = u32::try_from(vertices.len())?;
    for side in 0..SIDES {
        let angle = f32::from(side) / f32::from(SIDES) * core::f32::consts::TAU;
        let (sin_angle, cos_angle) = angle.sin_cos();
        let brown = [0.35, 0.22, 0.12, 1.0];
        vertices.push(TreeVertex {
            position: [0.085 * sin_angle, -0.89, 0.085 * cos_angle, 1.0],
            color: brown,
            uv: [0.5, 0.5],
        });
        vertices.push(TreeVertex {
            position: [0.069 * sin_angle, 0.37, 0.069 * cos_angle, 1.0],
            color: brown,
            uv: [0.5, 0.5],
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
                // GPU-only camera position. Canonical world IDs and positions
                // retain full f64 precision, independent of this visual pose.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "finite canonical fractions [0,1) are projected to camera-relative f32"
                )]
                let (coord_u, coord_v, visual_size) = (
                    xy[0] as f32,
                    xy[1] as f32,
                    instance.candidate.size_unit as f32,
                );
                let world_x = f32::from(i16::try_from(cell.x)?);
                let world_z = f32::from(i16::try_from(cell.y)?);
                let radius = 5.0_f32.mul_add(visual_size, 8.0);
                let position = [
                    (world_x + coord_u) * 12.0,
                    -6.0 + visual_size,
                    (world_z + coord_v).mul_add(12.0, 45.0),
                ];
                packed.push(PackedInstance::from_instance(
                    &instance,
                    RenderPose::upright(position, radius, instance.candidate.yaw_turns),
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

fn pitched_view(pitch: f32) -> Result<(TreeCamera, GpuView), Box<dyn Error>> {
    // Camera aims along +Z when pitch=0, directly downward at pitch=PI/2.
    // The world-up vector stays +Y for both mesh orientation and atlas lookup.
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let aspect = 16.0_f32 / 9.0;
    let near = 0.5;
    let far = 50_000.0;
    let depth = far / (far - near);
    let camera = TreeCamera::new(
        [
            [1.0 / aspect, 0.0, 0.0, 0.0],
            [0.0, cos_pitch, -depth * sin_pitch, -sin_pitch],
            [0.0, sin_pitch, depth * cos_pitch, cos_pitch],
            [0.0, 0.0, -near * depth, 0.0],
        ],
        [0.0, -sin_pitch, cos_pitch],
    )?;
    let nx = aspect.mul_add(aspect, 1.0).sqrt();
    let ny = 2.0_f32.sqrt();
    let view = GpuView::new(
        360.0,
        30.0,
        3.5,
        [0.0, -sin_pitch, cos_pitch],
        [
            [
                1.0 / nx,
                -aspect * sin_pitch / nx,
                aspect * cos_pitch / nx,
                0.0,
            ],
            [
                -1.0 / nx,
                -aspect * sin_pitch / nx,
                aspect * cos_pitch / nx,
                0.0,
            ],
            [
                0.0,
                (cos_pitch - sin_pitch) / ny,
                (sin_pitch + cos_pitch) / ny,
                0.0,
            ],
            [
                0.0,
                (-cos_pitch - sin_pitch) / ny,
                (-sin_pitch + cos_pitch) / ny,
                0.0,
            ],
            [0.0, -sin_pitch, cos_pitch, -near],
            [0.0, sin_pitch, -cos_pitch, far],
        ],
    )?;
    Ok((camera, view))
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
@fragment
fn fs_overhead(in: Out) -> @location(0) vec4<f32> {
    let green = mix(
        vec3<f32>(0.35, 0.44, 0.26),
        vec3<f32>(0.45, 0.51, 0.33),
        clamp(in.uv.y, 0.0, 1.0)
    );
    return vec4<f32>(green, 1.0);
}
";

fn background_pipeline(device: &wgpu::Device, fragment_entry: &str) -> wgpu::RenderPipeline {
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
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[derive(Clone, Copy)]
struct ForestPass<'a> {
    queue: &'a wgpu::Queue,
    renderer: &'a TreeRenderer,
    candidates: &'a [PackedInstance],
    view: GpuView,
    color: &'a wgpu::TextureView,
    depth: &'a wgpu::TextureView,
    background: &'a wgpu::RenderPipeline,
}

fn render_frame(
    encoder: &mut wgpu::CommandEncoder,
    scene: &ForestPass<'_>,
) -> Result<(), Box<dyn Error>> {
    scene
        .renderer
        .encode(scene.queue, encoder, scene.candidates, scene.view)?;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("i-love-proland real forest GPU draw"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: scene.color,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: scene.depth,
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
    pass.set_pipeline(scene.background);
    pass.draw(0..3, 0..1);
    scene.renderer.draw(&mut pass);
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
    for pixel in rgba.as_chunks::<4>().0 {
        image.extend_from_slice(&pixel[..3]);
    }
    fs::write(path, image)?;
    Ok(())
}

fn write_atlas_preview(path: &str, atlas: &AtlasImage) -> Result<(), Box<dyn Error>> {
    let mut composite = atlas.rgba().to_vec();
    for px in composite.as_chunks_mut::<4>().0 {
        let alpha = f32::from(px[3]) / 255.0;
        for channel in px.iter_mut().take(3) {
            let source = f32::from(*channel);
            let backdrop = 0.15 * 255.0;
            let mixed = source.mul_add(alpha, backdrop * (1.0 - alpha));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the RGBA blend is clamped to a valid u8 channel range"
            )]
            #[expect(
                clippy::cast_sign_loss,
                reason = "RGB values are explicitly clamped to nonnegative [0,255]"
            )]
            {
                *channel = mixed.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    write_ppm(path, atlas.width(), atlas.height(), &composite)
}

struct FrameTargets {
    _depth: wgpu::Texture,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

fn make_frame_targets(device: &wgpu::Device) -> FrameTargets {
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
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    FrameTargets {
        _depth: depth,
        color,
        color_view,
        depth_view,
    }
}

fn render_overhead(
    device: &wgpu::Device,
    scene: &ForestPass<'_>,
    low_camera: TreeCamera,
    background: &wgpu::RenderPipeline,
    color: &wgpu::Texture,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    // Same canonical objects, rebased 480m for the 90 degree downwards view.
    let mut overhead = scene.candidates.to_vec();
    for tree in &mut overhead {
        tree.position_radius[1] -= 480.0;
    }
    let (overhead_camera, overhead_view) = pitched_view(core::f32::consts::FRAC_PI_2)?;
    scene.renderer.set_camera(scene.queue, overhead_camera);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fully overhead forest atlas test"),
    });
    render_frame(
        &mut encoder,
        &ForestPass {
            candidates: &overhead,
            view: overhead_view,
            background,
            ..*scene
        },
    )?;
    let pixels = capture(device, scene.queue, encoder, color)?;
    write_ppm(&format!("{prefix}-overhead.ppm"), WIDTH, HEIGHT, &pixels)?;
    scene.renderer.set_camera(scene.queue, low_camera);
    Ok(())
}

// Render the same canonical objects twice at a fixed camera, changing only
// the wind time. Then restore the exact original unanimated camera.
fn validate_wind_animation(
    device: &wgpu::Device,
    scene: &ForestPass<'_>,
    camera: TreeCamera,
    color: &wgpu::Texture,
) -> Result<(), Box<dyn Error>> {
    use i_love_proland::wind::WindFrame;
    if std::env::var_os("PROLAND_TEST_WIND").is_none() {
        return Ok(());
    }
    let render = |camera, label| -> Result<Vec<u8>, Box<dyn Error>> {
        scene.renderer.set_camera(scene.queue, camera);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        render_frame(&mut encoder, scene)?;
        capture(device, scene.queue, encoder, color)
    };
    let axes = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let start = camera.with_wind(WindFrame::new([0., 600_000., 0.], axes, 0.0)?);
    let end = camera.with_wind(WindFrame::new([0., 600_000., 0.], axes, 3.0)?);
    let first = render(start, "wind t=0")?;
    let second = render(end, "wind t=3")?;
    let repeat = render(start, "wind t=0 repeat")?;
    let count_changed = |left: &[u8], right: &[u8]| {
        left.as_chunks::<4>()
            .0
            .iter()
            .zip(right.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count()
    };
    let moved = count_changed(&first, &second);
    let repeat_changed = count_changed(&first, &repeat);
    assert!(moved > 0, "wind animation must change pixels");
    // GPU atomic output order can resolve depth-tied edges differently.
    // Compare repeated-time noise against the actual animation signal.
    assert!(
        moved > repeat_changed * 10,
        "wind signal must exceed repeat rendering noise"
    );
    println!("FOREST_WIND_PASS changed_pixels={moved} repeated_time_changed={repeat_changed}");
    scene.renderer.set_camera(scene.queue, camera);
    Ok(())
}

// Real GPU regression: a nonuniformly scaled card is edge-on when
// static, but its original-source billboard must face the camera. A
// second view-facing card with the same canonical pose is not generated.
fn validate_source_billboards(
    device: &wgpu::Device,
    scene: &ForestPass<'_>,
    targets: &FrameTargets,
    atlas: &AtlasTexture,
    white: &wgpu::TextureView,
) -> Result<(), Box<dyn Error>> {
    use i_love_proland::billboard::Billboard;
    if std::env::var_os("PROLAND_TEST_BILLBOARD").is_none() {
        return Ok(());
    }
    let vertices = [[0.0_f32, 0.0_f32], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].map(|uv| TreeVertex {
        position: [uv[0] - 0.5, uv[1] - 0.5, 0.0, 1.0],
        color: [1.0, 0.0, 0.0, 1.0],
        uv,
    });
    let card = TreeMesh::new(device, &vertices, &[0, 1, 2, 2, 1, 3])?;
    let candidates = [PackedInstance {
        position_radius: [0.0, 0.4, 8.0, 1.7],
        rotation: [
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
        ],
        axis_radii: [1.4, 0.8, 1.7, 1.0],
        object_id: [1; 4],
        metadata: [2; 4],
        tint: [1.0; 4],
    }];
    let (camera, view) = pitched_view(0.0)?;
    let render = |billboard| -> Result<usize, Box<dyn Error>> {
        let compactor = GpuCompactor::new(
            device,
            1,
            MeshLods::new(vec![MeshLod {
                start_m: 0.0,
                index_count: card.index_count(),
                prefer_mesh: false,
            }])?,
            6,
        )?;
        let renderer = TreeRenderer::new(
            device,
            compactor,
            vec![vec![TreeMaterial {
                mesh: card.clone(),
                source_view: white,
                alpha_cutoff: 0.0,
                wind: None,
                billboard,
            }]],
            atlas,
            TreeRenderConfig {
                color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
                depth_format: wgpu::TextureFormat::Depth24Plus,
                sample_count: 1,
            },
        )?;
        renderer.set_camera(scene.queue, camera);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("authored source card regression"),
        });
        render_frame(
            &mut encoder,
            &ForestPass {
                candidates: &candidates,
                renderer: &renderer,
                view,
                ..*scene
            },
        )?;
        let pixels = capture(device, scene.queue, encoder, &targets.color)?;
        Ok(pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] > 245 && p[1] < 10 && p[2] < 10)
            .count())
    };
    let static_pixels = render(None)?;
    let billboard = Some(Billboard::new([0.0, 0.5, 0.0], 1.0)?);
    let animated_pixels = render(billboard)?;
    let repeated_pixels = render(billboard)?;
    assert_eq!(static_pixels, 0);
    assert!(animated_pixels > 1000);
    assert_eq!(animated_pixels, repeated_pixels);
    println!(
        "FOREST_BILLBOARD_PASS static_pixels={static_pixels} facing_pixels={animated_pixels} repeat_pixels={repeated_pixels}"
    );
    Ok(())
}

fn benchmark_forest(device: &wgpu::Device, scene: &ForestPass<'_>) -> Result<f64, Box<dyn Error>> {
    const FRAMES: u32 = 20;
    let start = Instant::now();
    for _ in 0..FRAMES {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forest throughput probe"),
        });
        render_frame(&mut encoder, scene)?;
        scene.queue.submit(Some(encoder.finish()));
    }
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(FRAMES))
}

fn load_source_image(args: &[String]) -> Result<SourceImage, Box<dyn Error>> {
    match args {
        [] => Ok(SourceImage::white()),
        [path, width, height] => Ok(SourceImage::new(
            width.parse()?,
            height.parse()?,
            fs::read(path)?,
        )?),
        _ => Err("expected no material or <rgba-file> <width> <height>".into()),
    }
}

/// Independent nonuniform RGB wind fixture; never reuse the white default
/// albedo or depend on external asset files for the animation regression.
fn wind_fixture() -> Result<SourceImage, Box<dyn Error>> {
    let mut rgba = Vec::new();
    for y in 0_u8..8 {
        for x in 0_u8..8 {
            rgba.extend_from_slice(&[x * 32, y * 32, (x ^ y) * 32, 255]);
        }
    }
    Ok(SourceImage::new(8, 8, rgba)?)
}

struct SourceViews<'a> {
    bark: &'a wgpu::TextureView,
    foliage: &'a wgpu::TextureView,
    wind: &'a wgpu::TextureView,
}

/// Share two immutable source meshes across every tier; keep their cutoffs.
fn make_source_renderer(
    device: &wgpu::Device,
    compactor: GpuCompactor,
    trunk_mesh: &TreeMesh,
    leaf_mesh: &TreeMesh,
    atlas_texture: &AtlasTexture,
    views: &SourceViews<'_>,
) -> Result<TreeRenderer, Box<dyn Error>> {
    let materials = compactor
        .mesh_lods()
        .tiers()
        .iter()
        .map(|_| {
            vec![
                TreeMaterial {
                    mesh: trunk_mesh.clone(),
                    source_view: views.bark,
                    alpha_cutoff: 0.0,
                    billboard: None,
                    wind: None,
                },
                TreeMaterial {
                    mesh: leaf_mesh.clone(),
                    source_view: views.foliage,
                    alpha_cutoff: 0.45,
                    billboard: None,
                    wind: Some(i_love_proland::impostor::MeshWind {
                        parameters: i_love_proland::wind::WindParameters {
                            spatial_scale: 0.08,
                            height_start: 0.0,
                            height_exponent: 1.62,
                            speed: 2.1,
                            intensity: 0.7,
                        },
                        map: views.wind,
                        source_center_y: 1.0,
                        source_radius: 1.0,
                    }),
                },
            ]
        })
        .collect();
    let count = compactor.mesh_lods().tiers().len();
    let renderer = TreeRenderer::new(
        device,
        compactor,
        materials,
        atlas_texture,
        TreeRenderConfig {
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            depth_format: wgpu::TextureFormat::Depth24Plus,
            sample_count: 1,
        },
    )?;
    assert_eq!(renderer.material_count(), 2 * count);
    Ok(renderer)
}

fn validate_green_forest(image: &[u8]) -> Result<usize, Box<dyn Error>> {
    let greens = image
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[1] > p[0] && p[1] > p[2])
        .count();
    if greens < 1_000 {
        return Err(format!("GPU forest capture appears empty: only {greens} green pixels").into());
    }
    Ok(greens)
}

fn example_compactor(
    device: &wgpu::Device,
    count: usize,
    index_count: u32,
) -> Result<GpuCompactor, Box<dyn Error>> {
    let starts = if std::env::var_os("PROLAND_SINGLE_MESH_TIER").is_some() {
        vec![0.0]
    } else {
        vec![0.0, 45.0, 110.0]
    };
    let lods = MeshLods::new(
        starts
            .into_iter()
            .map(|start_m| MeshLod {
                start_m,
                index_count,
                prefer_mesh: false,
            })
            .collect(),
    )?;
    Ok(GpuCompactor::new(device, u32::try_from(count)?, lods, 6)?)
}

fn validate_and_benchmark(
    device: &wgpu::Device,
    scene: &ForestPass<'_>,
    camera: TreeCamera,
    targets: &FrameTargets,
    atlas: &AtlasTexture,
    white: &wgpu::TextureView,
) -> Result<f64, Box<dyn Error>> {
    validate_wind_animation(device, scene, camera, &targets.color)?;
    validate_source_billboards(device, scene, targets, atlas, white)?;
    benchmark_forest(device, scene)
}

fn capture_initial_forest(
    device: &wgpu::Device,
    scene: &ForestPass<'_>,
    color: &wgpu::Texture,
    prefix: &str,
) -> Result<usize, Box<dyn Error>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("low-altitude forest test scene"),
    });
    render_frame(&mut encoder, scene)?;
    let image = capture(device, scene.queue, encoder, color)?;
    let greens = validate_green_forest(&image)?;
    write_ppm(&format!("{prefix}-scene.ppm"), WIDTH, HEIGHT, &image)?;
    Ok(greens)
}

/// Exercise every ordinary mesh tier and the impostor bin before/after
/// capacity growth. Nonoverlapping screen positions avoid depth-tie noise.
fn validate_capacity_growth(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut TreeRenderer,
    targets: &FrameTargets,
    background: &wgpu::RenderPipeline,
    view: GpuView,
    required: usize,
) -> Result<(), Box<dyn Error>> {
    let fixture = [
        [-15.0, -3.0, 30.0, 8.0],
        [0.0, -3.0, 65.0, 8.0],
        [65.0, -3.0, 130.0, 8.0],
        [-650.0, -3.0, 800.0, 8.0],
    ]
    .map(|position_radius| PackedInstance {
        position_radius,
        rotation: [0.0, 0.0, 0.0, 1.0],
        axis_radii: [8.0, 8.0, 8.0, 1.0],
        object_id: [1; 4],
        metadata: [2; 4],
        tint: [1.0; 4],
    });
    let lods = renderer.compactor().mesh_lods();
    for (tier, object) in fixture[..3].iter().enumerate() {
        assert_eq!(
            classify_cpu(object, view, lods),
            VisualLod::Mesh(tier.min(lods.tiers().len() - 1))
        );
    }
    assert_eq!(classify_cpu(&fixture[3], view, lods), VisualLod::Impostor);
    let subset_capture = |renderer: &TreeRenderer, candidates: &[PackedInstance]| {
        let scene = ForestPass {
            queue,
            renderer,
            candidates,
            view,
            color: &targets.color_view,
            depth: &targets.depth_view,
            background,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        render_frame(&mut encoder, &scene)?;
        capture(device, queue, encoder, &targets.color)
    };
    let initial_count = renderer.compactor().capacity();
    let empty = subset_capture(renderer, &[])?;
    let before = subset_capture(renderer, &fixture)?;
    assert_ne!(empty, before, "fixture must render visible geometry");
    let bin = renderer.compactor().bin(0).ok_or("missing first bin")?;
    let previous_indirect = bin.indirect.clone();
    let previous_visible = bin.indices.buffer.clone();
    let growth_started = Instant::now();
    assert!(renderer.reserve(device, u32::try_from(required)?)?);
    let growth_ms = growth_started.elapsed().as_secs_f64() * 1000.0;
    let bin = renderer
        .compactor()
        .bin(0)
        .ok_or("missing grown first bin")?;
    assert_eq!(&previous_indirect, bin.indirect);
    assert_ne!(&previous_visible, bin.indices.buffer);
    assert!(!renderer.reserve(device, 1)?);
    assert!(renderer.reserve(device, u32::MAX).is_err());
    let after = subset_capture(renderer, &fixture)?;
    assert_eq!(
        before, after,
        "capacity growth changed the same submitted scene"
    );
    println!(
        "FOREST_RESERVE_PASS initial={initial_count} capacity={} growth_ms={growth_ms:.3} pixel_identical=true",
        renderer.compactor().capacity()
    );
    Ok(())
}

fn bake_reference_atlas(
    trunk_vertices: &[TreeVertex],
    trunk_indices: &[u32],
    leaf_vertices: &[TreeVertex],
    leaf_indices: &[u32],
    bark: &SourceImage,
    source: &SourceImage,
) -> Result<(AtlasImage, f64), Box<dyn Error>> {
    let start = Instant::now();
    let atlas = bake_scene(
        &[
            BakeSurface {
                vertices: trunk_vertices,
                indices: trunk_indices,
                source: bark,
                alpha_cutoff: 0.0,
                billboard: None,
            },
            BakeSurface {
                vertices: leaf_vertices,
                indices: leaf_indices,
                source,
                alpha_cutoff: 0.45,
                billboard: None,
            },
        ],
        64,
        2,
    )?;
    let atlas_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((atlas, atlas_ms))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let prefix = args.next().ok_or(
        "usage: cargo run --features gpu --example forest_render -- /path/to/output-prefix",
    )?;
    let (vertices, indices) = sample_tree()?;
    let source = load_source_image(&args.collect::<Vec<_>>())?;
    // The synthetic source consists of a 10-sided trunk (20 vertices, 60
    // indices), followed by several UV-mapped crowns. They use different
    // source textures and alpha cutoffs but share one physical tree instance.
    let trunk_vertices = &vertices[..20];
    let trunk_indices = &indices[..60];
    let leaf_vertices = &vertices[20..];
    let leaf_indices = indices[60..]
        .iter()
        .map(|index| index - 20)
        .collect::<Vec<_>>();
    let bark = SourceImage::white();
    let (atlas, atlas_ms) = bake_reference_atlas(
        trunk_vertices,
        trunk_indices,
        leaf_vertices,
        &leaf_indices,
        &bark,
        &source,
    )?;
    write_atlas_preview(&format!("{prefix}-atlas.ppm"), &atlas)?;

    let (candidates, generation_ms) = generate_forest()?;
    let (device, queue) = create_device()?;
    let trunk_mesh = TreeMesh::new(&device, trunk_vertices, trunk_indices)?;
    let leaf_mesh = TreeMesh::new(&device, leaf_vertices, &leaf_indices)?;
    let compactor =
        example_compactor(&device, candidates.len().min(256), trunk_mesh.index_count())?;
    let atlas_texture = atlas.upload(&device, &queue)?;
    let bark_texture = bark.upload(&device, &queue)?;
    let source_texture = source.upload(&device, &queue)?;
    let wind_texture = wind_fixture()?.upload(&device, &queue)?;
    let mut renderer = make_source_renderer(
        &device,
        compactor,
        &trunk_mesh,
        &leaf_mesh,
        &atlas_texture,
        &SourceViews {
            bark: bark_texture.view(),
            foliage: source_texture.view(),
            wind: wind_texture.view(),
        },
    )?;
    let (low_camera, view) = pitched_view(0.0)?;
    renderer.set_camera(&queue, low_camera);
    let targets = make_frame_targets(&device);
    let color = &targets.color;
    let background = background_pipeline(&device, "fs_main");
    let overhead_background = background_pipeline(&device, "fs_overhead");
    validate_capacity_growth(
        &device,
        &queue,
        &mut renderer,
        &targets,
        &background,
        view,
        candidates.len(),
    )?;
    let scene = ForestPass {
        queue: &queue,
        renderer: &renderer,
        candidates: &candidates,
        view,
        color: &targets.color_view,
        depth: &targets.depth_view,
        background: &background,
    };
    let greens = capture_initial_forest(&device, &scene, color, &prefix)?;

    render_overhead(
        &device,
        &scene,
        low_camera,
        &overhead_background,
        color,
        &prefix,
    )?;

    let frame_ms = validate_and_benchmark(
        &device,
        &scene,
        low_camera,
        &targets,
        &atlas_texture,
        bark_texture.view(),
    )?;
    println!(
        "FOREST_RENDER_PASS candidates={} mesh_indices={} atlas_ms={atlas_ms:.2} canonical_ms={generation_ms:.2} capture_green_pixels={greens} gpu_compact_and_indirect_render_ms={frame_ms:.3} images={prefix}-scene.ppm,{prefix}-atlas.ppm",
        candidates.len(),
        trunk_indices.len() + leaf_indices.len(),
    );
    Ok(())
}
