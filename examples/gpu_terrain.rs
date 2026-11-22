#[path = "gpu_terrain/physics.rs"]
mod physics;
// Actual device probe: deterministic overlapping grids, independent
// collision snapshot ownership, real channel samples, batched timings.
use i_love_proland::terrain::{
    Reach, Region, RegionCell, RegionChart, Section,
    gpu::{CollisionGrid, TerrainGpu, TerrainPointRequest, TerrainRequest},
};
use std::{error::Error, io, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn device() -> Result<(wgpu::Device, wgpu::Queue)> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;
    println!("GPU_TERRAIN adapter={}", adapter.get_info().name);
    Ok(pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU terrain probe"),
            ..Default::default()
        },
    ))?)
}

fn synthetic() -> io::Result<Region> {
    let chart = RegionChart {
        radius_m: 600_000.0,
        latitude_rad: 0.0,
        longitude_rad: 0.0,
        spacing_m: 250.0,
        cells: 8,
    };
    let mut cells = Vec::new();
    for y in 0..=chart.cells {
        for x in 0..=chart.cells {
            let slope = (f64::from(y) - 4.0).abs() * 10.0;
            cells.push(RegionCell::new(
                f64::from(x).mul_add(0.25, 104.0 + slope),
                None,
                [0.3, 0.5, 0.2],
            )?);
        }
    }
    let section = |x| Section {
        xy: [x, 1000.0],
        stage: 100.0,
        width: 40.0,
        bend: 0.0,
    };
    Region::new(
        chart,
        cells,
        vec![Reach {
            a: section(400.0),
            b: section(1600.0),
            valley: 1.0,
        }],
    )
}

fn encoder(device: &wgpu::Device) -> wgpu::CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("terrain probe"),
    })
}

fn wait(
    device: &wgpu::Device,
    mut pending: i_love_proland::terrain::gpu::PendingTerrain,
) -> Result<CollisionGrid> {
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(pending
        .try_take()?
        .ok_or_else(|| io::Error::other("readback not complete"))?)
}

fn check_overlap(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let region = synthetic()?;
    let mut gpu = TerrainGpu::new(device, &region, 65, 3, 0x7654_3210, 1.0)?;
    let coarse = TerrainRequest {
        origin_ticks: [12_800, 14_400],
        spacing_ticks: 64,
        slot: 0,
    };
    let fine = TerrainRequest {
        origin_ticks: coarse.origin_ticks,
        spacing_ticks: 32,
        slot: 1,
    };
    let adjacent = TerrainRequest {
        origin_ticks: [coarse.origin_ticks[0] + 64 * 64, coarse.origin_ticks[1]],
        spacing_ticks: 64,
        slot: 2,
    };
    let mut commands = encoder(device);
    assert!(gpu.snapshot(device, &mut commands, 0).is_err());
    assert!(gpu.encode(device, &mut commands, &[fine, coarse]).is_err());
    gpu.encode(device, &mut commands, &[coarse, fine, adjacent])?;
    let first = gpu.snapshot(device, &mut commands, 0)?;
    let second = gpu.snapshot(device, &mut commands, 1)?;
    let third = gpu.snapshot(device, &mut commands, 2)?;
    // Deliberately replace slot zero after recording its snapshot.
    let reused = TerrainRequest {
        origin_ticks: [0, 0],
        spacing_ticks: 32,
        slot: 0,
    };
    gpu.encode(device, &mut commands, &[reused])?;
    let fourth = gpu.snapshot(device, &mut commands, 0)?;
    queue.submit([commands.finish()]);
    let pending = [first.begin(), second.begin(), third.begin(), fourth.begin()];
    let [a, b, c, d]: [CollisionGrid; 4] = pending
        .map(|p| wait(device, p))
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| io::Error::other("four snapshots expected"))?;
    assert_eq!(a.request, coarse);
    assert_eq!(d.request, reused);
    assert_ne!(a.samples, d.samples);
    for row in 0..33 {
        for col in 0..33 {
            assert_eq!(
                bytemuck::bytes_of(&a.samples[row * 65 + col]),
                bytemuck::bytes_of(&b.samples[row * 130 + col * 2])
            );
        }
    }
    for row in 0..65 {
        assert_eq!(
            bytemuck::bytes_of(&a.samples[row * 65 + 64]),
            bytemuck::bytes_of(&c.samples[row * 65])
        );
    }
    let wet = a
        .samples
        .iter()
        .filter(|sample| sample.water_kind == 2)
        .count();
    assert!(wet > 0);
    assert!(a.samples.iter().all(|sample| sample.bed_m.is_finite()
        && sample.water_m.is_finite()
        && sample.water_m >= sample.bed_m));
    println!(
        "GPU_TERRAIN overlap={} shared_edges=65 wet_samples={wet} immutable_snapshot=true",
        33 * 33
    );
    Ok(())
}

fn check_explicit_points(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let region = synthetic()?;
    let side = 17_u16;
    let mut gpu = TerrainGpu::new(device, &region, side, 3, 0x7654_3210, 1.0)?;
    let grid = TerrainRequest {
        origin_ticks: [12800, 14400],
        spacing_ticks: 192,
        slot: 0,
    };
    let width = u32::from(side);
    let addresses = (0..width * width)
        .map(|index| {
            [
                grid.origin_ticks[0] + grid.spacing_ticks * (index % width),
                grid.origin_ticks[1] + grid.spacing_ticks * (index / width),
            ]
        })
        .collect::<Vec<_>>();
    let reversed = addresses.iter().rev().copied().collect::<Vec<_>>();
    let mut commands = encoder(device);
    gpu.encode(device, &mut commands, &[grid])?;
    let expected = gpu.snapshot_samples(device, &mut commands, 0)?;
    let malformed = TerrainPointRequest {
        slot: 0,
        addresses: &addresses[..1],
    };
    assert!(
        gpu.encode_points(device, &mut commands, &[malformed])
            .is_err()
    );
    let outside = vec![[u32::MAX, 0]; addresses.len()];
    assert!(
        gpu.encode_points(
            device,
            &mut commands,
            &[TerrainPointRequest {
                slot: 0,
                addresses: &outside,
            }]
        )
        .is_err()
    );
    // Neither rejected batch may mutate an existing grid.
    assert!(gpu.snapshot(device, &mut commands, 0).is_ok());
    gpu.encode_points(
        device,
        &mut commands,
        &[
            TerrainPointRequest {
                slot: 0,
                addresses: &addresses,
            },
            TerrainPointRequest {
                slot: 2,
                addresses: &reversed,
            },
        ],
    )?;
    assert!(gpu.snapshot(device, &mut commands, 0).is_err());
    assert!(gpu.snapshot_samples(device, &mut commands, 1).is_err());
    let ordered = gpu.snapshot_samples(device, &mut commands, 0)?;
    let unordered = gpu.snapshot_samples(device, &mut commands, 2)?;
    // Reuse an explicit slot again with an aligned grid.
    gpu.encode(device, &mut commands, &[TerrainRequest { slot: 2, ..grid }])?;
    assert!(gpu.snapshot(device, &mut commands, 2).is_ok());
    queue.submit([commands.finish()]);
    let mut pending = [expected.begin(), ordered.begin(), unordered.begin()];
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let [expected, ordered, reversed] = pending.each_mut().map(|value| {
        value
            .try_take()
            .and_then(|result| result.ok_or_else(|| io::Error::other("point readback unfinished")))
    });
    let expected = expected?;
    let ordered = ordered?;
    let mut reversed = reversed?;
    assert_eq!(expected.len(), addresses.len());
    reversed.reverse();
    let bytes = |samples: &[i_love_proland::terrain::gpu::TerrainSample]| {
        bytemuck::cast_slice::<_, u8>(samples).to_vec()
    };
    assert_eq!(bytes(&expected), bytes(&ordered));
    assert_eq!(bytes(&expected), bytes(&reversed));
    println!(
        "GPU_TERRAIN explicit_points={} identical_bytes={} invalid_preserves_grid=true",
        addresses.len() * 2,
        bytes(&expected).len()
    );
    Ok(())
}

fn profile(device: &wgpu::Device, queue: &wgpu::Queue, region: &Region, side: u16) -> Result<()> {
    let mut gpu = TerrainGpu::new(device, region, side, 32, 0x7654_3210, 1.0)?;
    let extent = region.chart().extent();
    let spacing = 16;
    let span = f64::from(side - 1);
    let reach = region
        .reaches()
        .iter()
        .max_by(|a, b| a.a.width.total_cmp(&b.a.width));
    let center = reach.map_or([extent * 0.5; 2], |r| r.a.xy);
    let center = center.map(|v| span.mul_add(-0.5, v).clamp(0.0, extent - span));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated regional chart extent in integer ticks"
    )]
    #[expect(clippy::cast_sign_loss, reason = "clamped nonnegative chart origin")]
    let base = center.map(|v| (v * 16.0).floor() as u32);
    let mut requests = Vec::new();
    for slot in 0..32 {
        requests.push(TerrainRequest {
            origin_ticks: [base[0], base[1]],
            spacing_ticks: spacing,
            slot,
        });
    }
    let mut warmup = encoder(device);
    gpu.encode(device, &mut warmup, &requests)?;
    queue.submit([warmup.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely())?;
    for count in [1, 8, 32] {
        let start = Instant::now();
        for _ in 0..5 {
            let mut commands = encoder(device);
            gpu.encode(device, &mut commands, &requests[..count])?;
            queue.submit([commands.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
        }
        println!(
            "GPU_TERRAIN side={side} batch={count} samples={} end_to_end_ms={:.3}",
            usize::from(side).pow(2) * count,
            start.elapsed().as_secs_f64() * 200.0
        );
    }
    let start = Instant::now();
    let mut commands = encoder(device);
    let snapshot = gpu.snapshot(device, &mut commands, 0)?;
    queue.submit([commands.finish()]);
    let data = wait(device, snapshot.begin())?;
    let wet = data.samples.iter().filter(|s| s.water_kind == 2).count();
    println!(
        "GPU_TERRAIN collision side={side} bytes={} map_ms={:.3} river_samples={wet}",
        data.samples.len() * 16,
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn main() -> Result<()> {
    let (device, queue) = device()?;
    if std::env::args_os()
        .nth(1)
        .is_some_and(|argument| argument == "--physics")
    {
        return physics::run(&device, &queue);
    }
    if std::env::args_os()
        .nth(1)
        .is_some_and(|argument| argument == "--points")
    {
        return check_explicit_points(&device, &queue);
    }
    check_overlap(&device, &queue)?;
    check_explicit_points(&device, &queue)?;
    let region = if let Some(path) = std::env::args_os().nth(1) {
        Region::read(&std::fs::read(Path::new(&path))?)?
    } else {
        synthetic()?
    };
    profile(&device, &queue, &region, 65)?;
    profile(&device, &queue, &region, 129)?;
    Ok(())
}
