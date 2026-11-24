//! Verify that a sleeping contact patch can be represented by its
//! canonical coordinates, not by a stored height/triangle snapshot.
use super::{Result, encoder};
use i_love_proland::terrain::{
    Region,
    gpu::{CollisionGrid, TerrainGpu, TerrainPointRequest, TerrainRequest, TerrainSample},
};
use std::{io, path::Path, time::Instant};

const SIDE: u16 = 129;
const SMALL: u16 = 65;
const SPACING: u32 = 16;
const SPAN: u32 = (SIDE as u32 - 1) * SPACING;
const LOCATIONS: u32 = 16;

fn extent_ticks(region: &Region) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "integer-addressed chart extent"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "validated nonnegative regional chart"
    )]
    let extent = (region.chart().extent() * 16.0).floor() as u32;
    extent
}

fn tick_origin(center: [f64; 2], limit: u32) -> [u32; 2] {
    // All independently generated grids lie on the same one-metre
    // lattice, including near the clipped chart boundary.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped integer lattice address"
    )]
    #[expect(clippy::cast_sign_loss, reason = "clamped nonnegative coordinate")]
    let origin = center.map(|value| {
        (value - f64::from(SPAN / SPACING) * 0.5)
            .floor()
            .clamp(0.0, f64::from(limit / SPACING)) as u32
            * SPACING
    });
    origin
}

fn requests(region: &Region) -> Result<Vec<TerrainRequest>> {
    let limit = extent_ticks(region)
        .checked_sub(SPAN)
        .ok_or_else(|| io::Error::other("regional chart is smaller than the regeneration grid"))?;
    let aligned = limit / SPACING * SPACING;
    let mut origins = vec![[0, 0], [aligned, 0], [0, aligned], [aligned, aligned]];
    // Include the major channel, a wide range of relief and arbitrary
    // addresses rather than only synthetic flat terrain.
    let mut reaches = region.reaches().iter().collect::<Vec<_>>();
    reaches.sort_by(|a, b| b.a.width.total_cmp(&a.a.width));
    origins.extend(
        reaches
            .iter()
            .take(4)
            .map(|reach| tick_origin(reach.a.xy, limit)),
    );
    let mut state = 0x83d8_551b_u32;
    while origins.len() < LOCATIONS as usize {
        origins.push(std::array::from_fn(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state % (aligned / SPACING + 1)) * SPACING
        }));
    }
    Ok((0..LOCATIONS)
        .zip(origins)
        .map(|(slot, origin_ticks)| TerrainRequest {
            origin_ticks,
            spacing_ticks: SPACING,
            slot,
        })
        .collect())
}

fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut TerrainGpu,
    requests: &[TerrainRequest],
) -> Result<Vec<CollisionGrid>> {
    let mut commands = encoder(device);
    gpu.encode(device, &mut commands, requests)?;
    let snapshots = requests
        .iter()
        .map(|request| gpu.snapshot(device, &mut commands, request.slot))
        .collect::<io::Result<Vec<_>>>()?;
    queue.submit([commands.finish()]);
    let pending = snapshots
        .into_iter()
        .map(|snapshot| snapshot.begin())
        .collect::<Vec<_>>();
    device.poll(wgpu::PollType::wait_indefinitely())?;
    pending
        .into_iter()
        .map(|mut pending| {
            pending
                .try_take()?
                .ok_or_else(|| io::Error::other("regeneration readback unfinished").into())
        })
        .collect()
}

#[derive(Default)]
struct Differences {
    count: usize,
    samples: usize,
    max_bed: f64,
    max_water: f64,
    material: usize,
}

impl Differences {
    fn add(&mut self, expected: &TerrainSample, actual: &TerrainSample) {
        self.samples += 1;
        self.count += usize::from(bytemuck::bytes_of(expected) != bytemuck::bytes_of(actual));
        self.max_bed = self
            .max_bed
            .max((f64::from(expected.bed_m) - f64::from(actual.bed_m)).abs());
        self.max_water = self
            .max_water
            .max((f64::from(expected.water_m) - f64::from(actual.water_m)).abs());
        self.material +=
            usize::from(expected.rgba != actual.rgba || expected.water_kind != actual.water_kind);
    }

    fn compare(&mut self, expected: &[TerrainSample], actual: &[TerrainSample]) {
        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(actual) {
            self.add(a, b);
        }
    }

    fn verify(&self, kind: &str) -> Result<()> {
        println!(
            "TERRAIN_REGEN mode={kind} samples={} mismatches={} max_bed_m={:.9} max_water_m={:.9} material_mismatches={}",
            self.samples, self.count, self.max_bed, self.max_water, self.material
        );
        if self.count != 0 {
            return Err(io::Error::other("nonidentical canonical terrain samples").into());
        }
        Ok(())
    }
}

fn fresh_pool(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    region: &Region,
    baseline: &[CollisionGrid],
    original: &[TerrainRequest],
) -> Result<()> {
    let mut gpu = TerrainGpu::new(device, region, SIDE, LOCATIONS)?;
    // Reverse the spatial request order and slot association, overwrite
    // all resident allocations, then reproduce the original coordinates.
    let shuffled = original
        .iter()
        .rev()
        .zip(0..LOCATIONS)
        .map(|(r, slot)| TerrainRequest { slot, ..*r })
        .collect::<Vec<_>>();
    let _ = capture(device, queue, &mut gpu, &shuffled)?;
    let mut differences = Differences::default();
    for _ in 0..3 {
        let grids = capture(device, queue, &mut gpu, original)?;
        for (a, b) in baseline.iter().zip(&grids) {
            differences.compare(&a.samples, &b.samples);
            // Canonical body-fixed vertices and fixed diagonal must match.
            assert_eq!(
                a.body_vertices(region.chart())?,
                b.body_vertices(region.chart())?
            );
            assert!(a.triangles().eq(b.triangles()));
        }
        let _ = capture(device, queue, &mut gpu, &shuffled)?;
    }
    differences.verify("fresh-pool-slot-reuse")
}

fn retile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    region: &Region,
    baseline: &[CollisionGrid],
) -> Result<()> {
    let mut gpu = TerrainGpu::new(device, region, SMALL, 4)?;
    let mut differences = Differences::default();
    let side = usize::from(SMALL);
    for grid in baseline {
        let requests = (0..4)
            .map(|slot| TerrainRequest {
                origin_ticks: [
                    grid.request.origin_ticks[0] + slot % 2 * (u32::from(SMALL) - 1) * SPACING,
                    grid.request.origin_ticks[1] + slot / 2 * (u32::from(SMALL) - 1) * SPACING,
                ],
                spacing_ticks: SPACING,
                slot,
            })
            .collect::<Vec<_>>();
        for (slot, patch) in capture(device, queue, &mut gpu, &requests)?
            .iter()
            .enumerate()
        {
            for (i, sample) in patch.samples.iter().enumerate() {
                let x = slot % 2 * (side - 1) + i % side;
                let y = slot / 2 * (side - 1) + i / side;
                differences.add(&grid.samples[y * usize::from(SIDE) + x], sample);
            }
        }
    }
    differences.verify("retiled-overlapping-65")
}

fn explicit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    region: &Region,
    baseline: &[CollisionGrid],
) -> Result<()> {
    let mut gpu = TerrainGpu::new(device, region, SIDE, 1)?;
    let mut differences = Differences::default();
    let side = u32::from(SIDE);
    for grid in baseline {
        let addresses = (0..side * side)
            .rev()
            .map(|i| {
                [
                    grid.request.origin_ticks[0] + i % side * SPACING,
                    grid.request.origin_ticks[1] + i / side * SPACING,
                ]
            })
            .collect::<Vec<_>>();
        let mut commands = encoder(device);
        gpu.encode_points(
            device,
            &mut commands,
            &[TerrainPointRequest {
                slot: 0,
                addresses: &addresses,
            }],
        )?;
        let snapshot = gpu.snapshot_samples(device, &mut commands, 0)?;
        queue.submit([commands.finish()]);
        let mut pending = snapshot.begin();
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let mut samples = pending
            .try_take()?
            .ok_or_else(|| io::Error::other("explicit readback unfinished"))?;
        samples.reverse();
        differences.compare(&grid.samples, &samples);
    }
    differences.verify("reversed-explicit-addresses")
}

fn latency(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    region: &Region,
    request: TerrainRequest,
) -> Result<()> {
    for side in [9_u16, 17, 33, 65, 129] {
        let started = Instant::now();
        let mut gpu = TerrainGpu::new(device, region, side, 1)?;
        let init_ms = started.elapsed().as_secs_f64() * 1000.0;
        let request = TerrainRequest { slot: 0, ..request };
        let _ = capture(device, queue, &mut gpu, &[request])?;
        let mut times = Vec::new();
        for _ in 0..32 {
            let start = Instant::now();
            let _ = capture(device, queue, &mut gpu, &[request])?;
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "TERRAIN_REGEN_LATENCY side={side} samples={} snapshot_bytes={} pool_init_ms={init_ms:.3} median_ms={:.3} p95_ms={:.3} max_ms={:.3}",
            u32::from(side).pow(2),
            u32::from(side).pow(2) * 16,
            times[16],
            times[30],
            times[31]
        );
    }
    Ok(())
}

pub(super) fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    region: &Region,
    output: Option<&Path>,
) -> Result<()> {
    let original = requests(region)?;
    let baseline = {
        let mut gpu = TerrainGpu::new(device, region, SIDE, LOCATIONS)?;
        capture(device, queue, &mut gpu, &original)?
    };
    let wet = baseline
        .iter()
        .flat_map(|grid| &grid.samples)
        .filter(|s| s.water_kind != 0)
        .count();
    let (min, max) = baseline
        .iter()
        .flat_map(|grid| &grid.samples)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s.bed_m), hi.max(s.bed_m))
        });
    println!(
        "TERRAIN_REGEN_BASELINE grids={LOCATIONS} side={SIDE} samples={} wet={wet} min_m={min:.3} max_m={max:.3}",
        LOCATIONS * u32::from(SIDE).pow(2)
    );
    if let Some(path) = output {
        use std::io::Write;
        let mut file = io::BufWriter::new(std::fs::File::create(path)?);
        for grid in &baseline {
            file.write_all(bytemuck::cast_slice(&grid.samples))?;
        }
        file.flush()?;
    }
    fresh_pool(device, queue, region, &baseline, &original)?;
    retile(device, queue, region, &baseline)?;
    explicit(device, queue, region, &baseline)?;
    latency(device, queue, region, original[4])?;
    Ok(())
}
