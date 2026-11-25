//! GPU timestamps with balanced rotating execution order.
use super::{MeshContact, MeshQueries, Result};
use std::{io, sync::mpsc};

const ROUNDS: usize = 8;
const REPEATS: u32 = 8;

pub fn compare(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entries: &[(String, MeshQueries, Vec<MeshContact>)],
) -> Result<()> {
    let count = u32::try_from(2 * ROUNDS * entries.len())?;
    let set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("mesh packed timing"),
        ty: wgpu::QueryType::Timestamp,
        count,
    });
    let size = u64::from(count) * 8;
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mesh packed timing resolve"),
        size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mesh packed timing readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for round in 0..ROUNDS {
        for offset in 0..entries.len() {
            let index = (round + offset) % entries.len();
            let query = u32::try_from((round * entries.len() + index) * 2)?;
            encoder.write_timestamp(&set, query);
            for _ in 0..REPEATS {
                entries[index].1.encode(&mut encoder);
            }
            encoder.write_timestamp(&set, query + 1);
        }
    }
    encoder.resolve_query_set(&set, 0..count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let (send, receive) = mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receive.recv()??;
    let times = {
        let mapped = readback.slice(..).get_mapped_range()?;
        mapped
            .as_chunks::<8>()
            .0
            .iter()
            .copied()
            .map(u64::from_le_bytes)
            .collect::<Vec<_>>()
    };
    readback.unmap();
    summarize(entries, &times, queue.get_timestamp_period())
}

fn summarize(
    entries: &[(String, MeshQueries, Vec<MeshContact>)],
    times: &[u64],
    period: f32,
) -> Result<()> {
    for (i, (label, _, results)) in entries.iter().enumerate() {
        let mut samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let start = (round * entries.len() + i) * 2;
            let elapsed = times[start + 1]
                .checked_sub(times[start])
                .ok_or_else(|| io::Error::other("invalid GPU timestamp order"))?;
            // Each measured dispatch is far shorter than the exact f64 integer range.
            let ticks = u32::try_from(elapsed)?;
            samples.push(f64::from(ticks) * f64::from(period) / f64::from(REPEATS) * 1e-6);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "MESH_GPU_BENCH mode={label} pairs={} median_ms={:.6}",
            results.len(),
            samples[ROUNDS / 2]
        );
    }
    Ok(())
}
