//! Offline rainfall-weighted drainage-to-channel generation.
use super::{Grid, Rivers, lerp, mix, xy};
use crate::smooth;
pub(in super::super) use i_love_proland::terrain::{ChannelField, Weathered};
use i_love_proland::terrain::{Reach, Section};
use std::io;
const MIN_DISCHARGE: f64 = 14.0;
fn water_width(discharge: f64) -> f64 {
    (4.5 * discharge.sqrt()).clamp(3.0, 180.0)
}

fn unit(t: f64) -> f64 {
    smooth(t.clamp(0.0, 1.0))
}

fn norm(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn difference(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn ground_at(grid: &Grid, p: [f64; 2]) -> f64 {
    let q = p.map(|v| (v / grid.spacing).clamp(0.0, f64::from(grid.cells)));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "grid coordinates clamped to u16 extent"
    )]
    #[expect(clippy::cast_sign_loss, reason = "nonnegative grid coordinates")]
    let [x, y] = q.map(|v| v.floor() as u16);
    let nx = (x + 1).min(grid.cells);
    let ny = (y + 1).min(grid.cells);
    lerp(
        lerp(grid.at(x, y), grid.at(nx, y), q[0].fract()),
        lerp(grid.at(x, ny), grid.at(nx, ny), q[0].fract()),
        q[1].fract(),
    )
}

fn initial_sections(grid: &Grid, flow: &Rivers, inflow: &[usize]) -> Vec<Section> {
    // Filled heights are monotone along the receiver graph, and width
    // increases downstream. Subtracting a width-dependent incision
    // therefore yields a monotone water profile, including confluences.
    let section = |i: usize| {
        let width = water_width(flow.discharge[i]);
        let stage = if grid.heights[i] <= 0.0 {
            0.0
        } else {
            (flow.filled[i] - 0.043f64.mul_add(width, 1.4)).max(0.0)
        };
        Section {
            xy: xy(grid, i).map(|v| v * grid.spacing),
            stage,
            width,
            bend: 0.0,
        }
    };
    let mut nodes: Vec<_> = (0..grid.heights.len()).map(section).collect();
    // Remove D8 staircase turns over several nodes on low-gradient
    // reaches. Every edge uses the same smoothed endpoint, including
    // tributary junctions. Do not displace mouths or steep cascades.
    for _ in 0..4 {
        let old = nodes.clone();
        for (i, node) in nodes.iter_mut().enumerate() {
            let j = flow.receiver[i];
            let k = inflow[i];
            if j == usize::MAX
                || k == usize::MAX
                || flow.discharge[i] < MIN_DISCHARGE
                || grid.heights[i] <= 0.0
                || grid.heights[j] <= 0.0
            {
                continue;
            }
            let a = old[k];
            let b = old[j];
            let span = norm(difference(a.xy, b.xy)).max(grid.spacing);
            let slope = (a.stage - b.stage).abs() / span;
            let flat = 1.0 - unit((slope - 0.012) / 0.11);
            let middle = mix(a.xy, b.xy, 0.5);
            let target = mix(old[i].xy, middle, 0.65 * flat);
            let original = section(i).xy;
            let delta = difference(target, original);
            let limit = grid.spacing * 0.9;
            let scale = (limit / norm(delta).max(1e-9)).min(1.0);
            node.xy = [
                delta[0].mul_add(scale, original[0]),
                delta[1].mul_add(scale, original[1]),
            ];
        }
    }
    nodes
}

fn meander_sections(grid: &Grid, flow: &Rivers, inflow: &[usize], nodes: &mut [Section]) {
    // Integrate meander phase downstream-to-upstream so the
    // wavelength scales with discharge without phase jumps as width
    // changes. Branches share a phase at their confluence.
    let mut phase = vec![0.0_f64; nodes.len()];
    for &i in &flow.order {
        let j = flow.receiver[i];
        if j == usize::MAX {
            continue;
        }
        let width = nodes[i].width.midpoint(nodes[j].width);
        let wavelength = (11.0 * width).clamp(400.0, 1600.0);
        phase[i] = phase[j]
            + norm(difference(nodes[i].xy, nodes[j].xy)) * std::f64::consts::TAU / wavelength;
    }
    let coarse = nodes.to_vec();
    for (index, node) in nodes.iter_mut().enumerate() {
        let downstream = flow.receiver[index];
        let upstream = inflow[index];
        if downstream == usize::MAX
            || upstream == usize::MAX
            || flow.discharge[index] < MIN_DISCHARGE
            || grid.heights[index] <= 0.0
            || grid.heights[downstream] <= 0.0
        {
            continue;
        }
        let axis = difference(coarse[downstream].xy, coarse[upstream].xy);
        let span = norm(axis).max(1.0);
        let grade = (coarse[upstream].stage - coarse[downstream].stage).abs() / span;
        let (x, y) = grid.coordinates(index);
        let [gx, gy] = super::super::gradient(grid, x, y);
        let slope = gx.hypot(gy);
        let flat = (1.0 - unit((grade - 0.006) / 0.055)) * (1.0 - unit((slope - 0.065) / 0.22));
        let wavelength_phase = phase[index];
        let amplitude = (node.width * 0.95).min(100.0) * flat;
        let offset = amplitude
            * 0.24f64.mul_add(
                f64::mul_add(wavelength_phase, 0.47, 1.2).sin(),
                0.76 * wavelength_phase.sin(),
            );
        let target = [
            node.xy[0] - offset * axis[1] / span,
            node.xy[1] + offset * axis[0] / span,
        ];
        // Reject large excursions into opposing hillsides. This is
        // deliberately local and does not change the regional graph.
        let relief = ground_at(grid, target) - node.stage;
        let accept = unit((relief + 5.0) / 6.0) * (1.0 - unit((relief - 18.0) / 22.0));
        node.xy = mix(node.xy, target, accept);
        node.width *= (0.12 * flat).mul_add(f64::mul_add(wavelength_phase, 1.7, 0.5).sin(), 1.0);
        node.width = node.width.clamp(3.0, 180.0);
    }
    let moved = nodes.to_vec();
    for (i, node) in nodes.iter_mut().enumerate() {
        let j = flow.receiver[i];
        let k = inflow[i];
        if j == usize::MAX || k == usize::MAX {
            continue;
        }
        let a = difference(moved[i].xy, moved[k].xy);
        let b = difference(moved[j].xy, moved[i].xy);
        let bend = a[1].mul_add(-b[0], a[0] * b[1]) / (norm(a) * norm(b)).max(1.0);
        node.bend = (bend * 2.5).clamp(-1.0, 1.0);
    }
}

pub(in super::super) fn build(grid: &Grid, flow: &Rivers) -> io::Result<ChannelField> {
    let mut inflow = vec![usize::MAX; flow.receiver.len()];
    for (i, &j) in flow.receiver.iter().enumerate() {
        if j == usize::MAX || flow.discharge[i] < MIN_DISCHARGE || grid.heights[i] <= 0.0 {
            continue;
        }
        if inflow[j] == usize::MAX || flow.discharge[i] > flow.discharge[inflow[j]] {
            inflow[j] = i;
        }
    }
    let mut nodes = initial_sections(grid, flow, &inflow);
    meander_sections(grid, flow, &inflow, &mut nodes);
    let mut reaches = Vec::new();
    for (i, &j) in flow.receiver.iter().enumerate() {
        if j == usize::MAX || flow.discharge[i] < MIN_DISCHARGE || grid.heights[i] <= 0.0 {
            continue;
        }
        let a = nodes[i];
        let b = nodes[j];
        let start = if inflow[i] == usize::MAX {
            a
        } else {
            nodes[inflow[i]].mix(a, 0.5)
        };
        let end = a.mix(b, 0.5);
        let grade =
            (start.stage - end.stage).max(0.0) / norm(difference(start.xy, end.xy)).max(1.0);
        let valley = 1.0 - unit((grade - 0.018) / 0.16);
        let mut previous = start;
        for k in 1..=4 {
            let t = f64::from(k) * 0.25;
            let next = start.mix(a, t).mix(a.mix(end, t), t);
            reaches.push(Reach {
                a: previous,
                b: next,
                valley,
            });
            previous = next;
        }
        if inflow[j] != i || flow.receiver[j] == usize::MAX || grid.heights[j] <= 0.0 {
            reaches.push(Reach { a: end, b, valley });
        }
    }
    ChannelField::new(f64::from(grid.cells) * grid.spacing, reaches)
}

pub(in super::super) fn viewpoint(
    channels: &ChannelField,
    near: [f64; 2],
    extent: f64,
) -> Option<([f64; 2], [f64; 2])> {
    channels
        .reaches()
        .iter()
        .filter(|r| {
            r.valley > 0.7
                && r.a.width > 25.0
                && r.a.stage > 20.0
                && r.a.xy.iter().all(|&v| v > 1500.0 && v < extent - 1500.0)
        })
        .min_by(|a, b| norm(difference(a.a.xy, near)).total_cmp(&norm(difference(b.a.xy, near))))
        .map(|r| (r.a.xy, difference(r.b.xy, r.a.xy)))
}
