//! Drainage-constrained local fluvial morphology. The coarse regional
//! drainage graph remains authoritative. Cross sections are evaluated in
//! metres at arbitrary positions, not tied to the 62.5 m terrain lattice.
use super::{Grid, Rivers, lerp, mix, xy};
use crate::{hash, smooth};

const MIN_DISCHARGE: f64 = 14.0;
const BUCKET: f64 = 250.0;
const MAX_REACH: f64 = 670.0;

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

fn sediment_variation(p: [f64; 2], scale: f64, seed: u64) -> f64 {
    let q = p.map(|v| v / scale);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "regional material coordinates fit i32"
    )]
    let [x, y] = q.map(|v| v.floor() as i32);
    let random = |dx: i32, dy: i32| {
        let key = hash(
            seed ^ u64::from(u32::from_ne_bytes((x + dx).to_ne_bytes())).rotate_left(19)
                ^ u64::from(u32::from_ne_bytes((y + dy).to_ne_bytes())).rotate_left(47),
        );
        f64::from(u32::try_from(key >> 32).unwrap_or_default()) / f64::from(u32::MAX) * 2.0 - 1.0
    };
    let [tx, ty] = q.map(|v| unit(v - v.floor()));
    lerp(
        lerp(random(0, 0), random(1, 0), tx),
        lerp(random(0, 1), random(1, 1), tx),
        ty,
    )
}

#[derive(Clone, Copy, Debug)]
struct Section {
    xy: [f64; 2],
    stage: f64,
    width: f64,
    bend: f64,
}

impl Section {
    fn mix(self, other: Self, t: f64) -> Self {
        Self {
            xy: mix(self.xy, other.xy, t),
            stage: lerp(self.stage, other.stage, t),
            width: lerp(self.width, other.width, t),
            bend: lerp(self.bend, other.bend, t),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Reach {
    a: Section,
    b: Section,
    valley: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in super::super) struct Weathered {
    /// Solid ground, including incision and local sediment deposits.
    pub(in super::super) bed: f64,
    /// Actual free surface, if present; never a color-only river overlay.
    pub(in super::super) water: Option<f64>,
    pub(in super::super) bank: f64,
    pub(in super::super) gravel: f64,
    pub(in super::super) floodplain: f64,
    pub(in super::super) point_bar: f64,
    pub(in super::super) scour: f64,
    pub(in super::super) deposit: f64,
    pub(in super::super) depth: f64,
}

impl Weathered {
    pub(in super::super) fn elevation(self) -> f64 {
        self.water.map_or(self.bed, |w| self.bed.max(w))
    }
}

pub(in super::super) struct ChannelField {
    reaches: Vec<Reach>,
    buckets: Vec<Vec<usize>>,
    side: usize,
}

impl ChannelField {
    pub(in super::super) fn new(grid: &Grid, flow: &Rivers) -> Self {
        let mut inflow = vec![usize::MAX; flow.receiver.len()];
        for (i, &j) in flow.receiver.iter().enumerate() {
            if j == usize::MAX || flow.discharge[i] < MIN_DISCHARGE || grid.heights[i] <= 0.0 {
                continue;
            }
            if inflow[j] == usize::MAX || flow.discharge[i] > flow.discharge[inflow[j]] {
                inflow[j] = i;
            }
        }
        // Filled heights are monotone along the receiver graph, and width
        // increases downstream. Subtracting a width-dependent incision
        // therefore yields a monotone water profile, including confluences.
        let section = |i: usize| {
            let width = water_width(flow.discharge[i]);
            let stage = if grid.heights[i] <= 0.0 {
                0.0
            } else {
                (flow.filled[i] - (1.4 + 0.043 * width)).max(0.0)
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
                    original[0] + delta[0] * scale,
                    original[1] + delta[1] * scale,
                ];
            }
        }
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
        let coarse = nodes.clone();
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
            let d = difference(coarse[j].xy, coarse[k].xy);
            let span = norm(d).max(1.0);
            let grade = (coarse[k].stage - coarse[j].stage).abs() / span;
            let (x, y) = grid.coordinates(i);
            let [gx, gy] = super::super::gradient(grid, x, y);
            let slope = gx.hypot(gy);
            let flat = (1.0 - unit((grade - 0.006) / 0.055)) * (1.0 - unit((slope - 0.065) / 0.22));
            let wavelength_phase = phase[i];
            let amplitude = (node.width * 0.95).min(100.0) * flat;
            let offset = amplitude
                * (0.76 * wavelength_phase.sin() + 0.24 * (wavelength_phase * 0.47 + 1.2).sin());
            let target = [
                node.xy[0] - offset * d[1] / span,
                node.xy[1] + offset * d[0] / span,
            ];
            // Reject large excursions into opposing hillsides. This is
            // deliberately local and does not change the regional graph.
            let relief = ground_at(grid, target) - node.stage;
            let accept = unit((relief + 5.0) / 6.0) * (1.0 - unit((relief - 18.0) / 22.0));
            node.xy = mix(node.xy, target, accept);
            node.width *= 1.0 + 0.12 * flat * (wavelength_phase * 1.7 + 0.5).sin();
            node.width = node.width.clamp(3.0, 180.0);
        }
        let moved = nodes.clone();
        for (i, node) in nodes.iter_mut().enumerate() {
            let j = flow.receiver[i];
            let k = inflow[i];
            if j == usize::MAX || k == usize::MAX {
                continue;
            }
            let a = difference(moved[i].xy, moved[k].xy);
            let b = difference(moved[j].xy, moved[i].xy);
            let bend = (a[0] * b[1] - a[1] * b[0]) / (norm(a) * norm(b)).max(1.0);
            node.bend = (bend * 2.5).clamp(-1.0, 1.0);
        }
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded regional grid bin count"
        )]
        #[expect(clippy::cast_sign_loss, reason = "positive regional grid extent")]
        let side = (f64::from(grid.cells) * grid.spacing / BUCKET).ceil() as usize + 1;
        let mut result = Self {
            reaches,
            buckets: vec![Vec::new(); side * side],
            side,
        };
        for (i, reach) in result.reaches.iter().enumerate() {
            let radius = reach.a.width.max(reach.b.width) * 0.5;
            let margin = radius * 1.2
                + 1.8 * (10.0 + 0.48 * radius * 1.2 + 24.0 * reach.valley)
                + 25.0
                + radius * 1.2 * (0.6 + 2.0 * reach.valley)
                + 100.0 * reach.valley
                + 20.0;
            let [xmin, ymin] =
                std::array::from_fn(|c| result.bin(reach.a.xy[c].min(reach.b.xy[c]) - margin));
            let [xmax, ymax] =
                std::array::from_fn(|c| result.bin(reach.a.xy[c].max(reach.b.xy[c]) + margin));
            for y in ymin..=ymax {
                for x in xmin..=xmax {
                    result.buckets[y * side + x].push(i);
                }
            }
        }
        result
    }

    fn bin(&self, p: f64) -> usize {
        #[expect(clippy::cast_possible_truncation, reason = "bounded regional bin")]
        #[expect(clippy::cast_sign_loss, reason = "nonnegative bin")]
        let i = (p / BUCKET).max(0.0).floor() as usize;
        i.min(self.side - 1)
    }

    fn candidates(&self, p: [f64; 2]) -> &[usize] {
        let [x, y] = p.map(|v| self.bin(v));
        &self.buckets[y * self.side + x]
    }

    fn projection(reach: Reach, p: [f64; 2]) -> (f64, f64) {
        let d = difference(reach.b.xy, reach.a.xy);
        let delta = difference(p, reach.a.xy);
        let length_sq = d[0].mul_add(d[0], d[1] * d[1]).max(1e-9);
        let t = (delta[0].mul_add(d[0], delta[1] * d[1]) / length_sq).clamp(0.0, 1.0);
        let dx = delta[0] - t * d[0];
        let dy = delta[1] - t * d[1];
        (dx.mul_add(dx, dy * dy), t)
    }

    pub(in super::super) const MAX_RAISE: f64 = 6.0;

    pub(in super::super) fn at(&self, p: [f64; 2], base: f64) -> Weathered {
        let candidates = self.candidates(p);
        let nearest = candidates
            .iter()
            .map(|&i| Self::projection(self.reaches[i], p).0)
            .fold(f64::INFINITY, f64::min)
            .sqrt();
        let mut result = Weathered {
            bed: base,
            ..Weathered::default()
        };
        if nearest >= MAX_REACH || base <= 0.0 {
            return result;
        }
        // Smoothly interpolate competing reaches within a short distance of
        // the closest centerline point. At segment joins this avoids a
        // discontinuous winner selection or a series of independent pits.
        let mut section = Section {
            xy: [0.0; 2],
            stage: 0.0,
            width: 0.0,
            bend: 0.0,
        };
        let mut valley = 0.0;
        let mut tangent = [0.0; 2];
        let mut sum = 0.0;
        for &i in candidates {
            let r = self.reaches[i];
            let (d2, t) = Self::projection(r, p);
            if d2 > (nearest + 18.0).powi(2) {
                continue;
            }
            let distance = d2.sqrt();
            let s = r.a.mix(r.b, t);
            let blend = (s.width * 0.16).clamp(4.0, 18.0);
            let weight = 1.0 - unit((distance - nearest) / blend);
            if weight == 0.0 {
                continue;
            }
            // Length weighting limits sensitivity to curve tessellation.
            let w = weight * norm(difference(r.a.xy, r.b.xy)).max(0.01);
            sum += w;
            section.stage += w * s.stage;
            section.xy[0] += w * s.xy[0];
            section.xy[1] += w * s.xy[1];
            section.bend += w * s.bend;
            let direction = difference(r.b.xy, r.a.xy);
            let length = norm(direction).max(1e-6);
            tangent[0] += w * direction[0] / length;
            tangent[1] += w * direction[1] / length;
            section.width += w * s.width;
            valley += w * r.valley;
        }
        if sum == 0.0 {
            return result;
        }
        let radius = section.width / (2.0 * sum);
        let stage = section.stage / sum;
        let valley = valley / sum;
        let center = section.xy.map(|v| v / sum);
        let tangent = tangent.map(|v| v / norm(tangent).max(1e-9));
        let delta = difference(p, center);
        let signed = (tangent[0] * delta[1] - tangent[1] * delta[0]) / radius.max(1.0);
        let side = signed.clamp(-1.0, 1.0);
        let bend = (section.bend / sum) * side * valley;
        let inner = bend.max(0.0);
        let outer = (-bend).max(0.0);
        let sediment = sediment_variation(p, 65.0, 0x534544);
        let radius = radius * (1.0 - 0.27 * inner + 0.13 * outer + 0.06 * sediment);
        let cross = nearest;
        let bank_width = (10.0 + 0.48 * radius + 24.0 * valley) * (1.0 + 0.8 * inner - 0.3 * outer);
        let terrace_width = 25.0 + radius * (0.6 + 2.0 * valley) + 100.0 * valley;
        let reach = radius + bank_width + terrace_width;
        if cross >= reach {
            return result;
        }
        // A continuous incised bed joins the free surface at its banks;
        // the outer profile grades into a floodplain or narrow rock terrace.
        let depth = (0.65 + 0.068 * radius) * (1.0 + 0.35 * outer - 0.35 * inner);
        let rise = (2.0 + 0.08 * radius) * (1.0 + 0.65 * outer - 0.45 * inner);
        let u = cross / radius;
        // Inner bends accumulate a shelving point bar that can emerge at
        // ordinary water level. Outside bends retain deeper scour.
        let bar_shelf = inner * unit((u - 0.38) / 0.47) * (1.0 - unit((u - 0.94) / 0.06));
        let sediment_fine = sediment_variation(p, 13.0, 0x524950);
        let target = if cross <= radius {
            stage - depth * (1.0 - u.powf(3.5 + 2.0 * outer - 2.2 * inner))
                + depth * 0.78 * bar_shelf * (1.0 + 0.22 * sediment)
                + 0.32 * bar_shelf * sediment_fine
        } else {
            let lateral = (cross - radius) / bank_width;
            let bar = inner * (1.0 - unit((lateral - 0.3).abs() / 0.8));
            stage
                + rise * unit(lateral)
                + 0.018 * (cross - radius - bank_width).max(0.0)
                + 0.9 * bar * unit(lateral * 3.0)
                + (0.3 + 0.65 * valley) * sediment * unit(lateral) * (1.0 - unit(lateral / 4.0))
                + (0.35 + 0.5 * outer)
                    * sediment_fine
                    * unit(lateral * 3.0)
                    * (1.0 - unit(lateral / 3.0))
        };
        let support = 1.0 - unit((cross - radius - bank_width) / terrace_width);
        let local = lerp(base, target, support);
        result.bed = local.clamp(base - 42.0 * support, base + Self::MAX_RAISE * support);
        result.bank = (1.0 - unit((cross - radius).abs() / bank_width)) * support;
        result.gravel =
            (1.0 - unit((cross - radius - bank_width * 0.45).abs() / (bank_width * 1.4))) * support;
        result.point_bar = (inner * result.gravel).max(bar_shelf);
        result.gravel = result.gravel.max(bar_shelf);
        result.scour = outer * result.bank;
        result.deposit = (result.bed - base).max(0.0);

        result.floodplain = support * valley;
        if cross <= radius && stage > result.bed + 0.02 && stage <= base + Self::MAX_RAISE {
            result.water = Some(stage);
            result.depth = stage - result.bed;
        }
        result
    }

    pub(in super::super) fn has_channel(&self, p: [f64; 2]) -> bool {
        !self.candidates(p).is_empty()
    }

    // A low-oblique diagnostic view in a substantial low-gradient reach
    // near the original flight camera. Selection is deterministic.
    pub(in super::super) fn viewpoint(
        &self,
        near: [f64; 2],
        extent: f64,
    ) -> Option<([f64; 2], [f64; 2])> {
        self.reaches
            .iter()
            .filter(|r| {
                r.valley > 0.7
                    && r.a.width > 25.0
                    && r.a.stage > 20.0
                    && r.a.xy.iter().all(|&v| v > 1500.0 && v < extent - 1500.0)
            })
            .min_by(|a, b| {
                norm(difference(a.a.xy, near)).total_cmp(&norm(difference(b.a.xy, near)))
            })
            .map(|r| (r.a.xy, difference(r.b.xy, r.a.xy)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_section_is_incised_and_bounded() {
        let reach = Reach {
            a: Section {
                xy: [100.0, 250.0],
                stage: 100.0,
                width: 40.0,
                bend: 0.0,
            },
            b: Section {
                xy: [400.0, 250.0],
                stage: 100.0,
                width: 40.0,
                bend: 0.0,
            },
            valley: 1.0,
        };
        let field = ChannelField {
            reaches: vec![reach],
            buckets: vec![vec![0]; 9],
            side: 3,
        };
        let center = field.at([250.0, 250.0], 104.0);
        assert!(center.bed < 100.0);
        assert_eq!(center.water, Some(100.0));
        assert_eq!(center.elevation(), 100.0);
        let bank = field.at([250.0, 290.0], 104.0);
        assert!(bank.water.is_none() && bank.bank > 0.0 && bank.bed > center.bed);
        assert_eq!(field.at([250.0, 800.0], 104.0).bed, 104.0);
        let low = field.at([250.0, 250.0], 0.0);
        assert_eq!(low.bed, 0.0);
        assert!(field.at([250.0, 250.0], 160.0).water.is_none());
    }
}
