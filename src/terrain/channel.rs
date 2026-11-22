//! Immutable fluvial cross sections; evaluated independently of render LOD.
use crate::random::mix64 as hash;
use std::io;

const BUCKET: f64 = 250.0;
const MAX_REACH: f64 = 670.0;
fn smooth(t: f64) -> f64 {
    t * t * 2.0_f64.mul_add(-t, 3.0)
}
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    (b - a).mul_add(t, a)
}
fn mix(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    std::array::from_fn(|i| lerp(a[i], b[i], t))
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
pub struct Section {
    pub xy: [f64; 2],
    pub stage: f64,
    pub width: f64,
    pub bend: f64,
}

impl Section {
    pub fn mix(self, other: Self, t: f64) -> Self {
        Self {
            xy: mix(self.xy, other.xy, t),
            stage: lerp(self.stage, other.stage, t),
            width: lerp(self.width, other.width, t),
            bend: lerp(self.bend, other.bend, t),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Reach {
    pub a: Section,
    pub b: Section,
    pub valley: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Weathered {
    /// Solid ground, including incision and local sediment deposits.
    pub bed: f64,
    /// Actual free surface, if present; never a color-only river overlay.
    pub water: Option<f64>,
    pub bank: f64,
    pub gravel: f64,
    pub floodplain: f64,
    pub point_bar: f64,
    pub scour: f64,
    pub deposit: f64,
    pub depth: f64,
}

impl Weathered {
    pub fn elevation(self) -> f64 {
        self.water.map_or(self.bed, |w| self.bed.max(w))
    }
}

#[derive(Debug)]
pub struct ChannelField {
    reaches: Vec<Reach>,
    buckets: Vec<Vec<usize>>,
    side: usize,
}

impl ChannelField {
    /// Build an index of validated reaches for a finite square chart.
    pub fn new(extent: f64, reaches: Vec<Reach>) -> io::Result<Self> {
        if !(1.0..=250_000.0).contains(&extent)
            || reaches.len() > 1_000_000
            || reaches.iter().any(|r| {
                !((0.0..=1.0).contains(&r.valley)
                    && [r.a, r.b].into_iter().all(|s| {
                        s.xy.into_iter()
                            .all(|v| (-1000.0..=extent + 1000.0).contains(&v))
                            && (0.0..=32768.0).contains(&s.stage)
                            && (3.0..=180.0).contains(&s.width)
                            && (-1.0..=1.0).contains(&s.bend)
                    }))
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid channel field",
            ));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded regional grid bin count"
        )]
        #[expect(clippy::cast_sign_loss, reason = "positive regional grid extent")]
        let side = (extent / BUCKET).ceil() as usize + 1;
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
        Ok(result)
    }

    /// Return the original directed segments, in deterministic graph order.
    pub fn reaches(&self) -> &[Reach] {
        &self.reaches
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

    pub const MAX_RAISE: f64 = 6.0;

    pub fn at(&self, p: [f64; 2], base: f64) -> Weathered {
        let candidates = self.candidates(p);
        if candidates.is_empty() || base <= 0.0 {
            return Weathered {
                bed: base,
                ..Weathered::default()
            };
        }
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

    pub fn has_channel(&self, p: [f64; 2]) -> bool {
        !self.candidates(p).is_empty()
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
