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
        (f64::from(u32::try_from(key >> 32).unwrap_or_default()) / f64::from(u32::MAX))
            .mul_add(2.0, -1.0)
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
    #[must_use = "Use the interpolated channel section"]
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
    #[must_use = "Use the computed visible surface height"]
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

struct Profile {
    radius: f64,
    stage: f64,
    valley: f64,
    inner: f64,
    outer: f64,
    sediment: f64,
    cross: f64,
}

impl ChannelField {
    /// Build an index of validated reaches for a finite square chart.
    ///
    /// # Errors
    /// Returns invalid-data errors for invalid chart extent or nonfinite or out-of-range reach parameters.
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
            let radius = reach.a.width.max(reach.b.width) * 0.6;
            let bank = 24.0_f64.mul_add(reach.valley, 0.48_f64.mul_add(radius, 10.0));
            let terrace = 100.0_f64.mul_add(
                reach.valley,
                radius.mul_add(2.0_f64.mul_add(reach.valley, 0.6), 25.0),
            );
            let margin = 1.8_f64.mul_add(bank, radius + terrace + 20.0);
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
    #[must_use = "Use the directed channel segments"]
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

    /// Broad-phase intersection against the conservative channel support bins.
    /// False means no channel can contribute to any sample in the rectangle.
    pub(super) fn intersects_bounds(&self, min: [f64; 2], max: [f64; 2]) -> bool {
        let [xmin, ymin] = min.map(|v| self.bin(v));
        let [xmax, ymax] = max.map(|v| self.bin(v));
        (ymin..=ymax).any(|y| (xmin..=xmax).any(|x| !self.buckets[y * self.side + x].is_empty()))
    }

    fn projection(reach: Reach, p: [f64; 2]) -> (f64, f64) {
        let d = difference(reach.b.xy, reach.a.xy);
        let delta = difference(p, reach.a.xy);
        let length_sq = d[0].mul_add(d[0], d[1] * d[1]).max(1e-9);
        let t = (delta[0].mul_add(d[0], delta[1] * d[1]) / length_sq).clamp(0.0, 1.0);
        let dx = t.mul_add(-d[0], delta[0]);
        let dy = t.mul_add(-d[1], delta[1]);
        (dx.mul_add(dx, dy * dy), t)
    }

    pub const MAX_RAISE: f64 = 6.0;

    fn profile(&self, point: [f64; 2]) -> Option<Profile> {
        let candidates = self.candidates(point);
        if candidates.is_empty() {
            return None;
        }
        let nearest = candidates
            .iter()
            .map(|&i| Self::projection(self.reaches[i], point).0)
            .fold(f64::INFINITY, f64::min)
            .sqrt();
        if nearest >= MAX_REACH {
            return None;
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
        let mut valley = 0.0_f64;
        let mut tangent = [0.0; 2];
        let mut sum = 0.0;
        for &index in candidates {
            let reach = self.reaches[index];
            let (distance_sq, fraction) = Self::projection(reach, point);
            if distance_sq > (nearest + 18.0).powi(2) {
                continue;
            }
            let distance = distance_sq.sqrt();
            let sample = reach.a.mix(reach.b, fraction);
            let blend = (sample.width * 0.16).clamp(4.0, 18.0);
            let weight = 1.0 - unit((distance - nearest) / blend);
            if weight == 0.0 {
                continue;
            }
            // Length weighting limits sensitivity to curve tessellation.
            let direction = difference(reach.b.xy, reach.a.xy);
            let length = norm(direction);
            let weighted_length = weight * length.max(0.01);
            sum += weighted_length;
            section.stage = weighted_length.mul_add(sample.stage, section.stage);
            section.xy[0] = weighted_length.mul_add(sample.xy[0], section.xy[0]);
            section.xy[1] = weighted_length.mul_add(sample.xy[1], section.xy[1]);
            section.bend = weighted_length.mul_add(sample.bend, section.bend);
            let length = length.max(1e-6);
            tangent[0] = (weighted_length / length).mul_add(direction[0], tangent[0]);
            tangent[1] = (weighted_length / length).mul_add(direction[1], tangent[1]);
            section.width = weighted_length.mul_add(sample.width, section.width);
            valley = weighted_length.mul_add(reach.valley, valley);
        }
        if sum == 0.0 {
            return None;
        }
        let radius = section.width / (2.0 * sum);
        let stage = section.stage / sum;
        let valley = valley / sum;
        let center = section.xy.map(|v| v / sum);
        let tangent = tangent.map(|v| v / norm(tangent).max(1e-9));
        let delta = difference(point, center);
        let signed = tangent[0].mul_add(delta[1], -tangent[1] * delta[0]) / radius.max(1.0);
        let side = signed.clamp(-1.0, 1.0);
        let bend = (section.bend / sum) * side * valley;
        let inner = bend.max(0.0);
        let outer = (-bend).max(0.0);
        let sediment = sediment_variation(point, 65.0, 0x53_45_44);
        Some(Profile {
            radius,
            stage,
            valley,
            inner,
            outer,
            sediment,
            cross: nearest,
        })
    }

    /// Sample the channel-conditioned solid ground and separate liquid stage.
    #[must_use = "Use the weathered bed and water stage"]
    pub fn at(&self, point: [f64; 2], base: f64) -> Weathered {
        let mut result = Weathered {
            bed: base,
            ..Weathered::default()
        };
        if base <= 0.0 {
            return result;
        }
        let Some(Profile {
            radius,
            stage,
            valley,
            inner,
            outer,
            sediment,
            cross,
        }) = self.profile(point)
        else {
            return result;
        };
        let radius = radius
            * 0.06f64.mul_add(
                sediment,
                0.13f64.mul_add(outer, 0.27f64.mul_add(-inner, 1.0)),
            );
        let bank_width = 24.0f64.mul_add(valley, 0.48f64.mul_add(radius, 10.0))
            * 0.3f64.mul_add(-outer, 0.8f64.mul_add(inner, 1.0));
        let terrace_width = 100.0f64.mul_add(
            valley,
            f64::mul_add(radius, 2.0f64.mul_add(valley, 0.6), 25.0),
        );
        let reach = radius + bank_width + terrace_width;
        if cross >= reach {
            return result;
        }
        // A continuous incised bed joins the free surface at its banks;
        // the outer profile grades into a floodplain or narrow rock terrace.
        let depth =
            0.068f64.mul_add(radius, 0.65) * 0.35f64.mul_add(-inner, 0.35f64.mul_add(outer, 1.0));
        let rise =
            0.08f64.mul_add(radius, 2.0) * 0.45f64.mul_add(-inner, 0.65f64.mul_add(outer, 1.0));
        let u = cross / radius;
        // Inner bends accumulate a shelving point bar that can emerge at
        // ordinary water level. Outside bends retain deeper scour.
        let bar_shelf = inner * unit((u - 0.38) / 0.47) * (1.0 - unit((u - 0.94) / 0.06));
        let sediment_fine = sediment_variation(point, 13.0, 0x52_49_50);
        let target = if cross <= radius {
            let exponent = 2.2_f64.mul_add(-inner, 2.0_f64.mul_add(outer, 3.5));
            let thalweg = depth.mul_add(u.powf(exponent) - 1.0, stage);
            let bar_deposit = depth * 0.78 * bar_shelf;
            let deposited = bar_deposit.mul_add(0.22_f64.mul_add(sediment, 1.0), thalweg);
            (0.32 * bar_shelf).mul_add(sediment_fine, deposited)
        } else {
            let lateral = (cross - radius) / bank_width;
            let bar = inner * (1.0 - unit((lateral - 0.3).abs() / 0.8));
            let bank_height = rise.mul_add(unit(lateral), stage);
            let terrace = 0.018_f64.mul_add((cross - radius - bank_width).max(0.0), bank_height);
            let with_bar = (0.9 * bar).mul_add(unit(lateral * 3.0), terrace);
            let sediment_coarse = 0.65_f64.mul_add(valley, 0.3) * sediment * unit(lateral);
            let with_coarse = sediment_coarse.mul_add(1.0 - unit(lateral / 4.0), with_bar);
            let fine = 0.5_f64.mul_add(outer, 0.35) * sediment_fine * unit(lateral * 3.0);
            fine.mul_add(1.0 - unit(lateral / 3.0), with_coarse)
        };
        let support = 1.0 - unit((cross - radius - bank_width) / terrace_width);
        let local = lerp(base, target, support);
        result.bed = local.clamp(
            42.0f64.mul_add(-support, base),
            f64::mul_add(Self::MAX_RAISE, support, base),
        );
        result.bank = (1.0 - unit((cross - radius).abs() / bank_width)) * support;
        result.gravel = (1.0
            - unit(f64::mul_add(bank_width, -0.45, cross - radius).abs() / (bank_width * 1.4)))
            * support;
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

    #[must_use = "Use the broad-phase channel query to select candidate regions"]
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
