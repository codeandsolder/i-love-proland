//! CPU broad phase only: envelope the GPU field without evaluating noise,
//! river profiles or fine meshes. Heights are relative to the body radius.
use super::{ChannelField, QUANTUM, Region};
use crate::terrain::MAX_RELIEF;

/// Coarse bilinear height range and the possible GPU refinement displacement
/// over a closed chart rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightEnvelope {
    pub coarse_min_m: f64,
    pub coarse_max_m: f64,
    pub downward_m: f64,
    pub upward_m: f64,
}

impl HeightEnvelope {
    #[must_use = "Use the lower bound for terrain contact tests"]
    pub fn min_m(self) -> f64 {
        self.coarse_min_m - self.downward_m
    }

    #[must_use = "Use the upper bound for terrain clearance tests"]
    pub fn max_m(self) -> f64 {
        self.coarse_max_m + self.upward_m
    }
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * 2.0_f64.mul_add(-t, 3.0)
}

// Also covers the 1/16 m address rounding and f32 coordinate arithmetic
// at the maximum supported chart extent. The expanded rectangle accounts
// for positional error independently from the vertical rounding margin.
const ADDRESS_MARGIN_M: f64 = 0.125;
const HEIGHT_MARGIN_M: f64 = 0.05;

impl Region {
    #[must_use = "Use the unrefined regional height"]
    pub fn coarse_height(&self, point: [f64; 2]) -> Option<f64> {
        self.interpolation(point)
            .map(|i| i.sample(|c| f64::from(c.height) * QUANTUM))
    }

    fn cell_index(&self, coordinate: f64) -> u16 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "validated chart coordinate"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "validated nonnegative chart coordinate"
        )]
        let index = (coordinate / self.chart.spacing_m).floor() as u16;
        index.min(self.chart.cells - 1)
    }

    fn height_node(&self, x: u16, y: u16) -> f64 {
        let side = usize::from(self.chart.cells) + 1;
        f64::from(self.cells[usize::from(y) * side + usize::from(x)].height) * QUANTUM
    }

    fn coarse_range(&self, min: [f64; 2], max: [f64; 2]) -> Option<[f64; 2]> {
        let lo = min.map(|v| self.cell_index(v));
        let hi = max.map(|v| self.cell_index(v));
        let s = self.chart.spacing_m;
        let mut range = [f64::INFINITY, f64::NEG_INFINITY];
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                // A bilinear function reaches its extrema on the corners
                // of each intersected rectangle, including partial cells.
                for px in [
                    min[0].max(f64::from(x) * s),
                    max[0].min(f64::from(x + 1) * s),
                ] {
                    for py in [
                        min[1].max(f64::from(y) * s),
                        max[1].min(f64::from(y + 1) * s),
                    ] {
                        let h = self.coarse_height([px, py])?;
                        range = [range[0].min(h), range[1].max(h)];
                    }
                }
            }
        }
        Some(range)
    }

    fn slope_bound(&self, min: [f64; 2], max: [f64; 2]) -> f64 {
        let s = self.chart.spacing_m;
        // The GPU uses centred differences over one control-cell spacing.
        // Every such difference is bounded by the maximum edge gradient
        // in the expanded rectangle. Bilinear interpolation cannot exceed
        // its endpoint gradients.
        let lo = min.map(|v| self.cell_index((v - s * 0.5).max(0.0)));
        let hi = max.map(|v| self.cell_index((v + s * 0.5).min(self.chart.extent())));
        let mut dx = 0.0_f64;
        let mut dy = 0.0_f64;
        for y in lo[1]..=hi[1] + 1 {
            for x in lo[0]..=hi[0] + 1 {
                let h = self.height_node(x, y);
                if x <= hi[0] {
                    dx = dx.max((self.height_node(x + 1, y) - h).abs());
                }
                if y <= hi[1] {
                    dy = dy.max((self.height_node(x, y + 1) - h).abs());
                }
            }
        }
        // Account for f32 interpolation/subtraction even on a flat cell.
        ((dx + HEIGHT_MARGIN_M) / s).hypot((dy + HEIGHT_MARGIN_M) / s)
    }

    /// Bounds the fully refined GPU field over an arbitrary chart rectangle.
    /// Does not generate refined samples. A caller may reject a trajectory
    /// interval if its conservative lower altitude exceeds max_m().
    /// Returning None means the rectangle is invalid or outside this chart;
    /// it must not be treated as clear terrain.
    #[must_use = "Use the refined envelope to reject or subdivide encounter candidates"]
    pub fn refined_height_envelope(&self, min: [f64; 2], max: [f64; 2]) -> Option<HeightEnvelope> {
        let extent = self.chart.extent();
        if (0..2).any(|axis| {
            !(0.0..=extent).contains(&min[axis]) || !(min[axis]..=extent).contains(&max[axis])
        }) {
            return None;
        }
        let min = min.map(|v| (v - ADDRESS_MARGIN_M).max(0.0));
        let max = max.map(|v| (v + ADDRESS_MARGIN_M).min(extent));
        let [coarse_min_m, coarse_max_m] = self.coarse_range(min, max)?;
        let relief = MAX_RELIEF
            * smooth(self.slope_bound(min, max) / 0.5)
            * smooth((coarse_max_m + HEIGHT_MARGIN_M) / 30.0);
        let channels = self.channel_candidates(min, max) && coarse_max_m + relief > 0.0;
        Some(HeightEnvelope {
            coarse_min_m,
            coarse_max_m,
            downward_m: relief
                + if channels { ChannelField::MAX_CUT } else { 0.0 }
                + HEIGHT_MARGIN_M,
            upward_m: relief
                + if channels {
                    ChannelField::MAX_RAISE
                } else {
                    0.0
                }
                + HEIGHT_MARGIN_M,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{Reach, RegionCell, RegionChart, Section};
    use std::io;

    fn region(slope: f64, channel: bool) -> io::Result<Region> {
        let chart = RegionChart {
            radius_m: 600_000.0,
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            spacing_m: 100.0,
            cells: 20,
        };
        let cells = (0..=20)
            .flat_map(|_| {
                (0..=20).map(move |x| {
                    RegionCell::new(
                        f64::from(x).mul_add(slope * 100.0, 100.0),
                        None,
                        [0.3, 0.4, 0.2],
                    )
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let reaches = if channel {
            let section = |x| Section {
                xy: [x, 1000.0],
                stage: 100.0,
                width: 40.0,
                bend: 0.0,
            };
            vec![Reach {
                a: section(400.0),
                b: section(1600.0),
                valley: 1.0,
            }]
        } else {
            vec![]
        };
        Region::new(chart, cells, reaches)
    }

    #[test]
    fn flat_terrain_is_cheap_to_certify() -> io::Result<()> {
        let r = region(0.0, false)?;
        let e = r
            .refined_height_envelope([500.0, 500.0], [520.0, 520.0])
            .ok_or_else(super::super::invalid)?;
        assert!(e.max_m() < 100.06 && e.min_m() > 99.94);
        assert!(
            r.refined_height_envelope([-1.0, 0.0], [10.0, 10.0])
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn slope_and_channel_allow_full_refinement() -> io::Result<()> {
        let r = region(0.7, true)?;
        let e = r
            .refined_height_envelope([900.0, 1000.0], [920.0, 1000.0])
            .ok_or_else(super::super::invalid)?;
        assert!(e.upward_m >= MAX_RELIEF + ChannelField::MAX_RAISE);
        assert!(e.downward_m >= MAX_RELIEF + ChannelField::MAX_CUT);
        assert!(e.coarse_max_m - e.coarse_min_m < 15.0);
        let dry = r
            .refined_height_envelope([900.0, 0.0], [920.0, 0.0])
            .ok_or_else(super::super::invalid)?;
        assert!(dry.upward_m < 10.01 && dry.downward_m < 10.01);
        Ok(())
    }
}
