//! A versioned, finite body-fixed region. Compact regional geography plus
//! an analytic local channel field; immutable and safe to query concurrently.
use super::{ChannelField, Reach, Section, Weathered};
use std::{
    f64::consts::{PI, TAU},
    io::{self, Cursor, Read, Write},
};

const MAGIC: &[u8; 8] = b"PLTRN001";
const QUANTUM: f64 = 0.25;
const NO_LAKE: i16 = i16::MIN;

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid Proland terrain region")
}
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    (b - a).mul_add(t, a)
}
fn read_array<const N: usize>(input: &mut impl Read) -> io::Result<[u8; N]> {
    let mut bytes = [0; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn read_f64(input: &mut impl Read) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_array(input)?))
}

#[derive(Clone, Copy, Debug)]
pub struct RegionChart {
    pub radius_m: f64,
    /// Radians, in the runtime body's direction convention (Y north).
    pub latitude_rad: f64,
    pub longitude_rad: f64,
    pub spacing_m: f64,
    pub cells: u16,
}

impl RegionChart {
    #[must_use = "Use the chart extent for addressing"]
    pub fn extent(self) -> f64 {
        self.spacing_m * f64::from(self.cells)
    }
    fn valid(self) -> bool {
        let extent = self.extent();
        self.radius_m.is_finite()
            && self.radius_m > 0.0
            && (1..=2048).contains(&self.cells)
            && (1.0..=250_000.0).contains(&extent)
            && self.spacing_m.is_finite()
            && self.latitude_rad.is_finite()
            && self.longitude_rad.is_finite()
            && self.longitude_rad.abs() <= PI
            && self.latitude_rad.abs() + extent / (2.0 * self.radius_m) < PI * 0.5
            && extent / (2.0 * self.radius_m * self.latitude_rad.cos()) < PI
    }
    /// Same tangent-metric chart used by the regional terrain generator:
    /// X increases west, Y increases north. No render tile identity enters it.
    #[must_use = "Use the chart coordinates for terrain sampling"]
    pub fn project(self, direction: [f64; 3]) -> Option<[f64; 2]> {
        if !direction.iter().all(|v| v.is_finite())
            || direction.iter().all(|&component| component == 0.0)
        {
            return None;
        }
        let latitude = direction[1].atan2(direction[0].hypot(direction[2]));
        let half = self.extent() * 0.5;
        let y = (latitude - self.latitude_rad).mul_add(self.radius_m, half);
        if !(0.0..=self.extent()).contains(&y) {
            return None;
        }
        let longitude = direction[2].atan2(direction[0]);
        let delta = (longitude - self.longitude_rad + PI).rem_euclid(TAU) - PI;
        let x = (-delta).mul_add(self.radius_m * self.latitude_rad.cos(), half);
        (0.0..=self.extent()).contains(&x).then_some([x, y])
    }

    /// Inverse chart projection, convenient for tile/point validation.
    #[must_use = "Use the body-fixed direction"]
    pub fn direction(self, point: [f64; 2]) -> [f64; 3] {
        let half = self.extent() * 0.5;
        let lat = (point[1] - half) / self.radius_m + self.latitude_rad;
        let lon =
            self.longitude_rad - (point[0] - half) / (self.radius_m * self.latitude_rad.cos());
        let (sin_lat, cos_lat) = lat.sin_cos();
        let (sin_lon, cos_lon) = lon.sin_cos();
        [cos_lat * cos_lon, sin_lat, cos_lat * sin_lon]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RegionCell {
    height: i16,
    lake: i16,
    rgb: [u8; 3],
}

impl RegionCell {
    /// The 0.25m quantization covers -8191.75 to +8191.75 m. Reject
    /// out-of-range inputs rather than silently distorting the geography.
    ///
    /// # Errors
    /// Returns invalid data for heights outside the quantization range or invalid RGB.
    pub fn new(height: f64, lake: Option<f64>, rgb: [f64; 3]) -> io::Result<Self> {
        fn quantize(v: f64) -> io::Result<i16> {
            let q = (v / QUANTUM).round();
            if !(-32767.0..=32767.0).contains(&q) {
                return Err(invalid());
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "validated signed quarter-metre quantization"
            )]
            let out = q as i16;
            Ok(out)
        }
        if !rgb.into_iter().all(|v| (0.0..=1.0).contains(&v)) {
            return Err(invalid());
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "validated RGB8 quantization"
        )]
        #[expect(clippy::cast_sign_loss, reason = "validated nonnegative RGB8 channels")]
        let rgb = rgb.map(|v| (v * 255.0).round() as u8);
        Ok(Self {
            height: quantize(height)?,
            lake: lake.map(quantize).transpose()?.unwrap_or(NO_LAKE),
            rgb,
        })
    }
}

/// Distinguishes standing ocean/lake water from independently meshed rivers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaterKind {
    Standing,
    River,
}

#[derive(Clone, Copy, Debug)]
pub struct RegionSample {
    pub weather: Weathered,
    pub water_kind: Option<WaterKind>,
    pub rgb: [f64; 3],
    /// Blend with the host planet over the outer 4 km of the chart.
    pub weight: f64,
}

#[derive(Debug)]
pub struct Region {
    chart: RegionChart,
    cells: Vec<RegionCell>,
    channels: ChannelField,
    max_elevation: f64,
    min_elevation: f64,
}

struct CellInterpolation {
    points: [RegionCell; 4],
    fraction: [f64; 2],
}

impl CellInterpolation {
    fn sample(&self, f: impl Fn(RegionCell) -> f64) -> f64 {
        let [southwest, southeast, northwest, northeast] = self.points.map(f);
        let [tx, ty] = self.fraction;
        lerp(
            lerp(southwest, southeast, tx),
            lerp(northwest, northeast, tx),
            ty,
        )
    }
}

fn read_section(input: &mut impl Read) -> io::Result<Section> {
    Ok(Section {
        xy: [read_f64(input)?, read_f64(input)?],
        stage: read_f64(input)?,
        width: read_f64(input)?,
        bend: read_f64(input)?,
    })
}

impl Region {
    /// Construct a validated immutable regional surface.
    ///
    /// # Errors
    /// Returns invalid data for bad chart, cell count or channel geometry.
    pub fn new(
        chart: RegionChart,
        cells: Vec<RegionCell>,
        reaches: Vec<Reach>,
    ) -> io::Result<Self> {
        if !chart.valid() || cells.len() != (usize::from(chart.cells) + 1).pow(2) {
            return Err(invalid());
        }
        let max_elevation = cells
            .iter()
            .map(|c| f64::from(c.height) * QUANTUM)
            .fold(0.0, f64::max)
            + ChannelField::MAX_RAISE;
        let min_elevation = cells
            .iter()
            .map(|c| f64::from(c.height) * QUANTUM)
            .fold(0.0, f64::min)
            - ChannelField::MAX_CUT;
        let channels = ChannelField::new(chart.extent(), reaches)?;
        Ok(Self {
            chart,
            cells,
            channels,
            max_elevation,
            min_elevation,
        })
    }

    #[must_use = "Use the chart coordinate system"]
    pub const fn chart(&self) -> RegionChart {
        self.chart
    }
    #[must_use = "Use the conservative terrain bound"]
    pub const fn max_elevation(&self) -> f64 {
        self.max_elevation
    }
    /// Conservative lower bound for the solid regional field.
    #[must_use = "Use the conservative terrain bound"]
    pub const fn min_elevation(&self) -> f64 {
        self.min_elevation
    }
    #[must_use = "Use the directed channel segments"]
    pub fn reaches(&self) -> &[Reach] {
        self.channels.reaches()
    }

    /// Canonical body-fixed query, with no allocation or mutable cache.
    #[must_use = "Use the canonical terrain sample"]
    pub fn sample(&self, direction: [f64; 3]) -> Option<RegionSample> {
        let p = self.chart.project(direction)?;
        self.sample_chart(p)
    }
    #[must_use = "Use the canonical terrain sample"]
    pub fn sample_chart(&self, point: [f64; 2]) -> Option<RegionSample> {
        let interpolation = self.interpolation(point)?;
        let interpolate = |f: &dyn Fn(RegionCell) -> f64| interpolation.sample(f);
        let (weather, water_kind) = self.weather_at(point, &interpolation);
        let extent = self.chart.extent();
        let rgb = std::array::from_fn(|i| {
            let base = interpolate(&|c| f64::from(c.rgb[i]) / 255.0);
            let vegetated = lerp(base, [0.19, 0.31, 0.125][i], weather.floodplain * 0.30);
            let alluvium = lerp(
                vegetated,
                [0.50, 0.465, 0.385][i],
                0.65f64
                    .mul_add(weather.point_bar, 0.25 * weather.gravel)
                    .clamp(0.0, 1.0),
            );
            lerp(
                alluvium,
                [0.25, 0.245, 0.20][i],
                0.28f64
                    .mul_add(weather.scour, 0.22 * weather.bank)
                    .clamp(0.0, 1.0),
            )
        });
        let edge = point[0]
            .min(point[1])
            .min(extent - point[0])
            .min(extent - point[1]);
        let t = (edge / 4000.0_f64.min(extent * 0.25)).clamp(0.0, 1.0);
        let weight = t * t * 2.0_f64.mul_add(-t, 3.0);
        Some(RegionSample {
            weather,
            water_kind,
            rgb,
            weight,
        })
    }

    fn interpolation(&self, point: [f64; 2]) -> Option<CellInterpolation> {
        let extent = self.chart.extent();
        if !point.iter().all(|&v| (0.0..=extent).contains(&v)) {
            return None;
        }
        let grid = point.map(|v| v / self.chart.spacing_m);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates bounded by the chart's u16 cell count"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "validated nonnegative chart coordinates"
        )]
        let [column, row] = grid.map(|v| v.floor() as u16);
        let column = column.min(self.chart.cells - 1);
        let row = row.min(self.chart.cells - 1);
        let side = usize::from(self.chart.cells) + 1;
        let index = usize::from(row) * side + usize::from(column);
        let points = [
            self.cells[index],
            self.cells[index + 1],
            self.cells[index + side],
            self.cells[index + side + 1],
        ];
        let [tx, ty] = [grid[0] - f64::from(column), grid[1] - f64::from(row)];
        Some(CellInterpolation {
            points,
            fraction: [tx, ty],
        })
    }
    fn weather_at(
        &self,
        point: [f64; 2],
        interpolation: &CellInterpolation,
    ) -> (Weathered, Option<WaterKind>) {
        let height = interpolation.sample(|cell| f64::from(cell.height) * QUANTUM);
        let water = interpolation.sample(|cell| {
            let ground = f64::from(cell.height) * QUANTUM;
            if cell.lake == NO_LAKE {
                ground.max(0.0)
            } else {
                ground.max(f64::from(cell.lake) * QUANTUM)
            }
        });
        if water > height + 0.02 {
            (
                Weathered {
                    bed: height,
                    water: Some(water),
                    depth: water - height,
                    ..Weathered::default()
                },
                Some(WaterKind::Standing),
            )
        } else {
            let weather = self.channels.at(point, height);
            (weather, weather.water.map(|_| WaterKind::River))
        }
    }
    pub(super) fn channel_candidates(&self, min: [f64; 2], max: [f64; 2]) -> bool {
        self.channels.minimum_width(min, max).is_some()
    }

    /// Minimum nominal width of any potential channel affecting a spherical
    /// cap on this region, in metres. This is a conservative broad-phase
    /// detail hint: a candidate does not imply water at every point.
    /// A cap crossing the longitude seam or a geographic pole is handled
    /// without assuming the centre lies inside the chart.
    #[must_use = "Use the local feature scale to select terrain detail"]
    pub fn channel_width_in_cap(&self, direction: [f64; 3], half_angle: f64) -> Option<f64> {
        if !direction.iter().all(|v| v.is_finite())
            || direction.iter().all(|&v| v == 0.0)
            || !half_angle.is_finite()
            || half_angle < 0.0
            || self.channels.reaches().is_empty()
        {
            return None;
        }
        let chart = self.chart;
        // A small angular padding covers trig/projection roundoff at closed
        // cap and chart boundaries. It is independent of planetary radius.
        let angle = 64.0_f64.mul_add(f64::EPSILON, half_angle).min(PI);
        let latitude = direction[1].atan2(direction[0].hypot(direction[2]));
        let half = chart.extent() * 0.5;
        let center = (latitude - chart.latitude_rad).mul_add(chart.radius_m, half);
        let span = angle * chart.radius_m;
        let ymin = (center - span).max(0.0);
        let ymax = (center + span).min(chart.extent());
        if ymin > ymax {
            return None;
        }
        // If a pole is inside the cap, its longitude interval is the
        // entire circle. Otherwise the extremal meridian is tangent to it.
        if latitude.abs() + angle >= PI * 0.5 {
            return self
                .channels
                .minimum_width([0.0, ymin], [chart.extent(), ymax]);
        }
        let longitude = direction[2].atan2(direction[0]);
        let delta = (longitude - chart.longitude_rad + PI).rem_euclid(TAU) - PI;
        let scale = chart.radius_m * chart.latitude_rad.cos();
        let lon_span = (angle.sin() / latitude.cos()).clamp(0.0, 1.0).asin() * scale;
        let mut width = f64::INFINITY;
        for shift in [-TAU, 0.0, TAU] {
            let chart_x = (-(delta + shift)).mul_add(scale, half);
            let xmin = (chart_x - lon_span).max(0.0);
            let xmax = (chart_x + lon_span).min(chart.extent());
            if xmin <= xmax
                && let Some(candidate) = self.channels.minimum_width([xmin, ymin], [xmax, ymax])
            {
                width = width.min(candidate);
            }
        }
        width.is_finite().then_some(width)
    }

    #[cfg(feature = "gpu")]
    pub(super) fn gpu_controls(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        self.cells.iter().map(|cell| {
            let height = u32::from(u16::from_le_bytes(cell.height.to_le_bytes()));
            let lake = u32::from(u16::from_le_bytes(cell.lake.to_le_bytes()));
            [
                height | (lake << 16),
                u32::from_le_bytes([cell.rgb[0], cell.rgb[1], cell.rgb[2], 255]),
            ]
        })
    }

    #[cfg(feature = "gpu")]
    pub(super) const fn gpu_channels(&self) -> &ChannelField {
        &self.channels
    }

    /// Little-endian, versioned format. No machine-dependent struct layout.
    ///
    /// # Errors
    /// Propagates write failures.
    pub fn write(&self, mut out: impl Write) -> io::Result<()> {
        out.write_all(MAGIC)?;
        for v in [
            self.chart.radius_m,
            self.chart.latitude_rad,
            self.chart.longitude_rad,
            self.chart.spacing_m,
        ] {
            out.write_all(&v.to_le_bytes())?;
        }
        out.write_all(&self.chart.cells.to_le_bytes())?;
        out.write_all(
            &u32::try_from(self.channels.reaches().len())
                .map_err(|_| invalid())?
                .to_le_bytes(),
        )?;
        for c in &self.cells {
            out.write_all(&c.height.to_le_bytes())?;
            out.write_all(&c.lake.to_le_bytes())?;
            out.write_all(&c.rgb)?;
        }
        for r in self.channels.reaches() {
            for s in [r.a, r.b] {
                for v in [s.xy[0], s.xy[1], s.stage, s.width, s.bend] {
                    out.write_all(&v.to_le_bytes())?;
                }
            }
            out.write_all(&r.valley.to_le_bytes())?;
        }
        Ok(())
    }

    /// Decode and validate one complete regional asset.
    ///
    /// # Errors
    /// Rejects invalid, truncated or trailing data and out-of-range chart or channel geometry.
    pub fn read(bytes: &[u8]) -> io::Result<Self> {
        let mut input = Cursor::new(bytes);
        if &read_array::<8>(&mut input)? != MAGIC {
            return Err(invalid());
        }
        let chart = RegionChart {
            radius_m: read_f64(&mut input)?,
            latitude_rad: read_f64(&mut input)?,
            longitude_rad: read_f64(&mut input)?,
            spacing_m: read_f64(&mut input)?,
            cells: u16::from_le_bytes(read_array(&mut input)?),
        };
        let count =
            usize::try_from(u32::from_le_bytes(read_array(&mut input)?)).map_err(|_| invalid())?;
        if !chart.valid() || count > 1_000_000 {
            return Err(invalid());
        }
        let n = (usize::from(chart.cells) + 1).pow(2);
        if bytes.len() != 46 + n * 7 + count * 88 {
            return Err(invalid());
        }
        let mut cells = Vec::with_capacity(n);
        for _ in 0..n {
            let height = i16::from_le_bytes(read_array(&mut input)?);
            let lake = i16::from_le_bytes(read_array(&mut input)?);
            if height == NO_LAKE {
                return Err(invalid());
            }
            cells.push(RegionCell {
                height,
                lake,
                rgb: read_array(&mut input)?,
            });
        }
        let mut reaches = Vec::with_capacity(count);
        for _ in 0..count {
            reaches.push(Reach {
                a: read_section(&mut input)?,
                b: read_section(&mut input)?,
                valley: read_f64(&mut input)?,
            });
        }
        Self::new(chart, cells, reaches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn region_shares_bed_water_and_edge_weight() -> io::Result<()> {
        let chart = RegionChart {
            radius_m: 600_000.0,
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            spacing_m: 250.0,
            cells: 8,
        };
        let cell = RegionCell::new(104.0, None, [0.2, 0.4, 0.6])?;
        let section = |x| Section {
            xy: [x, 1000.0],
            stage: 100.0,
            width: 40.0,
            bend: 0.0,
        };
        let region = Region::new(
            chart,
            vec![cell; 81],
            vec![Reach {
                a: section(400.0),
                b: section(1600.0),
                valley: 1.0,
            }],
        )?;
        let mut bytes = Vec::new();
        region.write(&mut bytes)?;
        let region = Region::read(&bytes)?;
        let wet = region
            .sample(chart.direction([1000.0, 1000.0]))
            .ok_or_else(invalid)?;
        assert!(wet.weather.bed < 100.0);
        assert_eq!(wet.weather.water, Some(100.0));
        assert_eq!(wet.weight, 1.0);
        let edge = region.sample_chart([0.0, 1000.0]).ok_or_else(invalid)?;
        assert_eq!(edge.weight, 0.0);
        assert!(region.sample([0.0; 3]).is_none());
        assert!(region.sample(chart.direction([3000.0, 1000.0])).is_none());
        Ok(())
    }

    #[test]
    fn channel_detail_cap_encloses_wet_samples_at_seams_and_poles() -> io::Result<()> {
        for latitude_rad in [0.0, 1.55, -1.55] {
            let chart = RegionChart {
                radius_m: 600_000.0,
                latitude_rad,
                longitude_rad: PI - 0.001,
                spacing_m: 250.0,
                cells: 16,
            };
            let cell = RegionCell::new(104.0, None, [0.2, 0.4, 0.6])?;
            let section = |x| Section {
                xy: [x, 2000.0],
                stage: 100.0,
                width: 40.0,
                bend: 0.0,
            };
            let region = Region::new(
                chart,
                vec![cell; 289],
                vec![Reach {
                    a: section(1000.0),
                    b: section(3000.0),
                    valley: 1.0,
                }],
            )?;
            let wet = chart.direction([2000.0, 2000.0]);
            assert_eq!(region.channel_width_in_cap(wet, 0.0), Some(40.0));
            for dx in [-5000.0, -1000.0, 0.0, 1000.0, 5000.0] {
                for dy in [-5000.0, -1000.0, 0.0, 1000.0, 5000.0] {
                    let center = chart.direction([2000.0 + dx, 2000.0 + dy]);
                    let dot = center.iter().zip(wet).map(|(a, b)| a * b).sum::<f64>();
                    let half_angle = dot.clamp(-1.0, 1.0).acos() + 1e-8;
                    assert_eq!(region.channel_width_in_cap(center, half_angle), Some(40.0));
                }
            }
            assert!(region.channel_width_in_cap(wet.map(|v| -v), 0.01).is_none());
            assert!(region.channel_width_in_cap([0.0; 3], 1.0).is_none());
        }
        Ok(())
    }

    #[test]
    fn region_roundtrip_and_dateline_projection() -> io::Result<()> {
        let chart = RegionChart {
            radius_m: 600_000.0,
            latitude_rad: 0.1,
            longitude_rad: PI - 0.001,
            spacing_m: 500.0,
            cells: 4,
        };
        let cell = RegionCell::new(100.1, None, [0.2, 0.4, 0.6])?;
        let region = Region::new(chart, vec![cell; 25], vec![])?;
        let mut bytes = Vec::new();
        region.write(&mut bytes)?;
        let loaded = Region::read(&bytes)?;
        let p = [500.0, 1200.0];
        let direction = chart.direction(p);
        let sample = loaded.sample(direction).ok_or_else(invalid)?;
        assert!((sample.weather.bed - 100.0).abs() < 1e-9);
        assert!((sample.weight - 1.0).abs() < 1e-9);
        let actual = chart.project(direction).ok_or_else(invalid)?;
        assert!(actual.iter().zip(p).all(|(a, b)| (a - b).abs() < 1e-8));
        for n in 0..bytes.len() {
            assert!(Region::read(&bytes[..n]).is_err());
        }
        Ok(())
    }
}
