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
    pub fn project(self, direction: [f64; 3]) -> Option<[f64; 2]> {
        if !direction.iter().all(|v| v.is_finite()) || direction == [0.0; 3] {
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
    pub fn direction(self, p: [f64; 2]) -> [f64; 3] {
        let half = self.extent() * 0.5;
        let lat = (p[1] - half) / self.radius_m + self.latitude_rad;
        let lon = self.longitude_rad - (p[0] - half) / (self.radius_m * self.latitude_rad.cos());
        let (s, c) = lat.sin_cos();
        let (z, x) = lon.sin_cos();
        [c * x, s, c * z]
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

#[derive(Clone, Copy, Debug)]
pub struct RegionSample {
    pub weather: Weathered,
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
}

impl Region {
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
        let channels = ChannelField::new(chart.extent(), reaches)?;
        Ok(Self {
            chart,
            cells,
            channels,
            max_elevation,
        })
    }

    pub fn chart(&self) -> RegionChart {
        self.chart
    }
    pub fn max_elevation(&self) -> f64 {
        self.max_elevation
    }
    pub fn reaches(&self) -> &[Reach] {
        self.channels.reaches()
    }

    /// Canonical body-fixed query, with no allocation or mutable cache.
    pub fn sample(&self, direction: [f64; 3]) -> Option<RegionSample> {
        let p = self.chart.project(direction)?;
        self.sample_chart(p)
    }
    pub fn sample_chart(&self, p: [f64; 2]) -> Option<RegionSample> {
        let extent = self.chart.extent();
        if !p.iter().all(|&v| (0.0..=extent).contains(&v)) {
            return None;
        }
        let q = p.map(|v| v / self.chart.spacing_m);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates bounded by the chart's u16 cell count"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "validated nonnegative chart coordinates"
        )]
        let [x, y] = q.map(|v| v.floor() as u16);
        let x = x.min(self.chart.cells - 1);
        let y = y.min(self.chart.cells - 1);
        let side = usize::from(self.chart.cells) + 1;
        let i = usize::from(y) * side + usize::from(x);
        let points = [
            self.cells[i],
            self.cells[i + 1],
            self.cells[i + side],
            self.cells[i + side + 1],
        ];
        let [tx, ty] = [q[0] - f64::from(x), q[1] - f64::from(y)];
        let interpolate = |f: &dyn Fn(RegionCell) -> f64| {
            let [a, b, c, d] = points.map(f);
            lerp(lerp(a, b, tx), lerp(c, d, tx), ty)
        };
        let h = interpolate(&|c| f64::from(c.height) * QUANTUM);
        let water = interpolate(&|c| {
            let ground = f64::from(c.height) * QUANTUM;
            if c.lake == NO_LAKE {
                ground.max(0.0)
            } else {
                ground.max(f64::from(c.lake) * QUANTUM)
            }
        });
        let weather = if water > h + 0.02 {
            Weathered {
                bed: h,
                water: Some(water),
                depth: water - h,
                ..Weathered::default()
            }
        } else {
            self.channels.at(p, h)
        };
        let rgb = std::array::from_fn(|i| {
            let base = interpolate(&|c| f64::from(c.rgb[i]) / 255.0);
            let vegetated = lerp(base, [0.19, 0.31, 0.125][i], weather.floodplain * 0.30);
            let alluvium = lerp(
                vegetated,
                [0.50, 0.465, 0.385][i],
                (0.25 * weather.gravel + 0.65 * weather.point_bar).clamp(0.0, 1.0),
            );
            lerp(
                alluvium,
                [0.25, 0.245, 0.20][i],
                (0.22 * weather.bank + 0.28 * weather.scour).clamp(0.0, 1.0),
            )
        });
        let edge = p[0].min(p[1]).min(extent - p[0]).min(extent - p[1]);
        let t = (edge / 4000.0_f64.min(extent * 0.25)).clamp(0.0, 1.0);
        let weight = t * t * 2.0_f64.mul_add(-t, 3.0);
        Some(RegionSample {
            weather,
            rgb,
            weight,
        })
    }

    /// Little-endian, versioned format. No machine-dependent struct layout.
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
        fn section(input: &mut impl Read) -> io::Result<Section> {
            Ok(Section {
                xy: [read_f64(input)?, read_f64(input)?],
                stage: read_f64(input)?,
                width: read_f64(input)?,
                bend: read_f64(input)?,
            })
        }
        for _ in 0..count {
            reaches.push(Reach {
                a: section(&mut input)?,
                b: section(&mut input)?,
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
