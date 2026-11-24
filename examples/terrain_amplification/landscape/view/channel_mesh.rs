//! Adaptive sampling of the analytic fluvial field for the CPU preview.
//! Only cells visited by the rays are refined; shared vertices are evaluated
//! in the same world coordinates. Intersect bilinear patches quadratically
//! rather than marching at tiny steps along grazing rays.
use super::{ChannelField, Grid, Sample, Surface, V3, Weathered, lerp, normalize, weather_at};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct Vertex {
    bed: f64,
    top: f64,
}

struct FineCell {
    side: usize,
    vertices: Box<[Vertex]>,
}

impl FineCell {
    fn new(
        grid: &Grid,
        surface: &Surface,
        channels: &ChannelField,
        x: u16,
        y: u16,
        subdiv: u16,
    ) -> Self {
        let spacing = grid.spacing / f64::from(subdiv);
        let side = usize::from(subdiv) + 1;
        let mut vertices = Vec::with_capacity(side * side);
        for j in 0..=subdiv {
            for i in 0..=subdiv {
                let p = [
                    f64::mul_add(f64::from(i), spacing, f64::from(x) * grid.spacing),
                    f64::mul_add(f64::from(j), spacing, f64::from(y) * grid.spacing),
                ];
                // At the outer regional border sample the shared last vertex
                // directly rather than asking the half-open sample domain.
                let sample = Sample {
                    ids: [
                        grid.index(x, y),
                        grid.index(x + 1, y),
                        grid.index(x, y + 1),
                        grid.index(x + 1, y + 1),
                    ],
                    tx: f64::from(i) / f64::from(subdiv),
                    ty: f64::from(j) / f64::from(subdiv),
                };
                let w = weather_at(grid, surface, channels, &sample, p);
                vertices.push(Vertex {
                    bed: w.bed,
                    top: w.elevation(),
                });
            }
        }
        Self {
            side,
            vertices: vertices.into_boxed_slice(),
        }
    }

    fn patch(&self, x: u16, y: u16) -> [Vertex; 4] {
        let i = usize::from(y) * self.side + usize::from(x);
        [
            self.vertices[i],
            self.vertices[i + 1],
            self.vertices[i + self.side],
            self.vertices[i + self.side + 1],
        ]
    }
}

fn bilinear(values: [f64; 4], u: f64, v: f64) -> f64 {
    let [southwest, southeast, northwest, northeast] = values;
    lerp(
        lerp(southwest, southeast, u),
        lerp(northwest, northeast, u),
        v,
    )
}

fn hit_patch(
    values: [f64; 4],
    origin: [f64; 2],
    spacing: f64,
    eye: V3,
    ray: V3,
    start: f64,
    end: f64,
) -> Option<f64> {
    let eval = |t: f64| {
        let x = (ray[0].mul_add(t, eye[0]) - origin[0]) / spacing;
        let y = (ray[1].mul_add(t, eye[1]) - origin[1]) / spacing;
        ray[2].mul_add(t, eye[2]) - bilinear(values, x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
    };
    let a = eval(start);
    if a <= 0.0 {
        return Some(start);
    }
    let c = eval(end);
    let middle = eval(start.midpoint(end));
    // gap(u)=quadratic*u²+linear*u+a, u in [0,1].
    let quadratic = 2.0 * 2.0f64.mul_add(-middle, a + c);
    let linear = c - a - quadratic;
    let mut upper = end;
    if quadratic > 0.0 {
        let u = -linear / (2.0 * quadratic);
        if (0.0..1.0).contains(&u) {
            let t = lerp(start, end, u);
            // Even when both endpoints are above the patch, a grazing ray
            // may have entered and exited the bilinear saddle.
            if eval(t) > 0.0 {
                return None;
            }
            upper = t;
        } else if c > 0.0 {
            return None;
        }
    } else if c > 0.0 {
        return None;
    }
    let mut lower = start;
    for _ in 0..22 {
        let mid = lower.midpoint(upper);
        if eval(mid) > 0.0 {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    Some(upper)
}

/// Walk a square lattice in increasing ray parameter. Coordinates refer to
/// cells; start and end can be inside cells. Simultaneous edge crossings
/// advance both axes, including at exact grid corners.
fn walk<T>(
    eye: V3,
    ray: V3,
    origin: [f64; 2],
    spacing: f64,
    cells: u16,
    interval: [f64; 2],
    mut visit: impl FnMut(u16, u16, f64, f64) -> Option<T>,
) -> Option<T> {
    let [start, end] = interval;
    let mut cell: [i32; 2] = std::array::from_fn(|i| {
        let q = (ray[i].mul_add(start, eye[i]) - origin[i]) / spacing;
        let index = if ray[i] < 0.0 {
            q.ceil() - 1.0
        } else {
            q.floor()
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "clamped to lattice cell extent"
        )]
        let result = index.clamp(0.0, f64::from(cells - 1)) as i32;
        result
    });
    let direction: [i32; 2] = std::array::from_fn(|i| {
        if ray[i] > 0.0 {
            1
        } else if ray[i] < 0.0 {
            -1
        } else {
            0
        }
    });
    let mut current = start;
    while current < end && cell.iter().all(|&v| (0..i32::from(cells)).contains(&v)) {
        let next: [f64; 2] = std::array::from_fn(|i| {
            if direction[i] == 0 {
                return f64::INFINITY;
            }
            let edge = cell[i] + i32::from(direction[i] > 0);
            (f64::mul_add(f64::from(edge), spacing, origin[i]) - eye[i]) / ray[i]
        });
        let stop = next[0].min(next[1]).min(end).max(current);
        if stop > current {
            let (Ok(x), Ok(y)) = (u16::try_from(cell[0]), u16::try_from(cell[1])) else {
                return None;
            };
            if let Some(result) = visit(x, y, current, stop) {
                return Some(result);
            }
        }
        if stop >= end {
            break;
        }
        for i in 0..2 {
            if next[i] <= stop + 1e-9 {
                cell[i] += direction[i];
            }
        }
        current = stop;
    }
    None
}

pub(super) struct ChannelMesh<'a> {
    grid: &'a Grid,
    surface: &'a Surface,
    channels: &'a ChannelField,
    cells: HashMap<usize, FineCell>,
    subdiv: u16,
}

pub(super) struct Hit {
    pub(super) sample: Sample,
    pub(super) distance: f64,
    pub(super) weather: Weathered,
    pub(super) normal: Option<V3>,
}

impl<'a> ChannelMesh<'a> {
    pub(super) fn new(
        grid: &'a Grid,
        surface: &'a Surface,
        channels: &'a ChannelField,
        subdiv: u16,
    ) -> Self {
        Self {
            grid,
            surface,
            channels,
            cells: HashMap::new(),
            subdiv,
        }
    }

    pub(super) fn trace(&mut self, eye: V3, ray: V3, start: f64, end: f64) -> Option<Hit> {
        let grid = self.grid;
        let surface = self.surface;
        let channels = self.channels;
        let subdiv = self.subdiv;
        walk(
            eye,
            ray,
            [0.0; 2],
            grid.spacing,
            grid.cells,
            [start, end],
            |x, y, t0, t1| {
                let indices = [
                    grid.index(x, y),
                    grid.index(x + 1, y),
                    grid.index(x, y + 1),
                    grid.index(x + 1, y + 1),
                ];
                let top = indices.map(|i| grid.heights[i].max(surface.water_level[i]).max(0.0));
                let origin = [f64::from(x) * grid.spacing, f64::from(y) * grid.spacing];
                let highest =
                    top.into_iter().fold(f64::NEG_INFINITY, f64::max) + ChannelField::MAX_RAISE;
                if ray[2].mul_add(t0, eye[2]).min(ray[2].mul_add(t1, eye[2])) > highest {
                    return None;
                }
                // Bucket size is exactly four coarse cells in the laboratory;
                // use the center plus corners for arbitrary future grid spacing.
                let active = [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
                    .into_iter()
                    .any(|(u, v)| {
                        channels.has_channel([
                            f64::mul_add(u, grid.spacing, origin[0]),
                            f64::mul_add(v, grid.spacing, origin[1]),
                        ])
                    });
                if !active {
                    let t = hit_patch(top, origin, grid.spacing, eye, ray, t0, t1)?;
                    let p = [ray[0].mul_add(t, eye[0]), ray[1].mul_add(t, eye[1])];
                    let sample = Sample::at(grid, p[0], p[1])?;
                    let weather = weather_at(grid, surface, channels, &sample, p);
                    return Some(Hit {
                        sample,
                        distance: t,
                        weather,
                        normal: None,
                    });
                }
                let fine = self
                    .cells
                    .entry(indices[0])
                    .or_insert_with(|| FineCell::new(grid, surface, channels, x, y, subdiv));
                let spacing = grid.spacing / f64::from(subdiv);
                walk(
                    eye,
                    ray,
                    origin,
                    spacing,
                    subdiv,
                    [t0, t1],
                    |ix, iy, a, b| {
                        let vertices = fine.patch(ix, iy);
                        let corner = vertices.map(|v| v.top);
                        if ray[2].mul_add(a, eye[2]).min(ray[2].mul_add(b, eye[2]))
                            > corner.into_iter().fold(f64::NEG_INFINITY, f64::max)
                        {
                            return None;
                        }
                        let cell_origin = [
                            f64::mul_add(f64::from(ix), spacing, origin[0]),
                            f64::mul_add(f64::from(iy), spacing, origin[1]),
                        ];
                        let distance = hit_patch(corner, cell_origin, spacing, eye, ray, a, b)?;
                        let p = [
                            ray[0].mul_add(distance, eye[0]),
                            ray[1].mul_add(distance, eye[1]),
                        ];
                        let sample = Sample::at(grid, p[0], p[1])?;
                        let u = ((p[0] - cell_origin[0]) / spacing).clamp(0.0, 1.0);
                        let v = ((p[1] - cell_origin[1]) / spacing).clamp(0.0, 1.0);
                        let mut weather = weather_at(grid, surface, channels, &sample, p);
                        let bed = vertices.map(|v| v.bed);
                        weather.bed = bilinear(bed, u, v);
                        let top = bilinear(corner, u, v);
                        weather.depth = (top - weather.bed).max(0.0);
                        weather.water = (weather.depth > 0.025).then_some(top);
                        let [southwest, southeast, northwest, northeast] = bed;
                        let dx = lerp(southeast - southwest, northeast - northwest, v) / spacing;
                        let dy = lerp(northwest - southwest, northeast - southeast, u) / spacing;
                        let normal = normalize([-dx, -dy, 1.0]);
                        Some(Hit {
                            sample,
                            distance,
                            weather,
                            normal: Some(normal),
                        })
                    },
                )
            },
        )
    }

    pub(super) fn cells(&self) -> usize {
        self.cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_enter_and_exit_of_bilinear_saddle() {
        // Height is 4u(1-u) along diagonal, ray at z=0.75;
        // endpoints are clear but first intersection is u=0.25.
        let hit = hit_patch(
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.75],
            [1.0, 1.0, 0.0],
            0.0,
            1.0,
        );
        assert!((hit.unwrap_or_default() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn grid_walk_covers_reversed_corner_crossings() {
        let mut seen = Vec::new();
        let _: Option<()> = walk(
            [2.0, 2.0, 1.0],
            [-1.0, -1.0, 0.0],
            [0.0, 0.0],
            1.0,
            2,
            [0.0, 2.0],
            |x, y, a, b| {
                seen.push((x, y, a, b));
                None
            },
        );
        assert_eq!(seen, vec![(1, 1, 0.0, 1.0), (0, 0, 1.0, 2.0)]);
    }
}
