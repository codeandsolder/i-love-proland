use super::{Mesh, MeshSource, Node, Triangle, invalid};
use std::io;

const LEAF: usize = 8;
type V = [f64; 3];

fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: V, b: V) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}
fn cross(a: V, b: V) -> V {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

impl Triangle {
    fn points(self) -> [V; 3] {
        [
            self.a.map(f64::from),
            [self.b[0], self.b[1], self.b[2]].map(f64::from),
            self.c.map(f64::from),
        ]
    }
}

// Separating-axis triangle/AABB test in f64, inflated by roundoff.
// Zero-area and thin triangles are retained, including boundary touches.
fn intersects(points: [V; 3], lo: V, hi: V) -> bool {
    let center = std::array::from_fn(|i| f64::midpoint(lo[i], hi[i]));
    let half: V = std::array::from_fn(|i| (hi[i] - lo[i]) * 0.5);
    let p = points.map(|v| sub(v, center));
    let edges = [sub(p[1], p[0]), sub(p[2], p[1]), sub(p[0], p[2])];
    let xyz = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let separated = |axis: V| {
        let r = dot(half, axis.map(f64::abs));
        let d = p.map(|v| dot(v, axis));
        let eps = 128.0 * f64::EPSILON * (r + d.iter().map(|v| v.abs()).sum::<f64>() + 1.0);
        d.iter().all(|&v| v > r + eps) || d.iter().all(|&v| v < -r - eps)
    };
    !xyz.into_iter()
        .chain([cross(edges[0], edges[1])])
        .chain(
            edges
                .into_iter()
                .flat_map(|e| xyz.map(|axis| cross(e, axis))),
        )
        .any(separated)
}

struct Grid {
    lo: V,
    step: V,
}

impl Grid {
    fn new(lo: [f32; 3], hi: [f32; 3]) -> Self {
        let lo = lo.map(f64::from);
        let hi = hi.map(f64::from);
        Self {
            lo,
            step: std::array::from_fn(|i| (hi[i] - lo[i]) * 0.25),
        }
    }

    fn cell(&self, index: [u32; 3]) -> (V, V) {
        let lo = std::array::from_fn(|j| self.step[j].mul_add(f64::from(index[j]), self.lo[j]));
        let hi = std::array::from_fn(|j| self.step[j].mul_add(f64::from(index[j] + 1), self.lo[j]));
        (lo, hi)
    }

    // Restrict SAT to cells which overlap the triangle's axis-aligned
    // projection. Include the full SAT axis-roundoff allowance, so this
    // inexpensive prefilter cannot change the exact occupancy mask.
    fn axis_range(&self, points: [V; 3], axis: usize) -> std::ops::RangeInclusive<u32> {
        let min = points.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min);
        let max = points
            .iter()
            .map(|p| p[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        let end = self.step[axis].mul_add(4.0, self.lo[axis]);
        let scale = min
            .abs()
            .max(max.abs())
            .max(self.lo[axis].abs())
            .max(end.abs());
        let error = 128.0 * f64::EPSILON * 8.0_f64.mul_add(scale, 1.0);
        let boundary = |i| self.step[axis].mul_add(f64::from(i), self.lo[axis]);
        let mut first = 0;
        let mut last = 3;
        while first < 3 && boundary(first + 1) < min - error {
            first += 1;
        }
        while last > 0 && boundary(last) > max + error {
            last -= 1;
        }
        first..=last
    }

    fn occupied(&self, triangles: &[Triangle]) -> u64 {
        let mut result = 0_u64;
        for t in triangles {
            let points = t.points();
            let [xs, ys, zs] = std::array::from_fn(|axis| self.axis_range(points, axis));
            for z in zs {
                for y in ys.clone() {
                    for x in xs.clone() {
                        let bit = 1_u64 << (x + y * 4 + z * 16);
                        if result & bit != 0 {
                            continue;
                        }
                        let (lo, hi) = self.cell([x, y, z]);
                        if intersects(points, lo, hi) {
                            result |= bit;
                        }
                    }
                }
            }
            if result == u64::MAX {
                break;
            }
        }
        result
    }
}

fn mask(triangles: &[Triangle], lo: [f32; 3], hi: [f32; 3]) -> [u32; 2] {
    let bytes = Grid::new(lo, hi).occupied(triangles).to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

// Exterior flood fill on the conservative surface supercover. A cell
// which contains an interior point cannot reach the outside through
// face-connected empty cells without crossing an occupied cell.
// Sealed cavities and disconnected open surfaces may overfill, never
// carve holes in the authoritative closed shell.
fn solid_cover(surface: [u32; 2]) -> [u32; 2] {
    let occupied = u64::from(surface[0]) | (u64::from(surface[1]) << 32);
    const X0: u64 = 0x1111_1111_1111_1111;
    const X3: u64 = X0 << 3;
    const Y0: u64 = 0x000f_000f_000f_000f;
    const Y3: u64 = Y0 << 12;
    const Z0: u64 = 0xffff;
    const Z3: u64 = Z0 << 48;
    let empty = !occupied;
    let mut exterior = empty & (X0 | X3 | Y0 | Y3 | Z0 | Z3);
    loop {
        let neighbors = ((exterior & !X0) >> 1)
            | ((exterior & !X3) << 1)
            | ((exterior & !Y0) >> 4)
            | ((exterior & !Y3) << 4)
            | ((exterior & !Z0) >> 16)
            | ((exterior & !Z3) << 16);
        let expanded = exterior | (neighbors & empty);
        if expanded == exterior {
            let bytes = (!exterior).to_le_bytes();
            return [
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            ];
        }
        exterior = expanded;
    }
}

fn index(n: usize) -> io::Result<u32> {
    u32::try_from(n).map_err(io::Error::other)
}

fn build(nodes: &mut Vec<Node>, triangles: &mut [Triangle], first: u32) -> io::Result<()> {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for t in triangles.iter() {
        for i in 0..3 {
            lo[i] = lo[i].min(t.a[i]).min(t.b[i]).min(t.c[i]);
            hi[i] = hi[i].max(t.a[i]).max(t.b[i]).max(t.c[i]);
        }
    }
    // Nonzero extent for planar geometry; outward pad is shared by
    // CPU voxelization and GPU query normalization.
    for i in 0..3 {
        let pad = 8.0 * f32::EPSILON * lo[i].abs().max(hi[i].abs()).max(1.0);
        lo[i] -= pad;
        hi[i] += pad;
    }
    let offset = nodes.len();
    nodes.push(Node {
        lo,
        hi,
        first,
        count: 0,
        escape: 0,
        mask: mask(triangles, lo, hi),
        end: first
            .checked_add(index(triangles.len())?)
            .ok_or_else(|| invalid("mesh triangle range overflow"))?,
    });
    if triangles.len() <= LEAF {
        nodes[offset].count = index(triangles.len())?;
    } else {
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap_or(0);
        let key = |t: &Triangle| t.a[axis] + t.b[axis] + t.c[axis];
        let mid = triangles.len() / 2;
        triangles
            .select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)).then(a.id.cmp(&b.id)));
        let (a, b) = triangles.split_at_mut(mid);
        build(nodes, a, first)?;
        build(
            nodes,
            b,
            first
                .checked_add(index(mid)?)
                .ok_or_else(|| invalid("mesh triangle offset overflow"))?,
        )?;
    }
    nodes[offset].escape = index(nodes.len())?;
    Ok(())
}

pub(super) fn catalog(
    sources: &[MeshSource<'_>],
) -> io::Result<(Vec<Node>, Vec<Triangle>, Vec<Mesh>)> {
    let mut nodes = Vec::new();
    let mut triangles = Vec::new();
    let mut meshes = Vec::new();
    for source in sources {
        if source.indices.is_empty()
            || !source.indices.len().is_multiple_of(3)
            || source
                .positions
                .iter()
                .flatten()
                .any(|&x| !x.is_finite() || x.abs() > 1e6)
        {
            return Err(invalid("invalid collision mesh"));
        }
        let topology = super::topology::classify(source)?;
        let start = triangles.len();
        let root = index(nodes.len())?;
        for (id, face) in source.indices.as_chunks::<3>().0.iter().enumerate() {
            let points = face.map(|i| source.positions.get(i as usize).copied());
            let [Some(a), Some(b), Some(c)] = points else {
                return Err(invalid("collision index out of range"));
            };
            triangles.push(Triangle {
                a,
                id: index(id)?,
                b: [b[0], b[1], b[2], 0.0],
                c,
                flags: u32::from(topology.representatives[id])
                    | (u32::from(topology.closed_faces[id]) << 1),
            });
        }
        build(&mut nodes, &mut triangles[start..], index(start)?)?;
        let surface = nodes[root as usize].mask;
        let solid_mask = if topology.closed {
            solid_cover(surface)
        } else {
            surface
        };
        meshes.push(Mesh {
            nodes: [root, index(nodes.len())?],
            triangles: [index(start)?, index(triangles.len())?],
            closed: u32::from(topology.closed),
            padding: 0,
            solid_mask,
        });
    }
    if meshes.is_empty() {
        return Err(invalid("empty collision catalog"));
    }
    Ok((nodes, triangles, meshes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filled_root_cover_keeps_solids_and_exterior_holes() {
        // A sealed 4³ shell encloses eight interior cells.
        let interior = 0x0000_0660_0660_0000_u64;
        let boundary = !interior;
        let bytes = boundary.to_le_bytes();
        let mask = [
            u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            u32::from_le_bytes(bytes[4..].try_into().unwrap()),
        ];
        assert_eq!(solid_cover(mask), [u32::MAX; 2]);
        // Remove one face cell adjacent to the interior; all connected
        // empty cells must now be reachable from the outside.
        let vented = boundary & !(1_u64 << 5);
        let bytes = vented.to_le_bytes();
        let mask = [
            u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            u32::from_le_bytes(bytes[4..].try_into().unwrap()),
        ];
        assert_eq!(solid_cover(mask), mask);
        assert_eq!(solid_cover([0; 2]), [0; 2]);
        assert_eq!(solid_cover([1, 1 << 31]), [1, 1 << 31]);
    }

    #[test]
    fn grid_projection_preserves_exhaustive_sat() {
        // Reproduce the exhaustive old implementation on isolated
        // triangles (not a quickly saturated union of many surfaces).
        let mut rng = 0x4d59_5df4_u32;
        for offset in [0.0_f32, 100.0, 100_000.0, -100_000.0] {
            for extent in [0.125_f32, 1.0, 256.0] {
                let grid = Grid::new([offset; 3], [offset + extent; 3]);
                for case in 0..128_u32 {
                    let mut point = || {
                        std::array::from_fn(|_| {
                            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            let value = f32::from(u16::from_le_bytes([
                                rng.to_le_bytes()[2],
                                rng.to_le_bytes()[3],
                            ])) / 65_535.0;
                            extent.mul_add(value, offset)
                        })
                    };
                    let a = point();
                    let mut b = point();
                    let mut c = point();
                    if case % 4 == 0 {
                        b = a;
                    }
                    if case % 7 == 0 {
                        c[2] = a[2];
                        b[2] = a[2];
                    }
                    let triangle = Triangle {
                        a,
                        id: case,
                        b: [b[0], b[1], b[2], 0.0],
                        c,
                        flags: 0,
                    };
                    let points = triangle.points();
                    let mut exhaustive = 0_u64;
                    for cell in 0..64_u32 {
                        let (lo, hi) = grid.cell([cell % 4, cell / 4 % 4, cell / 16]);
                        if intersects(points, lo, hi) {
                            exhaustive |= 1_u64 << cell;
                        }
                    }
                    assert_eq!(grid.occupied(&[triangle]), exhaustive);
                }
            }
        }
    }

    #[test]
    fn sparse_diagonal_and_planar_boundary_are_preserved() {
        let t = Triangle {
            a: [0., 0., 0.],
            id: 0,
            b: [1., 1., 0., 0.],
            c: [1., 1., 1.],
            flags: 1,
        };
        let m = mask(&[t], [0.; 3], [1.; 3]);
        assert_ne!(m, [u32::MAX; 2]);
        assert!(m[0] & 1 != 0);
        assert!(m[1] & (1 << 31) != 0);
        assert!(intersects(
            [[0., 0., 0.], [1., 1., 0.], [1., 1., 1.]],
            [0., 0., 0.],
            [0.25; 3]
        ));
    }
}
