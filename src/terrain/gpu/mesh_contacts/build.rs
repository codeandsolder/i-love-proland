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
            [self.c[0], self.c[1], self.c[2]].map(f64::from),
        ]
    }
}

// Separating-axis triangle/AABB test in f64, inflated by roundoff.
// Zero-area and thin triangles are retained, including boundary touches.
fn intersects(points: [V; 3], lo: V, hi: V) -> bool {
    let center = std::array::from_fn(|i| (lo[i] + hi[i]) * 0.5);
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

fn mask(triangles: &[Triangle], lo: [f32; 3], hi: [f32; 3]) -> [u32; 2] {
    let lo = lo.map(f64::from);
    let hi = hi.map(f64::from);
    let step: V = std::array::from_fn(|i| (hi[i] - lo[i]) * 0.25);
    let mut result = 0_u64;
    for t in triangles {
        let p = t.points();
        for i in 0..64_u32 {
            if result & (1 << i) != 0 {
                continue;
            }
            let index = [i % 4, i / 4 % 4, i / 16];
            let a = std::array::from_fn(|j| step[j].mul_add(f64::from(index[j]), lo[j]));
            let b = std::array::from_fn(|j| step[j].mul_add(f64::from(index[j] + 1), lo[j]));
            if intersects(p, a, b) {
                result |= 1 << i;
            }
        }
        if result == u64::MAX {
            break;
        }
    }
    let bytes = result.to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
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
        padding: 0,
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
        let start = triangles.len();
        let root = index(nodes.len())?;
        for (id, face) in source.indices.as_chunks::<3>().0.iter().enumerate() {
            let [a, b, c] = face.map(|i| source.positions.get(i as usize).copied());
            let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                return Err(invalid("collision index out of range"));
            };
            triangles.push(Triangle {
                a,
                id: index(id)?,
                b: [b[0], b[1], b[2], 0.0],
                c: [c[0], c[1], c[2], 0.0],
            });
        }
        build(&mut nodes, &mut triangles[start..], index(start)?)?;
        meshes.push(Mesh {
            nodes: [root, index(nodes.len())?],
            triangles: [index(start)?, index(triangles.len())?],
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
    fn sparse_diagonal_and_planar_boundary_are_preserved() {
        let t = Triangle {
            a: [0., 0., 0.],
            id: 0,
            b: [1., 1., 0., 0.],
            c: [1., 1., 1., 0.],
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
