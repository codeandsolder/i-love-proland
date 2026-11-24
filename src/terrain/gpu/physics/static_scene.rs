//! Immutable triangle collision acceleration. Collision geometry is provided
//! in the same small metric coordinate frame as the resident terrain grid.
//! The tree is independent of render LOD, and traversal is entirely on GPU.
use super::{invalid, storage};
use std::io;

const LEAF_TRIANGLES: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct StaticTriangle {
    pub vertices: [[f32; 3]; 3],
    /// Index into the physics material palette.
    pub material: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Triangle {
    a: [f32; 3],
    material: u32,
    b: [f32; 4],
    c: [f32; 4],
}

impl Triangle {
    const fn new(t: StaticTriangle) -> Self {
        let [a, b, c] = t.vertices;
        Self {
            a,
            material: t.material,
            b: [b[0], b[1], b[2], 0.0],
            c: [c[0], c[1], c[2], 0.0],
        }
    }

    fn bounds(self) -> Bounds {
        Bounds {
            min: std::array::from_fn(|i| self.a[i].min(self.b[i]).min(self.c[i])),
            max: std::array::from_fn(|i| self.a[i].max(self.b[i]).max(self.c[i])),
        }
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

impl Bounds {
    fn union(self, other: Self) -> Self {
        Self {
            min: std::array::from_fn(|i| self.min[i].min(other.min[i])),
            max: std::array::from_fn(|i| self.max[i].max(other.max[i])),
        }
    }

    fn widest_axis(self) -> usize {
        let span: [f32; 3] = std::array::from_fn(|i| self.max[i] - self.min[i]);
        (0..3)
            .max_by(|&a, &b| span[a].total_cmp(&span[b]))
            .unwrap_or(0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Node {
    min: [f32; 3],
    first: u32,
    max: [f32; 3],
    count: u32,
    escape: u32,
    padding: [u32; 3],
}

/// Preorder traversal with escape links. A leaf references a contiguous
/// triangle range. No traversal stack, recursion or fixed depth on GPU.
fn build(nodes: &mut Vec<Node>, triangles: &mut [Triangle], first: u32) -> io::Result<()> {
    let bounds = triangles
        .iter()
        .map(|t| t.bounds())
        .reduce(Bounds::union)
        .ok_or_else(|| invalid("empty static collision node"))?;
    let index = nodes.len();
    nodes.push(Node {
        min: bounds.min,
        max: bounds.max,
        first,
        count: 0,
        escape: 0,
        padding: [0; 3],
    });
    if triangles.len() <= LEAF_TRIANGLES {
        nodes[index].count = u32::try_from(triangles.len()).map_err(io::Error::other)?;
    } else {
        let axis = bounds.widest_axis();
        let mid = triangles.len() / 2;
        let key = |t: &Triangle| {
            let b = t.bounds();
            b.min[axis].mul_add(0.5, b.max[axis] * 0.5)
        };
        triangles.select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)));
        let (left, right) = triangles.split_at_mut(mid);
        build(nodes, left, first)?;
        let offset = first
            .checked_add(u32::try_from(mid).map_err(io::Error::other)?)
            .ok_or_else(|| invalid("static collision index overflow"))?;
        build(nodes, right, offset)?;
    }
    nodes[index].escape = u32::try_from(nodes.len()).map_err(io::Error::other)?;
    Ok(())
}

/// Immutable resident BVH and full-resolution collision triangles.
pub struct StaticScene {
    pub(super) nodes: wgpu::Buffer,
    pub(super) triangles: wgpu::Buffer,
    pub(super) max_material: u32,
}

impl StaticScene {
    /// Construct a scene-local BVH for arbitrary triangle soup. The CPU
    /// builds only immutable acceleration metadata, never contact points.
    ///
    /// # Errors
    /// Rejects invalid coordinates, material IDs or adapter-sized buffers.
    pub fn new(device: &wgpu::Device, source: &[StaticTriangle]) -> io::Result<Self> {
        if source.iter().any(|t| {
            t.material == u32::MAX
                || t.vertices
                    .iter()
                    .flatten()
                    .any(|x| !x.is_finite() || x.abs() > 1e7)
        }) {
            return Err(invalid("invalid static collision triangle"));
        }
        let mut triangles = source
            .iter()
            .copied()
            .map(Triangle::new)
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        if triangles.is_empty() {
            nodes.push(Node {
                min: [1.0; 3],
                max: [-1.0; 3],
                first: 0,
                count: 0,
                escape: 1,
                padding: [0; 3],
            });
            triangles.push(bytemuck::Zeroable::zeroed());
        } else {
            build(&mut nodes, &mut triangles, 0)?;
        }
        Ok(Self {
            nodes: storage(device, "static collision BVH", bytemuck::cast_slice(&nodes))?,
            triangles: storage(
                device,
                "static collision triangles",
                bytemuck::cast_slice(&triangles),
            )?,
            max_material: source.iter().map(|t| t.material).max().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_leaf_range_covered_once_and_escape_links_are_forward() -> io::Result<()> {
        let mut triangles = (0..257_u16)
            .map(|x| {
                let x = f32::from(x);
                Triangle::new(StaticTriangle {
                    vertices: [[x, 0.0, 0.0], [x, 1.0, 0.0], [x, 0.0, 1.0]],
                    material: 0,
                })
            })
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        build(&mut nodes, &mut triangles, 0)?;
        let mut counts = vec![0; triangles.len()];
        for (i, n) in nodes.iter().enumerate() {
            assert!(usize::try_from(n.escape).map_err(io::Error::other)? > i);
            assert!(usize::try_from(n.escape).map_err(io::Error::other)? <= nodes.len());
            if n.count > 0 {
                for j in n.first..n.first + n.count {
                    counts[usize::try_from(j).map_err(io::Error::other)?] += 1;
                    let b = triangles[usize::try_from(j).map_err(io::Error::other)?].bounds();
                    assert!((0..3).all(|i| n.min[i] <= b.min[i] && n.max[i] >= b.max[i]));
                }
            }
        }
        assert!(counts.iter().all(|&n| n == 1));
        assert_eq!(
            usize::try_from(nodes[0].escape).map_err(io::Error::other)?,
            nodes.len()
        );
        Ok(())
    }
}
