//! Exact-coordinate welding for closed-shell eligibility and component seeds.
//!
//! Rendering meshes frequently duplicate vertices across materials or UV
//! seams. The physical topology must not mistake those seams for holes.
use super::MeshSource;
use std::{
    collections::{HashMap, HashSet},
    io,
};

pub(super) struct Topology {
    pub closed: bool,
    pub representatives: Vec<bool>,
}
const fn root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        let next = parent[x];
        parent[x] = parent[next];
        x = next;
    }
    x
}
fn unite(parent: &mut [usize], a: usize, b: usize) {
    let a = root(parent, a);
    let b = root(parent, b);
    if a != b {
        parent[a.max(b)] = a.min(b);
    }
}
fn bits(p: [f32; 3]) -> [u32; 3] {
    p.map(|x| if x == 0.0 { 0 } else { x.to_bits() })
}

pub(super) fn classify(source: &MeshSource<'_>) -> io::Result<Topology> {
    let mut welded = HashMap::new();
    let ids = source
        .positions
        .iter()
        .map(|&p| {
            let next = welded.len();
            *welded.entry(bits(p)).or_insert(next)
        })
        .collect::<Vec<_>>();
    let mut parent = (0..welded.len()).collect::<Vec<_>>();
    let mut edges = HashMap::<(usize, usize), (usize, isize)>::new();
    let mut closed = true;
    let faces = source
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|face| {
            let vertices = face.map(|i| ids.get(i as usize).copied());
            let [Some(a), Some(b), Some(c)] = vertices else {
                return Err(super::invalid("collision index out of range"));
            };
            unite(&mut parent, a, b);
            unite(&mut parent, b, c);
            if a == b || b == c || c == a {
                closed = false;
            }
            for (a, b) in [(a, b), (b, c), (c, a)] {
                let edge = edges.entry((a.min(b), a.max(b))).or_default();
                edge.0 += 1;
                edge.1 += if a < b { 1 } else { -1 };
            }
            Ok(a)
        })
        .collect::<io::Result<Vec<_>>>()?;
    closed &= edges
        .values()
        .all(|&(count, winding)| count == 2 && winding == 0);
    let mut seen = HashSet::new();
    let representatives = faces
        .into_iter()
        .map(|v| seen.insert(root(&mut parent, v)))
        .collect();
    Ok(Topology {
        closed,
        representatives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn welded_shell_and_disconnected_surfaces() -> io::Result<()> {
        // Tetrahedron with duplicated UV seam vertex.
        let positions = [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [-0., 0., 0.],
            [10., 0., 0.],
        ];
        let indices = [0, 2, 1, 4, 1, 3, 0, 3, 2, 1, 2, 3];
        let shell = classify(&MeshSource {
            positions: &positions,
            indices: &indices,
        })?;
        assert!(shell.closed);
        assert_eq!(shell.representatives.iter().filter(|&&v| v).count(), 1);
        let open = classify(&MeshSource {
            positions: &positions,
            indices: &[0, 2, 1, 5, 5, 5],
        })?;
        assert!(!open.closed);
        assert_eq!(open.representatives.iter().filter(|&&v| v).count(), 2);
        Ok(())
    }
}
