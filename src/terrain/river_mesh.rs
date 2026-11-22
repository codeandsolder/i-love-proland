//! On-demand river geometry in a globally indexed metric chart lattice.
//! A tile is not persisted and does not depend on camera traversal order.
//! Water is sampled from the same field that defines the solid channel bed.
use super::Region;
use std::{collections::HashMap, io};

/// Indexed, counter-clockwise water triangles in regional [west, north, stage] metres.
/// The host planet converts them to its body-fixed reference frame.
#[derive(Debug, Default)]
pub struct RiverPatch {
    pub vertices: Vec<[f64; 3]>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Node {
    point: [f64; 2],
    stage: Option<f64>,
}

struct Builder<'a> {
    region: &'a Region,
    nodes: Vec<Node>,
    vertex_ids: Vec<Option<u32>>,
    edges: HashMap<[usize; 2], u32>,
    mesh: RiverPatch,
}

impl Builder<'_> {
    fn vertex(&mut self, node: Node) -> u32 {
        // At most 256^2 cells and three vertices per triangle. Far below u32.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "maximum patch size bounds vertex IDs"
        )]
        let id = self.mesh.vertices.len() as u32;
        let [x, y] = node.point;
        self.mesh
            .vertices
            .push([x, y, node.stage.unwrap_or_default()]);
        id
    }

    fn wet_vertex(&mut self, index: usize) -> u32 {
        if let Some(id) = self.vertex_ids[index] {
            return id;
        }
        let id = self.vertex(self.nodes[index]);
        self.vertex_ids[index] = Some(id);
        id
    }

    fn crossing(&mut self, wet: usize, dry: usize) -> u32 {
        let key = [wet.min(dry), wet.max(dry)];
        if let Some(&id) = self.edges.get(&key) {
            return id;
        }
        // Always approach from the wet side. Neighbor patches use identical
        // point coordinates and arithmetic regardless of triangle traversal.
        let mut inside = self.nodes[wet];
        let mut outside = self.nodes[dry].point;
        for _ in 0..12 {
            let point = std::array::from_fn(|axis| inside.point[axis].midpoint(outside[axis]));
            if let Some(stage) = self.region.river_stage(point) {
                inside = Node {
                    point,
                    stage: Some(stage),
                };
            } else {
                outside = point;
            }
        }
        let id = self.vertex(inside);
        self.edges.insert(key, id);
        id
    }

    fn triangle(&mut self, triangle: [usize; 3]) {
        let mut polygon = [0; 4];
        let mut count = 0;
        for (index, &current) in triangle.iter().enumerate() {
            let previous = triangle[(index + 2) % 3];
            let current_wet = self.nodes[current].stage.is_some();
            let previous_wet = self.nodes[previous].stage.is_some();
            if current_wet != previous_wet {
                polygon[count] = if current_wet {
                    self.crossing(current, previous)
                } else {
                    self.crossing(previous, current)
                };
                count += 1;
            }
            if current_wet {
                polygon[count] = self.wet_vertex(current);
                count += 1;
            }
        }
        for index in 2..count {
            self.mesh
                .indices
                .extend([polygon[0], polygon[index - 1], polygon[index]]);
        }
    }
}

impl Region {
    /// Tessellate one canonical regional heightfield cell into independent
    /// river water geometry. Unlike a coarse terrain tile, this mesh samples
    /// freely at a resolution selected for the channel width. At each wet/dry
    /// crossing the true field is bisected (not painted over dry ground).
    ///
    /// Use identical subdivisions on adjoining cells for matching boundaries.
    /// The caller must render sufficiently detailed bed geometry as well:
    /// a coarse terrain approximation can otherwise occlude the water.
    ///
    /// # Errors
    /// Returns an invalid-input error for an invalid cell or a subdivision
    /// count outside 1..=256 or not a power of two.
    pub fn river_patch(&self, cell: [u16; 2], subdivisions: u16) -> io::Result<RiverPatch> {
        let chart = self.chart();
        if cell.iter().any(|&v| v >= chart.cells)
            || subdivisions > 256
            || !subdivisions.is_power_of_two()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid river patch lattice",
            ));
        }
        let min = cell.map(|v| f64::from(v) * chart.spacing_m);
        let max = cell.map(|v| f64::from(v + 1) * chart.spacing_m);
        if !self.channel_candidates(min, max) {
            return Ok(RiverPatch::default());
        }
        let side = usize::from(subdivisions) + 1;
        let mut nodes = Vec::with_capacity(side * side);
        let global = cell.map(|v| u32::from(v) * u32::from(subdivisions));
        for row in 0..=subdivisions {
            for column in 0..=subdivisions {
                // A lattice coordinate has one expression, independent of
                // the tile that generated it (including shared outer edges).
                let point = [global[0] + u32::from(column), global[1] + u32::from(row)]
                    .map(|v| f64::from(v) / f64::from(subdivisions) * chart.spacing_m);
                nodes.push(Node {
                    point,
                    stage: self.river_stage(point),
                });
            }
        }
        if nodes.iter().all(|node| node.stage.is_none()) {
            return Ok(RiverPatch::default());
        }
        let mut builder = Builder {
            region: self,
            nodes,
            vertex_ids: vec![None; side * side],
            edges: HashMap::new(),
            mesh: RiverPatch::default(),
        };
        for row in 0..side - 1 {
            for column in 0..side - 1 {
                let southwest = row * side + column;
                let southeast = southwest + 1;
                let northwest = southwest + side;
                let northeast = northwest + 1;
                builder.triangle([southwest, southeast, northeast]);
                builder.triangle([southwest, northeast, northwest]);
            }
        }
        Ok(builder.mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{Reach, RegionCell, RegionChart, Section};

    fn region() -> io::Result<Region> {
        let chart = RegionChart {
            radius_m: 600_000.0,
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            spacing_m: 250.0,
            cells: 8,
        };
        let cell = RegionCell::new(104.0, None, [0.3, 0.5, 0.2])?;
        let section = |x| Section {
            xy: [x, 1125.0],
            stage: 100.0,
            width: 40.0,
            bend: 0.0,
        };
        Region::new(
            chart,
            vec![cell; 81],
            vec![Reach {
                a: section(400.0),
                b: section(1600.0),
                valley: 1.0,
            }],
        )
    }

    #[test]
    fn independent_water_geometry_matches_channel_and_tile_edges() -> io::Result<()> {
        let region = region()?;
        // The 40m river lies between the corners of a 250m heightfield
        // cell. A coarse quadrilateral has no wet corner at all.
        for point in [
            [750.0, 1000.0],
            [750.0, 1250.0],
            [1000.0, 1000.0],
            [1000.0, 1250.0],
        ] {
            assert!(region.river_stage(point).is_none());
        }
        let left = region.river_patch([3, 4], 128)?;
        let right = region.river_patch([4, 4], 128)?;
        assert!(!left.indices.is_empty() && !right.indices.is_empty());
        let boundary = |patch: &RiverPatch| {
            let mut seam: Vec<_> = patch
                .vertices
                .iter()
                .filter(|p| (p[0] - 1000.0).abs() < 1e-10)
                .map(|p| p[1].to_bits())
                .collect();
            seam.sort_unstable();
            seam.dedup();
            seam
        };
        assert!(!boundary(&left).is_empty());
        assert_eq!(boundary(&left), boundary(&right));
        for patch in [&left, &right] {
            for triangle in patch.indices.chunks_exact(3) {
                let [a, b, c] = [triangle[0], triangle[1], triangle[2]]
                    .map(|index| patch.vertices[index as usize]);
                let area = (b[0] - a[0]).mul_add(c[1] - a[1], -(b[1] - a[1]) * (c[0] - a[0]));
                assert!(area >= 0.0);
            }
            for &[x, y, stage] in &patch.vertices {
                let canonical = region
                    .sample_chart([x, y])
                    .ok_or_else(|| io::Error::other("water vertex outside chart"))?;
                assert_eq!(canonical.weather.water, Some(stage));
                assert!(canonical.weather.bed < stage);
            }
        }
        assert!(region.river_patch([0, 0], 64)?.indices.is_empty());
        assert!(region.river_patch([8, 0], 64).is_err());
        assert!(region.river_patch([0, 0], 0).is_err());
        Ok(())
    }
}
