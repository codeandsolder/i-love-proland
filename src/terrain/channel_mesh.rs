//! On-demand channel-conditioned land and independent river geometry.
//! Both surfaces use one shoreline-conforming triangulation, so a coarser
//! approximation of the solid terrain cannot obscure the river in this mesh.
use super::{Region, RegionSample, WaterKind};
use std::{collections::HashMap, io};

/// A canonical chart coordinate and its terrain/material/water sample.
#[derive(Clone, Copy, Debug)]
pub struct ChannelVertex {
    pub point: [f64; 2],
    pub sample: RegionSample,
}

impl ChannelVertex {
    fn wet(self) -> bool {
        self.sample.water_kind == Some(WaterKind::River)
    }
}

/// Independent indexed land and river triangles sharing a vertex pool.
/// Coordinates are regional west/north metres; bed and stage are radial
/// elevations. Indices are counter-clockwise in chart space.
#[derive(Debug, Default)]
pub struct ChannelPatch {
    pub vertices: Vec<ChannelVertex>,
    pub land_indices: Vec<u32>,
    pub river_indices: Vec<u32>,
}

struct Builder<'a> {
    region: &'a Region,
    nodes: Vec<ChannelVertex>,
    vertex_ids: Vec<Option<u32>>,
    edges: HashMap<[usize; 2], u32>,
    mesh: ChannelPatch,
}

#[derive(Default)]
struct Polygon {
    vertices: [u32; 4],
    count: usize,
}

impl Polygon {
    const fn push(&mut self, id: u32) {
        if self.count > 0 && self.vertices[self.count - 1] == id {
            return;
        }
        self.vertices[self.count] = id;
        self.count += 1;
    }

    fn triangulate(&self, out: &mut Vec<u32>) {
        for index in 2..self.count {
            let triangle = [
                self.vertices[0],
                self.vertices[index - 1],
                self.vertices[index],
            ];
            if triangle[0] != triangle[1]
                && triangle[0] != triangle[2]
                && triangle[1] != triangle[2]
            {
                out.extend(triangle);
            }
        }
    }
}

fn vertex(region: &Region, point: [f64; 2]) -> io::Result<ChannelVertex> {
    let sample = region
        .sample_chart(point)
        .ok_or_else(|| io::Error::other("channel mesh sample outside chart"))?;
    Ok(ChannelVertex { point, sample })
}

impl Builder<'_> {
    fn insert(&mut self, node: ChannelVertex) -> u32 {
        // At most 256^2 cells and three vertices per triangle. Far below u32.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "maximum patch size bounds vertex IDs"
        )]
        let id = self.mesh.vertices.len() as u32;
        self.mesh.vertices.push(node);
        id
    }

    fn node_vertex(&mut self, index: usize) -> u32 {
        if let Some(id) = self.vertex_ids[index] {
            return id;
        }
        let id = self.insert(self.nodes[index]);
        self.vertex_ids[index] = Some(id);
        id
    }

    fn crossing(&mut self, wet: usize, dry: usize) -> io::Result<u32> {
        let key = [wet.min(dry), wet.max(dry)];
        if let Some(&id) = self.edges.get(&key) {
            return Ok(id);
        }
        // Always approach from the wet side so an independently generated
        // neighbor uses exactly the same arithmetic on shared edges.
        let mut inside = self.nodes[wet];
        let mut outside = self.nodes[dry].point;
        for _ in 0..12 {
            let point = std::array::from_fn(|axis| inside.point[axis].midpoint(outside[axis]));
            let sample = vertex(self.region, point)?;
            if sample.wet() {
                inside = sample;
            } else {
                outside = point;
            }
        }
        let id = if inside.point.map(f64::to_bits) == self.nodes[wet].point.map(f64::to_bits) {
            self.node_vertex(wet)
        } else {
            self.insert(inside)
        };
        self.edges.insert(key, id);
        Ok(id)
    }

    fn triangle(&mut self, triangle: [usize; 3]) -> io::Result<()> {
        let mut wet = Polygon::default();
        let mut dry = Polygon::default();
        for (index, &current) in triangle.iter().enumerate() {
            let previous = triangle[(index + 2) % 3];
            let current_wet = self.nodes[current].wet();
            let previous_wet = self.nodes[previous].wet();
            if current_wet != previous_wet {
                let id = if current_wet {
                    self.crossing(current, previous)?
                } else {
                    self.crossing(previous, current)?
                };
                wet.push(id);
                dry.push(id);
            }
            let id = self.node_vertex(current);
            if current_wet {
                wet.push(id);
            } else {
                dry.push(id);
            }
        }
        // Wet ground and water are literally the same triangles, evaluated
        // at different canonical elevations. Dry ground uses the same
        // shoreline endpoints, so there are no internal T-junctions.
        let start = self.mesh.land_indices.len();
        wet.triangulate(&mut self.mesh.land_indices);
        self.mesh
            .river_indices
            .extend_from_slice(&self.mesh.land_indices[start..]);
        dry.triangulate(&mut self.mesh.land_indices);
        Ok(())
    }
}

impl Region {
    /// Tessellate a canonical chart cell with channel-conditioned ground
    /// and independent river water. Callers may omit the land draw, but
    /// rendering the supplied land eliminates mismatched bed triangulation.
    /// A conservative broad-phase miss returns None; other cells return a
    /// complete land partition even if no river water is sampled.
    ///
    /// Use identical subdivisions on adjoining cells for matching edges.
    /// This samples a finite lattice and must be resolved finely enough
    /// to detect dry/wet transitions; it is not a global exact isocontour.
    ///
    /// # Errors
    /// Rejects invalid cell coordinates, non-power-of-two subdivision
    /// counts or counts greater than 256 and invalid chart samples.
    pub fn channel_patch(
        &self,
        cell: [u16; 2],
        subdivisions: u16,
    ) -> io::Result<Option<ChannelPatch>> {
        let chart = self.chart();
        if cell.iter().any(|&v| v >= chart.cells)
            || subdivisions > 256
            || !subdivisions.is_power_of_two()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid channel patch lattice",
            ));
        }
        let min = cell.map(|v| f64::from(v) * chart.spacing_m);
        let max = cell.map(|v| f64::from(v + 1) * chart.spacing_m);
        if !self.channel_candidates(min, max) {
            return Ok(None);
        }
        let side = usize::from(subdivisions) + 1;
        let mut nodes = Vec::with_capacity(side * side);
        let global = cell.map(|v| u32::from(v) * u32::from(subdivisions));
        for row in 0..=subdivisions {
            for column in 0..=subdivisions {
                let point = [global[0] + u32::from(column), global[1] + u32::from(row)]
                    .map(|v| f64::from(v) / f64::from(subdivisions) * chart.spacing_m);
                nodes.push(vertex(self, point)?);
            }
        }
        let mut builder = Builder {
            region: self,
            nodes,
            vertex_ids: vec![None; side * side],
            edges: HashMap::new(),
            mesh: ChannelPatch::default(),
        };
        for row in 0..side - 1 {
            for column in 0..side - 1 {
                let southwest = row * side + column;
                let southeast = southwest + 1;
                let northwest = southwest + side;
                let northeast = northwest + 1;
                builder.triangle([southwest, southeast, northeast])?;
                builder.triangle([southwest, northeast, northwest])?;
            }
        }
        Ok(Some(builder.mesh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{Reach, RegionCell, RegionChart, Section};
    use std::collections::HashSet;

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

    fn assert_partition(region: &Region, patch: &ChannelPatch) -> io::Result<()> {
        let land = patch.land_indices.as_chunks::<3>().0;
        let river = patch.river_indices.as_chunks::<3>().0;
        let mesh_area = |triangle: [u32; 3]| {
            let [a, b, c] = triangle.map(|index| patch.vertices[index as usize].point);
            (b[0] - a[0]).mul_add(c[1] - a[1], -(b[1] - a[1]) * (c[0] - a[0])) * 0.5
        };
        assert!(land.iter().all(|&triangle| mesh_area(triangle) >= 0.0));
        let total = land
            .iter()
            .map(|&triangle| mesh_area(triangle))
            .sum::<f64>();
        assert!((total - 250.0_f64.powi(2)).abs() < 1e-7);
        assert!(!river.is_empty());
        let land_set: HashSet<_> = land.iter().copied().collect();
        for triangle in river {
            assert!(land_set.contains(triangle));
            assert!(triangle.iter().all(|&index| {
                let node = patch.vertices[index as usize];
                node.wet()
                    && node
                        .sample
                        .weather
                        .water
                        .is_some_and(|stage| stage > node.sample.weather.bed)
            }));
        }
        for node in &patch.vertices {
            let canonical = region
                .sample_chart(node.point)
                .ok_or_else(|| io::Error::other("vertex outside chart"))?;
            assert_eq!(canonical.weather.bed, node.sample.weather.bed);
            assert_eq!(canonical.weather.water, node.sample.weather.water);
            assert_eq!(canonical.water_kind, node.sample.water_kind);
        }
        Ok(())
    }

    #[test]
    fn shoreline_partition_keeps_subcell_river_above_its_bed_and_matches_edges() -> io::Result<()> {
        let region = region()?;
        for point in [
            [750.0, 1000.0],
            [750.0, 1250.0],
            [1000.0, 1000.0],
            [1000.0, 1250.0],
        ] {
            assert!(!vertex(&region, point)?.wet());
        }
        let left = region
            .channel_patch([3, 4], 128)?
            .ok_or_else(|| io::Error::other("missing left patch"))?;
        let right = region
            .channel_patch([4, 4], 128)?
            .ok_or_else(|| io::Error::other("missing right patch"))?;
        let boundary = |patch: &ChannelPatch| {
            let mut seam: Vec<_> = patch
                .vertices
                .iter()
                .filter(|node| (node.point[0] - 1000.0).abs() < 1e-10)
                .map(|node| {
                    [
                        node.point[1].to_bits(),
                        node.sample.weather.bed.to_bits(),
                        node.sample.weather.water.unwrap_or_default().to_bits(),
                    ]
                })
                .collect();
            seam.sort_unstable();
            seam.dedup();
            seam
        };
        assert!(!boundary(&left).is_empty());
        assert_eq!(boundary(&left), boundary(&right));
        assert_partition(&region, &left)?;
        assert_partition(&region, &right)?;
        assert!(
            region
                .channel_patch([0, 0], 64)?
                .is_none_or(|patch| patch.river_indices.is_empty())
        );
        assert!(region.channel_patch([8, 0], 64).is_err());
        assert!(region.channel_patch([0, 0], 0).is_err());
        Ok(())
    }
}
