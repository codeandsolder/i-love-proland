use crate::domain::{CubeFace, Domain, TileKey};
use crate::error::{Error, Result};
use crate::generator::Generator;
use core::iter::FusedIterator;

/// Half-open rectangle of canonical cells, independent of the current
/// rendering quadtree or batching tile size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellRect {
    min: [i64; 2],
    end: [i64; 2],
}

impl CellRect {
    /// # Errors
    /// The rectangle must have strictly positive extent in both dimensions.
    pub const fn new(min: [i64; 2], end_exclusive: [i64; 2]) -> Result<Self> {
        if min[0] >= end_exclusive[0] || min[1] >= end_exclusive[1] {
            return Err(Error::InvalidBounds);
        }
        Ok(Self {
            min,
            end: end_exclusive,
        })
    }

    #[must_use]
    pub const fn min(self) -> [i64; 2] {
        self.min
    }

    #[must_use]
    pub const fn end_exclusive(self) -> [i64; 2] {
        self.end
    }

    /// Tests membership using the canonical cell, not its rendering tile.
    #[must_use]
    pub const fn contains(self, x: i64, y: i64) -> bool {
        x >= self.min[0] && x < self.end[0] && y >= self.min[1] && y < self.end[1]
    }
}

/// A lazy, deterministic, duplicate-free row-major enumeration of tiles.
///
/// This describes a *cover*: edge tiles can contain canonical cells outside
/// the requested rectangle. Filter accepted candidates by their cell address
/// if an application requires exactly the requested half-open region.
#[derive(Clone, Debug)]
pub struct TileWindow {
    face: Option<CubeFace>,
    first_x: i64,
    next_x: i64,
    next_y: i64,
    end_x: i64,
    remaining: usize,
}

impl Iterator for TileWindow {
    type Item = TileKey;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let key = match self.face {
            Some(face) => TileKey::spherical(face, self.next_x, self.next_y),
            None => TileKey::planar(self.next_x, self.next_y),
        };
        self.remaining -= 1;
        if self.next_x + 1 < self.end_x {
            self.next_x += 1;
        } else {
            self.next_x = self.first_x;
            // Last item does not need an unused row advance.
            if self.remaining != 0 {
                self.next_y += 1;
            }
        }
        Some(key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for TileWindow {}
impl FusedIterator for TileWindow {}

impl Generator {
    /// Enumerate only the tiles intersecting a requested canonical cell
    /// rectangle. Call once per cube face for multi-face planetary views.
    ///
    /// A finite `max_tiles` budget is mandatory, preventing accidental
    /// planet-wide enumeration from an unbounded streaming query.
    ///
    /// # Errors
    /// Returns an error for the wrong face/domain combination, cells outside
    /// a cube face, an invalid rectangle or a budget/coordinate overflow.
    pub fn tiles_covering_cells(
        self,
        face: Option<CubeFace>,
        rect: CellRect,
        max_tiles: usize,
    ) -> Result<TileWindow> {
        match (self.domain(), face) {
            (Domain::Planar { .. }, None) => {}
            (Domain::Spherical { face_cells, .. }, Some(_)) => {
                let limit = i64::from(face_cells);
                if rect.min.iter().any(|&v| v < 0) || rect.end.iter().any(|&v| v > limit) {
                    return Err(Error::InvalidBounds);
                }
            }
            _ => return Err(Error::InvalidTile),
        }

        let edge = i64::from(self.config().cells_per_tile);
        let lower_x = rect.min[0].div_euclid(edge);
        let lower_y = rect.min[1].div_euclid(edge);
        let end_x = (rect.end[0] - 1)
            .div_euclid(edge)
            .checked_add(1)
            .ok_or(Error::IndexOverflow)?;
        let end_y = (rect.end[1] - 1)
            .div_euclid(edge)
            .checked_add(1)
            .ok_or(Error::IndexOverflow)?;
        let nx = u128::try_from(i128::from(end_x) - i128::from(lower_x))
            .map_err(|_| Error::IndexOverflow)?;
        let ny = u128::try_from(i128::from(end_y) - i128::from(lower_y))
            .map_err(|_| Error::IndexOverflow)?;
        let total = nx.checked_mul(ny).ok_or(Error::IndexOverflow)?;
        if total > max_tiles as u128 {
            return Err(Error::TileLimitExceeded);
        }
        let remaining = usize::try_from(total).map_err(|_| Error::TileLimitExceeded)?;
        Ok(TileWindow {
            face,
            first_x: lower_x,
            next_x: lower_x,
            next_y: lower_y,
            end_x,
            remaining,
        })
    }

    /// Convert a half-open planar world-space AABB into a canonical cell rect.
    ///
    /// World-space cell indices must be exactly representable in f64. Very
    /// large worlds should issue canonical integer cell queries instead.
    ///
    /// # Errors
    /// Rejects spherical domains, non-finite or empty bounds, and coordinates
    /// beyond exact f64 cell addressing.
    pub fn planar_cell_rect(self, min_m: [f64; 2], max_m: [f64; 2]) -> Result<CellRect> {
        const MAX_EXACT_CELL: f64 = 4_503_599_627_370_496.0;
        let Domain::Planar { cell_size_m } = self.domain() else {
            return Err(Error::InvalidTile);
        };
        let mut lower = [0_i64; 2];
        let mut upper = [0_i64; 2];
        for axis in 0..2 {
            if !min_m[axis].is_finite() || !max_m[axis].is_finite() || min_m[axis] >= max_m[axis] {
                return Err(Error::InvalidBounds);
            }
            let first = (min_m[axis] / cell_size_m).floor();
            let after = (max_m[axis] / cell_size_m).ceil();
            if !first.is_finite()
                || !after.is_finite()
                || first.abs() > MAX_EXACT_CELL
                || after.abs() > MAX_EXACT_CELL
            {
                return Err(Error::IndexOverflow);
            }
            // These are checked exact integer-valued floats within +/-2^52.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "rounded values are validated exact integers within 2^52"
            )]
            {
                lower[axis] = first as i64;
                upper[axis] = after as i64;
            }
        }
        CellRect::new(lower, upper)
    }

    /// Return a lazy, bounded tile window covering a planar AABB.
    ///
    /// # Errors
    /// Rejects invalid bounds, an unsupported spherical domain, or a tile
    /// count beyond the requested budget.
    pub fn tiles_covering_planar_aabb(
        self,
        min_m: [f64; 2],
        max_m: [f64; 2],
        max_tiles: usize,
    ) -> Result<TileWindow> {
        self.tiles_covering_cells(None, self.planar_cell_rect(min_m, max_m)?, max_tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, GeneratorConfig, Layer, Variant};
    use std::collections::BTreeSet;

    fn flat(edge: u32) -> Generator {
        Generator::new(
            Domain::Planar { cell_size_m: 10.0 },
            GeneratorConfig {
                world_seed: 87,
                cells_per_tile: edge,
                ..GeneratorConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn negative_boundaries_are_half_open_and_duplicate_free() {
        let rect = CellRect::new([-32, -17], [-16, 1]).unwrap();
        let tiles: Vec<_> = flat(16)
            .tiles_covering_cells(None, rect, 12)
            .unwrap()
            .collect();
        assert_eq!(
            tiles,
            vec![
                TileKey::planar(-2, -2),
                TileKey::planar(-2, -1),
                TileKey::planar(-2, 0),
            ]
        );
        assert!(rect.contains(-32, -17));
        assert!(!rect.contains(-16, -17));
        assert!(!rect.contains(-32, 1));
        assert_eq!(tiles.len(), tiles.iter().collect::<BTreeSet<_>>().len());
    }

    #[test]
    fn planar_aabb_at_exact_grid_edges_does_not_load_neighbours() {
        let generator = flat(16);
        let tiles: Vec<_> = generator
            .tiles_covering_planar_aabb([-160.0, -160.0], [0.0, 160.0], 8)
            .unwrap()
            .collect();
        assert_eq!(tiles, [TileKey::planar(-1, -1), TileKey::planar(-1, 0)]);
        assert_eq!(
            generator
                .planar_cell_rect([-160.0, -160.0], [0.0, 160.0])
                .unwrap(),
            CellRect::new([-16, -16], [0, 16]).unwrap(),
        );
    }

    #[test]
    fn off_grid_aabb_includes_minimal_overlap() {
        let tiles: Vec<_> = flat(16)
            .tiles_covering_planar_aabb([-161.0, -1.0], [0.5, 160.1], 100)
            .unwrap()
            .collect();
        assert_eq!(tiles.len(), 9);
        assert_eq!(tiles.first(), Some(&TileKey::planar(-2, -1)));
        assert_eq!(tiles.last(), Some(&TileKey::planar(0, 1)));
    }

    #[test]
    fn changed_render_tile_size_preserves_generated_objects_in_cell_rect() {
        let rect = CellRect::new([-32, -32], [32, 32]).unwrap();
        let trees = Layer::new(
            9,
            0.01,
            Distribution::Poisson,
            vec![Variant {
                id: 100,
                weight: 1.0,
            }],
        )
        .unwrap();
        let collect = |generator: Generator| {
            let mut output = Vec::new();
            for tile in generator.tiles_covering_cells(None, rect, 1024).unwrap() {
                output.extend(
                    generator
                        .tile_with_habitat_in_cells(tile, rect, &trees, |_| 1.0)
                        .unwrap(),
                );
            }
            output.sort_by_key(|v| v.candidate.id);
            output
        };
        assert_eq!(collect(flat(16)), collect(flat(8)));
    }

    #[test]
    fn clipped_tile_matches_full_tile_and_skips_outside_habitat() {
        let generator = flat(16);
        let trees = Layer::new(
            9,
            0.01,
            Distribution::Poisson,
            vec![Variant {
                id: 100,
                weight: 1.0,
            }],
        )
        .unwrap();
        let tile = TileKey::planar(0, 0);
        let rect = CellRect::new([3, 7], [14, 11]).unwrap();
        let full = generator.tile(tile, &trees).unwrap();
        let matching = full
            .into_iter()
            .filter(|item| rect.contains(item.candidate.cell.x, item.candidate.cell.y))
            .collect::<Vec<_>>();
        let clipped = generator
            .tile_with_habitat_in_cells(tile, rect, &trees, |candidate| {
                assert!(rect.contains(candidate.cell.x, candidate.cell.y));
                1.0
            })
            .unwrap();
        assert_eq!(matching, clipped);
        assert!(
            generator
                .tile_with_habitat_in_cells(TileKey::planar(3, 3), rect, &trees, |_| panic!(
                    "outside tile"
                ))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn sphere_queries_check_faces_and_cell_bounds() {
        let generator = Generator::new(
            Domain::Spherical {
                radius_m: 600_000.0,
                face_cells: 64,
            },
            GeneratorConfig {
                cells_per_tile: 8,
                ..GeneratorConfig::default()
            },
        )
        .unwrap();
        let rect = CellRect::new([8, 8], [16, 17]).unwrap();
        let tiles: Vec<_> = generator
            .tiles_covering_cells(Some(CubeFace::NegativeX), rect, 8)
            .unwrap()
            .collect();
        assert_eq!(
            tiles,
            [
                TileKey::spherical(CubeFace::NegativeX, 1, 1),
                TileKey::spherical(CubeFace::NegativeX, 1, 2)
            ]
        );
        assert_eq!(
            generator.tiles_covering_cells(None, rect, 8).err(),
            Some(Error::InvalidTile)
        );
        let beyond = CellRect::new([63, 63], [65, 65]).unwrap();
        assert_eq!(
            generator
                .tiles_covering_cells(Some(CubeFace::PositiveZ), beyond, 100)
                .err(),
            Some(Error::InvalidBounds)
        );
    }

    #[test]
    fn arbitrary_signed_rectangles_cover_exactly_the_intersecting_tiles() {
        for cells_per_tile in [1_u32, 2, 3, 7, 16] {
            let generator = flat(cells_per_tile);
            let edge = i64::from(cells_per_tile);
            for i in 0..180_i64 {
                let low_x = (37 * i).rem_euclid(121) - 60;
                let low_y = (91 * i).rem_euclid(97) - 48;
                let high_x = low_x + (13 * i).rem_euclid(23) + 1;
                let high_y = low_y + (17 * i).rem_euclid(19) + 1;
                let rect = CellRect::new([low_x, low_y], [high_x, high_y]).unwrap();
                let expected: BTreeSet<_> = (low_y..high_y)
                    .flat_map(|y| {
                        (low_x..high_x)
                            .map(move |x| TileKey::planar(x.div_euclid(edge), y.div_euclid(edge)))
                    })
                    .collect();
                let actual: Vec<_> = generator
                    .tiles_covering_cells(None, rect, 1024)
                    .unwrap()
                    .collect();
                let unique: BTreeSet<_> = actual.iter().copied().collect();
                assert_eq!(unique, expected, "edge={edge}, rect={rect:?}");
                assert_eq!(actual.len(), unique.len());
            }
        }
    }

    #[test]
    fn huge_or_invalid_requests_fail_before_enumeration() {
        let generator = flat(16);
        assert_eq!(
            generator
                .tiles_covering_cells(
                    None,
                    CellRect::new([0, 0], [i64::MAX, i64::MAX]).unwrap(),
                    10,
                )
                .err(),
            Some(Error::TileLimitExceeded)
        );
        assert_eq!(
            generator
                .tiles_covering_planar_aabb([f64::NAN, 0.0], [100.0, 100.0], 20)
                .err(),
            Some(Error::InvalidBounds)
        );
        assert_eq!(
            generator
                .tiles_covering_planar_aabb([0.0, 0.0], [1e19, 1e19], 20)
                .err(),
            Some(Error::IndexOverflow)
        );
        assert_eq!(
            generator
                .tiles_covering_cells(
                    Some(CubeFace::PositiveZ),
                    CellRect::new([0, 0], [1, 1]).unwrap(),
                    1,
                )
                .err(),
            Some(Error::InvalidTile)
        );
    }

    #[test]
    fn iterator_is_exact_sized_and_fused() {
        let rect = CellRect::new([0, 0], [33, 17]).unwrap();
        let mut tiles = flat(16).tiles_covering_cells(None, rect, 6).unwrap();
        assert_eq!(tiles.len(), 6);
        assert_eq!(tiles.size_hint(), (6, Some(6)));
        for remaining in (0..6).rev() {
            assert!(tiles.next().is_some());
            assert_eq!(tiles.len(), remaining);
        }
        assert_eq!(tiles.next(), None);
        assert_eq!(tiles.next(), None);
    }
}
