use crate::error::{Error, Result};

/// Canonical cube-map face, independent of the camera and terrain LOD.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum CubeFace {
    PositiveX = 1,
    NegativeX = 2,
    PositiveY = 3,
    NegativeY = 4,
    PositiveZ = 5,
    NegativeZ = 6,
}

impl CubeFace {
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];

    fn direction(self, u: f64, v: f64) -> [f64; 3] {
        let xyz = match self {
            Self::PositiveX => [1.0, v, -u],
            Self::NegativeX => [-1.0, v, u],
            Self::PositiveY => [u, 1.0, -v],
            Self::NegativeY => [u, -1.0, v],
            Self::PositiveZ => [u, v, 1.0],
            Self::NegativeZ => [-u, v, -1.0],
        };
        let inv = (1.0 + u * u + v * v).sqrt().recip();
        [xyz[0] * inv, xyz[1] * inv, xyz[2] * inv]
    }
}

/// Render-tile batching is independent of canonical grid-cell identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TileKey {
    face: Option<CubeFace>,
    x: i64,
    y: i64,
}

impl TileKey {
    pub const fn planar(x: i64, y: i64) -> Self {
        Self { face: None, x, y }
    }

    pub const fn spherical(face: CubeFace, x: i64, y: i64) -> Self {
        Self {
            face: Some(face),
            x,
            y,
        }
    }

    pub const fn face(self) -> Option<CubeFace> {
        self.face
    }

    pub const fn x(self) -> i64 {
        self.x
    }

    pub const fn y(self) -> i64 {
        self.y
    }
}

/// An immutable horizontal anchor, to be projected onto visual or physics
/// terrain *after* object identity and habitat selection have been resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    Planar { x_m: f64, y_m: f64 },
    Spherical { direction: [f64; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Domain {
    /// An infinite XY plane with fixed canonical square cells.
    Planar { cell_size_m: f64 },
    /// A sphere with six identical face grids, independent of rendering quads.
    ///
    /// A cube-map's area Jacobian varies across the face. Generation uses
    /// exact Jacobian-aware thinning for uniform density per *surface area*.
    Spherical { radius_m: f64, face_cells: u32 },
}

impl Domain {
    pub(crate) fn validate(self, cells_per_tile: u32) -> Result<()> {
        if cells_per_tile == 0 {
            return Err(Error::InvalidConfiguration(
                "cells_per_tile must be positive",
            ));
        }
        match self {
            Self::Planar { cell_size_m } => {
                if !(cell_size_m.is_finite() && cell_size_m > 0.0) {
                    return Err(Error::InvalidConfiguration(
                        "planar cell size must be finite and positive",
                    ));
                }
            }
            Self::Spherical {
                radius_m,
                face_cells,
            } => {
                if !(radius_m.is_finite() && radius_m > 0.0) {
                    return Err(Error::InvalidConfiguration(
                        "sphere radius must be finite and positive",
                    ));
                }
                if face_cells == 0 || face_cells % cells_per_tile != 0 {
                    return Err(Error::InvalidConfiguration(
                        "face_cells must be nonzero and divisible by cells_per_tile",
                    ));
                }
                let step = 2.0 / f64::from(face_cells);
                let max_area = radius_m * radius_m * step * step;
                if !max_area.is_finite() {
                    return Err(Error::InvalidConfiguration(
                        "sphere cell area overflows f64",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn tiles_per_face(self, cells_per_tile: u32) -> Option<u32> {
        match self {
            Self::Planar { .. } => None,
            Self::Spherical { face_cells, .. }
                if cells_per_tile != 0 && face_cells % cells_per_tile == 0 =>
            {
                Some(face_cells / cells_per_tile)
            }
            Self::Spherical { .. } => None,
        }
    }

    pub(crate) fn validate_tile(self, tile: TileKey, cells_per_tile: u32) -> Result<()> {
        match self {
            Self::Planar { .. } if tile.face.is_none() => Ok(()),
            Self::Spherical { face_cells, .. } if tile.face.is_some() => {
                let n = i64::from(face_cells / cells_per_tile);
                if (0..n).contains(&tile.x) && (0..n).contains(&tile.y) {
                    Ok(())
                } else {
                    Err(Error::InvalidTile)
                }
            }
            _ => Err(Error::InvalidTile),
        }
    }
}

/// A stable global lattice cell, which may be queried through multiple tile
/// sizes without changing its seed, position or identity.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceCell {
    pub face: Option<CubeFace>,
    pub x: i64,
    pub y: i64,
    domain: Domain,
}

impl SurfaceCell {
    pub(crate) fn new(domain: Domain, face: Option<CubeFace>, x: i64, y: i64) -> Self {
        Self { face, x, y, domain }
    }

    /// Upper bound of the physical area density under the cell's cube-map
    /// parameterization. This is an exact Jacobian supremum, not a centre
    /// approximation, so thinning produces uniform spherical intensity.
    pub fn upper_area_m2(self) -> f64 {
        match self.domain {
            Domain::Planar { cell_size_m } => cell_size_m * cell_size_m,
            Domain::Spherical {
                radius_m,
                face_cells,
            } => {
                let (u0, u1) = bounds(self.x, face_cells);
                let (v0, v1) = bounds(self.y, face_cells);
                let min2 = closest_sq(u0, u1) + closest_sq(v0, v1);
                let step = 2.0 / f64::from(face_cells);
                radius_m * radius_m * step * step / (1.0 + min2).powf(1.5)
            }
        }
    }

    /// Given parameter-space uniform numbers, return an anchor and the
    /// thinning ratio in [0,1]. For a planar cell the ratio is always 1.
    pub fn sample(self, u: f64, v: f64) -> (Anchor, f64) {
        debug_assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
        match self.domain {
            Domain::Planar { cell_size_m } => (
                Anchor::Planar {
                    x_m: (self.x as f64 + u) * cell_size_m,
                    y_m: (self.y as f64 + v) * cell_size_m,
                },
                1.0,
            ),
            Domain::Spherical { face_cells, .. } => {
                let (u0, u1) = bounds(self.x, face_cells);
                let (v0, v1) = bounds(self.y, face_cells);
                let a = u0 + (u1 - u0) * u;
                let b = v0 + (v1 - v0) * v;
                let min2 = closest_sq(u0, u1) + closest_sq(v0, v1);
                let ratio = ((1.0 + min2) / (1.0 + a * a + b * b)).powf(1.5);
                (
                    Anchor::Spherical {
                        direction: self.face.expect("validated spherical tile").direction(a, b),
                    },
                    ratio.min(1.0),
                )
            }
        }
    }
}

fn bounds(index: i64, face_cells: u32) -> (f64, f64) {
    let lo = -1.0 + 2.0 * index as f64 / f64::from(face_cells);
    let hi = -1.0 + 2.0 * (index + 1) as f64 / f64::from(face_cells);
    (lo, hi)
}

fn closest_sq(lo: f64, hi: f64) -> f64 {
    let nearest = if lo > 0.0 {
        lo
    } else if hi < 0.0 {
        hi
    } else {
        0.0
    };
    nearest * nearest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_sphere_face_is_distinct_and_normalized() {
        let mut points = std::collections::HashSet::new();
        for face in CubeFace::ALL {
            let cell = SurfaceCell::new(
                Domain::Spherical {
                    radius_m: 600_000.0,
                    face_cells: 64,
                },
                Some(face),
                24,
                31,
            );
            let (Anchor::Spherical { direction }, weight) = cell.sample(0.3, 0.7) else {
                panic!("expected sphere");
            };
            assert!((weight > 0.0) && weight <= 1.0);
            let n = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-14);
            points.insert(direction.map(f64::to_bits));
        }
        assert_eq!(points.len(), 6);
    }

    #[test]
    fn cubemap_jacobian_decreases_toward_corners() {
        let domain = Domain::Spherical {
            radius_m: 100.0,
            face_cells: 16,
        };
        let centre = SurfaceCell::new(domain, Some(CubeFace::PositiveZ), 7, 7);
        let corner = SurfaceCell::new(domain, Some(CubeFace::PositiveZ), 0, 0);
        assert!(centre.upper_area_m2() > corner.upper_area_m2());
        assert!(corner.sample(0.8, 0.8).1 <= 1.0);
    }
}
