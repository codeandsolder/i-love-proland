use crate::domain::{Anchor, Domain, SurfaceCell, TileKey};
use crate::error::{Error, Result};
use crate::random::{Rng, hash_words};
use crate::selection::{Variant, choose_variant_validated};
use std::collections::HashSet;

/// Freeze this version for persistent worlds. If the PRNG, placement or
/// mapping changes incompatibly, introduce a *new* layout version explicitly.
pub const LAYOUT_VERSION: u32 = 1;

/// Cell addressing is never based on the currently visible terrain tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CellAddress {
    pub face: Option<crate::domain::CubeFace>,
    pub x: i64,
    pub y: i64,
}

/// Placement statistics are independent of all variant/model details.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Distribution {
    /// Area-correct, independent Poisson point process. Supports arbitrary
    /// intensity and uniform surface density even across cube-sphere faces.
    Poisson,
    /// At most one point per canonical grid cell. A centre-stratified
    /// jittered lattice avoids clumps/overlap, at the expense of no longer
    /// being an exact uniform independent Poisson process.
    ///
    /// The physical maximum density depends on the worst-case cell area.
    Stratified,
}

/// Independent stochastic layer: tree, rock, debris and structure populations
/// never consume one another's PRNG streams.
#[derive(Clone, Debug)]
pub struct Layer {
    pub id: u64,
    pub density_per_m2: f64,
    pub distribution: Distribution,
    variants: Vec<Variant>,
}

impl Layer {
    /// # Errors
    /// Rejects invalid densities, duplicate model IDs and invalid weights.
    pub fn new(
        id: u64,
        density_per_m2: f64,
        distribution: Distribution,
        variants: Vec<Variant>,
    ) -> Result<Self> {
        if !density_per_m2.is_finite() || density_per_m2 <= 0.0 {
            return Err(Error::InvalidConfiguration(
                "layer density must be finite and positive",
            ));
        }
        if variants.is_empty() {
            return Err(Error::InvalidConfiguration(
                "at least one variant is required",
            ));
        }
        let mut seen = HashSet::with_capacity(variants.len());
        for v in &variants {
            if !v.weight.is_finite() || v.weight <= 0.0 {
                return Err(Error::InvalidConfiguration(
                    "variant weight must be finite and positive",
                ));
            }
            if !seen.insert(v.id) {
                return Err(Error::InvalidConfiguration(
                    "variant IDs must be unique in a layer",
                ));
            }
        }
        Ok(Self {
            id,
            density_per_m2,
            distribution,
            variants,
        })
    }

    #[must_use]
    pub fn variants(&self) -> &[Variant] {
        &self.variants
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratorConfig {
    pub world_seed: u64,
    pub cells_per_tile: u32,
    /// Explicit resource budget. Exceeding it returns an error instead of
    /// silently dropping objects. Raise this when generating dense grass.
    pub max_proposals_per_tile: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            world_seed: 0,
            cells_per_tile: 16,
            max_proposals_per_tile: 1_000_000,
        }
    }
}

/// An object before variant and habitat selection. Coordinates and identity
/// are immutable; the caller can project onto any terrain LOD later.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub id: u128,
    pub layer_id: u64,
    pub cell: CellAddress,
    pub serial: u64,
    /// Exact canonical sub-cell fractions, retaining precision even when a
    /// planet-scale renderer cannot store huge absolute XY values precisely.
    pub local_uv: [f64; 2],
    pub anchor: Anchor,
    /// Stable fraction in [0,1) for biome/density/thinning acceptance.
    pub habitat_mark: f64,
    /// Independent stable fractions; map to yaw, size, color, and animation.
    pub yaw_turns: f64,
    pub size_unit: f64,
    pub appearance_unit: f64,
}

/// Accepted candidate with a stable, order-independent model/archetype ID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    pub candidate: Candidate,
    pub variant_id: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Generator {
    domain: Domain,
    config: GeneratorConfig,
}

impl Generator {
    /// # Errors
    /// Rejects invalid topologies, cell sizes and zero candidate budgets.
    pub fn new(domain: Domain, config: GeneratorConfig) -> Result<Self> {
        domain.validate(config.cells_per_tile)?;
        if config.max_proposals_per_tile == 0 {
            return Err(Error::InvalidConfiguration(
                "max_proposals_per_tile must be nonzero",
            ));
        }
        let Domain::Planar { cell_size_m } = domain else {
            return Ok(Self { domain, config });
        };
        if !(cell_size_m * cell_size_m).is_finite() {
            return Err(Error::InvalidConfiguration(
                "planar cell area overflows f64",
            ));
        }
        Ok(Self { domain, config })
    }

    #[must_use]
    pub const fn domain(self) -> Domain {
        self.domain
    }

    #[must_use]
    pub const fn config(self) -> GeneratorConfig {
        self.config
    }

    /// Generate a tile without any habitat filtering. Order and placement do
    /// not depend on camera, thread scheduling or neighbouring tile requests.
    /// # Errors
    /// Rejects invalid tile addresses or exhausted candidate budgets.
    pub fn tile(self, tile: TileKey, layer: &Layer) -> Result<Vec<Instance>> {
        self.tile_with_habitat(tile, layer, |_| 1.0)
    }

    /// A pure acceptance function returns a deterministic fraction [0,1]
    /// based on canonical habitat data. Visual mesh LOD must not decide
    /// object *existence*; only project the fixed object anchor.
    ///
    /// # Errors
    /// Rejects invalid tiles, invalid habitat fractions, excessive densities
    /// for stratification, and proposal budget or coordinate overflow.
    pub fn tile_with_habitat<F>(
        self,
        tile: TileKey,
        layer: &Layer,
        habitat: F,
    ) -> Result<Vec<Instance>>
    where
        F: Fn(&Candidate) -> f64,
    {
        self.tile_in_region(tile, None, layer, &habitat)
    }

    /// Generate only cells belonging to the requested half-open canonical
    /// rectangle. Objects outside the rectangle are never instantiated or
    /// passed to the habitat callback, even on partially overlapping tiles.
    ///
    /// This is equivalent to generating the entire tile and filtering by
    /// canonical cell address; it does not change random streams or IDs.
    ///
    /// # Errors
    /// Returns an error for invalid tiles, habitat values, density or
    /// resource budgets, just like the full tile generation method.
    pub fn tile_with_habitat_in_cells<F>(
        self,
        tile: TileKey,
        rect: crate::stream::CellRect,
        layer: &Layer,
        habitat: F,
    ) -> Result<Vec<Instance>>
    where
        F: Fn(&Candidate) -> f64,
    {
        self.tile_in_region(tile, Some(rect), layer, &habitat)
    }

    fn tile_in_region<F>(
        self,
        tile: TileKey,
        region: Option<crate::stream::CellRect>,
        layer: &Layer,
        habitat: &F,
    ) -> Result<Vec<Instance>>
    where
        F: Fn(&Candidate) -> f64,
    {
        self.domain
            .validate_tile(tile, self.config.cells_per_tile)?;
        let edge = i64::from(self.config.cells_per_tile);
        let origin_x = tile.x().checked_mul(edge).ok_or(Error::IndexOverflow)?;
        let origin_y = tile.y().checked_mul(edge).ok_or(Error::IndexOverflow)?;
        let end_x = origin_x.checked_add(edge).ok_or(Error::IndexOverflow)?;
        let end_y = origin_y.checked_add(edge).ok_or(Error::IndexOverflow)?;
        let (start_x, stop_x, start_y, stop_y) =
            region.map_or((origin_x, end_x, origin_y, end_y), |rect| {
                let min = rect.min();
                let end = rect.end_exclusive();
                (
                    origin_x.max(min[0]),
                    end_x.min(end[0]),
                    origin_y.max(min[1]),
                    end_y.min(end[1]),
                )
            });
        let mut out = Vec::new();
        let mut proposals = 0_usize;
        for cell_y in start_y..stop_y {
            for cell_x in start_x..stop_x {
                let cell = SurfaceCell::new(self.domain, tile.face(), cell_x, cell_y);
                self.collect_cell(cell, layer, habitat, &mut out, &mut proposals)?;
            }
        }
        Ok(out)
    }

    /// Canonical cells, not visible render tiles, seed every placement.
    #[expect(
        clippy::suboptimal_flops,
        reason = "layout v1 freezes the jitter arithmetic order"
    )]
    fn collect_cell<F>(
        self,
        cell: SurfaceCell,
        layer: &Layer,
        habitat: &F,
        out: &mut Vec<Instance>,
        proposals: &mut usize,
    ) -> Result<()>
    where
        F: Fn(&Candidate) -> f64,
    {
        let address = CellAddress {
            face: cell.face,
            x: cell.x,
            y: cell.y,
        };
        let seed = hash_words(&[
            0xd1d5_4cf6_1fe7_2b63,
            u64::from(LAYOUT_VERSION),
            self.config.world_seed,
            layer.id,
            u64::from(cell.face.map_or(0, |face| face as u8)),
            u64::from_ne_bytes(cell.x.to_ne_bytes()),
            u64::from_ne_bytes(cell.y.to_ne_bytes()),
        ]);
        let mut random = Rng::new(seed);
        let intensity = cell.upper_area_m2() * layer.density_per_m2;
        if !intensity.is_finite() {
            return Err(Error::InvalidConfiguration(
                "cell expected population overflows f64",
            ));
        }

        let mut emit = |serial: u64, local_u: f64, local_v: f64, random: &mut Rng| {
            let (anchor, jacobian_fraction) = cell.sample(local_u, local_v)?;
            if random.unit() > jacobian_fraction {
                return Ok(());
            }
            let id = (u128::from(hash_words(&[seed, serial, 0xb441_cac6_cafe_bab1])) << 64)
                | u128::from(hash_words(&[seed, serial, 0x7265_6e64_6572_6572]));
            let candidate = Candidate {
                id,
                layer_id: layer.id,
                cell: address,
                serial,
                local_uv: [local_u, local_v],
                anchor,
                habitat_mark: random.unit(),
                yaw_turns: random.unit(),
                size_unit: random.unit(),
                appearance_unit: random.unit(),
            };
            let fraction = habitat(&candidate);
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
                return Err(Error::InvalidHabitat);
            }
            if candidate.habitat_mark < fraction {
                out.push(Instance {
                    variant_id: choose_variant_validated(id, layer.variants()),
                    candidate,
                });
            }
            Ok::<(), Error>(())
        };

        match layer.distribution {
            Distribution::Poisson => {
                let mut distance = 0.0_f64;
                let mut serial = 0_u64;
                loop {
                    let step = -random.unit().ln();
                    let next = distance + step;
                    if next <= distance {
                        return Err(Error::CandidateLimitExceeded);
                    }
                    distance = next;
                    if distance > intensity {
                        break;
                    }
                    *proposals = proposals
                        .checked_add(1)
                        .ok_or(Error::CandidateLimitExceeded)?;
                    if *proposals > self.config.max_proposals_per_tile {
                        return Err(Error::CandidateLimitExceeded);
                    }
                    let u = random.unit();
                    let v = random.unit();
                    emit(serial, u, v, &mut random)?;
                    serial = serial.checked_add(1).ok_or(Error::CandidateLimitExceeded)?;
                }
            }
            Distribution::Stratified => {
                if intensity > 1.0 {
                    return Err(Error::DensityExceedsStratification);
                }
                if random.unit() < intensity {
                    *proposals = proposals
                        .checked_add(1)
                        .ok_or(Error::CandidateLimitExceeded)?;
                    if *proposals > self.config.max_proposals_per_tile {
                        return Err(Error::CandidateLimitExceeded);
                    }
                    let u = 0.25 + 0.5 * random.unit();
                    let v = 0.25 + 0.5 * random.unit();
                    emit(0, u, v, &mut random)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::CubeFace;

    fn trees() -> Layer {
        Layer::new(
            42,
            0.01,
            Distribution::Poisson,
            vec![
                Variant {
                    id: 10,
                    weight: 3.0,
                },
                Variant {
                    id: 11,
                    weight: 1.0,
                },
            ],
        )
        .unwrap()
    }

    fn plane(tile_cells: u32) -> Generator {
        Generator::new(
            Domain::Planar { cell_size_m: 10.0 },
            GeneratorConfig {
                world_seed: 123,
                cells_per_tile: tile_cells,
                ..GeneratorConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn repeated_generations_are_bitwise_stable() {
        let generator = plane(8);
        let tile = TileKey::planar(-17, 48);
        assert_eq!(
            generator.tile(tile, &trees()).unwrap(),
            generator.tile(tile, &trees()).unwrap()
        );
    }

    #[test]
    fn tile_granularity_does_not_change_identity_or_positions() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for y in -2..2 {
            for x in -2..2 {
                a.extend(plane(16).tile(TileKey::planar(x, y), &trees()).unwrap());
            }
        }
        for y in -4..4 {
            for x in -4..4 {
                b.extend(plane(8).tile(TileKey::planar(x, y), &trees()).unwrap());
            }
        }
        let by_id = |items: &mut Vec<Instance>| items.sort_by_key(|v| v.candidate.id);
        by_id(&mut a);
        by_id(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn order_of_tiles_does_not_change_instances() {
        let generator = plane(16);
        let keys = [
            TileKey::planar(1, -3),
            TileKey::planar(-7, 1),
            TileKey::planar(0, 0),
            TileKey::planar(-1, 5),
        ];
        let mut normal = Vec::new();
        for tile in keys {
            normal.extend(generator.tile(tile, &trees()).unwrap());
        }
        let mut reversed = Vec::new();
        for tile in keys.into_iter().rev() {
            reversed.extend(generator.tile(tile, &trees()).unwrap());
        }
        normal.sort_by_key(|v| v.candidate.id);
        reversed.sort_by_key(|v| v.candidate.id);
        assert_eq!(normal, reversed);
        let unique = normal
            .iter()
            .map(|v| v.candidate.id)
            .collect::<HashSet<_>>();
        assert_eq!(unique.len(), normal.len());
    }

    #[test]
    fn changing_habitat_only_filters_stable_objects() {
        let generator = plane(16);
        let tile = TileKey::planar(4, 8);
        let full = generator.tile(tile, &trees()).unwrap();
        let filtered = generator
            .tile_with_habitat(tile, &trees(), |_| 0.35)
            .unwrap();
        let full_ids = full.iter().map(|x| x.candidate.id).collect::<HashSet<_>>();
        assert!(filtered.iter().all(|x| full_ids.contains(&x.candidate.id)));
        assert!(filtered.len() < full.len());
    }

    #[test]
    fn layers_are_independent_and_variants_are_order_independent() {
        let generator = plane(16);
        let tile = TileKey::planar(2, 3);
        let a = generator.tile(tile, &trees()).unwrap();
        let shrub = Layer::new(
            99,
            0.2,
            Distribution::Poisson,
            vec![Variant {
                id: 55,
                weight: 1.0,
            }],
        )
        .unwrap();
        let _others = generator.tile(tile, &shrub).unwrap();
        assert_eq!(a, generator.tile(tile, &trees()).unwrap());
        let mut reversed = trees();
        reversed.variants.reverse();
        assert_eq!(a, generator.tile(tile, &reversed).unwrap());
    }

    #[test]
    fn planar_poisson_expected_density_is_reasonable() {
        let mut count = 0;
        for y in -10..10 {
            for x in -10..10 {
                count += plane(8)
                    .tile(TileKey::planar(x, y), &trees())
                    .unwrap()
                    .len();
            }
        }
        // Area 400 * (8*10)^2 = 2,560,000 m², expectation 25,600 trees.
        assert!((24_000..27_200).contains(&count), "count={count}");
    }

    #[test]
    fn spherical_sampling_is_uniform_per_surface_area() {
        let sphere = Generator::new(
            Domain::Spherical {
                radius_m: 100.0,
                face_cells: 32,
            },
            GeneratorConfig {
                world_seed: 10,
                cells_per_tile: 8,
                ..GeneratorConfig::default()
            },
        )
        .unwrap();
        let layer = Layer::new(
            7,
            0.3,
            Distribution::Poisson,
            vec![Variant { id: 1, weight: 1.0 }],
        )
        .unwrap();
        let mut bins = [0_usize; 5];
        let mut ids = HashSet::new();
        for face in CubeFace::ALL {
            for y in 0..4 {
                for x in 0..4 {
                    for item in sphere.tile(TileKey::spherical(face, x, y), &layer).unwrap() {
                        assert!(ids.insert(item.candidate.id));
                        let Anchor::Spherical { direction } = item.candidate.anchor else {
                            panic!("not spherical");
                        };
                        let latitude_bin = ((direction[1] + 1.0) * 2.5).floor() as usize;
                        bins[latitude_bin.min(4)] += 1;
                    }
                }
            }
        }
        let expected = std::f64::consts::PI * 4.0 * 100.0_f64.powi(2) * 0.3 / 5.0;
        for (i, actual) in bins.into_iter().enumerate() {
            assert!(
                (actual as f64 - expected).abs() < expected * 0.07,
                "bin {i}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn stratified_guarantees_no_overcrowded_cells_and_separate_ids() {
        let layer = Layer::new(
            9,
            0.009,
            Distribution::Stratified,
            vec![Variant {
                id: 12,
                weight: 1.0,
            }],
        )
        .unwrap();
        let generator = plane(16);
        let mut ids = HashSet::new();
        for y in -2..2 {
            for x in -2..2 {
                let items = generator.tile(TileKey::planar(x, y), &layer).unwrap();
                for item in items {
                    assert!(ids.insert(item.candidate.id));
                    let Anchor::Planar { x_m, y_m } = item.candidate.anchor else {
                        panic!("not planar");
                    };
                    let local_x = x_m / 10.0 - item.candidate.cell.x as f64;
                    let local_y = y_m / 10.0 - item.candidate.cell.y as f64;
                    assert!((0.25..0.75).contains(&local_x));
                    assert!((0.25..0.75).contains(&local_y));
                }
            }
        }
    }

    #[test]
    fn layout_v1_golden_spherical_placement() {
        // Frozen vectors from the first public v1 generator. Refactors must
        // not silently relocate persistent objects or change model selection.
        let scene = Generator::new(
            Domain::Spherical {
                radius_m: 600_000.0,
                face_cells: 1 << 18,
            },
            GeneratorConfig {
                world_seed: 0x5ec0_ffee,
                cells_per_tile: 16,
                ..GeneratorConfig::default()
            },
        )
        .unwrap();
        let oak = Layer::new(
            0x004f_414b,
            0.015,
            Distribution::Stratified,
            vec![
                Variant {
                    id: 1001,
                    weight: 3.0,
                },
                Variant {
                    id: 1002,
                    weight: 1.0,
                },
            ],
        )
        .unwrap();
        let instances = scene
            .tile_with_habitat(
                TileKey::spherical(CubeFace::PositiveZ, 65_000 / 16, 66_000 / 16),
                &oak,
                |candidate| if candidate.cell.x % 7 == 0 { 0.2 } else { 0.9 },
            )
            .unwrap();
        assert_eq!(instances.len(), 42);
        let first = instances.first().unwrap();
        assert_eq!(
            first.candidate.id,
            154533613205574063414866829586011445627_u128
        );
        assert_eq!(first.variant_id, 1002);
        assert_eq!(
            first.candidate.anchor,
            Anchor::Spherical {
                direction: [
                    -0.4115419164003406,
                    -0.40527162064537636,
                    0.8163260160897758
                ],
            }
        );
    }

    #[test]
    fn sphere_retiling_preserves_every_original_object() {
        let domain = Domain::Spherical {
            radius_m: 600.0,
            face_cells: 64,
        };
        let config = |cells_per_tile| GeneratorConfig {
            world_seed: 0x6793_b7e1_4a2d_57c0,
            cells_per_tile,
            ..GeneratorConfig::default()
        };
        let coarse = Generator::new(domain, config(16)).unwrap();
        let fine = Generator::new(domain, config(8)).unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for face in CubeFace::ALL {
            for y in 0..4 {
                for x in 0..4 {
                    a.extend(
                        coarse
                            .tile(TileKey::spherical(face, x, y), &trees())
                            .unwrap(),
                    );
                }
            }
            for y in 0..8 {
                for x in 0..8 {
                    b.extend(fine.tile(TileKey::spherical(face, x, y), &trees()).unwrap());
                }
            }
        }
        a.sort_by_key(|item| item.candidate.id);
        b.sort_by_key(|item| item.candidate.id);
        assert_eq!(a, b);
        assert!(a.len() > 10_000);
    }

    #[test]
    fn candidate_budget_is_an_error_not_truncation() {
        let generator = Generator::new(
            Domain::Planar { cell_size_m: 10.0 },
            GeneratorConfig {
                max_proposals_per_tile: 2,
                ..GeneratorConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            generator.tile(TileKey::planar(0, 0), &trees()),
            Err(Error::CandidateLimitExceeded)
        );
    }

    #[test]
    fn persisted_ids_are_unaffected_by_new_kinds() {
        let generator = plane(16);
        let tile = TileKey::planar(0, 0);
        let tree_instances = generator.tile(tile, &trees()).unwrap();
        let other = Layer::new(
            0x424c_4447,
            0.001,
            Distribution::Poisson,
            vec![Variant {
                id: 912,
                weight: 1.0,
            }],
        )
        .unwrap();
        let other_instances = generator.tile(tile, &other).unwrap();
        let unique = tree_instances
            .iter()
            .chain(other_instances.iter())
            .map(|o| o.candidate.id)
            .collect::<HashSet<_>>();
        assert_eq!(unique.len(), tree_instances.len() + other_instances.len());
        assert_eq!(tree_instances, generator.tile(tile, &trees()).unwrap());
    }

    #[test]
    fn validation_rejects_mismatched_faces_and_densities() {
        let generator = plane(16);
        let tree_layer = trees();
        assert_eq!(
            generator.tile(TileKey::spherical(CubeFace::PositiveZ, 0, 0), &tree_layer),
            Err(Error::InvalidTile)
        );
        assert!(
            Generator::new(
                Domain::Spherical {
                    radius_m: 1.0,
                    face_cells: 15
                },
                GeneratorConfig::default(),
            )
            .is_err()
        );
        assert!(
            Layer::new(
                1,
                f64::NAN,
                Distribution::Poisson,
                vec![Variant { id: 1, weight: 1.0 }]
            )
            .is_err()
        );
        assert!(
            generator
                .tile_with_habitat(TileKey::planar(0, 0), &tree_layer, |_| -0.1)
                .is_err()
        );
        let dense = Layer::new(
            1,
            1.0,
            Distribution::Stratified,
            vec![Variant { id: 1, weight: 1.0 }],
        )
        .unwrap();
        assert_eq!(
            generator.tile(TileKey::planar(0, 0), &dense),
            Err(Error::DensityExceedsStratification)
        );
    }
}
