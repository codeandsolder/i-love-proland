//! Deterministic, camera-independent placement of arbitrary world objects.
//!
//! Canonical cell identities and object IDs do not depend on *render* tiles,
//! traversal order, thread count, terrain resolution, or a particular renderer.
//! Use the same generator for trees, rocks, debris, buildings, or vegetation.
//!
//! # Reproducibility contract
//!
//! * `LAYOUT_VERSION` identifies the precise seed/placement algorithm.
//! * Preserve the seed, layer ID, topology, cell resolution, distribution,
//!   and source habitat mask to reproduce a saved scene.
//! * Changing `cells_per_tile` only changes batching, not object identities.
//! * Species/variant *array order* does not affect placement or selection.
//! * Callers own canonical biome/habitat sampling. Do **not** use a changing
//!   camera-dependent terrain LOD to decide whether an object *exists*.
//!   Rendering can project the same stable anchor onto any terrain LOD.
//!
//! No source code or assets from the original Proland project are included.

mod domain;
mod error;
mod generator;
mod random;
mod selection;

pub use domain::{Anchor, CubeFace, Domain, SurfaceCell, TileKey};
pub use error::{Error, Result};
pub use generator::{
    Candidate, Distribution, Generator, GeneratorConfig, Instance, LAYOUT_VERSION, Layer,
};
pub use selection::{Variant, choose_variant};
