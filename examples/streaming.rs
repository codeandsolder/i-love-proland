//! A moving view's changing tile requests never relocate persistent objects.
use i_love_proland::{Distribution, Domain, Generator, GeneratorConfig, Layer, Variant};
use std::collections::{BTreeMap, BTreeSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = Generator::new(
        Domain::Planar { cell_size_m: 8.0 },
        GeneratorConfig {
            world_seed: 2026,
            cells_per_tile: 16,
            ..GeneratorConfig::default()
        },
    )?;
    let forest = Layer::new(
        100,
        0.01,
        Distribution::Stratified,
        vec![
            Variant { id: 1, weight: 4.0 },
            Variant { id: 2, weight: 1.0 },
        ],
    )?;

    let views = [
        ([-150.0, -100.0], [150.0, 100.0]),
        ([-50.0, -100.0], [250.0, 100.0]),
        ([50.0, -100.0], [350.0, 100.0]),
    ];
    let mut seen_world = BTreeMap::new();
    for (min, max) in views {
        let cells = world.planar_cell_rect(min, max)?;
        let tiles = world.tiles_covering_cells(None, cells, 256)?;
        let mut seen_this_view = BTreeSet::new();
        let mut count = 0_usize;
        for tile in tiles {
            for tree in world.tile_with_habitat_in_cells(tile, cells, &forest, |_| 0.65)? {
                let id = tree.candidate.id;
                // Non-overlapping canonical-cell regions mean no object is
                // emitted twice within the same requested view.
                assert!(seen_this_view.insert(id));
                let anchor = tree.candidate.anchor;
                if let Some(previous) = seen_world.insert(id, anchor) {
                    assert_eq!(previous, anchor);
                }
                count += 1;
            }
        }
        println!(
            "view bounds {min:?}..{max:?}: {count} trees, {} seen cumulatively",
            seen_world.len(),
        );
    }
    Ok(())
}
