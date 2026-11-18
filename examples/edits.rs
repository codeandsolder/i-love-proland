//! Sparse object changes: reconstruct the same generated world after loading.
use i_love_proland::{
    AddedObject, CellAddress, CellRect, Distribution, Domain, GeneratedState, Generator,
    GeneratorConfig, Layer, TileKey, Variant, VisibleExtra, WorldEdits,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = Generator::new(
        Domain::Planar { cell_size_m: 8.0 },
        GeneratorConfig {
            world_seed: 2026,
            cells_per_tile: 16,
            ..GeneratorConfig::default()
        },
    )?;
    let trees = Layer::new(
        10,
        0.015,
        Distribution::Poisson,
        vec![Variant {
            id: 900,
            weight: 1.0,
        }],
    )?;
    let original = world
        .tile(TileKey::planar(0, 0), &trees)?
        .into_iter()
        .next()
        .ok_or(i_love_proland::Error::InvalidEdit)?;

    let destination = CellAddress {
        x: original.candidate.cell.x + 64,
        ..original.candidate.cell
    };

    let mut edits = WorldEdits::new();
    edits.modify_generated(&original, destination, "moved original tree".to_owned());
    edits.upsert_added(AddedObject {
        id: 101,
        layer_id: trees.id,
        cell: destination,
        payload: "user-planted tree".to_owned(),
    });

    // Application serialization only needs these sparse, sorted records.
    let saved_generated = edits
        .generated_changes()
        .map(|(id, change)| (id, change.clone()))
        .collect::<Vec<_>>();
    let saved_added = edits.added_objects().cloned().collect::<Vec<_>>();
    let loaded = WorldEdits::from_records(saved_generated, saved_added)?;

    if matches!(loaded.resolve_generated(&original)?, GeneratedState::Hidden) {
        println!("original tree is hidden in its source cell");
    }
    let view = CellRect::new(
        [destination.x, destination.y],
        [destination.x + 1, destination.y + 1],
    )?;
    for extra in loaded.extras_in_cells(destination.face, view) {
        match extra {
            VisibleExtra::Relocated { id, modification } => {
                println!("relocated {id:032x}: {}", modification.payload);
            }
            VisibleExtra::Added(object) => {
                println!("user-added {}: {}", object.id, object.payload);
            }
        }
    }
    Ok(())
}
