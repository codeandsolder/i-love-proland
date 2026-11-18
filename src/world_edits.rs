//! Sparse, type-independent edits to a deterministic generated world.
//!
//! Store only changes: generated objects are reconstructed from their stable
//! IDs; moved objects and user additions are independently indexed by current
//! canonical cell so they remain visible when the source tile is unloaded.

use crate::domain::CubeFace;
use crate::error::{Error, Result};
use crate::generator::{CellAddress, Instance};
use crate::stream::CellRect;
use std::collections::{BTreeMap, BTreeSet};

/// Generated and user-added IDs have distinct namespaces even when their raw
/// 128-bit integers happen to be equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ObjectKey {
    Generated(u128),
    Added(u128),
}

/// A changed generated object, including the authoritative cell it originated
/// in and the cell it currently occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Modification<T> {
    pub layer_id: u64,
    pub source: CellAddress,
    pub destination: CellAddress,
    pub payload: T,
}

/// A removed or modified object from deterministic source generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeneratedChange<T> {
    Removed,
    Modified(Modification<T>),
}

/// A fully user-authored object, independent of procedural generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedObject<T> {
    pub id: u128,
    pub layer_id: u64,
    pub cell: CellAddress,
    pub payload: T,
}

/// Result of applying the mutation overlay to a generated object.
pub enum GeneratedState<'a, T> {
    Original(&'a Instance),
    Modified {
        original: &'a Instance,
        payload: &'a T,
    },
    Hidden,
}

/// Non-generated visual records returned by a current-cell query.
pub enum VisibleExtra<'a, T> {
    /// A generated object moved away from its original canonical tile.
    Relocated {
        id: u128,
        modification: &'a Modification<T>,
    },
    Added(&'a AddedObject<T>),
}

/// A durable, sparse mutation overlay, with a secondary spatial index.
///
/// The canonical procedural layer is never copied. Persist records from
/// `generated_changes()` and `added_objects()` with the application's chosen
/// payload serializer and layout-version metadata. Rebuild the spatial index
/// with `from_records()` after loading a saved world.
#[derive(Clone, Debug)]
pub struct WorldEdits<T> {
    generated: BTreeMap<u128, GeneratedChange<T>>,
    added: BTreeMap<u128, AddedObject<T>>,
    by_x: BTreeMap<(Option<CubeFace>, i64), BTreeMap<i64, BTreeSet<ObjectKey>>>,
}

impl<T> Default for WorldEdits<T> {
    fn default() -> Self {
        Self {
            generated: BTreeMap::new(),
            added: BTreeMap::new(),
            by_x: BTreeMap::new(),
        }
    }
}

impl<T> WorldEdits<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from saved records in arbitrary order, while rebuilding the
    /// spatial index. Duplicate IDs in the *same* namespace are an error.
    ///
    /// # Errors
    /// Rejects duplicate generated or user-added IDs within a snapshot.
    pub fn from_records<G, A>(generated: G, added: A) -> Result<Self>
    where
        G: IntoIterator<Item = (u128, GeneratedChange<T>)>,
        A: IntoIterator<Item = AddedObject<T>>,
    {
        let mut result = Self::new();
        for (id, change) in generated {
            if result.generated.contains_key(&id) {
                return Err(Error::InvalidEdit);
            }
            result.update_generated(id, change);
        }
        for object in added {
            if result.added.contains_key(&object.id) {
                return Err(Error::InvalidEdit);
            }
            result.upsert_added(object);
        }
        Ok(result)
    }

    pub fn generated_changes(&self) -> impl Iterator<Item = (u128, &GeneratedChange<T>)> {
        self.generated.iter().map(|(&id, change)| (id, change))
    }

    pub fn added_objects(&self) -> impl Iterator<Item = &AddedObject<T>> {
        self.added.values()
    }

    #[must_use]
    pub fn get_generated(&self, id: u128) -> Option<&GeneratedChange<T>> {
        self.generated.get(&id)
    }

    #[must_use]
    pub fn get_added(&self, id: u128) -> Option<&AddedObject<T>> {
        self.added.get(&id)
    }

    /// Hide an automatically generated instance by its stable ID. Returns the
    /// previous edit, if one existed.
    pub fn hide_generated(&mut self, id: u128) -> Option<GeneratedChange<T>> {
        self.update_generated(id, GeneratedChange::Removed)
    }

    /// Supply a new state and destination canonical cell for a generated
    /// object. A moved object is visible from its *destination* cell even
    /// while its source tile is unloaded.
    pub fn modify_generated(
        &mut self,
        original: &Instance,
        destination: CellAddress,
        payload: T,
    ) -> Option<GeneratedChange<T>> {
        self.update_generated(
            original.candidate.id,
            GeneratedChange::Modified(Modification {
                layer_id: original.candidate.layer_id,
                source: original.candidate.cell,
                destination,
                payload,
            }),
        )
    }

    /// Remove an edit and resume the canonical generated instance.
    pub fn restore_generated(&mut self, id: u128) -> Option<GeneratedChange<T>> {
        let old = self.generated.remove(&id);
        if let Some(GeneratedChange::Modified(modification)) = &old {
            if modification.destination != modification.source {
                self.unindex(modification.destination, ObjectKey::Generated(id));
            }
        }
        old
    }

    /// Add or replace a player-authored object. A new cell automatically
    /// updates the spatial index, with no stale entry in its former location.
    pub fn upsert_added(&mut self, object: AddedObject<T>) -> Option<AddedObject<T>> {
        let id = object.id;
        let current_cell = object.cell;
        let previous = self.added.insert(id, object);
        if let Some(old) = &previous {
            self.unindex(old.cell, ObjectKey::Added(id));
        }
        self.index(current_cell, ObjectKey::Added(id));
        previous
    }

    pub fn remove_added(&mut self, id: u128) -> Option<AddedObject<T>> {
        let previous = self.added.remove(&id);
        if let Some(old) = &previous {
            self.unindex(old.cell, ObjectKey::Added(id));
        }
        previous
    }

    /// Apply an edit to a freshly generated instance.
    ///
    /// The source layer and cell must still match the saved-world record.
    /// A relocated instance is Hidden here and instead appears in the
    /// destination-cell extras, without being duplicated.
    ///
    /// # Errors
    /// Rejects saved edits whose source cell/layer disagrees with the canonical
    /// generator for this persistent object ID.
    pub fn resolve_generated<'a>(
        &'a self,
        instance: &'a Instance,
    ) -> Result<GeneratedState<'a, T>> {
        let Some(change) = self.generated.get(&instance.candidate.id) else {
            return Ok(GeneratedState::Original(instance));
        };
        match change {
            GeneratedChange::Removed => Ok(GeneratedState::Hidden),
            GeneratedChange::Modified(modification) => {
                if modification.source != instance.candidate.cell
                    || modification.layer_id != instance.candidate.layer_id
                {
                    return Err(Error::InvalidEdit);
                }
                if modification.destination == modification.source {
                    Ok(GeneratedState::Modified {
                        original: instance,
                        payload: &modification.payload,
                    })
                } else {
                    Ok(GeneratedState::Hidden)
                }
            }
        }
    }

    /// Enumerate added and relocated objects for any half-open canonical cell
    /// rectangle on a single face, without scanning all saved mutations.
    ///
    /// Entries are returned deterministically in cell and object-key order.
    /// They are a separate namespace from canonical generated tile contents.
    pub fn extras_in_cells(
        &self,
        face: Option<CubeFace>,
        rect: CellRect,
    ) -> impl Iterator<Item = VisibleExtra<'_, T>> + '_ {
        let min = rect.min();
        let end = rect.end_exclusive();
        self.by_x
            .range((face, min[0])..(face, end[0]))
            .flat_map(move |(_, rows)| rows.range(min[1]..end[1]))
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(move |key| match key {
                ObjectKey::Generated(id) => match self.generated.get(id) {
                    Some(GeneratedChange::Modified(modification))
                        if modification.destination != modification.source =>
                    {
                        Some(VisibleExtra::Relocated {
                            id: *id,
                            modification,
                        })
                    }
                    _ => None,
                },
                ObjectKey::Added(id) => self.added.get(id).map(VisibleExtra::Added),
            })
    }

    fn update_generated(
        &mut self,
        id: u128,
        change: GeneratedChange<T>,
    ) -> Option<GeneratedChange<T>> {
        let old = self.generated.remove(&id);
        if let Some(GeneratedChange::Modified(modification)) = &old {
            if modification.destination != modification.source {
                self.unindex(modification.destination, ObjectKey::Generated(id));
            }
        }
        if let GeneratedChange::Modified(modification) = &change {
            if modification.destination != modification.source {
                self.index(modification.destination, ObjectKey::Generated(id));
            }
        }
        self.generated.insert(id, change);
        old
    }

    fn index(&mut self, cell: CellAddress, key: ObjectKey) {
        self.by_x
            .entry((cell.face, cell.x))
            .or_default()
            .entry(cell.y)
            .or_default()
            .insert(key);
    }

    fn unindex(&mut self, cell: CellAddress, key: ObjectKey) {
        let column = (cell.face, cell.x);
        let remove_column = self.by_x.get_mut(&column).is_some_and(|rows| {
            if let Some(ids) = rows.get_mut(&cell.y) {
                ids.remove(&key);
                if ids.is_empty() {
                    rows.remove(&cell.y);
                }
            }
            rows.is_empty()
        });
        if remove_column {
            self.by_x.remove(&column);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant};

    fn generated() -> Instance {
        let generator = Generator::new(
            Domain::Planar { cell_size_m: 4.0 },
            GeneratorConfig::default(),
        )
        .unwrap();
        let trees = Layer::new(
            21,
            0.1,
            Distribution::Poisson,
            vec![Variant {
                id: 501,
                weight: 1.0,
            }],
        )
        .unwrap();
        generator.tile(TileKey::planar(0, 0), &trees).unwrap()[0]
    }

    fn rect(cell: CellAddress) -> CellRect {
        CellRect::new([cell.x, cell.y], [cell.x + 1, cell.y + 1]).unwrap()
    }

    #[test]
    fn removal_and_restore_apply_without_storing_generated_world() {
        let original = generated();
        let id = original.candidate.id;
        let mut edits = WorldEdits::<String>::new();
        assert!(matches!(
            edits.resolve_generated(&original),
            Ok(GeneratedState::Original(_))
        ));
        edits.hide_generated(id);
        assert!(matches!(
            edits.resolve_generated(&original),
            Ok(GeneratedState::Hidden)
        ));
        assert_eq!(edits.generated_changes().count(), 1);
        edits.restore_generated(id);
        assert!(matches!(
            edits.resolve_generated(&original),
            Ok(GeneratedState::Original(_))
        ));
        assert_eq!(edits.generated_changes().count(), 0);
    }

    #[test]
    fn same_cell_modification_is_resolved_once_without_extra() {
        let original = generated();
        let cell = original.candidate.cell;
        let mut edits = WorldEdits::new();
        edits.modify_generated(&original, cell, "bigger tree".to_owned());
        match edits.resolve_generated(&original).unwrap() {
            GeneratedState::Modified {
                payload,
                original: unchanged,
            } => {
                assert_eq!(payload, "bigger tree");
                assert_eq!(unchanged, &original);
            }
            _ => panic!("expected modified"),
        }
        assert_eq!(edits.extras_in_cells(cell.face, rect(cell)).count(), 0);
    }

    #[test]
    fn moving_generated_object_indexes_destination_not_unloaded_source() {
        let original = generated();
        let source = original.candidate.cell;
        let target = CellAddress {
            x: source.x + 200,
            y: source.y - 100,
            ..source
        };
        let later = CellAddress {
            x: source.x - 60,
            y: source.y + 100,
            ..source
        };
        let mut edits = WorldEdits::new();
        edits.modify_generated(&original, target, "moved oak");
        assert!(matches!(
            edits.resolve_generated(&original),
            Ok(GeneratedState::Hidden)
        ));
        assert_eq!(edits.extras_in_cells(source.face, rect(source)).count(), 0);
        {
            let mut moved = edits.extras_in_cells(target.face, rect(target));
            match moved.next().unwrap() {
                VisibleExtra::Relocated { id, modification } => {
                    assert_eq!(id, original.candidate.id);
                    assert_eq!(modification.layer_id, original.candidate.layer_id);
                    assert_eq!(modification.payload, "moved oak");
                }
                _ => panic!("expected relocated"),
            }
            assert!(moved.next().is_none());
        }
        edits.modify_generated(&original, later, "moved again");
        assert_eq!(edits.extras_in_cells(target.face, rect(target)).count(), 0);
        assert_eq!(edits.extras_in_cells(later.face, rect(later)).count(), 1);
        edits.restore_generated(original.candidate.id);
        assert_eq!(edits.extras_in_cells(later.face, rect(later)).count(), 0);
    }

    #[test]
    fn user_addition_id_namespace_is_separate_and_reindexing_is_exact() {
        let original = generated();
        let cell = original.candidate.cell;
        let target = CellAddress {
            x: cell.x + 25,
            ..cell
        };
        let id = original.candidate.id;
        let mut edits = WorldEdits::new();
        edits.hide_generated(id);
        edits.upsert_added(AddedObject {
            id,
            layer_id: 999,
            cell,
            payload: 1_u32,
        });
        assert_eq!(edits.extras_in_cells(cell.face, rect(cell)).count(), 1);
        edits.upsert_added(AddedObject {
            id,
            layer_id: 999,
            cell: target,
            payload: 2_u32,
        });
        assert_eq!(edits.extras_in_cells(cell.face, rect(cell)).count(), 0);
        let extras = edits
            .extras_in_cells(target.face, rect(target))
            .collect::<Vec<_>>();
        assert_eq!(extras.len(), 1);
        match &extras[0] {
            VisibleExtra::Added(addition) => assert_eq!(addition.payload, 2),
            _ => panic!("expected added"),
        }
        assert_eq!(edits.generated_changes().count(), 1);
        edits.remove_added(id);
        assert_eq!(edits.extras_in_cells(target.face, rect(target)).count(), 0);
    }

    #[test]
    fn roundtrip_snapshots_restore_current_cell_queries() {
        let original = generated();
        let cell = original.candidate.cell;
        let distant = CellAddress {
            x: cell.x - 25,
            y: cell.y + 25,
            ..cell
        };
        let mut edits = WorldEdits::new();
        edits.modify_generated(&original, distant, 11_u64);
        edits.upsert_added(AddedObject {
            id: 2,
            layer_id: 40,
            cell,
            payload: 6,
        });
        edits.upsert_added(AddedObject {
            id: 1,
            layer_id: 40,
            cell: distant,
            payload: 7,
        });
        let generated_records = edits
            .generated_changes()
            .map(|(id, change)| (id, change.clone()))
            .collect::<Vec<_>>();
        let added_records = edits.added_objects().cloned().collect::<Vec<_>>();
        let loaded = WorldEdits::from_records(generated_records, added_records).unwrap();
        assert_eq!(loaded.extras_in_cells(cell.face, rect(cell)).count(), 1);
        assert_eq!(
            loaded.extras_in_cells(distant.face, rect(distant)).count(),
            2
        );
        assert_eq!(loaded.generated_changes().count(), 1);
        assert_eq!(
            loaded.added_objects().map(|obj| obj.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn invalid_saved_duplicate_ids_fail_loudly() {
        let source = generated();
        let original = source.candidate.cell;
        let a = AddedObject {
            id: 42,
            layer_id: 1,
            cell: original,
            payload: 3_u32,
        };
        assert!(
            WorldEdits::from_records(
                [
                    (1, GeneratedChange::<u32>::Removed),
                    (1, GeneratedChange::Removed)
                ],
                std::iter::empty::<AddedObject<u32>>()
            )
            .is_err()
        );
        assert!(
            WorldEdits::from_records(
                std::iter::empty::<(u128, GeneratedChange<u32>)>(),
                [a.clone(), a]
            )
            .is_err()
        );
    }

    #[test]
    fn bad_loaded_source_mapping_is_detected_on_resolution() {
        let original = generated();
        let mut edits = WorldEdits::new();
        let wrong = Modification {
            layer_id: original.candidate.layer_id,
            source: CellAddress {
                x: original.candidate.cell.x + 1,
                ..original.candidate.cell
            },
            destination: original.candidate.cell,
            payload: (),
        };
        edits.update_generated(original.candidate.id, GeneratedChange::Modified(wrong));
        assert!(matches!(
            edits.resolve_generated(&original),
            Err(Error::InvalidEdit)
        ));
    }

    #[test]
    fn cell_index_respects_planet_faces_and_rectangle_bounds() {
        let mut edits = WorldEdits::new();
        let ground = CellAddress {
            face: Some(CubeFace::PositiveZ),
            x: -1,
            y: 20,
        };
        let sky = CellAddress {
            face: Some(CubeFace::NegativeZ),
            x: -1,
            y: 20,
        };
        edits.upsert_added(AddedObject {
            id: 1,
            layer_id: 1,
            cell: ground,
            payload: (),
        });
        edits.upsert_added(AddedObject {
            id: 2,
            layer_id: 1,
            cell: sky,
            payload: (),
        });
        assert_eq!(edits.extras_in_cells(ground.face, rect(ground)).count(), 1);
        assert_eq!(edits.extras_in_cells(sky.face, rect(sky)).count(), 1);
        let nearby = CellRect::new([-100, -100], [100, 100]).unwrap();
        assert_eq!(edits.extras_in_cells(None, nearby).count(), 0);
    }
}
