use crate::error::{Error, Result};
use crate::random::{Rng, hash_words};
use std::collections::HashSet;

/// A stable object type / model ID and its positive relative selection weight.
/// No tree-specific semantics: rocks, structures, debris and plants are equal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Variant {
    pub id: u64,
    pub weight: f64,
}

/// Weighted exponential race: selection is independent of variant slice order,
/// and independent of candidate generation or neighbouring tiles.
///
/// When additional variants are registered, existing object **identities** and
/// coordinates remain unchanged; only some model variants can change.
/// # Errors
/// Rejects missing variants, duplicate IDs and nonfinite or nonpositive weights.
pub fn choose_variant(object_id: u128, variants: &[Variant]) -> Result<u64> {
    if variants.is_empty() {
        return Err(Error::InvalidConfiguration(
            "at least one variant is required",
        ));
    }
    let mut ids = HashSet::with_capacity(variants.len());
    for variant in variants {
        if !variant.weight.is_finite() || variant.weight <= 0.0 {
            return Err(Error::InvalidConfiguration(
                "variant weight must be finite and positive",
            ));
        }
        if !ids.insert(variant.id) {
            return Err(Error::InvalidConfiguration(
                "variant IDs must be unique in a layer",
            ));
        }
    }
    Ok(choose_variant_validated(object_id, variants))
}

/// Hot-path version for a Layer whose variants were checked once at creation.
#[expect(
    clippy::cast_possible_truncation,
    reason = "extract low and high 64-bit limbs of the stable u128 object ID"
)]
pub fn choose_variant_validated(object_id: u128, variants: &[Variant]) -> u64 {
    let mut best = (f64::INFINITY, u64::MAX);
    for variant in variants {
        let mut random = Rng::new(hash_words(&[
            object_id as u64,
            (object_id >> 64) as u64,
            variant.id,
            0xd40c_e8aa_b5d6_74ae,
        ]));
        // A log-domain exponential race is stable for extreme finite weights.
        let score = (-random.unit().ln()).ln() - variant.weight.ln();
        if (score, variant.id).total_cmp_with_tie(best).is_lt() {
            best = (score, variant.id);
        }
    }
    best.1
}

trait CompareScore {
    fn total_cmp_with_tie(self, other: Self) -> core::cmp::Ordering;
}

impl CompareScore for (f64, u64) {
    fn total_cmp_with_tie(self, other: Self) -> core::cmp::Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.cmp(&other.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn variant_order_does_not_affect_selection() {
        let variants = [
            Variant {
                id: 100,
                weight: 0.1,
            },
            Variant { id: 2, weight: 1.0 },
            Variant {
                id: 999,
                weight: 4.0,
            },
        ];
        let mut reversed = variants.to_vec();
        reversed.reverse();
        for i in 0..10_000 {
            assert_eq!(
                choose_variant(i, &variants).unwrap(),
                choose_variant(i, &reversed).unwrap()
            );
        }
    }
    #[test]
    fn tiny_weights_keep_their_relative_probabilities() {
        let variants = [
            Variant {
                id: 1,
                weight: 1e-300,
            },
            Variant {
                id: 2,
                weight: 3e-300,
            },
        ];
        let count = (0..20_000)
            .filter(|&i| choose_variant(i, &variants).unwrap() == 2)
            .count();
        assert!((14_500..15_500).contains(&count), "count={count}");
    }

    #[test]
    fn relative_weights_are_respected() {
        let variants = [
            Variant { id: 1, weight: 1.0 },
            Variant { id: 2, weight: 3.0 },
        ];
        let count = (0..20_000)
            .filter(|&i| choose_variant(i, &variants).unwrap() == 2)
            .count();
        assert!((14_500..15_500).contains(&count), "count={count}");
    }
}
