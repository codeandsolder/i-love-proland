//! Fixed, specification-independent hashing. Never substitute the standard-library default hasher
//! or a platform RNG: persistent object IDs rely on these exact bits.

pub const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub fn hash_words(words: &[u64]) -> u64 {
    let mut h = 0x6a09_e667_f3bc_c909_u64;
    for &w in words {
        h = mix64(h ^ mix64(w.wrapping_add(0x9e37_79b9_7f4a_7c15)));
    }
    h
}

#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }

    /// Uniform open interval (0,1). The half-unit offset prevents ln(0).
    #[expect(
        clippy::cast_precision_loss,
        reason = "shifted integer has only 53 bits and converts exactly to f64"
    )]
    pub fn unit(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 1.0) / 9_007_199_254_740_994.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_known_vectors() {
        assert_eq!(mix64(0), 0);
        assert_eq!(mix64(1), 0x5692_161d_100b_05e5);
        let mut rng = Rng::new(42);
        let a = rng.next_u64();
        assert_eq!(a, 0xbdd7_3226_2feb_6e95);
        for _ in 0..100_000 {
            let x = rng.unit();
            assert!(x > 0.0 && x < 1.0);
        }
    }
}
