//! Utility functions (simple RNG for reproducible tests).

/// A simple deterministic PRNG for testing purposes.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Return a value in [0, 1).
    pub fn next(&mut self) -> f64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Return a value in [min, max).
    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next() * (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_reproducible() {
        let mut r1 = SimpleRng::new(42);
        let mut r2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert!((r1.next() - r2.next()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = SimpleRng::new(123);
        for _ in 0..1000 {
            let v = rng.next_range(-1.0, 1.0);
            assert!(v >= -1.0 && v < 1.0);
        }
    }

    #[test]
    fn test_rng_unit_interval() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let v = rng.next();
            assert!(v >= 0.0 && v < 1.0);
        }
    }
}
