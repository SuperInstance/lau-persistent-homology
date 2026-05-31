//! Betti curves: Betti numbers as a function of scale.

use crate::diagram::PersistenceDiagram;
use serde::{Deserialize, Serialize};

/// A Betti curve: Betti numbers plotted against scale parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BettiCurve {
    /// Scale values (x-axis).
    scales: Vec<f64>,
    /// Betti numbers at each scale.
    values: Vec<usize>,
    /// Homology dimension.
    dimension: usize,
}

impl BettiCurve {
    /// Compute a Betti curve from a persistence diagram.
    pub fn from_diagram(
        diagram: &PersistenceDiagram,
        dim: usize,
        resolution: usize,
        t_min: f64,
        t_max: f64,
    ) -> Self {
        let dt = if resolution > 1 {
            (t_max - t_min) / (resolution - 1) as f64
        } else {
            0.0
        };
        let scales: Vec<f64> = (0..resolution).map(|i| t_min + dt * i as f64).collect();
        let values: Vec<usize> = scales
            .iter()
            .map(|&t| diagram.betti_number(dim, t))
            .collect();

        BettiCurve {
            scales,
            values,
            dimension: dim,
        }
    }

    /// Get the scale values.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Get the Betti numbers.
    pub fn values(&self) -> &[usize] {
        &self.values
    }

    /// The homology dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Maximum Betti number reached.
    pub fn max_betti(&self) -> usize {
        self.values.iter().copied().max().unwrap_or(0)
    }

    /// Scale at which the Betti number first reaches its maximum.
    pub fn scale_of_max(&self) -> Option<f64> {
        let max = self.max_betti();
        self.values
            .iter()
            .position(|&v| v == max)
            .map(|i| self.scales[i])
    }

    /// Integral (area under the Betti curve).
    pub fn integral(&self) -> f64 {
        if self.scales.len() < 2 {
            return 0.0;
        }
        let dt = self.scales[1] - self.scales[0];
        self.values.iter().map(|&v| v as f64 * dt).sum()
    }
}

/// Compute L^p distance between two Betti curves.
pub fn betti_curve_distance(c1: &BettiCurve, c2: &BettiCurve, p: f64) -> f64 {
    assert_eq!(c1.scales.len(), c2.scales.len(), "Betti curves must have same resolution");
    let dt = if c1.scales.len() > 1 {
        c1.scales[1] - c1.scales[0]
    } else {
        1.0
    };

    let mut total = 0.0f64;
    for i in 0..c1.scales.len() {
        let diff = (c1.values[i] as f64 - c2.values[i] as f64).abs();
        if p.is_infinite() {
            total = total.max(diff);
        } else {
            total += diff.powf(p) * dt;
        }
    }

    if p.is_infinite() {
        total
    } else {
        total.powf(1.0 / p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagram::{PersistenceDiagram, PersistencePair};
    use crate::filtration::Filtration;

    #[test]
    fn test_betti_curve_triangle() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let curve = BettiCurve::from_diagram(&diag, 0, 100, 0.0, 2.0);

        // At scale 0, Betti 0 = 3 (three isolated points)
        assert_eq!(curve.values()[0], 3);

        // At large scale, Betti 0 = 1 (one component)
        assert_eq!(*curve.values().last().unwrap(), 1);
    }

    #[test]
    fn test_betti_curve_two_components() {
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![10.0],
            vec![11.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let curve = BettiCurve::from_diagram(&diag, 0, 100, 0.0, 15.0);

        assert_eq!(curve.max_betti(), 4); // 4 isolated points at scale 0
    }

    #[test]
    fn test_betti_curve_integral() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { birth_index: 2, death_index: Some(3), dimension: 0, birth: 0.0, death: 2.0 },
            PersistencePair { birth_index: 4, death_index: None, dimension: 0, birth: 0.0, death: f64::INFINITY },
        ]);
        let curve = BettiCurve::from_diagram(&diag, 0, 1000, 0.0, 3.0);

        let integral = curve.integral();
        assert!(integral > 0.0);
    }

    #[test]
    fn test_betti_curve_monotone_h0() {
        // H0 Betti number should be non-increasing (components merge)
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![3.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let curve = BettiCurve::from_diagram(&diag, 0, 100, 0.0, 5.0);

        for i in 1..curve.values.len() {
            assert!(curve.values[i] <= curve.values[i - 1]);
        }
    }

    #[test]
    fn test_betti_curve_distance() {
        let d1 = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
        ]);
        let d2 = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 2.0 },
        ]);
        let c1 = BettiCurve::from_diagram(&d1, 0, 100, 0.0, 3.0);
        let c2 = BettiCurve::from_diagram(&d2, 0, 100, 0.0, 3.0);

        let dist = betti_curve_distance(&c1, &c2, 2.0);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_betti_curve_identical_diagrams() {
        let d = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
        ]);
        let c1 = BettiCurve::from_diagram(&d, 0, 50, 0.0, 2.0);
        let c2 = BettiCurve::from_diagram(&d, 0, 50, 0.0, 2.0);

        let dist = betti_curve_distance(&c1, &c2, 2.0);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_scale_of_max() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
        ]);
        let curve = BettiCurve::from_diagram(&diag, 0, 100, 0.0, 2.0);
        assert_eq!(curve.scale_of_max(), Some(0.0)); // Max at scale 0
    }
}
