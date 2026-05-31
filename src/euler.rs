//! Euler characteristic curves.

use crate::diagram::PersistenceDiagram;
use crate::filtration::Filtration;
use serde::{Deserialize, Serialize};

/// An Euler characteristic curve: Euler characteristic as a function of scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EulerCurve {
    /// Scale values.
    scales: Vec<f64>,
    /// Euler characteristic at each scale.
    values: Vec<i64>,
}

impl EulerCurve {
    /// Compute an Euler characteristic curve from a filtration.
    pub fn from_filtration(filtration: &Filtration, resolution: usize, t_min: f64, t_max: f64) -> Self {
        let dt = if resolution > 1 {
            (t_max - t_min) / (resolution - 1) as f64
        } else {
            0.0
        };
        let scales: Vec<f64> = (0..resolution).map(|i| t_min + dt * i as f64).collect();

        let values: Vec<i64> = scales
            .iter()
            .map(|&t| {
                let complex = filtration.complex_at(t);
                complex.euler_characteristic()
            })
            .collect();

        EulerCurve { scales, values }
    }

    /// Compute from a persistence diagram.
    pub fn from_diagram(diagram: &PersistenceDiagram, resolution: usize, t_min: f64, t_max: f64) -> Self {
        let dt = if resolution > 1 {
            (t_max - t_min) / (resolution - 1) as f64
        } else {
            0.0
        };
        let scales: Vec<f64> = (0..resolution).map(|i| t_min + dt * i as f64).collect();

        let max_dim = diagram.max_dimension();
        let values: Vec<i64> = scales
            .iter()
            .map(|&t| {
                let mut chi: i64 = 0;
                for dim in 0..=max_dim {
                    let betti = diagram.betti_number(dim, t) as i64;
                    if dim % 2 == 0 {
                        chi += betti;
                    } else {
                        chi -= betti;
                    }
                }
                chi
            })
            .collect();

        EulerCurve { scales, values }
    }

    /// Get the scale values.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Get the Euler characteristic values.
    pub fn values(&self) -> &[i64] {
        &self.values
    }

    /// Maximum absolute Euler characteristic.
    pub fn max_abs(&self) -> i64 {
        self.values.iter().map(|v| v.abs()).max().unwrap_or(0)
    }

    /// Integral (area).
    pub fn integral(&self) -> f64 {
        if self.scales.len() < 2 {
            return 0.0;
        }
        let dt = self.scales[1] - self.scales[0];
        self.values.iter().map(|&v| v as f64 * dt).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filtration::Filtration;

    #[test]
    fn test_euler_curve_triangle() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let curve = EulerCurve::from_filtration(&filt, 100, 0.0, 2.0);

        // At scale 0: 3 vertices, chi = 3
        assert_eq!(curve.values()[0], 3);

        // At large scale: full triangle, chi = 1 (contractible)
        assert_eq!(*curve.values().last().unwrap(), 1);
    }

    #[test]
    fn test_euler_curve_contractible() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let diag = crate::diagram::PersistenceDiagram::from_filtration(&filt);
        let curve = EulerCurve::from_diagram(&diag, 100, 0.0, 2.0);

        // For a contractible space, chi = 1 at large scale
        assert_eq!(*curve.values().last().unwrap(), 1);
    }

    #[test]
    fn test_euler_curve_from_filtration() {
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let curve = EulerCurve::from_filtration(&filt, 50, 0.0, 3.0);

        // At scale 0: chi = 3 (3 vertices only)
        assert_eq!(curve.values()[0], 3);
        // At large scale: 3 vertices - 3 edges = 0 (contractible but just edges)
        // Actually with full Rips: 3V - 3E + 0T = 0 for dim 1, or 1 for a triangle
        // With max_dim=1 we only have vertices and edges: 3 - 3 = 0
        assert_eq!(*curve.values().last().unwrap(), 0);
    }

    #[test]
    fn test_euler_curve_nonempty() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let filt = Filtration::vietoris_rips(&points, 1);
        let curve = EulerCurve::from_filtration(&filt, 10, 0.0, 2.0);

        assert!(!curve.values().is_empty());
        assert!(curve.max_abs() > 0);
        assert!(curve.integral() != 0.0);
    }
}
