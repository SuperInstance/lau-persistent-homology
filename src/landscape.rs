//! Persistence landscapes — stable vectorization of persistence diagrams.

use crate::diagram::PersistenceDiagram;
use serde::{Deserialize, Serialize};

/// A persistence landscape: a sequence of piecewise-linear functions λ_k(t).
/// Each λ_k is the k-th largest persistence function value at parameter t.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistenceLandscape {
    /// Grid of t values where the landscape is sampled.
    grid: Vec<f64>,
    /// λ_k values: landscape_values[k][i] = λ_k(grid[i]).
    landscape_values: Vec<Vec<f64>>,
    /// The dimension this landscape represents.
    dimension: usize,
}

impl PersistenceLandscape {
    /// Compute a persistence landscape from a diagram at a given resolution.
    pub fn from_diagram(
        diagram: &PersistenceDiagram,
        dim: usize,
        resolution: usize,
        t_min: f64,
        t_max: f64,
    ) -> Self {
        let pairs = diagram.pairs_of_dim(dim);
        let mut tent_functions: Vec<(f64, f64, f64)> = Vec::new(); // (birth, death, max_value)

        for p in &pairs {
            if p.is_essential() {
                // For essential features, use a large death value or extend
                let death = if p.death.is_infinite() { t_max * 2.0 } else { p.death };
                tent_functions.push((p.birth, death, (death - p.birth) / 2.0));
            } else {
                let _mid = (p.birth + p.death) / 2.0;
                let height = (p.death - p.birth) / 2.0;
                tent_functions.push((p.birth, p.death, height));
            }
        }

        // Generate grid
        let dt = if resolution > 1 {
            (t_max - t_min) / (resolution - 1) as f64
        } else {
            0.0
        };
        let grid: Vec<f64> = (0..resolution).map(|i| t_min + dt * i as f64).collect();

        // Evaluate each tent function at each grid point
        let k = tent_functions.len();
        let mut all_values: Vec<Vec<f64>> = vec![vec![0.0; resolution]; k];

        for (idx, &(birth, death, _max_val)) in tent_functions.iter().enumerate() {
            let mid = (birth + death) / 2.0;
            for (i, &t) in grid.iter().enumerate() {
                if t < birth || t > death {
                    all_values[idx][i] = 0.0;
                } else if t <= mid {
                    all_values[idx][i] = t - birth;
                } else {
                    all_values[idx][i] = death - t;
                }
            }
        }

        // Sort at each grid point to get landscape functions
        let mut landscape_values: Vec<Vec<f64>> = Vec::new();
        for i in 0..resolution {
            let mut vals: Vec<f64> = all_values.iter().map(|v| v[i]).collect();
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
            for (k, val) in vals.iter().enumerate() {
                if landscape_values.len() <= k {
                    landscape_values.push(vec![0.0; resolution]);
                }
                landscape_values[k][i] = *val;
            }
        }

        if landscape_values.is_empty() {
            landscape_values.push(vec![0.0; resolution]);
        }

        PersistenceLandscape {
            grid,
            landscape_values,
            dimension: dim,
        }
    }

    /// Get the grid values.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Get the k-th landscape function values.
    pub fn landscape(&self, k: usize) -> &[f64] {
        if k < self.landscape_values.len() {
            &self.landscape_values[k]
        } else {
            &[0.0][..0] // empty slice if k is too large
        }
    }

    /// Number of landscape functions.
    pub fn num_landscapes(&self) -> usize {
        self.landscape_values.len()
    }

    /// The dimension of the homology this landscape represents.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute the L^p norm of the landscape.
    pub fn l_norm(&self, p: f64) -> f64 {
        if self.grid.len() < 2 {
            return 0.0;
        }
        let dt = if self.grid.len() > 1 {
            self.grid[1] - self.grid[0]
        } else {
            1.0
        };

        let mut total = 0.0f64;
        for k in 0..self.landscape_values.len() {
            for &val in &self.landscape_values[k] {
                if p.is_infinite() {
                    total = total.max(val);
                } else {
                    total += val.powf(p) * dt;
                }
            }
        }

        if p.is_infinite() {
            total
        } else {
            total.powf(1.0 / p)
        }
    }

    /// Integrate the first landscape function (area under λ_1).
    pub fn integral(&self) -> f64 {
        if self.grid.len() < 2 || self.landscape_values.is_empty() {
            return 0.0;
        }
        let dt = self.grid[1] - self.grid[0];
        self.landscape_values[0].iter().map(|&v| v * dt).sum()
    }

    /// Compute the L^∞ norm (maximum value).
    pub fn l_inf_norm(&self) -> f64 {
        self.landscape_values
            .iter()
            .flat_map(|v| v.iter())
            .copied()
            .fold(0.0f64, f64::max)
    }
}

/// Compute L^p distance between two landscapes.
pub fn landscape_distance(land1: &PersistenceLandscape, land2: &PersistenceLandscape, p: f64) -> f64 {
    assert_eq!(land1.grid.len(), land2.grid.len(), "Landscapes must have the same grid");
    let dt = if land1.grid.len() > 1 {
        land1.grid[1] - land1.grid[0]
    } else {
        1.0
    };

    let max_k = land1.num_landscapes().max(land2.num_landscapes());
    let mut total = 0.0f64;

    for k in 0..max_k {
        let l1 = if k < land1.num_landscapes() { land1.landscape(k) } else { &[] };
        let l2 = if k < land2.num_landscapes() { land2.landscape(k) } else { &[] };
        let len = l1.len().max(l2.len());

        for i in 0..len {
            let v1 = if i < l1.len() { l1[i] } else { 0.0 };
            let v2 = if i < l2.len() { l2[i] } else { 0.0 };
            let diff = (v1 - v2).abs();
            if p.is_infinite() {
                total = total.max(diff);
            } else {
                total += diff.powf(p) * dt;
            }
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

    #[test]
    fn test_landscape_basic() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 2.0 },
        ]);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 100, -1.0, 3.0);

        assert!(landscape.num_landscapes() >= 1);
        assert!(landscape.l_inf_norm() > 0.0);
        // Peak should be at t=1.0 with value 1.0
        let peak = landscape.landscape(0).iter().cloned().fold(0.0f64, f64::max);
        assert!((peak - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_landscape_nonnegativity() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { birth_index: 2, death_index: Some(3), dimension: 0, birth: 0.5, death: 2.0 },
        ]);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 50, -1.0, 3.0);

        for k in 0..landscape.num_landscapes() {
            for &v in landscape.landscape(k) {
                assert!(v >= -1e-10, "Landscape values should be non-negative");
            }
        }
    }

    #[test]
    fn test_landscape_ordering() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 2.0 },
            PersistencePair { birth_index: 2, death_index: Some(3), dimension: 0, birth: 0.0, death: 1.0 },
        ]);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 50, -1.0, 3.0);

        // λ_1 >= λ_2 >= λ_3 >= ...
        for k in 1..landscape.num_landscapes() {
            for i in 0..landscape.grid.len() {
                assert!(landscape.landscape(k - 1)[i] >= landscape.landscape(k)[i] - 1e-10);
            }
        }
    }

    #[test]
    fn test_landscape_integral() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 2.0 },
        ]);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 1000, -1.0, 3.0);

        // Triangle from 0 to 2 with height 1: area = 1.0
        let integral = landscape.integral();
        assert!((integral - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_landscape_l2_norm() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 2.0 },
        ]);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 1000, -1.0, 3.0);

        let norm = landscape.l_norm(2.0);
        assert!(norm > 0.0);
    }

    #[test]
    fn test_landscape_distance_identical() {
        let diag = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
        ]);
        let l1 = PersistenceLandscape::from_diagram(&diag, 0, 100, -1.0, 2.0);
        let l2 = PersistenceLandscape::from_diagram(&diag, 0, 100, -1.0, 2.0);

        let dist = landscape_distance(&l1, &l2, 2.0);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_landscape_from_point_cloud() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = crate::filtration::Filtration::vietoris_rips(&points, 2);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 100, 0.0, 2.0);

        assert!(landscape.num_landscapes() >= 1);
        assert!(landscape.l_inf_norm() > 0.0);
    }

    #[test]
    fn test_empty_diagram_landscape() {
        let diag = PersistenceDiagram::new(vec![]);
        let landscape = PersistenceLandscape::from_diagram(&diag, 0, 10, 0.0, 1.0);
        assert!(landscape.l_inf_norm().abs() < 1e-10);
    }
}
