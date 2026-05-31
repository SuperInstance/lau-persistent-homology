//! Filtration building from point clouds.

use crate::complex::SimplicialComplex;
use crate::simplex::Simplex;
use serde::{Deserialize, Serialize};

/// A filtration: a sequence of simplicial complexes ordered by increasing parameter.
/// Each simplex has an associated filtration value (the scale at which it appears).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filtration {
    /// Sorted simplices with their filtration values.
    entries: Vec<(Simplex, f64)>,
    /// Maximum dimension.
    max_dim: usize,
}

impl Filtration {
    /// Create a filtration from (simplex, value) pairs.
    /// Sorts by value, then by dimension, then lexicographically.
    pub fn new(mut entries: Vec<(Simplex, f64)>) -> Self {
        entries.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let max_dim = entries
            .iter()
            .map(|(s, _)| s.dimension())
            .max()
            .unwrap_or(0);
        Filtration { entries, max_dim }
    }

    /// Build a Vietoris-Rips filtration from a point cloud.
    pub fn vietoris_rips(points: &[Vec<f64>], max_dim: usize) -> Self {
        let n = points.len();
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = points[i]
                    .iter()
                    .zip(points[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        let mut entries: Vec<(Simplex, f64)> = Vec::new();

        // Vertices appear at 0
        for i in 0..n {
            entries.push((Simplex::vertex(i), 0.0));
        }

        // Build simplices by dimension
        // Track which simplices exist at each dimension
        let mut prev_simplices: Vec<Simplex> = (0..n).map(|i| Simplex::vertex(i)).collect();

        for dim in 1..=max_dim {
            let mut next_simplices = Vec::new();

            if dim == 1 {
                // Edges
                for i in 0..n {
                    for j in (i + 1)..n {
                        let val = dist[i][j];
                        entries.push((Simplex::edge(i, j), val));
                        next_simplices.push(Simplex::edge(i, j));
                    }
                }
            } else {
                // Higher simplices: value = max of all face values
                // Generate candidates by adding one vertex to existing simplices
                let mut seen = std::collections::HashSet::new();
                for simplex in &prev_simplices {
                    if simplex.dimension() != dim - 1 {
                        continue;
                    }
                    let verts = simplex.vertices();
                    // Try adding each vertex greater than max existing
                    let max_v = *verts.last().unwrap_or(&0);
                    for new_v in (max_v + 1)..n {
                        let mut new_verts = verts.to_vec();
                        new_verts.push(new_v);
                        let candidate = Simplex::new(new_verts);
                        if seen.contains(&candidate) {
                            continue;
                        }
                        seen.insert(candidate.clone());

                        // Filtration value = max pairwise distance
                        let mut max_dist = 0.0f64;
                        let cv = candidate.vertices();
                        for a in 0..cv.len() {
                            for b in (a + 1)..cv.len() {
                                max_dist = max_dist.max(dist[cv[a]][cv[b]]);
                            }
                        }
                        entries.push((candidate.clone(), max_dist));
                        next_simplices.push(candidate);
                    }
                }
            }
            prev_simplices = next_simplices;
        }

        Filtration::new(entries)
    }

    /// Build a Rips filtration with a custom distance function.
    pub fn vietoris_rips_with_dist<F>(
        n: usize,
        distance_fn: F,
        max_dim: usize,
    ) -> Self
    where
        F: Fn(usize, usize) -> f64,
    {
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = distance_fn(i, j);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        // Reuse the same logic but from dist matrix
        let _points: Vec<Vec<f64>> = (0..n).map(|_| vec![]).collect();
        // Actually let's just reconstruct from dist directly
        let mut entries: Vec<(Simplex, f64)> = Vec::new();
        for i in 0..n {
            entries.push((Simplex::vertex(i), 0.0));
        }

        let mut prev_simplices: Vec<Simplex> = (0..n).map(|i| Simplex::vertex(i)).collect();

        for dim in 1..=max_dim {
            let mut next_simplices = Vec::new();
            if dim == 1 {
                for i in 0..n {
                    for j in (i + 1)..n {
                        entries.push((Simplex::edge(i, j), dist[i][j]));
                        next_simplices.push(Simplex::edge(i, j));
                    }
                }
            } else {
                let mut seen = std::collections::HashSet::new();
                for simplex in &prev_simplices {
                    if simplex.dimension() != dim - 1 {
                        continue;
                    }
                    let verts = simplex.vertices();
                    let max_v = *verts.last().unwrap_or(&0);
                    for new_v in (max_v + 1)..n {
                        let mut new_verts = verts.to_vec();
                        new_verts.push(new_v);
                        let candidate = Simplex::new(new_verts);
                        if seen.contains(&candidate) {
                            continue;
                        }
                        seen.insert(candidate.clone());
                        let mut max_dist = 0.0f64;
                        let cv = candidate.vertices();
                        for a in 0..cv.len() {
                            for b in (a + 1)..cv.len() {
                                max_dist = max_dist.max(dist[cv[a]][cv[b]]);
                            }
                        }
                        entries.push((candidate.clone(), max_dist));
                        next_simplices.push(candidate);
                    }
                }
            }
            prev_simplices = next_simplices;
        }

        Filtration::new(entries)
    }

    /// Get the sorted entries.
    pub fn entries(&self) -> &[(Simplex, f64)] {
        &self.entries
    }

    /// Number of simplices in the filtration.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the filtration is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entries of a specific dimension.
    pub fn entries_of_dim(&self, dim: usize) -> Vec<(&Simplex, f64)> {
        self.entries
            .iter()
            .filter(|(s, _)| s.dimension() == dim)
            .map(|(s, v)| (s, *v))
            .collect()
    }

    /// Maximum dimension.
    pub fn max_dim(&self) -> usize {
        self.max_dim
    }

    /// Get the filtration value for a simplex.
    pub fn value_of(&self, simplex: &Simplex) -> Option<f64> {
        self.entries
            .iter()
            .find(|(s, _)| s == simplex)
            .map(|(_, v)| *v)
    }

    /// Get all unique filtration values, sorted.
    pub fn values(&self) -> Vec<f64> {
        let mut vals: Vec<f64> = self.entries.iter().map(|(_, v)| *v).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        vals
    }

    /// Extract the complex at a given threshold.
    pub fn complex_at(&self, threshold: f64) -> SimplicialComplex {
        let mut complex = SimplicialComplex::new();
        for (simplex, val) in &self.entries {
            if *val <= threshold {
                complex.insert_closed(simplex);
            }
        }
        complex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rips_filtration_three_points() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        assert_eq!(filt.len(), 7); // 3 vertices + 3 edges + 1 triangle

        // Vertices at 0
        assert_eq!(filt.value_of(&Simplex::vertex(0)), Some(0.0));
        assert_eq!(filt.value_of(&Simplex::vertex(1)), Some(0.0));

        // Edge 0-1 at distance 1.0
        assert_eq!(filt.value_of(&Simplex::edge(0, 1)), Some(1.0));

        // Edge 0-2 at distance 1.0
        assert_eq!(filt.value_of(&Simplex::edge(0, 2)), Some(1.0));

        // Edge 1-2 at sqrt(2)
        let v12 = filt.value_of(&Simplex::edge(1, 2)).unwrap();
        assert!((v12 - std::f64::consts::SQRT_2).abs() < 1e-10);

        // Triangle at sqrt(2)
        let tri = filt.value_of(&Simplex::triangle(0, 1, 2)).unwrap();
        assert!((tri - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_filtration_ordering() {
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![3.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let entries = filt.entries();

        // Check that entries are sorted by value
        for i in 1..entries.len() {
            assert!(entries[i].1 >= entries[i - 1].1 - 1e-10);
        }
    }

    #[test]
    fn test_complex_at_threshold() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);

        // At threshold 0, only vertices
        let c0 = filt.complex_at(0.0);
        assert_eq!(c0.size(), 3);
        assert!(c0.contains(&Simplex::vertex(0)));
        assert!(!c0.contains(&Simplex::edge(0, 1)));

        // At threshold 1.0, vertices + edges
        let c1 = filt.complex_at(1.0);
        assert!(c1.contains(&Simplex::edge(0, 1)));
        assert!(c1.contains(&Simplex::edge(0, 2)));

        // At threshold sqrt(2), full complex
        let c2 = filt.complex_at(std::f64::consts::SQRT_2);
        assert!(c2.contains(&Simplex::triangle(0, 1, 2)));
    }

    #[test]
    fn test_custom_distance() {
        let filt = Filtration::vietoris_rips_with_dist(
            3,
            |i, j| (i as f64 - j as f64).abs(),
            2,
        );
        assert!(filt.len() >= 6); // at least 3 vertices + 3 edges
    }
}
