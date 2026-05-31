//! Persistence diagrams and barcodes.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A single persistence pair (birth, death) in a given homology dimension.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistencePair {
    /// Index of the birth simplex in the filtration.
    pub birth_index: usize,
    /// Index of the death simplex (None = infinite persistence).
    pub death_index: Option<usize>,
    /// Homology dimension.
    pub dimension: usize,
    /// Filtration value at birth.
    pub birth: f64,
    /// Filtration value at death (infinity for essential features).
    pub death: f64,
}

impl PersistencePair {
    /// Persistence length (death - birth).
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Is this an essential (infinite persistence) feature?
    pub fn is_essential(&self) -> bool {
        self.death_index.is_none() || self.death.is_infinite()
    }

    /// Midpoint of the persistence interval.
    pub fn midpoint(&self) -> f64 {
        (self.birth + self.death) / 2.0
    }
}

/// A persistence diagram: collection of persistence pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pairs: Vec<PersistencePair>,
    max_dimension: usize,
}

impl PersistenceDiagram {
    /// Create a new persistence diagram from pairs.
    pub fn new(pairs: Vec<PersistencePair>) -> Self {
        let max_dimension = pairs.iter().map(|p| p.dimension).max().unwrap_or(0);
        PersistenceDiagram { pairs, max_dimension }
    }

    /// Compute a persistence diagram from a filtration by running reduction.
    pub fn from_filtration(filtration: &crate::filtration::Filtration) -> Self {
        use crate::boundary::BoundaryMatrix;
        use crate::reduction::reduce;

        let bm = BoundaryMatrix::from_filtration(filtration);
        let entries = filtration.entries();
        let (_reduced, pairs) = reduce(bm);

        // Fill in actual filtration values
        let filled_pairs: Vec<PersistencePair> = pairs
            .into_iter()
            .map(|p| {
                let birth = entries[p.birth_index].1;
                let death = p.death_index.map_or(f64::INFINITY, |di| entries[di].1);
                PersistencePair {
                    birth_index: p.birth_index,
                    death_index: p.death_index,
                    dimension: p.dimension,
                    birth,
                    death,
                }
            })
            .collect();

        PersistenceDiagram::new(filled_pairs)
    }

    /// Get all pairs.
    pub fn pairs(&self) -> &[PersistencePair] {
        &self.pairs
    }

    /// Get pairs of a specific dimension.
    pub fn pairs_of_dim(&self, dim: usize) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.dimension == dim).collect()
    }

    /// Get essential (infinite persistence) pairs.
    pub fn essential_pairs(&self) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.is_essential()).collect()
    }

    /// Get finite pairs only.
    pub fn finite_pairs(&self) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| !p.is_essential()).collect()
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Maximum dimension.
    pub fn max_dimension(&self) -> usize {
        self.max_dimension
    }

    /// Convert to barcode representation.
    /// A barcode is a list of (dimension, birth, death) intervals.
    pub fn to_barcode(&self) -> Barcode {
        Barcode {
            bars: self
                .pairs
                .iter()
                .map(|p| Bar {
                    dimension: p.dimension,
                    birth: p.birth,
                    death: p.death,
                })
                .collect(),
        }
    }

    /// Betti number at a given threshold (number of features alive at that scale).
    pub fn betti_number(&self, dim: usize, threshold: f64) -> usize {
        self.pairs
            .iter()
            .filter(|p| {
                p.dimension == dim && p.birth <= threshold &&
                    (p.death > threshold || p.death.is_infinite())
            })
            .count()
    }

    /// Get Betti numbers for all dimensions up to max_dimension at a threshold.
    pub fn betti_numbers(&self, threshold: f64) -> Vec<usize> {
        (0..=self.max_dimension)
            .map(|d| self.betti_number(d, threshold))
            .collect()
    }

    /// Filter pairs by minimum persistence.
    pub fn filter_by_persistence(&self, min_persistence: f64) -> PersistenceDiagram {
        let filtered: Vec<PersistencePair> = self
            .pairs
            .iter()
            .filter(|p| p.persistence() >= min_persistence || p.is_essential())
            .cloned()
            .collect();
        PersistenceDiagram::new(filtered)
    }

    /// Sort pairs by persistence (longest first).
    pub fn sorted_by_persistence(&self) -> Vec<&PersistencePair> {
        let mut ps: Vec<_> = self.pairs.iter().collect();
        ps.sort_by(|a, b| {
            b.persistence()
                .partial_cmp(&a.persistence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ps
    }
}

/// A barcode: collection of bars in various dimensions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Barcode {
    bars: Vec<Bar>,
}

/// A single bar in a barcode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bar {
    pub dimension: usize,
    pub birth: f64,
    pub death: f64,
}

impl Barcode {
    /// Get all bars.
    pub fn bars(&self) -> &[Bar] {
        &self.bars
    }

    /// Get bars of a specific dimension.
    pub fn bars_of_dim(&self, dim: usize) -> Vec<&Bar> {
        self.bars.iter().filter(|b| b.dimension == dim).collect()
    }

    /// Number of bars.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// Get unique scale values (births and deaths), sorted.
    pub fn scale_values(&self) -> Vec<f64> {
        let mut vals: Vec<f64> = self
            .bars
            .iter()
            .flat_map(|b| {
                if b.death.is_infinite() {
                    vec![b.birth]
                } else {
                    vec![b.birth, b.death]
                }
            })
            .filter(|v| v.is_finite())
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        vals
    }
}

impl fmt::Display for PersistenceDiagram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Persistence Diagram ({} pairs):", self.pairs.len())?;
        for dim in 0..=self.max_dimension {
            let dim_pairs: Vec<_> = self.pairs_of_dim(dim);
            if dim_pairs.is_empty() {
                continue;
            }
            writeln!(f, "  H{}:", dim)?;
            for p in &dim_pairs {
                if p.is_essential() {
                    writeln!(f, "    ({:.4}, ∞)", p.birth)?;
                } else {
                    writeln!(f, "    ({:.4}, {:.4}) pers={:.4}", p.birth, p.death, p.persistence())?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filtration::Filtration;

    #[test]
    fn test_diagram_triangle() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let diag = PersistenceDiagram::from_filtration(&filt);

        // Contractible triangle: H0 = 1, all else 0
        let h0_essential = diag.pairs_of_dim(0).iter().filter(|p| p.is_essential()).count();
        assert_eq!(h0_essential, 1);

        let h1 = diag.pairs_of_dim(1);
        // No persistent H1 features
        assert!(h1.iter().all(|p| !p.is_essential()));
    }

    #[test]
    fn test_diagram_two_components() {
        // Two pairs of close points
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let diag = PersistenceDiagram::from_filtration(&filt);

        // Full Rips eventually merges everything into 1 component
        let h0_essential = diag.pairs_of_dim(0).iter().filter(|p| p.is_essential()).count();
        assert_eq!(h0_essential, 1);
        
        // But we should see long bars for the two clusters
        let h0_finite = diag.finite_pairs().iter().filter(|p| p.dimension == 0).count();
        assert_eq!(h0_finite, 3); // 3 vertices die
    }

    #[test]
    fn test_barcode() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let barcode = diag.to_barcode();

        assert_eq!(barcode.len(), diag.len());
        assert!(barcode.scale_values().len() > 0);
    }

    #[test]
    fn test_betti_number() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let diag = PersistenceDiagram::from_filtration(&filt);

        // At threshold 0, H0 should have 3 components (3 isolated points)
        assert_eq!(diag.betti_number(0, 0.0), 3);

        // At threshold 1.0, H0 should have 1 component (all connected)
        assert_eq!(diag.betti_number(0, 1.0), 1);
    }

    #[test]
    fn test_filter_by_persistence() {
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![10.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let filtered = diag.filter_by_persistence(5.0);

        // Only essential features and long bars should survive
        for p in filtered.finite_pairs() {
            assert!(p.persistence() >= 5.0);
        }
    }

    #[test]
    fn test_persistence_pair_properties() {
        let pair = PersistencePair {
            birth_index: 0,
            death_index: Some(1),
            dimension: 0,
            birth: 0.0,
            death: 1.0,
        };
        assert!(!pair.is_essential());
        assert!((pair.persistence() - 1.0).abs() < 1e-10);
        assert!((pair.midpoint() - 0.5).abs() < 1e-10);

        let essential = PersistencePair {
            birth_index: 0,
            death_index: None,
            dimension: 0,
            birth: 0.0,
            death: f64::INFINITY,
        };
        assert!(essential.is_essential());
        assert!(essential.persistence().is_infinite());
    }

    #[test]
    fn test_sorted_by_persistence() {
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![3.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let sorted = diag.sorted_by_persistence();

        for i in 1..sorted.len() {
            assert!(sorted[i - 1].persistence() >= sorted[i].persistence()
                || sorted[i - 1].is_essential());
        }
    }

    #[test]
    fn test_diagram_display() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let diag = PersistenceDiagram::from_filtration(&filt);
        let s = format!("{}", diag);
        assert!(s.contains("Persistence Diagram"));
    }

    #[test]
    fn test_betti_numbers() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let diag = PersistenceDiagram::from_filtration(&filt);

        let bn = diag.betti_numbers(0.0);
        assert_eq!(bn[0], 3); // Three isolated points

        let bn1 = diag.betti_numbers(1.0);
        assert_eq!(bn1[0], 1); // One component
    }
}
