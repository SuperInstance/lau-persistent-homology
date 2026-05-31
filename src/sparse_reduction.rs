//! Sparse reduction for large complexes.

use crate::boundary::BoundaryMatrix;
use crate::diagram::{PersistenceDiagram, PersistencePair};

/// Sparse boundary matrix reduction using a hash-based pivot tracking.
/// More memory-efficient for large, sparse boundary matrices.
pub fn sparse_reduce(mut matrix: BoundaryMatrix) -> (BoundaryMatrix, Vec<PersistencePair>) {
    let n = matrix.n();
    use std::collections::HashMap;
    let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
    let mut pairs: Vec<PersistencePair> = Vec::new();
    let originally_zero: Vec<bool> = (0..n).map(|j| matrix.is_zero(j)).collect();

    for j in 0..n {
        // Reduce column j
        let mut current_low = matrix.low(j);
        while let Some(low) = current_low {
            if let Some(&k) = pivot_to_col.get(&low) {
                matrix.add_column(k, j);
                current_low = matrix.low(j);
            } else {
                pivot_to_col.insert(low, j);
                break;
            }
        }
    }

    // Extract pairs
    let mut paired: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (&low, &col) in &pivot_to_col {
        pairs.push(PersistencePair {
            birth_index: low,
            death_index: Some(col),
            dimension: matrix.dim(low),
            birth: 0.0,
            death: 0.0,
        });
        paired.insert(low);
        paired.insert(col);
    }

    // Unpaired = infinite persistence (only originally-zero columns)
    for j in 0..n {
        if !paired.contains(&j) && originally_zero[j] {
            pairs.push(PersistencePair {
                birth_index: j,
                death_index: None,
                dimension: matrix.dim(j),
                birth: 0.0,
                death: f64::INFINITY,
            });
        }
    }

    (matrix, pairs)
}

/// Compute a persistence diagram using sparse reduction.
pub fn sparse_reduce_to_diagram(matrix: BoundaryMatrix) -> PersistenceDiagram {
    let (_, pairs) = sparse_reduce(matrix);
    PersistenceDiagram::new(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filtration::Filtration;
    use crate::boundary::BoundaryMatrix;

    fn make_test_filtration() -> BoundaryMatrix {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        BoundaryMatrix::from_filtration(&filt)
    }

    #[test]
    fn test_sparse_reduce_basic() {
        let bm = make_test_filtration();
        let (_reduced, pairs) = sparse_reduce(bm);
        assert!(pairs.len() > 0);

        let h0_infinite: Vec<_> = pairs
            .iter()
            .filter(|p| p.dimension == 0 && p.death_index.is_none())
            .collect();
        assert_eq!(h0_infinite.len(), 1); // One component
    }

    #[test]
    fn test_sparse_matches_standard() {
        use crate::reduction::reduce;

        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);

        let bm1 = BoundaryMatrix::from_filtration(&filt);
        let bm2 = BoundaryMatrix::from_filtration(&filt);

        let (_, standard_pairs) = reduce(bm1);
        let (_, sparse_pairs) = sparse_reduce(bm2);

        // Both should find the same number of pairs
        assert_eq!(standard_pairs.len(), sparse_pairs.len());

        // Same number of infinite H0 features
        let std_inf = standard_pairs.iter().filter(|p| p.death_index.is_none() && p.dimension == 0).count();
        let sp_inf = sparse_pairs.iter().filter(|p| p.death_index.is_none() && p.dimension == 0).count();
        assert_eq!(std_inf, sp_inf);
    }

    #[test]
    fn test_sparse_single_point() {
        let points = vec![vec![0.0]];
        let filt = Filtration::vietoris_rips(&points, 0);
        let bm = BoundaryMatrix::from_filtration(&filt);
        let (_, pairs) = sparse_reduce(bm);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].death_index.is_none());
    }

    #[test]
    fn test_sparse_two_points() {
        let points = vec![vec![0.0], vec![1.0]];
        let filt = Filtration::vietoris_rips(&points, 1);
        let bm = BoundaryMatrix::from_filtration(&filt);
        let (_, pairs) = sparse_reduce(bm);

        // Two vertices merge into one component when edge appears
        let h0_infinite = pairs.iter().filter(|p| p.dimension == 0 && p.death_index.is_none()).count();
        assert_eq!(h0_infinite, 1);
    }
}
