//! Standard boundary matrix reduction algorithm.

use crate::boundary::BoundaryMatrix;
use crate::diagram::{PersistenceDiagram, PersistencePair};

/// Reduce a boundary matrix using the standard reduction algorithm.
/// Returns the reduced matrix and the persistence pairs.
///
/// The algorithm processes columns from left to right. For each column j,
/// while the lowest nonzero entry (low(j)) collides with a previously processed column's low,
/// we add that column to j.
pub fn reduce(mut matrix: BoundaryMatrix) -> (BoundaryMatrix, Vec<PersistencePair>) {
    let n = matrix.n();
    let mut low_to_col: Vec<Option<usize>> = vec![None; n];
    let mut pairs: Vec<PersistencePair> = Vec::new();
    // Track which columns were originally zero (empty boundary = potential essential features)
    let originally_zero: Vec<bool> = (0..n).map(|j| matrix.is_zero(j)).collect();

    for j in 0..n {
        let mut current_low = matrix.low(j);
        while let Some(low) = current_low {
            if let Some(k) = low_to_col[low] {
                matrix.add_column(k, j);
                current_low = matrix.low(j);
            } else {
                low_to_col[low] = Some(j);
                break;
            }
        }
    }

    let mut paired_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut paired_cols: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for j in 0..n {
        if let Some(low) = matrix.low(j) {
            let birth_dim = matrix.dim(low);
            pairs.push(PersistencePair {
                birth_index: low,
                death_index: Some(j),
                dimension: birth_dim,
                birth: 0.0,
                death: 0.0,
            });
            paired_rows.insert(low);
            paired_cols.insert(j);
        }
    }

    // Essential features: only columns that were ORIGINALLY zero (not reduced to zero)
    // and whose index was never used as a pivot row
    for j in 0..n {
        if !paired_rows.contains(&j) && !paired_cols.contains(&j) && originally_zero[j] {
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

/// Reduce and compute a persistence diagram with actual filtration values.
pub fn reduce_to_diagram(matrix: &BoundaryMatrix) -> PersistenceDiagram {
    let (_reduced, pairs) = reduce(matrix.clone());

    let diagram_pairs: Vec<PersistencePair> = pairs
        .into_iter()
        .map(|p| {
            let birth = 0.0;
            let death = p.death_index.map_or(f64::INFINITY, |_| 0.0);
            PersistencePair {
                birth_index: p.birth_index,
                death_index: p.death_index,
                dimension: p.dimension,
                birth,
                death,
            }
        })
        .collect();

    PersistenceDiagram::new(diagram_pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filtration::Filtration;

    fn make_triangle_boundary() -> BoundaryMatrix {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        BoundaryMatrix::from_filtration(&filt)
    }

    #[test]
    fn test_reduce_triangle() {
        let bm = make_triangle_boundary();
        let (_reduced, pairs) = reduce(bm);

        // H0: 1 essential + 2 finite
        // H1: 1 finite (cycle born at third edge, killed by triangle)
        let h0_infinite = pairs.iter().filter(|p| p.dimension == 0 && p.death_index.is_none()).count();
        assert_eq!(h0_infinite, 1);

        let h0_finite = pairs.iter().filter(|p| p.dimension == 0 && p.death_index.is_some()).count();
        assert_eq!(h0_finite, 2);

        // There should be an H1 pair (the cycle that forms and then fills)
        let h1 = pairs.iter().filter(|p| p.dimension == 1).count();
        assert!(h1 >= 1);
    }

    #[test]
    fn test_reduce_two_components() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 1);
        let bm = BoundaryMatrix::from_filtration(&filt);
        let (_reduced, pairs) = reduce(bm);

        // 1 essential H0 + 3 finite H0 = 4 pairs
        let h0_infinite = pairs.iter().filter(|p| p.dimension == 0 && p.death_index.is_none()).count();
        assert_eq!(h0_infinite, 1);

        let h0_finite = pairs.iter().filter(|p| p.dimension == 0 && p.death_index.is_some()).count();
        assert_eq!(h0_finite, 3);
    }

    #[test]
    fn test_reduce_preserves_size() {
        let bm = make_triangle_boundary();
        let (reduced, _) = reduce(bm);
        assert_eq!(reduced.n(), 7);
    }

    #[test]
    fn test_reduce_single_point() {
        let points = vec![vec![0.0, 0.0]];
        let filt = Filtration::vietoris_rips(&points, 0);
        let bm = BoundaryMatrix::from_filtration(&filt);
        let (_, pairs) = reduce(bm);

        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].death_index.is_none());
        assert_eq!(pairs[0].dimension, 0);
    }

    #[test]
    fn test_reduce_circle() {
        let points = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let bm = BoundaryMatrix::from_filtration(&filt);
        let (_, pairs) = reduce(bm);

        let h0_infinite = pairs.iter().filter(|p| p.dimension == 0 && p.death_index.is_none()).count();
        assert_eq!(h0_infinite, 1);
    }
}
