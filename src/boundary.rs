//! Boundary matrices for simplicial complexes.

use crate::simplex::Simplex;
use crate::filtration::Filtration;
use std::collections::HashMap;

/// A boundary matrix stored in sparse column format.
/// Column j represents the boundary of simplex j in the filtration order.
#[derive(Clone, Debug)]
pub struct BoundaryMatrix {
    /// Number of rows (and columns — it's square).
    n: usize,
    /// For each column j, the set of row indices with nonzero entries.
    columns: Vec<Vec<usize>>,
    /// Mapping from filtration index to simplex.
    index_to_simplex: Vec<Simplex>,
    /// Mapping from simplex to filtration index.
    simplex_to_index: HashMap<Simplex, usize>,
    /// Dimension of each column's simplex.
    dims: Vec<usize>,
}

impl BoundaryMatrix {
    /// Build a boundary matrix from a filtration.
    pub fn from_filtration(filtration: &Filtration) -> Self {
        let entries = filtration.entries();
        let n = entries.len();

        let index_to_simplex: Vec<Simplex> = entries.iter().map(|(s, _)| s.clone()).collect();
        let simplex_to_index: HashMap<Simplex, usize> = index_to_simplex
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let dims: Vec<usize> = entries.iter().map(|(s, _)| s.dimension()).collect();

        let mut columns = vec![Vec::new(); n];
        for j in 0..n {
            let simplex = &index_to_simplex[j];
            if simplex.dimension() == 0 {
                // Vertices have empty boundary
                columns[j] = Vec::new();
            } else {
                let faces = simplex.faces();
                let mut col: Vec<usize> = faces
                    .iter()
                    .filter_map(|f| simplex_to_index.get(f).copied())
                    .collect();
                col.sort_unstable();
                col.dedup();
                columns[j] = col;
            }
        }

        BoundaryMatrix {
            n,
            columns,
            index_to_simplex,
            simplex_to_index,
            dims,
        }
    }

    /// Get the column (boundary) of simplex at index j.
    pub fn column(&self, j: usize) -> &[usize] {
        &self.columns[j]
    }

    /// Number of columns/rows.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the simplex at index i.
    pub fn simplex_at(&self, i: usize) -> &Simplex {
        &self.index_to_simplex[i]
    }

    /// Get the index of a simplex.
    pub fn index_of(&self, simplex: &Simplex) -> Option<usize> {
        self.simplex_to_index.get(simplex).copied()
    }

    /// Get the dimension of simplex at index i.
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    /// Add column j to column k (mod 2).
    pub fn add_column(&mut self, j: usize, k: usize) {
        let col_j = self.columns[j].clone();
        let col_k = &mut self.columns[k];
        // XOR merge
        let mut result = Vec::new();
        let (mut i, mut l) = (0, 0);
        while i < col_j.len() && l < col_k.len() {
            if col_j[i] < col_k[l] {
                result.push(col_j[i]);
                i += 1;
            } else if col_j[i] > col_k[l] {
                result.push(col_k[l]);
                l += 1;
            } else {
                // Both present: cancel (mod 2)
                i += 1;
                l += 1;
            }
        }
        while i < col_j.len() {
            result.push(col_j[i]);
            i += 1;
        }
        while l < col_k.len() {
            result.push(col_k[l]);
            l += 1;
        }
        *col_k = result;
    }

    /// Get the lowest nonzero entry in column j (i.e., max row index).
    pub fn low(&self, j: usize) -> Option<usize> {
        self.columns[j].last().copied()
    }

    /// Check if column j is zero.
    pub fn is_zero(&self, j: usize) -> bool {
        self.columns[j].is_empty()
    }

    /// Get all column indices of a given dimension.
    pub fn columns_of_dim(&self, dim: usize) -> Vec<usize> {
        (0..self.n).filter(|&i| self.dims[i] == dim).collect()
    }

    /// Count nonzero entries (for density analysis).
    pub fn nnz(&self) -> usize {
        self.columns.iter().map(|c| c.len()).sum()
    }

    /// Deep clone (useful before reduction).
    pub fn clone_matrix(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_triangle() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let bm = BoundaryMatrix::from_filtration(&filt);

        // Vertices have empty boundary
        for i in 0..3 {
            assert!(bm.is_zero(i));
        }

        // Edge boundaries have one entry (the vertex not being the edge endpoint... actually
        // the boundary of an edge [a,b] is [a] + [b])
        for i in 3..6 {
            assert_eq!(bm.column(i).len(), 2);
        }

        // Triangle boundary has 3 entries
        let tri_col = bm.column(6);
        assert_eq!(tri_col.len(), 3);
    }

    #[test]
    fn test_add_column() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let mut bm = BoundaryMatrix::from_filtration(&filt);

        // Add column 3 to column 4
        let _c3 = bm.column(3).to_vec();
        let c4 = bm.column(4).to_vec();
        bm.add_column(3, 4);
        let c4_new = bm.column(4).to_vec();
        // Should be XOR of c3 and c4
        assert_ne!(c4_new, c4);
    }

    #[test]
    fn test_boundary_matrix_size() {
        let points = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        let filt = Filtration::vietoris_rips(&points, 2);
        let bm = BoundaryMatrix::from_filtration(&filt);
        assert_eq!(bm.n(), filt.len());
    }
}
