//! Simplices and simplex operations.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// A simplex represented as a sorted list of vertex indices.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Simplex {
    vertices: Vec<usize>,
}

impl Simplex {
    /// Create a new simplex from vertices. Sorts the vertices.
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort_unstable();
        vertices.dedup();
        Simplex { vertices }
    }

    /// Create a vertex (0-simplex).
    pub fn vertex(v: usize) -> Self {
        Simplex { vertices: vec![v] }
    }

    /// Create an edge (1-simplex).
    pub fn edge(a: usize, b: usize) -> Self {
        Simplex::new(vec![a, b])
    }

    /// Create a triangle (2-simplex).
    pub fn triangle(a: usize, b: usize, c: usize) -> Self {
        Simplex::new(vec![a, b, c])
    }

    /// Create a tetrahedron (3-simplex).
    pub fn tetrahedron(a: usize, b: usize, c: usize, d: usize) -> Self {
        Simplex::new(vec![a, b, c, d])
    }

    /// Dimension of the simplex (number of vertices - 1).
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Get the vertices.
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the boundary faces (all (dim-1)-faces).
    pub fn faces(&self) -> Vec<Simplex> {
        if self.vertices.is_empty() {
            return vec![];
        }
        let n = self.vertices.len();
        if n == 1 {
            return vec![]; // vertices have empty boundary
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            result.push(Simplex { vertices: face_vertices });
        }
        result
    }

    /// Check if this simplex contains another simplex (as a face).
    pub fn contains_face(&self, other: &Simplex) -> bool {
        let mut i = 0;
        for &v in other.vertices.iter() {
            while i < self.vertices.len() && self.vertices[i] < v {
                i += 1;
            }
            if i >= self.vertices.len() || self.vertices[i] != v {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Is this simplex a face of another?
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        other.contains_face(self)
    }
}

impl fmt::Display for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.vertices.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}

impl Ord for Simplex {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by dimension first, then lexicographically
        match self.dimension().cmp(&other.dimension()) {
            Ordering::Equal => self.vertices.cmp(&other.vertices),
            ord => ord,
        }
    }
}

impl PartialOrd for Simplex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_creation() {
        let s = Simplex::new(vec![3, 1, 2]);
        assert_eq!(s.vertices(), &[1, 2, 3]);
    }

    #[test]
    fn test_simplex_dimension() {
        assert_eq!(Simplex::vertex(0).dimension(), 0);
        assert_eq!(Simplex::edge(0, 1).dimension(), 1);
        assert_eq!(Simplex::triangle(0, 1, 2).dimension(), 2);
        assert_eq!(Simplex::tetrahedron(0, 1, 2, 3).dimension(), 3);
    }

    #[test]
    fn test_simplex_faces() {
        let tri = Simplex::triangle(0, 1, 2);
        let faces = tri.faces();
        assert_eq!(faces.len(), 3);
        assert!(faces.contains(&Simplex::edge(0, 1)));
        assert!(faces.contains(&Simplex::edge(0, 2)));
        assert!(faces.contains(&Simplex::edge(1, 2)));
    }

    #[test]
    fn test_vertex_has_no_faces() {
        assert!(Simplex::vertex(0).faces().is_empty());
    }

    #[test]
    fn test_tetrahedron_faces() {
        let tet = Simplex::tetrahedron(0, 1, 2, 3);
        let faces = tet.faces();
        assert_eq!(faces.len(), 4);
        assert!(faces.contains(&Simplex::triangle(0, 1, 2)));
        assert!(faces.contains(&Simplex::triangle(0, 1, 3)));
        assert!(faces.contains(&Simplex::triangle(0, 2, 3)));
        assert!(faces.contains(&Simplex::triangle(1, 2, 3)));
    }

    #[test]
    fn test_contains_face() {
        let tri = Simplex::triangle(0, 1, 2);
        assert!(tri.contains_face(&Simplex::edge(0, 1)));
        assert!(tri.contains_face(&Simplex::vertex(0)));
        assert!(!tri.contains_face(&Simplex::edge(3, 4)));
    }

    #[test]
    fn test_dedup() {
        let s = Simplex::new(vec![1, 1, 2, 2]);
        assert_eq!(s.vertices(), &[1, 2]);
    }
}
