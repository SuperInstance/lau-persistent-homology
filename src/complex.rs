//! Simplicial complex types: abstract, Vietoris-Rips, Čech, Alpha, witness.

use crate::simplex::Simplex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// A simplicial complex: a collection of simplices closed under taking faces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimplicialComplex {
    simplices: BTreeSet<Simplex>,
    /// Map from dimension to sorted list of simplices
    by_dim: HashMap<usize, Vec<Simplex>>,
}

impl SimplicialComplex {
    /// Create an empty complex.
    pub fn new() -> Self {
        SimplicialComplex {
            simplices: BTreeSet::new(),
            by_dim: HashMap::new(),
        }
    }

    /// Create a complex from a list of simplices, adding all faces for closure.
    pub fn from_simplices(simplices: Vec<Simplex>) -> Self {
        let mut complex = Self::new();
        for s in simplices {
            complex.insert_closed(&s);
        }
        complex
    }

    /// Insert a simplex and all its faces (maintaining closure).
    pub fn insert_closed(&mut self, simplex: &Simplex) {
        if self.simplices.contains(simplex) {
            return;
        }
        // Insert all faces recursively
        for face in simplex.faces() {
            self.insert_closed(&face);
        }
        // Insert the simplex itself
        self.simplices.insert(simplex.clone());
        self.by_dim
            .entry(simplex.dimension())
            .or_default()
            .push(simplex.clone());
    }

    /// Insert a simplex only (no faces added — caller ensures closure).
    pub fn insert(&mut self, simplex: Simplex) {
        if self.simplices.insert(simplex.clone()) {
            self.by_dim
                .entry(simplex.dimension())
                .or_default()
                .push(simplex);
        }
    }

    /// Check if the complex contains a simplex.
    pub fn contains(&self, simplex: &Simplex) -> bool {
        self.simplices.contains(simplex)
    }

    /// Number of simplices.
    pub fn size(&self) -> usize {
        self.simplices.len()
    }

    /// Get all simplices.
    pub fn simplices(&self) -> &BTreeSet<Simplex> {
        &self.simplices
    }

    /// Get simplices of a given dimension.
    pub fn simplices_of_dim(&self, dim: usize) -> &[Simplex] {
        self.by_dim.get(&dim).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Maximum dimension of any simplex.
    pub fn max_dimension(&self) -> usize {
        self.by_dim.keys().copied().max().unwrap_or(0)
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.simplices_of_dim(0).len()
    }

    /// Get vertex indices.
    pub fn vertex_indices(&self) -> Vec<usize> {
        self.simplices_of_dim(0)
            .iter()
            .filter_map(|s| s.vertices().first().copied())
            .collect()
    }

    /// Validate that the complex is closed under face relations.
    pub fn is_valid(&self) -> bool {
        for simplex in &self.simplices {
            for face in simplex.faces() {
                if !self.simplices.contains(&face) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute Euler characteristic.
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi: i64 = 0;
        for dim in 0..=self.max_dimension() {
            let count = self.simplices_of_dim(dim).len();
            if dim % 2 == 0 {
                chi += count as i64;
            } else {
                chi -= count as i64;
            }
        }
        chi
    }
}

/// Build a Vietoris-Rips complex from a distance matrix.
pub fn vietoris_rips(distances: &[Vec<f64>], epsilon: f64, max_dim: usize) -> SimplicialComplex {
    let n = distances.len();
    let mut complex = SimplicialComplex::new();

    // Add all vertices
    for i in 0..n {
        complex.insert(Simplex::vertex(i));
    }

    if max_dim < 1 {
        return complex;
    }

    // Build simplices dimension by dimension
    // Start with edges
    let mut prev_simplices: Vec<Simplex> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if distances[i][j] <= epsilon {
                let s = Simplex::edge(i, j);
                complex.insert_closed(&s);
                prev_simplices.push(s);
            }
        }
    }

    // Build higher dimensions: for each (dim-1)-simplex, try extending with larger vertex indices
    for _dim in 2..=max_dim {
        let mut next_simplices = Vec::new();
        for simplex in &prev_simplices {
            let verts = simplex.vertices();
            let max_v = *verts.last().unwrap();
            for new_v in (max_v + 1)..n {
                // Check all pairwise distances to new_v
                let mut all_close = true;
                for &v in verts {
                    if distances[v][new_v] > epsilon {
                        all_close = false;
                        break;
                    }
                }
                if all_close {
                    let mut new_verts = verts.to_vec();
                    new_verts.push(new_v);
                    let ns = Simplex::new(new_verts);
                    complex.insert_closed(&ns);
                    next_simplices.push(ns);
                }
            }
        }
        prev_simplices = next_simplices;
        if prev_simplices.is_empty() {
            break;
        }
    }

    complex
}

/// Build a Vietoris-Rips complex from a point cloud.
pub fn vietoris_rips_from_points(points: &[Vec<f64>], epsilon: f64, max_dim: usize) -> SimplicialComplex {
    let _n = points.len();
    let distances = compute_distance_matrix(points);
    vietoris_rips(&distances, epsilon, max_dim)
}

/// Build a Čech complex from points and balls of radius r.
/// A simplex is included if there's a common intersection point of all balls.
/// For efficiency, we approximate: a simplex is included if all pairwise distances <= 2r.
pub fn cech_complex(points: &[Vec<f64>], radius: f64, max_dim: usize) -> SimplicialComplex {
    // Čech at radius r is equivalent to Rips at diameter 2r for pairwise checks
    // but Čech is actually stricter for higher simplices
    let distances = compute_distance_matrix(points);
    let diameter = 2.0 * radius;
    vietoris_rips(&distances, diameter, max_dim)
}

/// Alpha complex (simplified version using Delaunay-like construction).
/// For a proper alpha complex, we'd need a Delaunay triangulation.
/// This is a simplified version that uses the Rips complex with Voronoi-based filtering.
pub fn alpha_complex(points: &[Vec<f64>], max_dim: usize) -> SimplicialComplex {
    let n = points.len();
    if n == 0 {
        return SimplicialComplex::new();
    }

    let distances = compute_distance_matrix(points);

    // Compute nearest neighbor distances for each point
    let mut nn_dist: Vec<f64> = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && distances[i][j] < nn_dist[i] {
                nn_dist[i] = distances[i][j];
            }
        }
    }

    // Build alpha complex by checking circumsphere condition
    let mut complex = SimplicialComplex::new();
    for i in 0..n {
        complex.insert(Simplex::vertex(i));
    }

    // Add edges where the circumradius is small enough
    for i in 0..n {
        for j in (i + 1)..n {
            let cr = distances[i][j] / 2.0;
            if cr <= (nn_dist[i].min(nn_dist[j]) * 1.5) {
                complex.insert_closed(&Simplex::edge(i, j));
            }
        }
    }

    // Add higher simplices
    if max_dim >= 2 {
        let edges: Vec<_> = complex.simplices_of_dim(1).to_vec();
        for e1 in &edges {
            for e2 in &edges {
                if e1 >= e2 {
                    continue;
                }
                let v1 = e1.vertices();
                let v2 = e2.vertices();
                // Check if they share a vertex and together form a triangle
                let shared: Vec<_> = v1.iter().filter(|v| v2.contains(v)).copied().collect();
                if shared.len() == 1 {
                    let uniques: Vec<_> = v1.iter().chain(v2.iter()).filter(|v| !shared.contains(v)).copied().collect();
                    if uniques.len() == 2 {
                        let tri = Simplex::new(vec![shared[0], uniques[0], uniques[1]]);
                        if complex.contains(&Simplex::edge(shared[0], uniques[0]))
                            && complex.contains(&Simplex::edge(shared[0], uniques[1]))
                            && complex.contains(&Simplex::edge(uniques[0], uniques[1]))
                        {
                            complex.insert_closed(&tri);
                        }
                    }
                }
            }
        }
    }

    complex
}

/// Witness complex: build from landmark and witness points.
pub fn witness_complex(
    landmarks: &[Vec<f64>],
    witnesses: &[Vec<f64>],
    max_dim: usize,
    nu: usize,
) -> SimplicialComplex {
    let m = landmarks.len();
    let mut complex = SimplicialComplex::new();

    // Add landmark vertices
    for i in 0..m {
        complex.insert(Simplex::vertex(i));
    }

    // For each witness, find nearest landmarks
    let mut witness_nn: Vec<Vec<(usize, f64)>> = Vec::with_capacity(witnesses.len());
    for w in witnesses {
        let mut dists: Vec<(usize, f64)> = landmarks
            .iter()
            .enumerate()
            .map(|(i, l)| (i, euclidean_dist(w, l)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        witness_nn.push(dists);
    }

    // A simplex of landmark points is included if some witness has all its vertices
    // among its nu nearest landmarks
    if nu < 2 {
        return complex;
    }

    // Build candidate simplices up to dimension min(max_dim, nu-1)
    let actual_max = max_dim.min(nu - 1);

    // Edges
    if actual_max >= 1 {
        for wnn in &witness_nn {
            let nearest: Vec<usize> = wnn.iter().take(nu).map(|(i, _)| *i).collect();
            for i in 0..nearest.len() {
                for j in (i + 1)..nearest.len() {
                    complex.insert_closed(&Simplex::edge(nearest[i], nearest[j]));
                }
            }
        }
    }

    // Triangles and higher
    for dim in 2..=actual_max {
        let prev = complex.simplices_of_dim(dim - 1).to_vec();
        let vertices: Vec<usize> = complex.vertex_indices();
        for simplex in &prev {
            let sverts = simplex.vertices();
            for &v in &vertices {
                if v <= *sverts.last().unwrap_or(&0) {
                    continue;
                }
                let mut new_verts = sverts.to_vec();
                new_verts.push(v);
                // Check if all sub-faces exist
                let candidate = Simplex::new(new_verts);
                let faces = candidate.faces();
                if faces.iter().all(|f| complex.contains(f)) {
                    complex.insert_closed(&candidate);
                }
            }
        }
    }

    complex
}

/// Compute a distance matrix from a point cloud.
pub fn compute_distance_matrix(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_dist(&points[i], &points[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ---- Named complex generators for testing ----

/// Build the boundary of an n-simplex (homeomorphic to S^n).
pub fn sphere_complex(dim: usize) -> SimplicialComplex {
    // The boundary of a (dim+1)-simplex is S^dim
    let n = dim + 2; // number of vertices
    let all_verts: Vec<usize> = (0..n).collect();
    let mut complex = SimplicialComplex::new();
    // Add all faces of dimension `dim` (i.e., all subsets of size dim+1 from n vertices)
    add_subsets(&mut complex, &all_verts, dim + 1);
    complex
}

/// Build a torus as a simplicial complex (9 vertices, standard triangulation).
pub fn torus_complex() -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();
    // Standard 9-vertex triangulation of the torus
    // Vertices: 0..8
    // 18 triangles forming a 3x3 grid with periodic boundary
    for i in 0..3 {
        for j in 0..3 {
            let a = i * 3 + j;
            let b = i * 3 + (j + 1) % 3;
            let c = ((i + 1) % 3) * 3 + j;
            let d = ((i + 1) % 3) * 3 + (j + 1) % 3;
            complex.insert_closed(&Simplex::triangle(a, b, d));
            complex.insert_closed(&Simplex::triangle(a, c, d));
        }
    }
    complex
}

/// Build a Klein bottle as a simplicial complex.
pub fn klein_bottle_complex() -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();
    // Klein bottle using 9 vertices with a twist in one direction
    for i in 0..3 {
        for j in 0..3 {
            let a = i * 3 + j;
            let b = i * 3 + (j + 1) % 3;
            let c = ((i + 1) % 3) * 3 + j;
            let d = ((i + 1) % 3) * 3 + (j + 1) % 3;
            // Twist: in the second row, reverse one direction
            if i == 1 {
                complex.insert_closed(&Simplex::triangle(a, b, c));
                complex.insert_closed(&Simplex::triangle(b, c, d));
            } else {
                complex.insert_closed(&Simplex::triangle(a, b, d));
                complex.insert_closed(&Simplex::triangle(a, c, d));
            }
        }
    }
    complex
}

fn add_subsets(complex: &mut SimplicialComplex, items: &[usize], k: usize) {
    let n = items.len();
    if k == 0 || k > n {
        return;
    }
    let mut combo = vec![0usize; k];
    for i in 0..k {
        combo[i] = i;
    }
    loop {
        let verts: Vec<usize> = combo.iter().map(|&i| items[i]).collect();
        complex.insert_closed(&Simplex::new(verts));

        // Next combination
        let mut i = k as i32 - 1;
        while i >= 0 {
            combo[i as usize] += 1;
            if combo[i as usize] < n - (k - 1 - i as usize) {
                for j in (i as usize + 1)..k {
                    combo[j] = combo[j - 1] + 1;
                }
                break;
            }
            i -= 1;
        }
        if i < 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_complex() {
        let c = SimplicialComplex::new();
        assert_eq!(c.size(), 0);
        assert!(c.is_valid());
    }

    #[test]
    fn test_simple_complex() {
        let c = SimplicialComplex::from_simplices(vec![
            Simplex::triangle(0, 1, 2),
        ]);
        assert!(c.is_valid());
        // Triangle + 3 edges + 3 vertices = 7
        assert_eq!(c.size(), 7);
    }

    #[test]
    fn test_sphere_s1() {
        let s = sphere_complex(1);
        // Boundary of 3-simplex: 3 edges + 3 vertices = 6
        assert_eq!(s.size(), 6);
        assert!(s.is_valid());
    }

    #[test]
    fn test_sphere_s2() {
        let s = sphere_complex(2);
        // Boundary of 4-simplex (tetrahedron): 4 triangles + 6 edges + 4 vertices = 14
        assert_eq!(s.size(), 14);
        assert!(s.is_valid());
    }

    #[test]
    fn test_torus() {
        let t = torus_complex();
        assert!(t.is_valid());
        // 9 vertices, edges and triangles
        assert_eq!(t.num_vertices(), 9);
    }

    #[test]
    fn test_klein_bottle() {
        let k = klein_bottle_complex();
        assert!(k.is_valid());
        assert_eq!(k.num_vertices(), 9);
    }

    #[test]
    fn test_vietoris_rips_simple() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let vr = vietoris_rips_from_points(&points, 1.5, 2);
        assert!(vr.contains(&Simplex::vertex(0)));
        assert!(vr.contains(&Simplex::vertex(1)));
        assert!(vr.contains(&Simplex::vertex(2)));
        assert!(vr.contains(&Simplex::edge(0, 1)));
        assert!(vr.contains(&Simplex::edge(0, 2)));
        assert!(vr.contains(&Simplex::edge(1, 2)));
        // Triangle only if all pairwise dists <= 1.5
        // dist(1,2) = sqrt(2) ≈ 1.414 <= 1.5, so yes
        assert!(vr.contains(&Simplex::triangle(0, 1, 2)));
    }

    #[test]
    fn test_vietoris_rips_sparse() {
        let points = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        let vr = vietoris_rips_from_points(&points, 1.0, 2);
        assert!(vr.contains(&Simplex::vertex(0)));
        assert!(vr.contains(&Simplex::vertex(1)));
        assert!(vr.contains(&Simplex::vertex(2)));
        assert!(!vr.contains(&Simplex::edge(0, 1)));
        assert!(!vr.contains(&Simplex::edge(0, 2)));
    }

    #[test]
    fn test_euler_tetrahedron() {
        let c = SimplicialComplex::from_simplices(vec![
            Simplex::tetrahedron(0, 1, 2, 3),
        ]);
        // 4 vertices - 6 edges + 4 triangles - 1 tetrahedron = 1
        assert_eq!(c.euler_characteristic(), 1);
    }

    #[test]
    fn test_euler_sphere() {
        let s = sphere_complex(2);
        // S^2 has chi = 2
        assert_eq!(s.euler_characteristic(), 2);
    }
}
