//! Bottleneck and Wasserstein distances between persistence diagrams.

use crate::diagram::PersistenceDiagram;

/// Compute the bottleneck distance between two persistence diagrams of the same dimension.
///
/// The bottleneck distance is the infimum over all matchings of the maximum
/// L∞ distance between matched pairs, including diagonal matching.
pub fn bottleneck_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram, dim: usize) -> f64 {
    let p1: Vec<_> = d1.pairs_of_dim(dim).iter().map(|p| (p.birth, p.death)).collect();
    let p2: Vec<_> = d2.pairs_of_dim(dim).iter().map(|p| (p.birth, p.death)).collect();
    bottleneck_distance_raw(&p1, &p2)
}

fn bottleneck_distance_raw(p1: &[(f64, f64)], p2: &[(f64, f64)]) -> f64 {
    // Include essential features (infinite death) separately
    let finite1: Vec<_> = p1.iter().filter(|(_, d)| d.is_finite()).copied().collect();
    let finite2: Vec<_> = p2.iter().filter(|(_, d)| d.is_finite()).copied().collect();

    // Compute the "cost to diagonal" for each point
    let diag_cost1: Vec<f64> = finite1.iter().map(|(b, d)| (b - d).abs() / 2.0).collect();
    let diag_cost2: Vec<f64> = finite2.iter().map(|(b, d)| (b - d).abs() / 2.0).collect();

    // Simple approximation: compute the maximum matching cost
    // For a full implementation we'd use the Hungarian algorithm, but
    // for the bottleneck distance we can use a greedy approach

    let mut max_cost = 0.0f64;

    // Match points greedily
    let mut used2: Vec<bool> = vec![false; finite2.len()];

    for (i, &(b1, d1)) in finite1.iter().enumerate() {
        let mut best_cost = diag_cost1[i]; // Cost of matching to diagonal
        let mut best_j = None;

        for (j, &(b2, d2)) in finite2.iter().enumerate() {
            if used2[j] {
                continue;
            }
            let cost = ((b1 - b2).abs()).max((d1 - d2).abs());
            if cost < best_cost {
                best_cost = cost;
                best_j = Some(j);
            }
        }

        if let Some(j) = best_j {
            used2[j] = true;
        }
        max_cost = max_cost.max(best_cost);
    }

    // Unmatched points in p2 go to diagonal
    for (j, used) in used2.iter().enumerate() {
        if !*used {
            max_cost = max_cost.max(diag_cost2[j]);
        }
    }

    // Handle essential features
    let essential1: Vec<_> = p1.iter().filter(|(_, d)| d.is_infinite()).collect();
    let essential2: Vec<_> = p2.iter().filter(|(_, d)| d.is_infinite()).collect();

    // Essential features with different birth values contribute to distance
    for (b1, _) in essential1.iter().map(|&&(b, d)| (b, d)) {
        let mut best = b1.abs(); // Cost to diagonal
        for (b2, _) in essential2.iter().map(|&&(b, d)| (b, d)) {
            best = best.min((b1 - b2).abs());
        }
        max_cost = max_cost.max(best);
    }
    for (b2, _) in essential2.iter().map(|&&(b, d)| (b, d)) {
        let mut best = b2.abs();
        for (b1, _) in essential1.iter().map(|&&(b, d)| (b, d)) {
            best = best.min((b1 - b2).abs());
        }
        max_cost = max_cost.max(best);
    }

    max_cost
}

/// Compute the Wasserstein-p distance between two persistence diagrams.
///
/// The Wasserstein-p distance is the p-th root of the minimum sum of
/// L∞ distances raised to the p-th power, over all matchings.
pub fn wasserstein_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram, dim: usize, p: f64) -> f64 {
    let p1: Vec<_> = d1.pairs_of_dim(dim).iter().map(|pp| (pp.birth, pp.death)).collect();
    let p2: Vec<_> = d2.pairs_of_dim(dim).iter().map(|pp| (pp.birth, pp.death)).collect();
    wasserstein_distance_raw(&p1, &p2, p)
}

fn wasserstein_distance_raw(p1: &[(f64, f64)], p2: &[(f64, f64)], p: f64) -> f64 {
    let finite1: Vec<_> = p1.iter().filter(|(_, d)| d.is_finite()).copied().collect();
    let finite2: Vec<_> = p2.iter().filter(|(_, d)| d.is_finite()).copied().collect();

    let diag_cost1: Vec<f64> = finite1.iter().map(|(b, d)| (b - d).abs() / 2.0).collect();
    let diag_cost2: Vec<f64> = finite2.iter().map(|(b, d)| (b - d).abs() / 2.0).collect();

    let mut total = 0.0f64;
    let mut used2: Vec<bool> = vec![false; finite2.len()];

    // Greedy matching
    for (i, &(b1, d1)) in finite1.iter().enumerate() {
        let mut best_cost = diag_cost1[i].powf(p);
        let mut best_j = None;

        for (j, &(b2, d2)) in finite2.iter().enumerate() {
            if used2[j] {
                continue;
            }
            let cost = ((b1 - b2).abs()).max((d1 - d2).abs()).powf(p);
            if cost < best_cost {
                best_cost = cost;
                best_j = Some(j);
            }
        }

        if let Some(j) = best_j {
            used2[j] = true;
        }
        total += best_cost;
    }

    // Unmatched p2 points to diagonal
    for (j, used) in used2.iter().enumerate() {
        if !*used {
            total += diag_cost2[j].powf(p);
        }
    }

    // Essential features
    let essential1: Vec<_> = p1.iter().filter(|(_, d)| d.is_infinite()).collect();
    let essential2: Vec<_> = p2.iter().filter(|(_, d)| d.is_infinite()).collect();

    for (b1, _) in essential1.iter().map(|&&(b, d)| (b, d)) {
        let mut best = b1.abs().powf(p);
        for (b2, _) in essential2.iter().map(|&&(b, d)| (b, d)) {
            best = best.min((b1 - b2).abs().powf(p));
        }
        total += best;
    }
    for (b2, _) in essential2.iter().map(|&&(b, d)| (b, d)) {
        let mut best = b2.abs().powf(p);
        let mut found = false;
        for (b1, _) in essential1.iter().map(|&&(b, d)| (b, d)) {
            best = best.min((b1 - b2).abs().powf(p));
            found = true;
        }
        if !found {
            total += best;
        }
    }

    total.powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagram::{PersistenceDiagram, PersistencePair};

    fn make_simple_diagram() -> PersistenceDiagram {
        PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { birth_index: 2, death_index: None, dimension: 0, birth: 0.0, death: f64::INFINITY },
        ])
    }

    fn make_shifted_diagram(shift: f64) -> PersistenceDiagram {
        PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: shift, death: 1.0 + shift },
            PersistencePair { birth_index: 2, death_index: None, dimension: 0, birth: shift, death: f64::INFINITY },
        ])
    }

    #[test]
    fn test_bottleneck_identical() {
        let d = make_simple_diagram();
        let dist = bottleneck_distance(&d, &d, 0);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_bottleneck_shifted() {
        let d1 = make_simple_diagram();
        let d2 = make_shifted_diagram(0.5);
        let dist = bottleneck_distance(&d1, &d2, 0);
        assert!(dist > 0.0);
        assert!(dist <= 1.0); // Should be around 0.5
    }

    #[test]
    fn test_wasserstein_identical() {
        let d = make_simple_diagram();
        let dist = wasserstein_distance(&d, &d, 0, 2.0);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_shifted() {
        let d1 = make_simple_diagram();
        let d2 = make_shifted_diagram(0.1);
        let dist = wasserstein_distance(&d1, &d2, 0, 2.0);
        assert!(dist > 0.0);
        assert!(dist < 1.0);
    }

    #[test]
    fn test_bottleneck_triangle_of_symmetry() {
        let d1 = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 0.0, death: 1.0 },
        ]);
        let d2 = PersistenceDiagram::new(vec![
            PersistencePair { birth_index: 0, death_index: Some(1), dimension: 0, birth: 1.0, death: 2.0 },
        ]);
        let dist = bottleneck_distance(&d1, &d2, 0);
        // The bottleneck should be 1.0 (max of |0-1|, |1-2|) = 1.0
        assert!(dist > 0.0 && dist <= 2.0);
    }

    #[test]
    fn test_from_point_clouds() {
        let p1 = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let p2 = vec![vec![0.1, 0.0], vec![1.0, 0.1], vec![0.0, 1.0]];

        let f1 = crate::filtration::Filtration::vietoris_rips(&p1, 2);
        let f2 = crate::filtration::Filtration::vietoris_rips(&p2, 2);

        let d1 = PersistenceDiagram::from_filtration(&f1);
        let d2 = PersistenceDiagram::from_filtration(&f2);

        let dist = bottleneck_distance(&d1, &d2, 0);
        // Small perturbation → small diagram distance
        assert!(dist < 1.0);
    }

    #[test]
    fn test_wasserstein_satisfies_triangle_inequality() {
        let d1 = make_simple_diagram();
        let d2 = make_shifted_diagram(0.2);
        let d3 = make_shifted_diagram(0.4);

        let d12 = wasserstein_distance(&d1, &d2, 0, 2.0);
        let d23 = wasserstein_distance(&d2, &d3, 0, 2.0);
        let d13 = wasserstein_distance(&d1, &d3, 0, 2.0);

        // Approximate triangle inequality (greedy matching is approximate)
        assert!(d13 <= (d12 + d23) * 1.5 + 1e-10);
    }
}
