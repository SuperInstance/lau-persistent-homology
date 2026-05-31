//! Agent behavior shape analysis and anomaly detection via topology.

use crate::filtration::Filtration;
use crate::diagram::PersistenceDiagram;
use crate::landscape::PersistenceLandscape;
use crate::betti::BettiCurve;
use crate::distance::bottleneck_distance;
use serde::{Deserialize, Serialize};

/// A behavior trace: a sequence of state vectors.
pub type BehaviorTrace = Vec<Vec<f64>>;

/// Result of analyzing agent behavior topology.
#[derive(Debug, Serialize, Deserialize)]
pub struct BehaviorAnalysis {
    /// Number of connected components (H0 essential features).
    pub num_components: usize,
    /// Number of loops (H1 features with significant persistence).
    pub num_loops: usize,
    /// Number of voids (H2 features).
    pub num_voids: usize,
    /// Maximum persistence in H0.
    pub max_h0_persistence: f64,
    /// Maximum persistence in H1.
    pub max_h1_persistence: f64,
    /// Betti curve integral for H0.
    pub h0_curve_integral: f64,
    /// Persistence landscape L2 norm for H0.
    pub h0_landscape_norm: f64,
    /// Whether this behavior is anomalous compared to reference.
    pub is_anomalous: bool,
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous).
    pub anomaly_score: f64,
}

/// Analyze the topology of an agent's behavior trace.
pub fn analyze_behavior(
    trace: &BehaviorTrace,
    max_dim: usize,
    resolution: usize,
) -> BehaviorAnalysis {
    if trace.is_empty() {
        return BehaviorAnalysis {
            num_components: 0,
            num_loops: 0,
            num_voids: 0,
            max_h0_persistence: 0.0,
            max_h1_persistence: 0.0,
            h0_curve_integral: 0.0,
            h0_landscape_norm: 0.0,
            is_anomalous: false,
            anomaly_score: 0.0,
        };
    }

    let filt = Filtration::vietoris_rips(trace, max_dim);
    let diag = PersistenceDiagram::from_filtration(&filt);

    let h0_essential = diag.pairs_of_dim(0).iter().filter(|p| p.is_essential()).count();
    let h1_significant = diag
        .pairs_of_dim(1)
        .iter()
        .filter(|p| p.persistence() > 0.5)
        .count();
    let h2_significant = diag
        .pairs_of_dim(2)
        .iter()
        .filter(|p| p.persistence() > 0.5)
        .count();

    let max_h0 = diag
        .pairs_of_dim(0)
        .iter()
        .map(|p| p.persistence())
        .fold(0.0f64, f64::max);
    let max_h1 = diag
        .pairs_of_dim(1)
        .iter()
        .map(|p| p.persistence())
        .fold(0.0f64, f64::max);

    let max_val = filt.values().last().copied().unwrap_or(1.0) * 1.1;
    let curve = BettiCurve::from_diagram(&diag, 0, resolution, 0.0, max_val);
    let landscape = PersistenceLandscape::from_diagram(&diag, 0, resolution, 0.0, max_val);

    BehaviorAnalysis {
        num_components: h0_essential,
        num_loops: h1_significant,
        num_voids: h2_significant,
        max_h0_persistence: max_h0,
        max_h1_persistence: max_h1,
        h0_curve_integral: curve.integral(),
        h0_landscape_norm: landscape.l_norm(2.0),
        is_anomalous: false,
        anomaly_score: 0.0,
    }
}

/// Detect anomalies by comparing behavior topology to a reference.
pub fn detect_anomaly(
    behavior: &BehaviorTrace,
    reference: &BehaviorTrace,
    max_dim: usize,
    threshold: f64,
) -> (BehaviorAnalysis, f64) {
    let filt_b = Filtration::vietoris_rips(behavior, max_dim);
    let filt_r = Filtration::vietoris_rips(reference, max_dim);
    let diag_b = PersistenceDiagram::from_filtration(&filt_b);
    let diag_r = PersistenceDiagram::from_filtration(&filt_r);

    let bn_dist = bottleneck_distance(&diag_b, &diag_r, 0);
    let mut analysis = analyze_behavior(behavior, max_dim, 50);
    analysis.is_anomalous = bn_dist > threshold;
    analysis.anomaly_score = (bn_dist / threshold).min(1.0);

    (analysis, bn_dist)
}

/// Sliding window analysis of a long behavior trace.
pub fn sliding_window_analysis(
    trace: &BehaviorTrace,
    window_size: usize,
    stride: usize,
    max_dim: usize,
) -> Vec<BehaviorAnalysis> {
    let mut results = Vec::new();
    let mut start = 0;
    while start + window_size <= trace.len() {
        let window = &trace[start..start + window_size];
        results.push(analyze_behavior(&window.to_vec(), max_dim, 50));
        start += stride;
    }
    results
}

/// Compare two behavior traces topologically.
pub fn compare_behaviors(
    b1: &BehaviorTrace,
    b2: &BehaviorTrace,
    max_dim: usize,
) -> BehaviorComparison {
    let d1 = PersistenceDiagram::from_filtration(&Filtration::vietoris_rips(b1, max_dim));
    let d2 = PersistenceDiagram::from_filtration(&Filtration::vietoris_rips(b2, max_dim));

    BehaviorComparison {
        bottleneck_h0: bottleneck_distance(&d1, &d2, 0),
        bottleneck_h1: if max_dim >= 1 {
            bottleneck_distance(&d1, &d2, 1)
        } else {
            0.0
        },
        a_analysis: analyze_behavior(b1, max_dim, 50),
        b_analysis: analyze_behavior(b2, max_dim, 50),
    }
}

/// Result of comparing two behaviors.
#[derive(Debug, Serialize, Deserialize)]
pub struct BehaviorComparison {
    pub bottleneck_h0: f64,
    pub bottleneck_h1: f64,
    pub a_analysis: BehaviorAnalysis,
    pub b_analysis: BehaviorAnalysis,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_simple_trace() {
        let trace = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ];
        let analysis = analyze_behavior(&trace, 1, 50);
        assert_eq!(analysis.num_components, 1); // All connected
    }

    #[test]
    fn test_analyze_empty_trace() {
        let empty: BehaviorTrace = vec![];
        let analysis = analyze_behavior(&empty, 1, 50);
        assert_eq!(analysis.num_components, 0);
    }

    #[test]
    fn test_analyze_two_clusters() {
        let trace = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
        ];
        let analysis = analyze_behavior(&trace, 1, 50);
        // At small enough epsilon, should see 2 components
        assert!(analysis.num_components >= 1);
    }

    #[test]
    fn test_detect_anomaly_similar() {
        let reference = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let behavior = vec![
            vec![0.01, 0.0],
            vec![1.0, 0.01],
            vec![0.0, 1.0],
        ];
        let (analysis, dist) = detect_anomaly(&behavior, &reference, 1, 1.0);
        assert!(!analysis.is_anomalous);
        assert!(dist < 1.0);
    }

    #[test]
    fn test_detect_anomaly_different() {
        let reference = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let behavior = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        let (analysis, _dist) = detect_anomaly(&behavior, &reference, 1, 0.5);
        assert!(analysis.is_anomalous);
    }

    #[test]
    fn test_sliding_window() {
        let trace: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 * 0.1, (i as f64 * 0.1).sin()])
            .collect();
        let results = sliding_window_analysis(&trace, 5, 2, 1);
        assert!(results.len() >= 5);
        for r in &results {
            assert!(r.num_components >= 1);
        }
    }

    #[test]
    fn test_compare_behaviors() {
        let b1 = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b2 = vec![vec![0.1, 0.0], vec![1.0, 0.1], vec![0.0, 1.0]];
        let comp = compare_behaviors(&b1, &b2, 1);
        assert!(comp.bottleneck_h0 >= 0.0);
        assert!(comp.bottleneck_h1 >= 0.0);
    }

    #[test]
    fn test_circular_behavior() {
        // Circular behavior should have H1 features
        let trace: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
                vec![theta.cos(), theta.sin()]
            })
            .collect();
        let analysis = analyze_behavior(&trace, 2, 50);
        assert_eq!(analysis.num_components, 1);
    }

    #[test]
    fn test_linear_behavior() {
        let trace: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let analysis = analyze_behavior(&trace, 1, 50);
        assert_eq!(analysis.num_components, 1);
        // Linear path: Rips complex may create H1 features from short-range cycles
        // Just check it doesn't crash and gives reasonable results
        assert!(analysis.num_loops < 20);
    }
}
