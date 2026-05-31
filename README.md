# lau-persistent-homology

The core algorithm of topological data analysis. Build a filtration of simplicial complexes from point cloud data, reduce boundary matrices, and extract persistence diagrams — birth-death pairs that reveal the shape of your data at every scale.

## The math in 60 seconds

Given a point cloud, build a **Vietoris-Rips complex** by connecting points within distance ε. As ε grows from 0 to ∞, you get a **filtration** — a nested sequence of simplicial complexes. The **boundary matrix reduction algorithm** identifies which topological features (connected components, loops, voids) are born and die at each scale.

Key results:

- **Persistence diagram:** (birth, death) pairs — features that persist across many scales are signal, not noise
- **Bottleneck distance:** d_B(D₁, D₂) = inf{sup{d(a,φ(a))}} — the natural metric on diagrams
- **Stability theorem:** small perturbation of data → small change in diagram
- **Persistence landscape:** a stable vectorization of diagrams for ML pipelines
- **Betti curves:** Betti numbers as a function of scale — the "topological fingerprint"

References: Edelsbrunner & Harer, *Computational Topology* (2010)

## Quick start

```rust
use lau_persistent_homology::{RipsComplex, Filtration, PersistenceDiagram};

// Point cloud — two clusters with noise
let points = vec![
    vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], // cluster 1
    vec![5.0, 5.0], vec![5.1, 5.0], vec![5.0, 5.1], // cluster 2
    vec![2.5, 2.5], // noise
];

// Build Vietoris-Rips filtration
let rips = RipsComplex::from_points(&points, 3.0); // max scale
let filtration = Filtration::from_rips(&rips);

// Compute persistence diagram
let diagram = filtration.compute_persistence();

// Betti numbers at scale 1.0
let betti = diagram.betti_at_scale(1.0); // [2, 0] = two components, no loops

// Persistence landscape for ML
let landscape = diagram.to_landscape();

// Bottleneck distance between two diagrams
let d = diagram_1.bottleneck_distance(&diagram_2);
```

## Key types

| Type | What it is |
|------|-----------|
| `SimplicialComplex` | Abstract, Vietoris-Rips, Čech, Alpha, or witness complex |
| `Filtration` | A nested sequence of complexes indexed by scale |
| `PersistenceDiagram` | Birth-death pairs with dimension labels |
| `Barcode` | Visual representation of persistence pairs |
| `PersistenceLandscape` | Stable vectorization for statistical analysis |
| `BettiCurve` | Betti numbers as a function of scale |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-persistent-homology/issues) or PR. We'd love:

- GPU-accelerated boundary reduction
- Sparse optimizations for large complexes
- More complex generators (torus, Klein bottle, real datasets)
