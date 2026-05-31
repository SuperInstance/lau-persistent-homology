//! # lau-persistent-homology
//!
//! Persistent homology for agent behavior analysis — filtrations, barcodes, and stability.
//!
//! This crate implements the core algorithms of topological data analysis (TDA):
//! simplicial complexes, filtrations, boundary matrix reduction, persistence diagrams,
//! and various distance and vectorization methods.

pub mod simplex;
pub mod complex;
pub mod filtration;
pub mod boundary;
pub mod reduction;
pub mod sparse_reduction;
pub mod diagram;
pub mod distance;
pub mod landscape;
pub mod betti;
pub mod euler;
pub mod stability;
pub mod behavior;
pub mod util;
