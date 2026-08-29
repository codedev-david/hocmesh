//! Executing a contiguous range of a model's transformer blocks.
//!
//! hocMESH could already say which layers a machine should hold, price the
//! work, settle it and move the bytes. It could not run a layer range: a node
//! handed an assignment loaded a whole model into an external llama.cpp, which
//! means a model that fits on no single participating machine could be
//! planned but never executed. This crate is that missing step.
//!
//! A [`stage::Stage`] holds blocks `[start, end)` and is handed the activation
//! the previous stage produced. Nothing in it reaches for a weight outside its
//! own range, so it runs on a machine that holds only its own layers — which
//! is the arrangement the whole system exists to make possible.

pub mod config;
pub mod dequant;
pub mod fixture;
pub mod stage;
pub mod weights;

pub use config::{ASSUME_ARCHITECTURE, ModelConfig, RopeStyle};
pub use stage::{Activation, Session, Stage};
pub use weights::{Tensor, WeightFile};
