//! The participant node, as a library.
//!
//! `hocmesh` is a command-line program and stays one. This library exists so
//! that a second front end -- the desktop app -- can drive the same node
//! without a second implementation of any of it. A UI that spoke to the
//! coordinator through its own copy of the client would drift from the CLI the
//! first time a signature or a route changed, and the two would disagree about
//! what this machine is doing while both looked correct.
//!
//! So the modules the CLI is built from are public here, and the binary in
//! `main.rs` is one caller of them among two.

pub mod client;
pub mod control;
pub mod daemon;
pub mod install;
pub mod loadtest;
pub mod pipeline;
