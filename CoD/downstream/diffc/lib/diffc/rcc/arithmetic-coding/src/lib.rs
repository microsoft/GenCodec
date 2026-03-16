//! Arithmetic coding library

#![deny(missing_docs, missing_debug_implementations)]

pub use arithmetic_coding_core::{fixed_length, max_length, one_shot, BitStore, Model};

pub mod decoder;
pub mod encoder;

pub use decoder::Decoder;
pub use encoder::Encoder;

/// Errors that can occur during encoding/decoding
#[derive(Debug, thiserror::Error)]
pub enum Error<E> {
    /// Io error when reading/writing bits from a stream
    Io(#[from] std::io::Error),

    /// Invalid symbol
    ValueError(E),
}
