#[cfg(any(
    all(feature = "cuda", feature = "coreml"),
    all(feature = "cuda", feature = "webgpu"),
    all(feature = "coreml", feature = "webgpu")
))]
compile_error!("enable only one of the cuda, coreml, or webgpu features at a time");

mod audio;
mod execution;
mod model;
mod session;
mod stream;
mod types;

pub use model::{Canary, ExecutionConfig, SessionConfig};
#[cfg(feature = "coreml")]
pub use ort::ep::coreml::{
    ComputeUnits as CoreMLComputeUnits, ModelFormat as CoreMLModelFormat,
    SpecializationStrategy as CoreMLSpecializationStrategy,
};
pub use session::CanarySession;
pub use stream::{CanaryStream, StreamChunk, StreamConfig};
pub use types::{BeamLogitsProcessor, CanaryError, CanaryResult, LogitsProcessor, Result, Token};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_config() {
        let config = ExecutionConfig::new().with_threads(2, 2);

        assert_eq!(config.inter_threads, 2);
        assert_eq!(config.intra_threads, 2);
    }
}
