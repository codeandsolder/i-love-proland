//! Immutable regional terrain and local fluvial morphology.
mod channel;
pub use channel::{ChannelField, Reach, Section, Weathered};
mod region;
pub use region::{Region, RegionCell, RegionChart, RegionSample, WaterKind};
mod channel_mesh;
pub use channel_mesh::{ChannelPatch, ChannelVertex, ShorelinePartition, partition_triangle};

#[cfg(feature = "gpu")]
pub mod gpu;

// The GPU is the sole procedural geometric refinement evaluator.
// Shared coefficients also define the CPU broad-phase envelope.
const REFINEMENT_AMPLITUDES: [f64; 4] = [7.0, 3.0, 0.65, 0.4];
#[expect(
    clippy::suboptimal_flops,
    reason = "constant arithmetic; fused operations are unnecessary for an outward-rounded compile-time bound"
)]
const MAX_RELIEF: f64 = {
    // Exact envelope of the four terms, rounded outward by a millimetre.
    // The coefficients are also the GPU uniform.
    let [a, b, c, d] = REFINEMENT_AMPLITUDES;
    a + 0.5 * b + c + 2.0 * d + 0.001
};
pub use region::HeightEnvelope;
