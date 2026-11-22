//! Immutable regional terrain and local fluvial morphology.
mod channel;
pub use channel::{ChannelField, Reach, Section, Weathered};
mod region;
pub use region::{Region, RegionCell, RegionChart, RegionSample, WaterKind};
mod channel_mesh;
pub use channel_mesh::{ChannelPatch, ChannelVertex};
