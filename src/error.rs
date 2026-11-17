use core::fmt;

/// Errors are explicit: placement never silently drops or caps candidates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidConfiguration(&'static str),
    InvalidTile,
    IndexOverflow,
    DensityExceedsStratification,
    CandidateLimitExceeded,
    InvalidHabitat,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(s) => write!(f, "invalid configuration: {s}"),
            Self::InvalidTile => write!(f, "tile is outside the configured domain"),
            Self::IndexOverflow => write!(f, "tile or cell coordinate overflow"),
            Self::DensityExceedsStratification => {
                write!(
                    f,
                    "stratified distribution supports at most one expected object per cell; decrease the cell size or use Poisson"
                )
            }
            Self::CandidateLimitExceeded => {
                write!(f, "explicit per-tile candidate budget exhausted")
            }
            Self::InvalidHabitat => write!(f, "habitat fraction must be finite and in [0, 1]"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
