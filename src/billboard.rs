//! Original-source billboard geometry, distinct from composite impostors.
//!
//! The source is rotated about its placement anchor before instance scaling.
//! Source forward is the camera-back vector dotted with the scaled axes;
//! horizontal right is normalized, forward is intentionally not normalized.
use crate::{Error, Result};

/// Source bounds used to recover the original placement anchor from
/// the normalized mesh coordinate system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Billboard {
    center: [f32; 3],
    radius: f32,
}

impl Billboard {
    /// # Errors
    /// Rejects nonfinite bounds or a nonpositive radius.
    pub fn new(center: [f32; 3], radius: f32) -> Result<Self> {
        if !center.iter().all(|v| v.is_finite()) || !radius.is_finite() || radius <= 0.0 {
            return Err(Error::InvalidConfiguration(
                "invalid billboard source bounds",
            ));
        }
        Ok(Self { center, radius })
    }

    #[must_use]
    pub const fn center(self) -> [f32; 3] {
        self.center
    }

    #[must_use]
    pub const fn radius(self) -> f32 {
        self.radius
    }

    /// Unit-source-scale transform for the static multiview atlas.
    /// Camera back is a unit vector in source coordinates. At the zenith
    /// use the atlas's analytic azimuth right vector as the degenerate
    /// fallback; the source's cylindrical rule is otherwise unchanged.
    #[must_use]
    pub(crate) fn atlas_point(
        self,
        normalized: [f32; 3],
        back: [f32; 3],
        fallback_right: [f32; 3],
    ) -> [f32; 3] {
        let local: [f32; 3] = core::array::from_fn(|i| normalized[i] + self.center[i]);
        let horizontal = back[0].hypot(back[2]);
        let right = if horizontal > 1.0e-6 {
            [-back[2] / horizontal, 0.0, back[0] / horizontal]
        } else {
            fallback_right
        };
        core::array::from_fn(|i| {
            local[0].mul_add(
                right[i],
                local[2].mul_add(back[i], if i == 1 { local[1] } else { 0.0 }),
            ) - self.center[i]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_pivot_survives_rotation_and_zenith() {
        let source = Billboard::new([0.2, 0.5, -0.1], 4.0).unwrap();
        let anchor = [-0.2, -0.5, 0.1];
        for (back, right) in [
            ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ] {
            assert_eq!(source.atlas_point(anchor, back, right), anchor);
        }
        let point = [0.3, 0.5, 0.1];
        let turned = source.atlas_point(point, [1., 0., 0.], [0., 0., 1.]);
        assert!((turned[0] + 0.2).abs() < 1e-6);
        assert!((turned[1] - 0.5).abs() < 1e-6);
        assert!((turned[2] - 0.6).abs() < 1e-6);
        assert!(Billboard::new([0.; 3], 0.0).is_err());
    }
}
