//! Material-authored, spatially coherent wind deformation for source meshes.
//!
//! The host supplies a stable field coordinate frame; no instance IDs or
//! camera-relative coordinates are used as random animation phases.

use crate::{Error, Result};

/// Texture-driven displacement parameters in original source mesh units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindParameters {
    pub spatial_scale: f32,
    pub height_start: f32,
    pub height_exponent: f32,
    pub speed: f32,
    pub intensity: f32,
}

impl WindParameters {
    /// # Errors
    /// Rejects nonfinite or physically invalid parameters.
    pub fn validate(self) -> Result<()> {
        if ![
            self.spatial_scale,
            self.height_start,
            self.height_exponent,
            self.speed,
            self.intensity,
        ]
        .iter()
        .all(|v| v.is_finite())
            || self.spatial_scale < 0.0
            || self.height_start < 0.0
            || self.height_exponent <= 0.0
            || self.speed < 0.0
            || self.intensity < 0.0
        {
            return Err(Error::InvalidConfiguration(
                "invalid source wind parameters",
            ));
        }
        Ok(())
    }

    /// Conservative displacement in render metres for RGB samples in [0,1].
    ///
    /// # Errors
    /// Rejects invalid inputs and overflowing displacement.
    pub fn max_displacement(self, maximum_height: f32) -> Result<f32> {
        self.validate()?;
        if !maximum_height.is_finite() {
            return Err(Error::InvalidConfiguration("nonfinite wind height"));
        }
        let displacement = if maximum_height > self.height_start {
            1.4 * self.intensity * maximum_height.powf(self.height_exponent)
        } else {
            0.0
        };
        if !displacement.is_finite() {
            return Err(Error::InvalidConfiguration("wind displacement overflow"));
        }
        Ok(displacement)
    }
}

/// Stable Cartesian wind field, with the host's render axes expressed in
/// field space. Camera translation remains f64 until periodic UV reduction.
#[derive(Clone, Copy, Debug)]
pub struct WindFrame {
    camera_in_field: [f64; 3],
    field_from_render: [[f32; 3]; 3],
    time_seconds: f64,
}

impl WindFrame {
    /// # Errors
    /// Rejects nonfinite coordinates/time or a nonorthonormal basis.
    pub fn new(
        camera_in_field: [f64; 3],
        field_from_render: [[f32; 3]; 3],
        time_seconds: f64,
    ) -> Result<Self> {
        if !camera_in_field
            .iter()
            .all(|v| v.is_finite() && v.abs() <= f64::from(f32::MAX))
            || !time_seconds.is_finite()
            || time_seconds.abs() > f64::from(f32::MAX)
            || !field_from_render.iter().flatten().all(|v| v.is_finite())
        {
            return Err(Error::InvalidConfiguration("nonfinite wind field frame"));
        }
        for (i, a) in field_from_render.iter().enumerate() {
            for (j, b) in field_from_render.iter().enumerate() {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                if (dot - f32::from(u8::from(i == j))).abs() > 1.0e-4 {
                    return Err(Error::InvalidConfiguration(
                        "wind field axes must be orthonormal",
                    ));
                }
            }
        }
        Ok(Self {
            camera_in_field,
            field_from_render,
            time_seconds,
        })
    }

    /// The periodic origin phase is reduced before f32 conversion so
    /// subtracting a new camera origin never jumps the texture by metres.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "periodic coordinates are bounded and GPU field origin is validated"
    )]
    pub(crate) fn uniforms(
        self,
        parameters: WindParameters,
    ) -> ([[f32; 4]; 3], [f32; 4], [f32; 4]) {
        let axes = self.field_from_render.map(|a| [a[0], a[1], a[2], 0.0]);
        let origin = self.camera_in_field.map(|v| v as f32);
        let phase = self
            .camera_in_field
            .map(|v| (v * f64::from(parameters.spatial_scale)).rem_euclid(1.0) as f32);
        // Unity _Time.x is elapsed seconds / 20.
        let scroll =
            (self.time_seconds * (f64::from(parameters.speed) / 20.0)).rem_euclid(1.0) as f32;
        (
            axes,
            [origin[0], origin[1], origin[2], 0.0],
            [phase[0], phase[1], phase[2], scroll],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const AXES: [[f32; 3]; 3] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    fn parameters() -> WindParameters {
        WindParameters {
            spatial_scale: 0.008,
            height_start: 2.14,
            height_exponent: 1.62,
            speed: 2.1,
            intensity: 0.04,
        }
    }
    #[test]
    fn phase_survives_large_camera_rebases_and_time_scroll() {
        let p = parameters();
        let a = WindFrame::new([6e8, 6e8, 6e8], AXES, 0.0)
            .unwrap()
            .uniforms(p)
            .2;
        let b = WindFrame::new([6e8 + 4.25, 6e8, 6e8], AXES, 10.0)
            .unwrap()
            .uniforms(p)
            .2;
        let delta = (b[0] - a[0]).rem_euclid(1.0);
        assert!(4.25_f32.mul_add(-p.spatial_scale, delta).abs() < 1e-6);
        assert!((b[3] - 0.05).abs() < 1e-6);
        assert_eq!(a[1], b[1]);
        assert!(WindFrame::new([0.; 3], [[0.; 3]; 3], 0.).is_err());
    }
    #[test]
    fn bound_respects_source_height_and_rejects_overflow() {
        let p = parameters();
        assert_eq!(p.max_displacement(2.14).unwrap(), 0.0);
        assert!(p.max_displacement(4.0).unwrap() > 0.0);
        assert!(p.max_displacement(f32::MAX).is_err());
        assert!(
            WindParameters {
                intensity: -1.0,
                ..p
            }
            .validate()
            .is_err()
        );
    }
}
