//! Generic texture input for exact UV/alpha-masked vegetation.
//!
//! The original source texture (DDS, PNG, etc.) is decoded by the host into
//! row-major sRGB RGBA bytes. No engine-specific asset decoder is linked.
//! UV origin is the first texture row, matching wgpu texture sampling.
//! Source UVs wrap (including negative UVs); diffuse RGB is sampled in
//! linear light and alpha is sampled without gamma conversion.

use crate::{Error, Result};
use wgpu::util::DeviceExt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceImage {
    width: u16,
    height: u16,
    rgba: Vec<u8>,
}

impl SourceImage {
    /// # Errors
    /// Rejects zero-sized/oversized images and mismatching byte counts.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self> {
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|v| v.checked_mul(4))
            .ok_or(Error::IndexOverflow)?;
        if width == 0
            || height == 0
            || width > 16_384
            || height > 16_384
            || usize::try_from(pixels).ok() != Some(rgba.len())
        {
            return Err(Error::InvalidConfiguration(
                "source texture dimensions/bytes invalid",
            ));
        }
        Ok(Self {
            width: u16::try_from(width).map_err(|_| Error::IndexOverflow)?,
            height: u16::try_from(height).map_err(|_| Error::IndexOverflow)?,
            rgba,
        })
    }

    #[must_use]
    pub fn white() -> Self {
        Self {
            width: 1,
            height: 1,
            rgba: vec![255; 4],
        }
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        u32::from(self.width)
    }
    #[must_use]
    pub fn height(&self) -> u32 {
        u32::from(self.height)
    }
    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// # Errors
    /// Rejects images exceeding this GPU's supported dimensions.
    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<SourceTexture> {
        if u32::from(self.width) > device.limits().max_texture_dimension_2d
            || u32::from(self.height) > device.limits().max_texture_dimension_2d
        {
            return Err(Error::InvalidConfiguration(
                "source texture too large for GPU",
            ));
        }
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("i-love-proland source UV/alpha-masked material"),
                size: wgpu::Extent3d {
                    width: u32::from(self.width),
                    height: u32::from(self.height),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &self.rgba,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(SourceTexture {
            _texture: texture,
            view,
        })
    }

    pub(crate) fn linear_texels(&self) -> Vec<[f32; 4]> {
        let mut transfer = [0.0_f32; 256];
        for (slot, value) in transfer.iter_mut().zip(0_u8..=255) {
            *slot = srgb_to_linear(value);
        }
        self.rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                [
                    transfer[usize::from(p[0])],
                    transfer[usize::from(p[1])],
                    transfer[usize::from(p[2])],
                    f32::from(p[3]) / 255.0,
                ]
            })
            .collect()
    }

    /// Bilinear sampling with repeat wrapping, matching the GPU source
    /// sampler's normalized UV convention and linear-light sRGB decoding.
    pub(crate) fn sample(&self, texels: &[[f32; 4]], uv: [f32; 2]) -> [f32; 4] {
        let width = f32::from(self.width);
        let height = f32::from(self.height);
        let position_x = uv[0].rem_euclid(1.0).mul_add(width, -0.5);
        let position_y = uv[1].rem_euclid(1.0).mul_add(height, -0.5);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "fractional texel locations are bounded by the validated <=16,384 dimensions"
        )]
        let (left, top) = (position_x.floor() as i32, position_y.floor() as i32);
        let fractions = [
            position_x - position_x.floor(),
            position_y - position_y.floor(),
        ];
        let width_i32 = i32::from(self.width);
        let height_i32 = i32::from(self.height);
        let texel = |dx: i32, dy: i32| {
            let x = (left + dx).rem_euclid(width_i32) as usize;
            let y = (top + dy).rem_euclid(height_i32) as usize;
            texels[y * usize::from(self.width) + x]
        };
        let top_left = texel(0, 0);
        let top_right = texel(1, 0);
        let bottom_left = texel(0, 1);
        let bottom_right = texel(1, 1);
        core::array::from_fn(|i| {
            let upper = top_left[i].mul_add(1.0 - fractions[0], top_right[i] * fractions[0]);
            let lower = bottom_left[i].mul_add(1.0 - fractions[0], bottom_right[i] * fractions[0]);
            upper.mul_add(1.0 - fractions[1], lower * fractions[1])
        })
    }
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = f32::from(v) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[must_use]
pub(crate) fn linear_to_srgb(value: f32) -> f32 {
    let c = value.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055_f32.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

/// GPU source material storage; kept alive through the host renderer.
pub struct SourceTexture {
    _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_source_images() {
        assert!(SourceImage::new(0, 1, Vec::new()).is_err());
        assert!(SourceImage::new(1, 0, Vec::new()).is_err());
        assert!(SourceImage::new(1, 2, vec![0; 4]).is_err());
        assert!(SourceImage::new(16_385, 1, vec![0; 16_385 * 4]).is_err());
    }

    #[test]
    fn repeats_uvs_and_preserves_srgb_alpha_semantics() {
        // Four different texels (row-major, top down).
        let src = SourceImage::new(
            2,
            2,
            vec![
                255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 128,
            ],
        )
        .unwrap();
        let texels = src.linear_texels();
        assert_eq!(src.sample(&texels, [0.25, 0.25]), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(src.sample(&texels, [1.25, -0.75]), [1.0, 0.0, 0.0, 0.0]);
        let center = src.sample(&texels, [0.5, 0.5]);
        assert!((center[0] - 0.5).abs() < 1.0e-6);
        assert!((center[3] - (0.0 + 1.0 + 1.0 + 128.0 / 255.0) / 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn correct_gamma_transfer_roundtrip() {
        for value in 0_u8..=255 {
            let decoded = srgb_to_linear(value);
            let encoded = linear_to_srgb(decoded);
            assert!((encoded - f32::from(value) / 255.0).abs() < 1.0e-5);
        }
    }
}
