//! A shared, source-textured foliage card for GPU modules and their baked
//! impostors. All expensive shape and lighting work is performed once.
use crate::{
    Error, Result,
    atlas_bake::to_byte,
    material::{SourceImage, linear_to_srgb},
};
use wgpu::util::DeviceExt;

const SIDE: u16 = 128;

fn random(x: u32, y: u32) -> f32 {
    let word = (u64::from(x) << 32) | u64::from(y);
    let [a, b, ..] = crate::random::mix64(word ^ 0x79b9_7f4a_85eb_ca6b).to_le_bytes();
    f32::from(u16::from_le_bytes([a, b])) / 65535.0
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

fn smooth(t: f32) -> f32 {
    t * t * 2.0_f32.mul_add(-t, 3.0)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "positive bounded card UVs are sampled on a small finite noise lattice"
)]
fn noise(uv: [f32; 2], frequency: f32) -> f32 {
    let [x, y] = uv.map(|v| v * frequency);
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let tx = smooth(x.fract());
    let ty = smooth(y.fract());
    mix(
        mix(random(ix, iy), random(ix + 1, iy), tx),
        mix(random(ix, iy + 1), random(ix + 1, iy + 1), tx),
        ty,
    )
}

fn coverage_and_shading(uv: [f32; 2]) -> (f32, f32) {
    let [x, y] = uv.map(|v| 2.0_f32.mul_add(v, -1.0));
    let radius = x.hypot(y);
    let theta = y.atan2(x);
    // A rounded envelope with multiple independent lobes, fine leaf-scale
    // silhouette variation and occasional small gaps between leaf groups.
    let coarse = noise(uv, 7.0);
    let fine = noise(uv, 27.0);
    let edge = 0.10_f32.mul_add(
        fine - 0.5,
        0.032_f32.mul_add(
            9.0_f32.mul_add(theta, -1.2).sin(),
            0.052_f32.mul_add(5.0_f32.mul_add(theta, 0.6).sin(), 0.95),
        ),
    );
    let envelope = ((edge - radius) / 0.025 + 0.5).clamp(0.0, 1.0);
    let gaps = ((0.6_f32.mul_add(coarse, 0.4 * fine) - 0.18) / 0.12).clamp(0.0, 1.0);
    // Source-local ambient occlusion and crown-top illumination; the
    // irregular mottling is in linear light and is stable under motion.
    let center = radius.mul_add(-radius, 1.0).max(0.0).sqrt();
    let light = (0.24 * center + 0.10 * y + 0.20 * coarse + 0.15 * fine + 0.43).clamp(0.0, 1.0);
    (envelope * gaps, light)
}

/// The immutable CPU representation used for both source baking and upload.
/// RGB is straight linear-light modulation encoded as sRGB; alpha is coverage.
pub struct CanopyCard {
    pub(crate) image: SourceImage,
}

/// GPU texture containing the same card and coverage-preserving mip chain.
#[derive(Clone)]
pub struct CanopyTexture {
    pub(crate) view: wgpu::TextureView,
}

impl CanopyCard {
    /// Generate a deterministic irregular leaf-clump material from a source.
    ///
    /// # Errors
    /// Propagates an invalid generated texture.
    pub fn new(source: &SourceImage) -> Result<Self> {
        let original = source.linear_texels();
        let mut bytes = Vec::with_capacity(usize::from(SIDE).pow(2) * 4);
        let inverse = f32::from(SIDE).recip();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let uv = [
                    (f32::from(x) + 0.5) * inverse,
                    (f32::from(y) + 0.5) * inverse,
                ];
                let texel = source.sample(&original, uv);
                let (coverage, light) = coverage_and_shading(uv);
                bytes.extend_from_slice(&[
                    to_byte(linear_to_srgb(texel[0] * light)),
                    to_byte(linear_to_srgb(texel[1] * light)),
                    to_byte(linear_to_srgb(texel[2] * light)),
                    to_byte(texel[3] * coverage),
                ]);
            }
        }
        Ok(Self {
            image: SourceImage::new(u32::from(SIDE), u32::from(SIDE), bytes)?,
        })
    }

    // Store premultiplied linear RGB in every mip, including the base,
    // so interpolation across transparent silhouettes cannot darken leaves.
    fn mip_bytes(&self) -> Vec<u8> {
        let mut level = self.image.linear_texels();
        for pixel in &mut level {
            let alpha = pixel[3];
            for channel in &mut pixel[..3] {
                *channel *= alpha;
            }
        }
        let mut side = usize::from(SIDE);
        let mut data = Vec::new();
        loop {
            for p in &level {
                data.extend_from_slice(&[
                    to_byte(linear_to_srgb(p[0])),
                    to_byte(linear_to_srgb(p[1])),
                    to_byte(linear_to_srgb(p[2])),
                    to_byte(p[3]),
                ]);
            }
            if side == 1 {
                return data;
            }
            let next = side / 2;
            let mut reduced = Vec::with_capacity(next * next);
            for y in 0..next {
                for x in 0..next {
                    let at = 2 * (y * side + x);
                    reduced.push(core::array::from_fn(|c| {
                        (level[at][c]
                            + level[at + 1][c]
                            + level[at + side][c]
                            + level[at + side + 1][c])
                            * 0.25
                    }));
                }
            }
            level = reduced;
            side = next;
        }
    }

    /// Upload once, then share across all module LODs for this material.
    ///
    /// # Errors
    /// Rejects a device unable to hold the card.
    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<CanopyTexture> {
        let side = u32::from(SIDE);
        if side > device.limits().max_texture_dimension_2d {
            return Err(Error::InvalidConfiguration(
                "canopy card exceeds GPU texture limit",
            ));
        }
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("shared procedural leaf-clump card"),
                size: wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
                mip_level_count: side.bit_width(),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &self.mip_bytes(),
        );
        Ok(CanopyTexture {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::srgb_to_linear;

    #[test]
    fn leaf_card_is_reproducible_variegated_and_preserves_mip_coverage() -> Result<()> {
        let card = CanopyCard::new(&SourceImage::white())?;
        assert_eq!(card.image, CanopyCard::new(&SourceImage::white())?.image);
        let pixels = card.image.linear_texels();
        let mut minimum = 1.0_f32;
        let mut maximum = 0.0_f32;
        let mut sum = [0.0_f64; 4];
        for p in pixels {
            if p[3] > 0.9 {
                minimum = minimum.min(p[0]);
                maximum = maximum.max(p[0]);
            }
            for c in 0..3 {
                sum[c] += f64::from(p[c] * p[3]);
            }
            sum[3] += f64::from(p[3]);
        }
        let n = f64::from(SIDE).powi(2);
        let coverage = sum[3] / n;
        println!(
            "CANOPY_CARD coverage={coverage:.5} opaque_luminance_range={minimum:.4}..{maximum:.4}"
        );
        assert!((0.60..0.78).contains(&coverage));
        assert!(maximum - minimum > 0.20);
        let bytes = card.mip_bytes();
        let last = &bytes[bytes.len() - 4..];
        assert!((f64::from(last[3]) / 255.0 - coverage).abs() < 1.0 / 255.0);
        for c in 0..3 {
            assert!((f64::from(srgb_to_linear(last[c])) - sum[c] / n).abs() < 0.005);
        }
        Ok(())
    }

    #[test]
    fn source_alpha_is_retained() -> Result<()> {
        let transparent = SourceImage::new(1, 1, vec![255, 255, 255, 0])?;
        let card = CanopyCard::new(&transparent)?;
        assert!(card.image.rgba().chunks_exact(4).all(|p| p[3] == 0));
        Ok(())
    }
}
