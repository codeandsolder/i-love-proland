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

const LEAF_COUNT: u32 = 640;

struct Leaflet {
    center: [f32; 2],
    radii: [f32; 2],
    basis: [f32; 2],
    illumination: f32,
}

impl Leaflet {
    fn seeded(index: u32) -> Self {
        let r = |stream| random(index, stream);
        let angle = core::f32::consts::TAU * r(0);
        let (sin, cos) = angle.sin_cos();
        let boundary = 0.035_f32.mul_add(
            9.0_f32.mul_add(angle, -1.2).sin(),
            0.055_f32.mul_add(5.0_f32.mul_add(angle, 0.6).sin(), 0.9),
        );
        let radial = boundary * r(1).sqrt();
        let length = r(2).mul_add(0.065, 0.052);
        let width = length * r(3).mul_add(0.37, 0.43);
        let orientation = core::f32::consts::TAU * r(4);
        let (sin_leaf, cos_leaf) = orientation.sin_cos();
        let center = [radial * cos, radial * sin];
        // Independent leaf orientation and shadowing, in linear light.
        // The upper crown is sunlit; the back layers retain cooler shading.
        let illumination = 0.16_f32.mul_add(center[1], r(5).mul_add(0.44, 0.44));
        Self {
            center,
            radii: [length, width],
            basis: [cos_leaf, sin_leaf],
            illumination,
        }
    }

    fn sample(&self, point: [f32; 2]) -> (f32, f32) {
        let [dx, dy] = std::array::from_fn(|i| point[i] - self.center[i]);
        let [c, s] = self.basis;
        let x = dx.mul_add(c, dy * s) / self.radii[0];
        let y = dy.mul_add(c, -dx * s) / self.radii[1];
        // Taper the two ends: this is a pointed leaflet, not a circular blob.
        let shape = y.mul_add(y, x.abs().mul_add(0.2, x * x));
        let edge = (1.0 - shape) * 3.2;
        let alpha = edge.clamp(0.0, 1.0);
        // A gentle central rib and curved lamina create legible texture at
        // near LOD; both disappear naturally under mip filtering.
        let rib = y.abs().mul_add(-7.0, 1.0).max(0.0);
        let curvature = y.mul_add(-y, 1.0).max(0.0);
        let light = self.illumination * 0.12_f32.mul_add(rib, 0.22_f32.mul_add(curvature, 0.68));
        (alpha, light.clamp(0.0, 1.0))
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "pixel bounds are explicitly rounded and clamped to the finite card extent"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "pixel bounds are clamped to nonnegative coordinates"
    )]
    fn composite(&self, texels: &mut [[f32; 2]]) {
        let half = f32::from(SIDE) * 0.5;
        let radius = self.radii[0];
        let minimum = self.center.map(|v| {
            ((v - radius + 1.0) * half)
                .floor()
                .clamp(0.0, f32::from(SIDE)) as u16
        });
        let maximum = self.center.map(|v| {
            ((v + radius + 1.0) * half)
                .ceil()
                .clamp(0.0, f32::from(SIDE)) as u16
        });
        let inverse = half.recip();
        for y in minimum[1]..maximum[1] {
            for x in minimum[0]..maximum[0] {
                let point = [x, y].map(|v| (f32::from(v) + 0.5).mul_add(inverse, -1.0));
                let (alpha, light) = self.sample(point);
                let pixel = &mut texels[usize::from(y) * usize::from(SIDE) + usize::from(x)];
                // Premultiplied back-to-front compositing of leaf layers.
                pixel[0] = light.mul_add(alpha, pixel[0] * (1.0 - alpha));
                pixel[1] = alpha.mul_add(1.0 - pixel[1], pixel[1]);
            }
        }
    }
}

fn leaf_cluster() -> Vec<[f32; 2]> {
    let mut texels = vec![[0.0; 2]; usize::from(SIDE).pow(2)];
    for index in 0..LEAF_COUNT {
        Leaflet::seeded(index).composite(&mut texels);
    }
    texels
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
        let cluster = leaf_cluster();
        let mut bytes = Vec::with_capacity(usize::from(SIDE).pow(2) * 4);
        let inverse = f32::from(SIDE).recip();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let uv = [
                    (f32::from(x) + 0.5) * inverse,
                    (f32::from(y) + 0.5) * inverse,
                ];
                let texel = source.sample(&original, uv);
                let [premultiplied, coverage] =
                    cluster[usize::from(y) * usize::from(SIDE) + usize::from(x)];
                let light = if coverage > 0.0 {
                    premultiplied / coverage
                } else {
                    0.0
                };
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
        assert!(
            card.image
                .rgba()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[3] == 0)
        );
        Ok(())
    }
}
