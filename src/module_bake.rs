//! Raster-exact angular atlas generation from the compact GPU module bank.
//!
//! Use the same tapered branch quads, source-ID-oriented canopy cards, source
//! texture and radial cutout as the module vertex/fragment shaders. Every
//! visual model is baked independently in its original local source frame;
//! the canonical tree ID never changes with LOD or camera angle.

use crate::{
    Error, Result,
    atlas_bake::{BakeSurface, bake_scene, to_byte},
    impostor::{AtlasImage, GpuModule, TreeVertex},
    material::{SourceImage, linear_to_srgb},
    procedural::CompiledModuleBank,
};

fn module_hash(mut value: u32) -> u32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

fn unit(vector: [f32; 3]) -> Result<[f32; 3]> {
    let length_squared = vector.iter().map(|v| v * v).sum::<f32>();
    if !length_squared.is_finite() || length_squared <= f32::EPSILON {
        return Err(Error::InvalidConfiguration(
            "degenerate foliage module branch axis",
        ));
    }
    let reciprocal = length_squared.sqrt().recip();
    Ok(vector.map(|v| v * reciprocal))
}

/// Build exactly the four source-space corners of one GPU module card.
fn quad(module: &GpuModule, side: u32) -> Result<[TreeVertex; 4]> {
    let color = module.color;
    let positions = match module.metadata[0] {
        1 => {
            let start = &module.origin_radius;
            let end = &module.extent_radius;
            let tangent = unit(std::array::from_fn(|axis| end[axis] - start[axis]))?;
            let reference = if tangent[1].abs() > 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let normal_a = unit(cross(tangent, reference))?;
            let normal = if side == 0 {
                normal_a
            } else {
                cross(tangent, normal_a)
            };
            [
                std::array::from_fn(|i| start[i] - normal[i] * start[3]),
                std::array::from_fn(|i| start[i] + normal[i] * start[3]),
                std::array::from_fn(|i| end[i] - normal[i] * end[3]),
                std::array::from_fn(|i| end[i] + normal[i] * end[3]),
            ]
        }
        2 => {
            let theta = (module_hash(module.metadata[1]) & 65535) as f32
                * (core::f32::consts::TAU / 65536.0)
                + side as f32 * core::f32::consts::FRAC_PI_2;
            let (sin, cos) = theta.sin_cos();
            let x = module.origin_radius[3];
            let y = module.extent_radius[0];
            let z = module.extent_radius[1];
            let center = &module.origin_radius;
            [
                [center[0] - cos * x, center[1] - y, center[2] - sin * z],
                [center[0] + cos * x, center[1] - y, center[2] + sin * z],
                [center[0] - cos * x, center[1] + y, center[2] - sin * z],
                [center[0] + cos * x, center[1] + y, center[2] + sin * z],
            ]
        }
        _ => {
            return Err(Error::InvalidConfiguration(
                "inert or unsupported module atlas component",
            ));
        }
    };
    let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    Ok(std::array::from_fn(|i| TreeVertex {
        position: [positions[i][0], positions[i][1], positions[i][2], 1.0],
        color,
        uv: uvs[i],
    }))
}

#[derive(Default)]
struct QuadMesh {
    vertices: Vec<TreeVertex>,
    indices: Vec<u32>,
}

impl QuadMesh {
    fn push(&mut self, corners: [TreeVertex; 4]) -> Result<()> {
        let base = u32::try_from(self.vertices.len()).map_err(|_| Error::IndexOverflow)?;
        self.vertices.extend(corners);
        for offset in [0, 1, 2, 2, 1, 3] {
            self.indices.push(base + offset);
        }
        Ok(())
    }

    fn surface<'a>(&'a self, texture: &'a SourceImage, cutoff: f32) -> Option<BakeSurface<'a>> {
        (!self.indices.is_empty()).then_some(BakeSurface {
            vertices: &self.vertices,
            indices: &self.indices,
            source: texture,
            alpha_cutoff: cutoff,
            billboard: None,
        })
    }
}

fn masked_canopy_texture(source: &SourceImage, tile_size: u32) -> Result<SourceImage> {
    let size = tile_size.max(128);
    let texels = source.linear_texels();
    let capacity =
        usize::try_from(u64::from(size) * u64::from(size) * 4).map_err(|_| Error::IndexOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    for y in 0..size {
        for x in 0..size {
            let uv = [
                (x as f32 + 0.5) / size as f32,
                (y as f32 + 0.5) / size as f32,
            ];
            let texel = source.sample(&texels, uv);
            let distance = (2.0 * uv[0] - 1.0).hypot(2.0 * uv[1] - 1.0);
            let shape = ((1.025 - distance) / 0.05).clamp(0.0, 1.0);
            bytes.extend_from_slice(&[
                to_byte(linear_to_srgb(texel[0])),
                to_byte(linear_to_srgb(texel[1])),
                to_byte(linear_to_srgb(texel[2])),
                to_byte(texel[3] * shape),
            ]);
        }
    }
    SourceImage::new(size, size, bytes)
}

impl CompiledModuleBank {
    /// Bake a 32-angle impostor per dense visual model, including the same
    /// source-index-dependent card orientation as the GPU module shader.
    /// The output is in precisely the same order as self.variant_ids.
    ///
    /// For an exact match, use the same source image for GPU module materials;
    /// the baker adds the module canopy's radial alpha envelope while keeping
    /// source-sampled linear RGB. A common source normalization frame must
    /// conservatively enclose every branch/card endpoint.
    ///
    /// # Errors
    /// Rejects geometry outside the common frame, invalid module geometry,
    /// inconsistent variant strides, and invalid angular atlas parameters.
    pub fn bake_variant_atlases(
        &self,
        source: &SourceImage,
        tile_size: u32,
        gutter: u32,
    ) -> Result<Vec<AtlasImage>> {
        let mask = masked_canopy_texture(source, tile_size)?;
        let stride = usize::try_from(self.modules_per_variant).map_err(|_| Error::IndexOverflow)?;
        if self.variant_ids.is_empty()
            || stride == 0
            || self.variant_ids.len().checked_mul(stride) != Some(self.data.len())
        {
            return Err(Error::InvalidConfiguration(
                "invalid foliage module atlas bank",
            ));
        }
        self.data
            .chunks_exact(stride)
            .map(|modules| {
                let mut branches = QuadMesh::default();
                let mut leaves = QuadMesh::default();
                for module in modules {
                    let dest = match module.metadata[0] {
                        0 => continue,
                        1 => &mut branches,
                        2 => &mut leaves,
                        _ => return Err(Error::InvalidConfiguration("invalid module kind")),
                    };
                    for side in [0, 1] {
                        dest.push(quad(module, side)?)?;
                    }
                }
                let materials = [branches.surface(source, 0.35), leaves.surface(&mask, 0.5)];
                let surfaces = materials.into_iter().flatten().collect::<Vec<_>>();
                bake_scene(&surfaces, tile_size, gutter)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::compile_modules;
    use lorax_rs::{Plant, ScreenLod, SourceFrame, Species};

    #[test]
    fn module_angular_atlas_depends_on_seed_and_matches_dense_catalog() -> Result<()> {
        let species = Species::default();
        let sources = [13, 17]
            .into_iter()
            .map(|seed| {
                let plant = Plant::generate(&species, seed)
                    .map_err(|_| Error::InvalidConfiguration("plant generation"))?;
                let tier = plant
                    .module_tier(
                        ScreenLod::new(48.0, 0.75, 1.0)
                            .and_then(|value| value.with_cluster_budget(8))
                            .map_err(|_| Error::InvalidConfiguration("foliage tier"))?,
                        species.wood_color,
                    )
                    .map_err(|_| Error::InvalidConfiguration("foliage module tier"))?;
                Ok((900, tier))
            })
            .collect::<Result<Vec<_>>>()?;
        let bank = compile_modules(
            &sources,
            SourceFrame::new([0.0, 4.0, 0.0], 12.0)
                .map_err(|_| Error::InvalidConfiguration("source frame"))?,
        )
        .map_err(|_| Error::InvalidConfiguration("module compilation"))?;
        let atlases = bank.bake_variant_atlases(&SourceImage::white(), 16, 2)?;
        assert_eq!(atlases.len(), sources.len());
        assert_ne!(atlases[0].rgba(), atlases[1].rgba());
        assert!(
            atlases
                .iter()
                .all(|atlas| atlas.rgba().chunks_exact(4).any(|pixel| pixel[3] != 0))
        );
        assert!(
            bank.bake_variant_atlases(&SourceImage::white(), 4, 2)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn orientation_hash_agrees_with_gpu_shader_test_vectors() {
        assert_eq!(module_hash(0), 0);
        assert_eq!(module_hash(1), 0x6889_90c0);
        assert_ne!(module_hash(17), module_hash(13));
    }
}
