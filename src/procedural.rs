//! Procedural foliage geometry adapter for the existing instanced tree renderer.
//!
//! Keep immutable plant geometry, canonical scatter identity, and view-driven LOD
//! separate. A host can cache meshes at chosen error levels, then use the
//! shared GPU compactor to select one mesh tier per instance.

use crate::{
    Result,
    impostor::{TreeMesh, TreeVertex},
};
use lorax_rs::{Bounds, Error as FoliageError, Plant, ScreenLod, SourceFrame};

/// Foliage mesh in the renderer's original source-unit-sphere coordinate frame.
///
/// The original authored source frame is invariant under LOD. Fitted
/// conservative bounds and cull radius are separate: source cards and
/// generated stems may extend outside the authored normalization sphere.
#[derive(Clone, Debug, PartialEq)]
pub struct ProceduralMesh {
    pub vertices: Vec<TreeVertex>,
    pub indices: Vec<u32>,
    pub source_bounds: Bounds,
    pub source_frame: SourceFrame,
    /// Conservative model-space radius for generated geometry. This is not
    /// the normalization radius used for instance transforms.
    pub required_culling_radius: f32,
}

impl ProceduralMesh {
    /// Upload the prepared mesh to the existing Proland tree GPU renderer.
    ///
    /// The material may use `SourceImage::white()` to display the generated
    /// vertex colors, or a more detailed foliage texture supplied by the host.
    ///
    /// # Errors
    /// Rejects invalid geometry, colors, or GPU indices.
    pub fn upload(&self, device: &wgpu::Device) -> Result<TreeMesh> {
        TreeMesh::new(device, &self.vertices, &self.indices)
    }

    /// Get the physical center and radius used when building the instance
    /// transform, which are shared by every LOD of a given plant.
    #[must_use = "Use the original source frame for every mesh LOD"]
    pub const fn source_frame(&self) -> SourceFrame {
        self.source_frame
    }

    /// Multiply the authored source sphere radius by this factor when
    /// culling fitted geometry. Do not apply it to the instance transform,
    /// which must remain normalized by the original authored source frame.
    #[must_use = "Apply this factor to conservative procedural visibility bounds"]
    pub fn culling_expansion(&self) -> f32 {
        self.required_culling_radius / self.source_frame.radius
    }
}

/// Convert an immutable Lorax plant into the existing Proland mesh vertex layout.
///
/// Generated normals remain in the original Lorax Mesh; this adapter emits
/// the current Proland `TreeVertex` layout (position/color/UV), which does not yet
/// carry vertex normals. Relightable procedural-material support is a separate
/// renderer change, not an undocumented change to the source shader.
///
/// # Errors
/// Returns the original generator error for an invalid screen target or a
/// geometry budget overflow.
pub fn bake(
    plant: &Plant,
    target: ScreenLod,
    wood_color: [f32; 4],
) -> std::result::Result<ProceduralMesh, FoliageError> {
    let mesh = plant.mesh(target, wood_color)?;
    let bounds = mesh.source_bounds;
    let frame = mesh.source_frame;
    let center = frame.center;
    let radius = frame.radius;
    let required_culling_radius = plant.required_culling_radius();
    if !required_culling_radius.is_finite() || required_culling_radius < radius {
        return Err(FoliageError::Invalid(
            "unrepresentable procedural culling radius",
        ));
    }
    let vertices: Vec<TreeVertex> = mesh
        .vertices
        .into_iter()
        .map(|vertex| TreeVertex {
            position: [
                (vertex.position[0] - center[0]) / radius,
                (vertex.position[1] - center[1]) / radius,
                (vertex.position[2] - center[2]) / radius,
                1.0,
            ],
            color: vertex.color,
            uv: vertex.uv,
        })
        .collect();
    if vertices
        .iter()
        .any(|vertex| !vertex.position.iter().all(|x| x.is_finite()))
    {
        return Err(FoliageError::Invalid(
            "non-finite normalized procedural geometry",
        ));
    }
    Ok(ProceduralMesh {
        vertices,
        indices: mesh.indices,
        source_bounds: bounds,
        source_frame: frame,
        required_culling_radius,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lorax_rs::Species;

    #[test]
    fn generated_tiers_use_one_source_frame_with_valid_gpu_vertices()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let species = Species::default();
        let plant = Plant::generate(&species, 7)?;
        let mut baseline = None;
        for pixels in [8.0, 16.0, 32.0, 64.0, 256.0] {
            let baked = bake(
                &plant,
                ScreenLod::new(pixels, 0.75, 12.0)?,
                species.wood_color,
            )?;
            let frame = baked.source_frame();
            assert!(frame.radius.is_finite() && frame.radius > 0.0);
            if let Some(original) = baseline {
                assert_eq!(frame, original);
            }
            baseline = Some(frame);
            assert!(!baked.indices.is_empty());
            assert!(
                baked
                    .indices
                    .iter()
                    .all(|&i| { usize::try_from(i).is_ok_and(|i| i < baked.vertices.len()) })
            );
            for vertex in &baked.vertices {
                assert!(vertex.position[..3].iter().all(|p| p.is_finite()));
                let distance_squared = vertex.position[..3].iter().map(|v| v * v).sum::<f32>();
                assert!(distance_squared.sqrt() <= baked.culling_expansion() + 0.0001);
                assert!(TreeVertex::valid_color(vertex.color));
            }
        }
        Ok(())
    }

    #[test]
    fn error_is_preserved_before_gpu_upload() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let species = Species::default();
        let plant = Plant::generate(&species, 3)?;
        assert!(bake(&plant, ScreenLod::new(10.0, 0.5, 12.0)?, [f32::NAN; 4]).is_err());
        Ok(())
    }
    #[test]
    fn imported_oak_frame_is_not_rescaled_by_generated_aabb()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use lorax_rs::{Foliage, Stem};
        let source = Plant::from_parts(
            vec![Stem {
                start: [0.0, -5.0, 0.0],
                end: [0.0, 5.0, 0.0],
                base_radius: 0.7,
                tip_radius: 0.1,
                parent: None,
            }],
            vec![Foliage {
                center: [3.0, 2.0, 0.0],
                radii: [2.0, 3.0, 2.0],
                color: [0.25, 0.5, 0.2, 1.0],
                stem: 0,
            }],
        )?
        .with_source_frame(SourceFrame::new([0.0, 0.0, 0.0], 5.1)?)?;
        assert_ne!(source.bounds().radius(), source.source_frame().radius);
        let mut baseline = None;
        for size in [8.0, 32.0, 128.0] {
            let output = bake(
                &source,
                ScreenLod::new(size, 0.75, 12.0)?,
                [0.4, 0.2, 0.1, 1.0],
            )?;
            assert_eq!(output.source_frame(), source.source_frame());
            assert_eq!(output.source_bounds, source.bounds());
            assert_eq!(
                output.required_culling_radius,
                source.required_culling_radius()
            );
            assert!(output.culling_expansion() >= 1.0);
            if let Some((original_frame, original_expansion)) = baseline {
                assert_eq!(output.source_frame(), original_frame);
                assert_eq!(output.culling_expansion(), original_expansion);
            }
            baseline = Some((output.source_frame(), output.culling_expansion()));
            for vertex in &output.vertices {
                let norm = vertex.position[..3]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();
                assert!(norm <= output.culling_expansion() + 0.0001);
            }
        }
        Ok(())
    }
}

/// Compact, source-independent GPU representation for one visual LOD tier.
///
/// The caller passes an explicit common frame so variants of different height
/// and crown size are not accidentally normalized into identical silhouettes.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledModuleBank {
    pub data: Vec<crate::impostor::GpuModule>,
    pub source_frame: SourceFrame,
    /// Source variant IDs in dense GPU module-slot order, never rehashed.
    pub variant_ids: Vec<u64>,
    pub modules_per_variant: u32,
    /// Conservative local physical culling radii, indexed by the same dense
    /// visual form slot as \`variant_ids\`. Bounds are measured about the
    /// *shared source frame center*, not each plant's private center, so
    /// instances keep one immutable physical anchor through all LOD tiers.
    /// Each value also encloses the full source geometry under this module
    /// tier, including tapered stems and crossed canopy card corners.
    pub variant_culling_radii: Vec<f32>,
}

impl CompiledModuleBank {
    /// Deterministically pick one of many visual forms belonging to this
    /// persistent source variant. The caller supplies a stable tree/biome seed,
    /// never a camera-dependent or frame-dependent key.
    ///
    /// A source catalog may contain repeated IDs because different
    /// procedural seeds can represent the same logical source asset.
    #[must_use = "Use this local GPU slot without changing the persistent source ID"]
    pub fn model_slot_for_seed(&self, canonical_variant: u64, variation_seed: u64) -> Option<u32> {
        let forms = self
            .variant_ids
            .iter()
            .filter(|&&id| id == canonical_variant)
            .count();
        if forms == 0 {
            return None;
        }
        let mut hash = variation_seed ^ 0x9e37_79b9_7f4a_7c15;
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        hash ^= hash >> 31;
        let chosen = usize::try_from(hash % u64::try_from(forms).ok()?).ok()?;
        self.variant_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == canonical_variant)
            .nth(chosen)
            .and_then(|(index, _)| u32::try_from(index).ok())
    }

    /// Upload the validated CPU module bank without regenerating triangles.
    ///
    /// # Errors
    /// Returns a GPU/device error when the immutable bank exceeds its limits.
    pub fn upload(&self, device: &wgpu::Device) -> Result<crate::impostor::TreeModules> {
        crate::impostor::TreeModules::new(
            device,
            &self.data,
            self.variant_ids.clone(),
            self.modules_per_variant,
        )
    }
}

/// Normalize a single Lorax branch or spatial canopy group into the
/// shared source frame, retaining original component identity.
fn pack_module(
    module: &lorax_rs::Module,
    frame: SourceFrame,
) -> std::result::Result<crate::impostor::GpuModule, FoliageError> {
    let normalize = |point: [f32; 3]| {
        std::array::from_fn::<_, 3, _>(|i| (point[i] - frame.center[i]) / frame.radius)
    };
    let packed = match module {
        lorax_rs::Module::Stem {
            source_index,
            start,
            end,
            base_radius,
            tip_radius,
            color,
        } => {
            let mut a = [0.0_f32; 4];
            let mut b = [0.0_f32; 4];
            a[..3].copy_from_slice(&normalize(*start));
            b[..3].copy_from_slice(&normalize(*end));
            a[3] = base_radius / frame.radius;
            b[3] = tip_radius / frame.radius;
            crate::impostor::GpuModule {
                origin_radius: a,
                extent_radius: b,
                color: *color,
                metadata: [1, *source_index, 0, 0],
            }
        }
        lorax_rs::Module::Canopy {
            source_index,
            source_ids,
            bounds,
            color,
        } => {
            let center = bounds.center();
            let mut a = [0.0_f32; 4];
            a[..3].copy_from_slice(&normalize(center));
            a[3] = (bounds.max[0] - center[0]) / frame.radius;
            crate::impostor::GpuModule {
                origin_radius: a,
                extent_radius: [
                    (bounds.max[1] - center[1]) / frame.radius,
                    (bounds.max[2] - center[2]) / frame.radius,
                    0.0,
                    0.0,
                ],
                color: *color,
                metadata: [
                    2,
                    *source_index,
                    u32::try_from(source_ids.len()).map_err(|_| FoliageError::TooLarge)?,
                    0,
                ],
            }
        }
    };
    let coordinates = packed
        .origin_radius
        .iter()
        .chain(packed.extent_radius.iter());
    if !coordinates.clone().all(|v| v.is_finite())
        || packed.origin_radius[3] <= 0.0
        || (packed.metadata[0] == 1 && packed.extent_radius[3] <= 0.0)
        || (packed.metadata[0] == 2
            && (packed.extent_radius[0] <= 0.0 || packed.extent_radius[1] <= 0.0))
    {
        return Err(FoliageError::Invalid("unrepresentable normalized module"));
    }
    Ok(packed)
}

/// Conservative radius in the shared model frame, including protruding
/// fitted branch and foliage proxies outside original source spheres.
fn module_radius(
    tier: &lorax_rs::ModuleTier,
    frame: SourceFrame,
) -> std::result::Result<f32, FoliageError> {
    SourceFrame::new(tier.source_frame.center, tier.source_frame.radius)?;
    if !tier.required_culling_radius.is_finite() || tier.required_culling_radius <= 0.0 {
        return Err(FoliageError::Invalid("invalid source variant cull radius"));
    }
    let b = tier.source_bounds;
    if !b.valid() {
        return Err(FoliageError::Invalid("invalid module source bounds"));
    }
    let farthest = std::array::from_fn::<_, 3, _>(|i| {
        (b.max[i] - frame.center[i])
            .abs()
            .max((b.min[i] - frame.center[i]).abs())
    });
    let outer = farthest.into_iter().map(|v| v * v).sum::<f32>().sqrt();
    let delta = tier
        .source_frame
        .center
        .iter()
        .zip(frame.center)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let radius = outer.max(delta + tier.required_culling_radius);
    if !radius.is_finite() {
        return Err(FoliageError::Invalid(
            "unrepresentable variant culling bound",
        ));
    }
    Ok(radius)
}

/// Normalize and pad arbitrary Lorax source variants into one GPU module bank.
///
/// Source IDs may repeat: one canonical species/asset variant may have many
/// different seed-generated visual forms, all preserving source identity.
/// Select the dense GPU slot using a stable seed independent from the camera.
/// Padding is inert, allowing species with different counts of branches and
/// foliage to share an indirect draw. Every module retains original source
/// indices so source-textured cluster atlases can be produced separately.
///
/// # Errors
/// Rejects missing models, degenerate source frames, non-finite transformed
/// extents, or module/index overflow.
pub fn compile_modules(
    variants: &[(u64, lorax_rs::ModuleTier)],
    shared_frame: SourceFrame,
) -> std::result::Result<CompiledModuleBank, FoliageError> {
    let frame = SourceFrame::new(shared_frame.center, shared_frame.radius)?;
    if variants.is_empty() {
        return Err(FoliageError::Invalid("missing module variants"));
    }
    let max_modules = variants
        .iter()
        .map(|(_, t)| t.modules.len())
        .max()
        .ok_or(FoliageError::Invalid("missing module variants"))?;
    if max_modules == 0 {
        return Err(FoliageError::Invalid("empty module geometry"));
    }
    let module_count = u32::try_from(max_modules).map_err(|_| FoliageError::TooLarge)?;
    let total = variants
        .len()
        .checked_mul(max_modules)
        .ok_or(FoliageError::TooLarge)?;
    u32::try_from(variants.len()).map_err(|_| FoliageError::TooLarge)?;
    module_count.checked_mul(12).ok_or(FoliageError::TooLarge)?;
    let mut ids = Vec::new();
    let mut data = Vec::new();
    let mut variant_culling_radii = Vec::new();
    ids.try_reserve(variants.len())
        .map_err(|_| FoliageError::TooLarge)?;
    data.try_reserve(total)
        .map_err(|_| FoliageError::TooLarge)?;
    variant_culling_radii
        .try_reserve(variants.len())
        .map_err(|_| FoliageError::TooLarge)?;
    for (id, tier) in variants {
        ids.push(*id);
        variant_culling_radii.push(module_radius(tier, frame)?);
        for module in &tier.modules {
            data.push(pack_module(module, frame)?);
        }
        data.resize(
            data.len()
                .checked_add(max_modules - tier.modules.len())
                .ok_or(FoliageError::TooLarge)?,
            crate::impostor::GpuModule::default(),
        );
    }
    Ok(CompiledModuleBank {
        data,
        source_frame: frame,
        variant_ids: ids,
        modules_per_variant: module_count,
        variant_culling_radii,
    })
}

#[cfg(test)]
mod module_tests {
    use super::*;
    use lorax_rs::Species;

    #[test]
    fn heterogeneous_species_share_explicit_frame_and_padded_gpu_stride()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let broadleaf = Species::default();
        let mut conifer = broadleaf;
        conifer.height *= 1.7;
        conifer.crown_radius *= 0.55;
        conifer.levels = 2;
        let frame = SourceFrame::new([0.0, 8.0, 0.0], 22.0)?;
        let variants = [
            (&broadleaf, 0xdead_beef_u64),
            (&conifer, 2_u64),
            (&broadleaf, 99999_u64),
        ]
        .into_iter()
        .map(|(species, seed)| {
            let tree = Plant::generate(species, seed)?;
            Ok((
                seed,
                tree.module_tier(
                    ScreenLod::new(80.0, 0.75, 0.01)?.with_cluster_budget(24)?,
                    species.wood_color,
                )?,
            ))
        })
        .collect::<std::result::Result<Vec<_>, FoliageError>>()?;
        let bank = compile_modules(&variants, frame)?;
        assert_eq!(bank.model_slot_for_seed(0xdead_beef, 12), Some(0));
        assert_eq!(bank.model_slot_for_seed(2, 12), Some(1));
        assert_eq!(bank.model_slot_for_seed(99999, 12), Some(2));
        assert_eq!(bank.model_slot_for_seed(0, 12), None);
        assert_eq!(
            bank.data.len(),
            usize::try_from(bank.modules_per_variant)? * variants.len()
        );
        assert_eq!(bank.variant_culling_radii.len(), variants.len());
        assert!(
            bank.variant_culling_radii
                .iter()
                .all(|radius| *radius > 0.0)
        );
        assert!(
            bank.variant_culling_radii
                .iter()
                .all(|radius| radius.is_finite())
        );
        assert_eq!(bank.source_frame, frame);
        for ((_, tier), &radius) in variants.iter().zip(&bank.variant_culling_radii) {
            let bounds = tier.source_bounds;
            let farthest: [f32; 3] = std::array::from_fn(|axis| {
                (bounds.max[axis] - frame.center[axis])
                    .abs()
                    .max((bounds.min[axis] - frame.center[axis]).abs())
            });
            assert!(farthest.into_iter().map(|x| x * x).sum::<f32>().sqrt() <= radius);
        }
        assert!(bank.data.iter().any(|record| record.metadata[0] == 0));
        assert!(bank.data.iter().any(|record| record.metadata[0] == 1));
        assert!(bank.data.iter().any(|record| record.metadata[0] == 2));
        assert!(bank.data.iter().all(|record| {
            record
                .origin_radius
                .iter()
                .chain(record.extent_radius.iter())
                .all(|v| v.is_finite())
        }));
        Ok(())
    }

    #[test]
    fn canonical_source_id_can_have_many_distinct_visual_seed_forms()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let species = Species::default();
        let mut variants = Vec::new();
        for seed in 3..11_u64 {
            let plant = Plant::generate(&species, seed)?;
            variants.push((
                900,
                plant.module_tier(
                    ScreenLod::new(96.0, 0.75, 0.01)?.with_cluster_budget(24)?,
                    species.wood_color,
                )?,
            ));
        }
        let shared_frame = SourceFrame::new([0.0, 4.0, 0.0], 15.0)?;
        let bank = compile_modules(&variants, shared_frame)?;
        assert_eq!(bank.variant_ids, vec![900; 8]);
        assert_eq!(bank.variant_culling_radii.len(), 8);
        let slots = (0..300_u64)
            .map(|seed| bank.model_slot_for_seed(900, seed))
            .collect::<Vec<_>>();
        assert_eq!(
            slots,
            (0..300_u64)
                .map(|seed| bank.model_slot_for_seed(900, seed))
                .collect::<Vec<_>>()
        );
        assert!(slots.iter().all(|slot| slot.is_some_and(|index| index < 8)));
        assert!(slots.windows(2).any(|pair| pair[0] != pair[1]));
        assert_eq!(bank.model_slot_for_seed(901, 5), None);
        let stride = usize::try_from(bank.modules_per_variant)?;
        assert_ne!(bank.data[0..stride], bank.data[stride..2 * stride]);
        Ok(())
    }

    #[test]
    fn heterogeneous_visual_forms_keep_independent_conservative_bounds()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let wide = Species {
            height: 24.0,
            crown_radius: 10.0,
            ..Species::default()
        };
        let small = Species {
            height: 5.0,
            crown_radius: 1.5,
            ..Species::default()
        };
        let frame = SourceFrame::new([0.0, 10.0, 0.0], 45.0)?;
        let variants = [&wide, &small]
            .into_iter()
            .map(|species| {
                let plant = Plant::generate(species, 17)?;
                Ok((
                    900,
                    plant.module_tier(
                        ScreenLod::new(32.0, 0.75, 0.01)?.with_cluster_budget(8)?,
                        species.wood_color,
                    )?,
                ))
            })
            .collect::<std::result::Result<Vec<_>, FoliageError>>()?;
        let bank = compile_modules(&variants, frame)?;
        let [large, small] = bank.variant_culling_radii.as_slice() else {
            return Err("expected exactly two source radii".into());
        };
        assert!(*large > *small * 1.25);
        // These bounds are shared-frame-relative rather than normalized to
        // the individual source plant's independently chosen source center.
        assert!(*large <= frame.radius);
        assert!(*small <= frame.radius);
        for ((_, tier), radius) in variants.iter().zip(&bank.variant_culling_radii) {
            assert!(module_radius(tier, frame)? <= *radius);
            assert!(radius.is_finite() && *radius > 0.0);
        }
        assert!(bank.model_slot_for_seed(900, 6).is_some());

        // Both forms have identical world transforms and GPU source-frame
        // dimensions, but independently conservative projected diameters.
        // Their visible material, impostor billboard size, canonical source
        // metadata and handoff anchor are deliberately unchanged.
        use crate::gpu::{GpuView, MeshLod, MeshLods, PackedInstance, VisualLod, classify_cpu};
        let mature = PackedInstance {
            position_radius: [0.0, 0.0, 300.0, *large],
            rotation: [0.0, 0.0, 0.0, 1.0],
            axis_radii: [frame.radius, frame.radius, frame.radius, 0.0],
            object_id: [37, 0, 0, 0],
            metadata: [900, 0, 0, 0],
            render_model: [0; 4],
            tint: [1.0; 4],
            handoff_anchor_rank: [0.0, 0.0, 300.0, 0.5],
        };
        let shrub = PackedInstance {
            position_radius: [0.0, 0.0, 300.0, *small],
            render_model: [1, 0, 0, 0],
            ..mature
        };
        let lods = MeshLods::new(vec![MeshLod {
            start_m: 0.0,
            index_count: 12,
            prefer_mesh: false,
        }])?
        .with_projected_diameters(vec![0.0])?;
        let view = GpuView::unculled(300.0, *large + *small, 0.15)?;
        assert_eq!(classify_cpu(&mature, view, &lods), VisualLod::Mesh(0));
        assert_eq!(classify_cpu(&shrub, view, &lods), VisualLod::Impostor);
        assert_eq!(mature.axis_radii, shrub.axis_radii);
        assert_eq!(mature.object_id, shrub.object_id);
        assert_eq!(mature.handoff_anchor_rank, shrub.handoff_anchor_rank);
        Ok(())
    }

    #[test]
    fn missing_models_and_invalid_frame_rejected()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let species = Species::default();
        let plant = Plant::generate(&species, 5)?;
        let tier = plant.module_tier(ScreenLod::new(40.0, 0.5, 12.0)?, species.wood_color)?;
        let frame = tier.source_frame;
        assert!(compile_modules(&[], frame).is_err());
        let bank = compile_modules(&[(12, tier.clone()), (12, tier)], frame)?;
        assert_eq!(bank.variant_ids, vec![12, 12]);
        for seed in 0..100 {
            assert!(matches!(bank.model_slot_for_seed(12, seed), Some(0 | 1)));
            assert_eq!(
                bank.model_slot_for_seed(12, seed),
                bank.model_slot_for_seed(12, seed)
            );
        }
        assert_eq!(bank.model_slot_for_seed(13, 0), None);
        assert!(
            compile_modules(
                &[(
                    4,
                    plant.module_tier(ScreenLod::new(40.0, 0.5, 12.0)?, species.wood_color)?
                )],
                SourceFrame {
                    center: [0.0; 3],
                    radius: 0.0
                }
            )
            .is_err()
        );
        Ok(())
    }
}
