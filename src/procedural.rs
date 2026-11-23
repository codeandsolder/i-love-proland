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
