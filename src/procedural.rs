//! Procedural foliage geometry adapter for the existing instanced tree renderer.
//!
//! Keep immutable plant geometry, canonical scatter identity, and view-driven LOD
//! separate. A host can cache meshes at chosen error levels, then use the
//! shared GPU compactor to select one mesh tier per instance.

use crate::{
    Result,
    impostor::{TreeMesh, TreeVertex},
};
use lorax_rs::{Bounds, Error as FoliageError, Plant, ScreenLod};

/// Foliage mesh in the renderer's original source-unit-sphere coordinate frame.
///
/// The center/radius describe the *entire* plant, not only the present visual
/// LOD. This is essential to retain matching transform, culling and wind bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct ProceduralMesh {
    pub vertices: Vec<TreeVertex>,
    pub indices: Vec<u32>,
    pub source_bounds: Bounds,
}

impl ProceduralMesh {
    /// Upload the prepared mesh to the existing Proland tree GPU renderer.
    ///
    /// The material may use SourceImage::white() to display the generated
    /// vertex colors, or a more detailed foliage texture supplied by the host.
    ///
    /// # Errors
    /// Rejects invalid geometry, colors, or GPU indices.
    pub fn upload(&self, device: &wgpu::Device) -> Result<TreeMesh> {
        TreeMesh::new(device, &self.vertices, &self.indices)
    }

    /// Get the physical center and radius used when building the instance
    /// transform, which are shared by every LOD of a given plant.
    #[must_use = "Use the same source bounds for all visual LODs"]
    pub fn source_frame(&self) -> ([f32; 3], f32) {
        (self.source_bounds.center(), self.source_bounds.radius())
    }
}

/// Convert an immutable Lorax plant into the existing Proland mesh vertex layout.
///
/// Generated normals remain in the original Lorax Mesh; this adapter emits
/// the current Proland TreeVertex layout (position/color/UV), which does not yet
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
    let center = bounds.center();
    let radius = bounds.radius();
    let vertices = mesh
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
    Ok(ProceduralMesh {
        vertices,
        indices: mesh.indices,
        source_bounds: bounds,
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
            assert!(frame.1.is_finite() && frame.1 > 0.0);
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
                assert!(
                    vertex.position[..3]
                        .iter()
                        .all(|p| p.is_finite() && p.abs() <= 1.001)
                );
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
}
