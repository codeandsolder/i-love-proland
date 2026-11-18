//! CPU reference baker for 8×4 multiview impostor atlases.
//!
//! Accepts arbitrary colored triangulated source meshes. A host importing
//! textured leaves can pre-sample its source materials into per-vertex colors
//! or provide its own atlas with the same 32-view convention. Supersampling,
//! per-view depth, and independent padded cells are deterministic.

use crate::impostor::{AZIMUTH_VIEWS, AtlasImage, ELEVATION_VIEWS, TreeVertex};
use crate::material::{SourceImage, linear_to_srgb};
use crate::{Error, Result};
use core::f32::consts::{FRAC_PI_2, TAU};

#[derive(Clone, Copy)]
struct Sample {
    pixel: [f32; 2],
    depth: f32,
    color: [f32; 4],
    uv: [f32; 2],
}

#[derive(Clone, Copy)]
struct RasterMaterial<'a> {
    source: &'a SourceImage,
    texels: &'a [[f32; 4]],
    cutoff: f32,
}

/// A textured source-mesh draw within a shared impostor atlas.
///
/// All surfaces must be expressed in the same normalized local coordinate
/// system. Depth is resolved across surfaces, including transparency cutouts:
/// separate leaf/trunk atlases cannot correctly represent those overlaps.
#[derive(Clone, Copy)]
pub struct BakeSurface<'a> {
    pub vertices: &'a [TreeVertex],
    pub indices: &'a [u32],
    pub source: &'a SourceImage,
    pub alpha_cutoff: f32,
}

struct PreparedSurface<'a> {
    vertices: &'a [TreeVertex],
    indices: Vec<usize>,
    source: &'a SourceImage,
    texels: Vec<[f32; 4]>,
    alpha_cutoff: f32,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn basis(azimuth: u16, elevation: u16) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let azi = f32::from(azimuth) * TAU / 8.0;
    let elev = f32::from(elevation) * FRAC_PI_2 / 3.0;
    let (sin_azi, cos_azi) = azi.sin_cos();
    let (sin_elev, cos_elev) = elev.sin_cos();
    let eye = [sin_azi * cos_elev, sin_elev, cos_azi * cos_elev];
    // Analytic right basis remains well-defined even directly overhead.
    let right = [-cos_azi, 0.0, sin_azi];
    let up = [-sin_elev * sin_azi, cos_elev, -sin_elev * cos_azi];
    (eye, right, up)
}

fn project(
    vertex: &TreeVertex,
    eye: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    pixels: f32,
) -> Sample {
    let position = [vertex.position[0], vertex.position[1], vertex.position[2]];
    let u = dot(position, right);
    let v = dot(position, up);
    Sample {
        pixel: [(u + 1.0) * pixels * 0.5, (1.0 - v) * pixels * 0.5],
        depth: dot(position, eye),
        color: vertex.color,
        uv: vertex.uv,
    }
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (p[0] - a[0]).mul_add(b[1] - a[1], -(p[1] - a[1]) * (b[0] - a[0]))
}

fn raster_triangle(
    triangle: [Sample; 3],
    side: u16,
    depth: &mut [f32],
    color: &mut [[f32; 4]],
    material: RasterMaterial<'_>,
) {
    let [a, b, c] = triangle;
    let area = edge(a.pixel, b.pixel, c.pixel);
    if area.abs() < 0.00001 {
        return;
    }
    let upper = f32::from(side.saturating_sub(1));
    let lower_x = a.pixel[0]
        .min(b.pixel[0])
        .min(c.pixel[0])
        .floor()
        .clamp(0.0, upper);
    let upper_x = a.pixel[0]
        .max(b.pixel[0])
        .max(c.pixel[0])
        .ceil()
        .clamp(0.0, upper);
    let lower_y = a.pixel[1]
        .min(b.pixel[1])
        .min(c.pixel[1])
        .floor()
        .clamp(0.0, upper);
    let upper_y = a.pixel[1]
        .max(b.pixel[1])
        .max(c.pixel[1])
        .ceil()
        .clamp(0.0, upper);

    // Bounded by at most 2 * the configured tile size; raster coordinates
    // are finite because model vertices were validated before projection.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "finite raster values are clamped into the allocated square"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamping to zero excludes negative raster coordinates"
    )]
    let (min_x, max_x, min_y, max_y) = (
        lower_x as u16,
        upper_x as u16,
        lower_y as u16,
        upper_y as u16,
    );
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [f32::from(x) + 0.5, f32::from(y) + 0.5];
            let weights = [
                edge(b.pixel, c.pixel, point) / area,
                edge(c.pixel, a.pixel, point) / area,
                edge(a.pixel, b.pixel, point) / area,
            ];
            if weights.iter().any(|&w| w < -0.000_001) {
                continue;
            }
            let z = weights[0].mul_add(a.depth, weights[1].mul_add(b.depth, weights[2] * c.depth));
            let at = usize::from(y) * usize::from(side) + usize::from(x);
            if z <= depth[at] {
                continue;
            }
            let rgba: [f32; 4] = core::array::from_fn(|i| {
                weights[0].mul_add(
                    a.color[i],
                    weights[1].mul_add(b.color[i], weights[2] * c.color[i]),
                )
            });
            let uv: [f32; 2] = core::array::from_fn(|i| {
                weights[0].mul_add(a.uv[i], weights[1].mul_add(b.uv[i], weights[2] * c.uv[i]))
            });
            let texel = material.source.sample(material.texels, uv);
            if texel[3] * rgba[3] < material.cutoff {
                continue;
            }
            depth[at] = z;
            color[at] = [
                texel[0] * rgba[0],
                texel[1] * rgba[1],
                texel[2] * rgba[2],
                1.0,
            ];
        }
    }
}

fn render_view(surfaces: &[PreparedSurface<'_>], side: u16, azi: u16, elev: u16) -> Vec<[f32; 4]> {
    let extent = usize::from(side);
    let mut depth = vec![f32::NEG_INFINITY; extent * extent];
    let mut color = vec![[0.0_f32; 4]; extent * extent];
    let (eye, right, up) = basis(azi, elev);
    for surface in surfaces {
        let projected: Vec<_> = surface
            .vertices
            .iter()
            .map(|vertex| project(vertex, eye, right, up, f32::from(side)))
            .collect();
        let material = RasterMaterial {
            source: surface.source,
            texels: &surface.texels,
            cutoff: surface.alpha_cutoff,
        };
        for face in surface.indices.as_chunks::<3>().0 {
            let samples = [projected[face[0]], projected[face[1]], projected[face[2]]];
            raster_triangle(samples, side, &mut depth, &mut color, material);
        }
    }
    color
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "caller explicitly clamps the float to [0,1] before 8-bit quantization"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "saturating clamp excludes negative values before u8 quantization"
)]
fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn put_pixel(dest: &mut [u8], stride: usize, x: usize, y: usize, pixel: [u8; 4]) {
    let index = 4 * (y * stride + x);
    dest[index..index + 4].copy_from_slice(&pixel);
}

fn downsample(
    result: &mut [u8],
    canvas_width: usize,
    tile_origin: [usize; 2],
    tile_size: usize,
    gutter: usize,
    supersampled: &[[f32; 4]],
) {
    let content = tile_size - 2 * gutter;
    let side = content * 2;
    for y in 0..content {
        for x in 0..content {
            let mut alpha = 0.0_f32;
            let mut premultiplied = [0.0_f32; 3];
            for sy in 0..2 {
                for sx in 0..2 {
                    let p = supersampled[(2 * y + sy) * side + 2 * x + sx];
                    alpha += p[3];
                    for c in 0..3 {
                        premultiplied[c] = p[c].mul_add(p[3], premultiplied[c]);
                    }
                }
            }
            let color = if alpha > 0.0 {
                [
                    to_byte(linear_to_srgb(premultiplied[0] / alpha)),
                    to_byte(linear_to_srgb(premultiplied[1] / alpha)),
                    to_byte(linear_to_srgb(premultiplied[2] / alpha)),
                    to_byte(alpha * 0.25),
                ]
            } else {
                [0_u8; 4]
            };
            put_pixel(
                result,
                canvas_width,
                tile_origin[0] + gutter + x,
                tile_origin[1] + gutter + y,
                color,
            );
        }
    }
    // Clamp all four sides into each cell's own content. This prevents
    // bilinear filtering from sampling the next view at tile boundaries.
    for y in 0..tile_size {
        for x in 0..tile_size {
            if (gutter..tile_size - gutter).contains(&x)
                && (gutter..tile_size - gutter).contains(&y)
            {
                continue;
            }
            let nx = x.clamp(gutter, tile_size - gutter - 1);
            let ny = y.clamp(gutter, tile_size - gutter - 1);
            let src = 4 * ((tile_origin[1] + ny) * canvas_width + tile_origin[0] + nx);
            let mut copied = [0_u8; 4];
            copied.copy_from_slice(&result[src..src + 4]);
            put_pixel(
                result,
                canvas_width,
                tile_origin[0] + x,
                tile_origin[1] + y,
                copied,
            );
        }
    }
}

/// Bake a 32-angle, 2× supersampled atlas from multiple source-material
/// surfaces with a single shared per-sample depth buffer.
///
/// Every source is normalized to the same unit bounding sphere around
/// the canonical object center. Each supersample independently tests the
/// interpolated texture alpha * vertex alpha against the source cutoff,
/// with a depth buffer so foliage behind holes remains visible.
/// Resolve uses premultiplied linear color and preserves fractional pixel
/// coverage. The output atlas is encoded back to sRGB.
///
/// # Errors
/// Rejects empty/non-triangular/out-of-range meshes, non-finite geometry,
/// source points outside the unit sphere, and invalid tile parameters.
pub fn bake_scene(surfaces: &[BakeSurface<'_>], tile_size: u32, gutter: u32) -> Result<AtlasImage> {
    if surfaces.is_empty() {
        return Err(Error::InvalidConfiguration("atlas scene has no surfaces"));
    }
    for surface in surfaces {
        let vertices = surface.vertices;
        let indices = surface.indices;
        if vertices.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err(Error::InvalidConfiguration(
                "atlas source must contain triangles",
            ));
        }
        if vertices.iter().any(|v| {
            !v.position.iter().all(|x| x.is_finite())
                || !v.uv.iter().all(|x| x.is_finite())
                || !v
                    .color
                    .iter()
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
                || dot(
                    [v.position[0], v.position[1], v.position[2]],
                    [v.position[0], v.position[1], v.position[2]],
                ) > 1.00001
        }) || indices
            .iter()
            .any(|&i| usize::try_from(i).map_or(true, |i| i >= vertices.len()))
        {
            return Err(Error::InvalidConfiguration(
                "invalid atlas source vertex/index",
            ));
        }
        if !surface.alpha_cutoff.is_finite() || !(0.0..=1.0).contains(&surface.alpha_cutoff) {
            return Err(Error::InvalidConfiguration(
                "invalid source material alpha cutoff",
            ));
        }
    }
    if !(8..=512).contains(&tile_size) || gutter == 0 || gutter * 4 >= tile_size {
        return Err(Error::InvalidConfiguration(
            "invalid atlas tile size/gutter",
        ));
    }
    let width = usize::try_from(tile_size * AZIMUTH_VIEWS).map_err(|_| Error::IndexOverflow)?;
    let height = usize::try_from(tile_size * ELEVATION_VIEWS).map_err(|_| Error::IndexOverflow)?;
    let tile = usize::try_from(tile_size).map_err(|_| Error::IndexOverflow)?;
    let padding = usize::try_from(gutter).map_err(|_| Error::IndexOverflow)?;
    let prepared = surfaces
        .iter()
        .map(|surface| {
            Ok(PreparedSurface {
                vertices: surface.vertices,
                indices: surface
                    .indices
                    .iter()
                    .map(|&i| usize::try_from(i).map_err(|_| Error::IndexOverflow))
                    .collect::<Result<Vec<_>>>()?,
                source: surface.source,
                texels: surface.source.linear_texels(),
                alpha_cutoff: surface.alpha_cutoff,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let row_count = u16::try_from(ELEVATION_VIEWS).map_err(|_| Error::IndexOverflow)?;
    let col_count = u16::try_from(AZIMUTH_VIEWS).map_err(|_| Error::IndexOverflow)?;
    let side = u16::try_from((tile - 2 * padding) * 2).map_err(|_| Error::IndexOverflow)?;
    let mut rgba = vec![0_u8; width * height * 4];
    for row in 0..row_count {
        for col in 0..col_count {
            let supersampled = render_view(&prepared, side, col, row);
            downsample(
                &mut rgba,
                width,
                [usize::from(col) * tile, usize::from(row) * tile],
                tile,
                padding,
                &supersampled,
            );
        }
    }
    AtlasImage::new(tile_size, gutter, rgba)
}

/// Convenience entrypoint for a source with one material.
pub fn bake_mesh(
    vertices: &[TreeVertex],
    indices: &[u32],
    source: &SourceImage,
    alpha_cutoff: f32,
    tile_size: u32,
    gutter: u32,
) -> Result<AtlasImage> {
    bake_scene(
        &[BakeSurface {
            vertices,
            indices,
            source,
            alpha_cutoff,
        }],
        tile_size,
        gutter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra() -> (Vec<TreeVertex>, Vec<u32>) {
        let positions = [
            [0.0, 0.75, 0.0, 1.0],
            [-0.6, -0.5, 0.3, 1.0],
            [0.6, -0.5, 0.3, 1.0],
            [0.0, -0.5, -0.6, 1.0],
        ];
        let vertices = positions
            .into_iter()
            .map(|position| TreeVertex {
                position,
                color: [0.1, 0.8, 0.2, 1.0],
                uv: [0.0, 0.0],
            })
            .collect();
        (vertices, vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2])
    }

    #[test]
    fn atlas_baker_deterministic_and_populated_for_all_view_rows() {
        let (vertices, indices) = tetra();
        let atlas = bake_mesh(&vertices, &indices, &SourceImage::white(), 0.45, 32, 2).unwrap();
        assert_eq!((atlas.width(), atlas.height()), (256, 128));
        assert_eq!(
            atlas,
            bake_mesh(&vertices, &indices, &SourceImage::white(), 0.45, 32, 2).unwrap()
        );
        let width = usize::try_from(atlas.width()).unwrap();
        let pixels = atlas.rgba();
        for row in 0..4 {
            for col in 0..8 {
                let samples = (0..32_usize).flat_map(|y| {
                    (0..32_usize).map(move |x| {
                        let idx = 4 * ((row * 32 + y) * width + col * 32 + x);
                        pixels[idx + 3]
                    })
                });
                assert!(
                    samples.map(u64::from).sum::<u64>() > 0,
                    "empty impostor view {col},{row}"
                );
            }
        }
    }

    #[test]
    fn atlas_padding_is_filled_only_from_its_own_tile() {
        let (vertices, indices) = tetra();
        let atlas = bake_mesh(&vertices, &indices, &SourceImage::white(), 0.45, 16, 2).unwrap();
        let width = usize::try_from(atlas.width()).unwrap();
        for row in 0..4_usize {
            for col in 0..8_usize {
                for y in 0..16_usize {
                    for x in 0..16_usize {
                        if (2..14).contains(&x) && (2..14).contains(&y) {
                            continue;
                        }
                        let sample = |px: usize, py: usize| {
                            let off = 4 * ((row * 16 + py) * width + col * 16 + px);
                            &atlas.rgba()[off..off + 4]
                        };
                        assert_eq!(sample(x, y), sample(x.clamp(2, 13), y.clamp(2, 13)));
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_bake_source_rejected() {
        let (mut vertices, indices) = tetra();
        assert!(bake_mesh(&vertices, &[999, 1, 2], &SourceImage::white(), 0.45, 16, 2).is_err());
        assert!(bake_mesh(&vertices, &indices, &SourceImage::white(), 0.45, 16, 5).is_err());
        vertices[0].position[0] = f32::NAN;
        assert!(bake_mesh(&vertices, &indices, &SourceImage::white(), 0.45, 16, 2).is_err());
    }
}

#[cfg(test)]
mod textured_tests {
    use super::*;

    fn leaf_card() -> (Vec<TreeVertex>, Vec<u32>) {
        // A single front-facing 1.6m wide card (normalized local units).
        // The UVs must be interpolated per supersample, not pre-baked to
        // per-vertex colors: transparent texel holes expose the background.
        let corners = [
            ([-0.8, -0.6, 0.0, 1.0], [0.0, 1.0]),
            ([0.8, -0.6, 0.0, 1.0], [1.0, 1.0]),
            ([-0.8, 0.6, 0.0, 1.0], [0.0, 0.0]),
            ([0.8, 0.6, 0.0, 1.0], [1.0, 0.0]),
        ];
        let vertices = corners
            .into_iter()
            .map(|(position, uv)| TreeVertex {
                position,
                color: [1.0; 4],
                uv,
            })
            .collect();
        (vertices, vec![0, 1, 2, 2, 1, 3])
    }

    fn alpha_coverage(atlas: &AtlasImage) -> usize {
        let width = usize::try_from(atlas.width()).unwrap();
        let pixels = atlas.rgba();
        // First atlas view, seen face-on (source viewing direction +Z).
        (0..32_usize)
            .flat_map(|row| {
                (0..32_usize).map(move |col| {
                    let index = 4 * (row * width + col);
                    usize::from(pixels[index + 3] > 0)
                })
            })
            .sum()
    }

    #[test]
    fn alpha_masked_uvs_produce_half_coverage_not_opaque_cards() {
        let (vertices, indices) = leaf_card();
        let white = SourceImage::white();
        let opaque = bake_mesh(&vertices, &indices, &white, 0.5, 32, 2).unwrap();
        let masked_rgba = (0..8_u32)
            .flat_map(|_row| {
                (0..8_u32).flat_map(|col| [40_u8, 210, 50, if col < 4 { 255 } else { 0 }])
            })
            .collect();
        let masked = SourceImage::new(8, 8, masked_rgba).unwrap();
        let cutout = bake_mesh(&vertices, &indices, &masked, 0.5, 32, 2).unwrap();
        let total = alpha_coverage(&opaque);
        let masked_count = alpha_coverage(&cutout);
        assert!(total > 100, "front-facing source geometry missing");
        assert!(masked_count > total / 4, "alpha test removed too much");
        assert!(
            masked_count < total * 3 / 4,
            "transparent half was incorrectly opaque"
        );
        assert_eq!(
            cutout,
            bake_mesh(&vertices, &indices, &masked, 0.5, 32, 2).unwrap()
        );
    }

    #[test]
    fn atlas_baking_preserves_srgb_and_uses_source_tint() {
        let (mut vertices, indices) = leaf_card();
        let gray = SourceImage::new(1, 1, vec![128, 128, 128, 255]).unwrap();
        let atlas = bake_mesh(&vertices, &indices, &gray, 0.5, 32, 2).unwrap();
        let w = usize::try_from(atlas.width()).unwrap();
        let center = 4 * (16 * w + 16);
        for channel in &atlas.rgba()[center..center + 3] {
            assert!((i32::from(*channel) - 128).abs() <= 1);
        }
        for vertex in &mut vertices {
            vertex.color = [0.5, 0.5, 0.5, 1.0];
        }
        let tinted = bake_mesh(&vertices, &indices, &gray, 0.5, 32, 2).unwrap();
        assert!(tinted.rgba()[center] < 128);
        assert!(tinted.rgba()[center] > 75);
    }

    #[test]
    fn malformed_cutoff_and_uv_are_rejected() {
        let (mut vertices, indices) = leaf_card();
        assert!(bake_mesh(&vertices, &indices, &SourceImage::white(), f32::NAN, 32, 2).is_err());
        vertices[0].uv = [f32::INFINITY, 0.0];
        assert!(bake_mesh(&vertices, &indices, &SourceImage::white(), 0.5, 32, 2).is_err());
    }
}

#[cfg(test)]
mod multi_material_tests {
    use super::*;

    fn square(z: f32) -> (Vec<TreeVertex>, Vec<u32>) {
        let vertices = [
            ([-0.6, -0.6, z, 1.0], [0.0, 0.0]),
            ([0.6, -0.6, z, 1.0], [1.0, 0.0]),
            ([-0.6, 0.6, z, 1.0], [0.0, 1.0]),
            ([0.6, 0.6, z, 1.0], [1.0, 1.0]),
        ]
        .map(|(position, uv)| TreeVertex {
            position,
            uv,
            color: [1.0; 4],
        });
        (vertices.into(), vec![0, 1, 2, 2, 1, 3])
    }

    #[test]
    fn multi_material_depth_and_alpha_cutout_are_draw_order_independent() {
        let (trunk, indices) = square(-0.2);
        let (leaves, leaf_indices) = square(0.2);
        let red = SourceImage::new(1, 1, vec![255, 20, 20, 255]).unwrap();
        let green = SourceImage::new(
            8,
            8,
            (0..64)
                .flat_map(|pixel| [20, 230, 20, if pixel % 8 < 4 { 255 } else { 0 }])
                .collect(),
        )
        .unwrap();
        let materials = [
            BakeSurface {
                vertices: &trunk,
                indices: &indices,
                source: &red,
                alpha_cutoff: 0.0,
            },
            BakeSurface {
                vertices: &leaves,
                indices: &leaf_indices,
                source: &green,
                alpha_cutoff: 0.5,
            },
        ];
        let a = bake_scene(&materials, 32, 2).unwrap();
        let b = bake_scene(&[materials[1], materials[0]], 32, 2).unwrap();
        assert_eq!(a, b, "independent materials must share one depth buffer");
        let width = usize::try_from(a.width()).unwrap();
        let rgba = a.rgba();
        let colors = (3..29)
            .flat_map(|y| {
                (3..29).map(move |x| {
                    let at = 4 * (y * width + x);
                    [rgba[at], rgba[at + 1], rgba[at + 3]]
                })
            })
            .filter(|pixel| pixel[2] > 100)
            .collect::<Vec<_>>();
        assert!(
            colors.iter().filter(|p| p[0] > p[1]).count() > 20,
            "opaque trunk should show through leaf holes"
        );
        assert!(
            colors.iter().filter(|p| p[1] > p[0]).count() > 20,
            "cutout leaves should occlude trunk where opaque"
        );
    }

    #[test]
    fn scene_validates_every_source_draw() {
        let (vertices, indices) = square(0.0);
        let white = SourceImage::white();
        let valid = BakeSurface {
            vertices: &vertices,
            indices: &indices,
            source: &white,
            alpha_cutoff: 0.1,
        };
        let invalid = BakeSurface {
            vertices: &vertices,
            indices: &[4, 1, 2],
            source: &white,
            alpha_cutoff: 0.1,
        };
        assert!(bake_scene(&[], 32, 2).is_err());
        assert!(bake_scene(&[valid, invalid], 32, 2).is_err());
        let invalid_cutoff = BakeSurface {
            vertices: &vertices,
            indices: &indices,
            source: &white,
            alpha_cutoff: f32::NAN,
        };
        assert!(bake_scene(&[invalid_cutoff], 32, 2).is_err());
    }
}
