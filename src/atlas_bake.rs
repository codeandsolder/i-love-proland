//! CPU reference baker for 8×4 multiview impostor atlases.
//!
//! Accepts arbitrary colored triangulated source meshes. A host importing
//! textured leaves can pre-sample its source materials into per-vertex colors
//! or provide its own atlas with the same 32-view convention. Supersampling,
//! per-view depth, and independent padded cells are deterministic.

use crate::impostor::{AZIMUTH_VIEWS, AtlasImage, ELEVATION_VIEWS, TreeVertex};
use crate::{Error, Result};
use core::f32::consts::{FRAC_PI_2, TAU};

#[derive(Clone, Copy)]
struct Sample {
    pixel: [f32; 2],
    depth: f32,
    color: [f32; 4],
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn basis(azimuth: u32, elevation: u32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let azi = azimuth as f32 * TAU / AZIMUTH_VIEWS as f32;
    let elev = elevation as f32 * FRAC_PI_2 / (ELEVATION_VIEWS - 1) as f32;
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
    }
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (p[0] - a[0]).mul_add(b[1] - a[1], -(p[1] - a[1]) * (b[0] - a[0]))
}

fn raster_triangle(triangle: [Sample; 3], side: usize, depth: &mut [f32], color: &mut [[f32; 4]]) {
    let [a, b, c] = triangle;
    let area = edge(a.pixel, b.pixel, c.pixel);
    if area.abs() < 0.00001 {
        return;
    }
    let upper = side.saturating_sub(1) as f32;
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
        clippy::cast_sign_loss,
        reason = "finite clamped raster coordinates are within the allocated square"
    )]
    let (min_x, max_x, min_y, max_y) = (
        lower_x as usize,
        upper_x as usize,
        lower_y as usize,
        upper_y as usize,
    );
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let weights = [
                edge(b.pixel, c.pixel, point) / area,
                edge(c.pixel, a.pixel, point) / area,
                edge(a.pixel, b.pixel, point) / area,
            ];
            if weights.iter().any(|&w| w < -0.000001) {
                continue;
            }
            let z = weights[0].mul_add(a.depth, weights[1].mul_add(b.depth, weights[2] * c.depth));
            let at = y * side + x;
            if z <= depth[at] {
                continue;
            }
            let rgba = core::array::from_fn(|i| {
                weights[0].mul_add(
                    a.color[i],
                    weights[1].mul_add(b.color[i], weights[2] * c.color[i]),
                )
            });
            if rgba[3] < 0.45 {
                continue;
            }
            depth[at] = z;
            color[at] = rgba;
        }
    }
}

fn render_view(
    vertices: &[TreeVertex],
    indices: &[usize],
    side: usize,
    azi: u32,
    elev: u32,
) -> Vec<[f32; 4]> {
    let mut depth = vec![f32::NEG_INFINITY; side * side];
    let mut color = vec![[0.0_f32; 4]; side * side];
    let (eye, right, up) = basis(azi, elev);
    let projected: Vec<_> = vertices
        .iter()
        .map(|vertex| project(vertex, eye, right, up, side as f32))
        .collect();
    for face in indices.chunks_exact(3) {
        let samples = [projected[face[0]], projected[face[1]], projected[face[2]]];
        raster_triangle(samples, side, &mut depth, &mut color);
    }
    color
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "caller explicitly clamps the float to [0,1] before 8-bit quantization"
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
                    to_byte(premultiplied[0] / alpha),
                    to_byte(premultiplied[1] / alpha),
                    to_byte(premultiplied[2] / alpha),
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

/// Bake a deterministic 32-angle, 2× supersampled atlas from colored triangles.
///
/// Geometry is in a unit bounding sphere around the canonical object center.
/// This basic reference baker uses vertex colors and cutout opacity. Hosts
/// with textured or wind-animated source foliage may instead supply their
/// own pre-rendered AtlasImage using the same camera-basis/view ordering.
///
/// # Errors
/// Rejects empty/non-triangular/out-of-range meshes, non-finite geometry,
/// source points outside the unit sphere, and invalid tile parameters.
pub fn bake_colored_mesh(
    vertices: &[TreeVertex],
    indices: &[u32],
    tile_size: u32,
    gutter: u32,
) -> Result<AtlasImage> {
    if vertices.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(Error::InvalidConfiguration(
            "atlas source must contain triangles",
        ));
    }
    if vertices.iter().any(|v| {
        !v.position.iter().all(|x| x.is_finite())
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
    if !(8..=512).contains(&tile_size) || gutter == 0 || gutter * 4 >= tile_size {
        return Err(Error::InvalidConfiguration(
            "invalid atlas tile size/gutter",
        ));
    }
    let width = usize::try_from(tile_size * AZIMUTH_VIEWS).map_err(|_| Error::IndexOverflow)?;
    let height = usize::try_from(tile_size * ELEVATION_VIEWS).map_err(|_| Error::IndexOverflow)?;
    let tile = usize::try_from(tile_size).map_err(|_| Error::IndexOverflow)?;
    let padding = usize::try_from(gutter).map_err(|_| Error::IndexOverflow)?;
    let checked_indices = indices
        .iter()
        .map(|&i| usize::try_from(i).map_err(|_| Error::IndexOverflow))
        .collect::<Result<Vec<_>>>()?;
    let mut rgba = vec![0_u8; width * height * 4];
    for row in 0..ELEVATION_VIEWS {
        for col in 0..AZIMUTH_VIEWS {
            let supersampled = render_view(
                vertices,
                &checked_indices,
                (tile - 2 * padding) * 2,
                col,
                row,
            );
            downsample(
                &mut rgba,
                width,
                [
                    usize::try_from(col).map_err(|_| Error::IndexOverflow)? * tile,
                    usize::try_from(row).map_err(|_| Error::IndexOverflow)? * tile,
                ],
                tile,
                padding,
                &supersampled,
            );
        }
    }
    AtlasImage::new(tile_size, gutter, rgba)
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
            })
            .collect();
        (vertices, vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2])
    }

    #[test]
    fn atlas_baker_deterministic_and_populated_for_all_view_rows() {
        let (vertices, indices) = tetra();
        let atlas = bake_colored_mesh(&vertices, &indices, 32, 2).unwrap();
        assert_eq!((atlas.width(), atlas.height()), (256, 128));
        assert_eq!(
            atlas,
            bake_colored_mesh(&vertices, &indices, 32, 2).unwrap()
        );
        let width = usize::try_from(atlas.width()).unwrap();
        for row in 0..4 {
            for col in 0..8 {
                let samples = (0..32_usize).flat_map(|y| {
                    (0..32_usize).map(move |x| {
                        let idx = 4 * ((row * 32 + y) * width + col * 32 + x);
                        atlas.rgba()[idx + 3]
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
        let atlas = bake_colored_mesh(&vertices, &indices, 16, 2).unwrap();
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
        assert!(bake_colored_mesh(&vertices, &[999, 1, 2], 16, 2).is_err());
        assert!(bake_colored_mesh(&vertices, &indices, 16, 5).is_err());
        vertices[0].position[0] = f32::NAN;
        assert!(bake_colored_mesh(&vertices, &indices, 16, 2).is_err());
    }
}
