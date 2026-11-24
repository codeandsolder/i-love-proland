//! Independent f64 closest-distance check against the actual generated mesh.
use i_love_proland::terrain::{RegionChart, gpu::CollisionGrid};
use std::io;

type V = [f64; 3];

fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: V, b: V) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn cross(a: V, b: V) -> V {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}
fn scale(a: V, factor: f64) -> V {
    a.map(|x| x * factor)
}
fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}

fn edge_distance_sq(p: V, a: V, b: V) -> f64 {
    let axis = sub(b, a);
    let offset = sub(p, a);
    let t = (dot(offset, axis) / dot(axis, axis).max(f64::MIN_POSITIVE)).clamp(0.0, 1.0);
    let residual = sub(offset, scale(axis, t));
    dot(residual, residual)
}

fn triangle_distance_sq(p: V, [a, b, c]: [V; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let normal = cross(ab, ac);
    let n2 = dot(normal, normal);
    let signed = dot(sub(p, a), normal);
    // Edge half-spaces determine whether the orthogonal projection lies
    // within the actual triangle. Otherwise the nearest point is on an edge.
    if n2 > 0.0
        && [(a, b), (b, c), (c, a)]
            .into_iter()
            .all(|(u, v)| dot(cross(sub(v, u), sub(p, u)), normal) >= 0.0)
    {
        return signed * signed / n2;
    }
    [(a, b), (b, c), (c, a)]
        .into_iter()
        .map(|(u, v)| edge_distance_sq(p, u, v))
        .fold(f64::INFINITY, f64::min)
}

pub(super) fn sphere_clearance(
    grid: &CollisionGrid,
    chart: RegionChart,
    position: [f32; 4],
) -> io::Result<f64> {
    let half = f64::from(grid.side - 1) * f64::from(grid.request.spacing_ticks) / 32.0;
    let centre = grid
        .request
        .origin_ticks
        .map(|v| f64::from(v) / 16.0 + half);
    let up = chart.direction(centre);
    let west = scale([up[2], 0.0, -up[0]], up[0].hypot(up[2]).recip());
    let north = cross(up, west);
    let [x, y, z, _] = position.map(f64::from);
    let body = add(
        scale(up, chart.radius_m + z),
        add(scale(west, x - half), scale(north, y - half)),
    );
    let vertices = grid.body_vertices(chart)?;
    Ok(grid
        .triangles()
        .map(|indices| triangle_distance_sq(body, indices.map(|i| vertices[i as usize])))
        .fold(f64::INFINITY, f64::min)
        .sqrt())
}
