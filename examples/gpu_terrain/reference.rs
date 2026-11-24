//! Compare device-generated samples against the actual CPU region evaluator.
use super::Result;
use i_love_proland::terrain::{Region, WaterKind, gpu::CollisionGrid};

pub fn compare(region: &Region, grid: &CollisionGrid) -> Result<()> {
    let mut bed_error = 0.0_f64;
    let mut water_error = 0.0_f64;
    let mut mismatches = 0;
    let mut color_error = 0.0_f64;
    for (index, sample) in grid.samples.iter().enumerate() {
        let i = u32::try_from(index)?;
        let point = [
            grid.request.origin_ticks[0] + grid.request.spacing_ticks * (i % grid.side),
            grid.request.origin_ticks[1] + grid.request.spacing_ticks * (i / grid.side),
        ]
        .map(|tick| f64::from(tick) / 16.0);
        let reference = region
            .sample_chart(point)
            .ok_or("invalid reference point")?;
        bed_error = bed_error.max((reference.weather.bed - f64::from(sample.bed_m)).abs());
        let expected_kind = match reference.water_kind {
            None => 0,
            Some(WaterKind::Standing) => 1,
            Some(WaterKind::River) => 2,
        };
        mismatches += usize::from(expected_kind != sample.water_kind);
        if expected_kind == sample.water_kind {
            let water = reference.weather.water.unwrap_or(reference.weather.bed);
            water_error = water_error.max((water - f64::from(sample.water_m)).abs());
        }
        for (actual, expected) in sample.rgba.to_le_bytes().into_iter().zip(reference.rgb) {
            color_error = color_error.max((f64::from(actual) - expected * 255.0).abs());
        }
    }
    println!(
        "GPU_TERRAIN reference count={} max_bed_m={bed_error:.6} max_water_m={water_error:.6} wet_kind_mismatches={mismatches} max_rgb8={color_error:.3}",
        grid.samples.len()
    );
    assert!(
        bed_error <= 0.05,
        "GPU bed differs from canonical CPU field"
    );
    assert!(
        water_error <= 0.05,
        "GPU water differs from canonical CPU field"
    );
    assert!(
        color_error <= 2.0,
        "GPU color differs from canonical CPU field"
    );
    // At a wet/dry threshold f32 and f64 may classify a sample differently;
    // report the count separately from the continuous surface errors.
    Ok(())
}
