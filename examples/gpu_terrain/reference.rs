//! Validate the conservative CPU envelope against the authoritative GPU field.
use super::Result;
use i_love_proland::terrain::{Region, gpu::CollisionGrid};

pub fn compare(region: &Region, grid: &CollisionGrid) -> Result<()> {
    let mut largest_raise = 0.0_f64;
    let mut largest_cut = 0.0_f64;
    let mut min_slack = f64::INFINITY;
    for (index, sample) in grid.samples.iter().enumerate() {
        let i = u32::try_from(index)?;
        let point = [
            grid.request.origin_ticks[0] + grid.request.spacing_ticks * (i % grid.side),
            grid.request.origin_ticks[1] + grid.request.spacing_ticks * (i / grid.side),
        ]
        .map(|tick| f64::from(tick) / 16.0);
        let coarse = region.coarse_height(point).ok_or("invalid coarse point")?;
        let envelope = region
            .refined_height_envelope(point, point)
            .ok_or("invalid envelope")?;
        let bed = f64::from(sample.bed_m);
        largest_raise = largest_raise.max(bed - coarse);
        largest_cut = largest_cut.max(coarse - bed);
        let slack = (bed - envelope.min_m()).min(envelope.max_m() - bed);
        assert!(
            slack >= 0.0,
            "GPU bed {bed} outside {envelope:?} at {point:?}"
        );
        min_slack = min_slack.min(slack);
    }
    println!(
        "GPU_TERRAIN envelope count={} max_raise_m={largest_raise:.6} max_cut_m={largest_cut:.6} minimum_slack_m={min_slack:.6}",
        grid.samples.len()
    );
    Ok(())
}
