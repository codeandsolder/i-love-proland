//! Generate one stable species model and report the real GPU-compatible mesh
//! counts at multiple projected crown diameters, without requiring a device.

use i_love_proland::procedural::bake;
use lorax_rs::{Plant, ScreenLod, Species};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let species = Species::default();
    let plant = Plant::generate(&species, 0x4f414b)?;
    let bounds = plant.bounds();
    println!(
        "Source sphere center {:?}, radius {:.3}",
        bounds.center(),
        bounds.radius()
    );
    for pixels in [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0] {
        let geometry = bake(
            &plant,
            ScreenLod::new(pixels, 0.75, 12.0)?,
            species.wood_color,
        )?;
        println!(
            "{pixels:>5.0}px: {:>5} triangles",
            geometry.indices.len() / 3
        );
    }
    Ok(())
}
