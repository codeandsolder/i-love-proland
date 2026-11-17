//! Small mixed-species example: trees, rocks and debris use identical logic.
use i_love_proland::{
    CubeFace, Distribution, Domain, Generator, GeneratorConfig, Layer, TileKey, Variant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = Generator::new(
        Domain::Spherical {
            radius_m: 600_000.0,
            face_cells: 1 << 18,
        },
        GeneratorConfig {
            world_seed: 0x5ec0_ffee,
            cells_per_tile: 16,
            ..GeneratorConfig::default()
        },
    )?;

    let populations = [
        Layer::new(
            0x004f_414b,
            0.015,
            Distribution::Stratified,
            vec![
                Variant {
                    id: 1001,
                    weight: 3.0,
                },
                Variant {
                    id: 1002,
                    weight: 1.0,
                },
            ],
        )?,
        Layer::new(
            0x524f_434b,
            0.10,
            Distribution::Poisson,
            vec![
                Variant {
                    id: 2001,
                    weight: 4.0,
                },
                Variant {
                    id: 2002,
                    weight: 1.0,
                },
            ],
        )?,
        Layer::new(
            0x4445_4252_4953,
            0.006,
            Distribution::Poisson,
            vec![Variant {
                id: 3001,
                weight: 1.0,
            }],
        )?,
    ];

    let tile = TileKey::spherical(CubeFace::PositiveZ, 65_000 / 16, 66_000 / 16);
    for layer in &populations {
        let instances = scene.tile_with_habitat(tile, layer, |candidate| {
            // Deterministic example. Production code samples canonical
            // biome/noise/altitude data instead of using an arbitrary region.
            if candidate.cell.x % 7 == 0 { 0.2 } else { 0.9 }
        })?;
        println!(
            "layer={} stable objects={} first object={:?}",
            layer.id,
            instances.len(),
            instances.first(),
        );
    }
    Ok(())
}
