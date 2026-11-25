//! Compact normal/tint input synthesized from a real elevation residual.
//! One atlas is reusable across regions and does not contain planetary
//! coordinates. Source samples have a halo for overlapping patch synthesis.
use std::io::{self, Write};

const MAGIC: &[u8; 8] = b"PLRLF001";
const MIN_SIDE: u16 = 129;
const MAX_SIDE: u16 = 2048;
// These fixed scales form part of the on-disk texture contract.
// Signed normalized slopes cover roughly +/-2; residual heights are
// divided by source sample spacing and cover roughly +/-8.
const SLOPE_SCALE: f64 = 63.0;
const HEIGHT_SCALE: f64 = 15.0;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid geological relief atlas",
    )
}

#[derive(Debug)]
pub struct ReliefAtlas {
    side: u16,
    rgb: Vec<u8>,
}

fn pixel_count(side: u16) -> io::Result<usize> {
    if !(MIN_SIDE..=MAX_SIDE).contains(&side) {
        return Err(invalid());
    }
    Ok(usize::from(side).pow(2))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite residual is rounded and clamped to a bounded RGB8 code"
)]
const fn encode(value: f64, scale: f64) -> u8 {
    value.mul_add(scale, 128.0).round().clamp(1.0, 255.0) as u8
}

impl ReliefAtlas {
    /// Build from a high-pass elevation grid with known sample spacing.
    ///
    /// R/G encode normalized central-difference slopes and B encodes the
    /// residual height divided by source spacing. Zero maps exactly to 128.
    /// Extreme values saturate to the format range. On shader sampling,
    /// decode slopes as (code - 128) / 63 and height as (code - 128) / 15.
    ///
    /// # Errors
    /// Rejects invalid dimensions, spacing or nonfinite input.
    pub fn from_residual(side: u16, spacing_m: f64, heights: &[f64]) -> io::Result<Self> {
        let count = pixel_count(side)?;
        if !spacing_m.is_finite()
            || spacing_m <= 0.0
            || heights.len() != count
            || heights.iter().any(|&h| !(h / spacing_m).is_finite())
        {
            return Err(invalid());
        }
        let mut rgb = Vec::with_capacity(count * 3);
        let side_usize = usize::from(side);
        let at = |x: usize, y: usize| heights[y * side_usize + x] / spacing_m;
        for y in 0..side_usize {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(side_usize - 1);
            let dy = f64::from(if y0 == y || y1 == y { 1 } else { 2 });
            for x in 0..side_usize {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(side_usize - 1);
                let dx = f64::from(if x0 == x || x1 == x { 1 } else { 2 });
                rgb.extend([
                    encode((at(x1, y) - at(x0, y)) / dx, SLOPE_SCALE),
                    encode((at(x, y1) - at(x, y0)) / dy, SLOPE_SCALE),
                    encode(at(x, y), HEIGHT_SCALE),
                ]);
            }
        }
        Ok(Self { side, rgb })
    }

    /// Width and height of the square source image.
    #[must_use = "Use the validated texture extent"]
    pub const fn side(&self) -> u16 {
        self.side
    }

    /// Linear RGB8 texels in increasing source X and Y order.
    #[must_use = "Use the packed derivative and residual texture"]
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// Serialize the compact source texture, without duplicated mip levels.
    ///
    /// # Errors
    /// Propagates I/O errors.
    pub fn write(&self, mut output: impl Write) -> io::Result<()> {
        output.write_all(MAGIC)?;
        output.write_all(&self.side.to_le_bytes())?;
        output.write_all(&self.rgb)
    }

    /// Validate and decode one complete atlas.
    ///
    /// # Errors
    /// Rejects invalid geometry, version, length or reserved zero codes.
    pub fn read(bytes: &[u8]) -> io::Result<Self> {
        if bytes.get(..8) != Some(MAGIC) {
            return Err(invalid());
        }
        let side = u16::from_le_bytes(
            bytes
                .get(8..10)
                .ok_or_else(invalid)?
                .try_into()
                .map_err(|_| invalid())?,
        );
        let count = pixel_count(side)?;
        let rgb = bytes.get(10..).ok_or_else(invalid)?;
        if rgb.len() != 3 * count || rgb.contains(&0) {
            return Err(invalid());
        }
        Ok(Self {
            side,
            rgb: rgb.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_residual_preserves_dimensionless_gradient_and_roundtrips() -> io::Result<()> {
        let side = 129;
        let spacing = 62.5;
        let heights: Vec<_> = (0..side)
            .flat_map(|y| {
                (0..side).map(move |x| {
                    (f64::from(x) - 64.0).mul_add(0.25, (f64::from(y) - 64.0) * -0.5) * spacing
                })
            })
            .collect();
        let atlas = ReliefAtlas::from_residual(side, spacing, &heights)?;
        assert_eq!(atlas.side(), side);
        assert!(
            atlas
                .rgb()
                .as_chunks::<3>()
                .0
                .iter()
                .all(|p| p[0] == 144 && p[1] == 97)
        );
        assert_eq!(atlas.rgb()[3 * (64 * usize::from(side) + 64) + 2], 128);
        let mut bytes = Vec::new();
        atlas.write(&mut bytes)?;
        let decoded = ReliefAtlas::read(&bytes)?;
        assert_eq!(decoded.rgb(), atlas.rgb());
        assert!(ReliefAtlas::read(&bytes[..bytes.len() - 1]).is_err());
        bytes.push(0);
        assert!(ReliefAtlas::read(&bytes).is_err());
        assert!(ReliefAtlas::from_residual(side, 0.0, &heights).is_err());
        Ok(())
    }
}
