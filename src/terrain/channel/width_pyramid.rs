//! Exact rectangular minimum over the channel support grid.
//! Coarse LOD queries usually cover large rectangles; reduce them through
//! min-mip levels without visiting all the underlying bins.
#[derive(Debug)]
pub(super) struct MinWidthPyramid {
    levels: Vec<Level>,
}

#[derive(Debug)]
struct Level {
    side: usize,
    widths: Vec<f64>,
}

impl Level {
    fn at(&self, x: usize, y: usize) -> f64 {
        self.widths[y * self.side + x]
    }

    fn reduced(&self) -> Self {
        let side = self.side.div_ceil(2);
        let mut widths = Vec::with_capacity(side * side);
        for y in 0..side {
            for x in 0..side {
                let x0 = 2 * x;
                let y0 = 2 * y;
                let x1 = (x0 + 1).min(self.side - 1);
                let y1 = (y0 + 1).min(self.side - 1);
                widths.push(
                    self.at(x0, y0)
                        .min(self.at(x1, y0))
                        .min(self.at(x0, y1))
                        .min(self.at(x1, y1)),
                );
            }
        }
        Self { side, widths }
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    min: [usize; 2],
    max: [usize; 2],
}

impl Bounds {
    fn intersects(self, other: Self) -> bool {
        (0..2).all(|axis| self.min[axis] <= other.max[axis] && other.min[axis] <= self.max[axis])
    }

    fn contains(self, other: Self) -> bool {
        (0..2).all(|axis| self.min[axis] <= other.min[axis] && self.max[axis] >= other.max[axis])
    }
}

impl MinWidthPyramid {
    pub(super) fn new(side: usize, widths: Vec<f64>) -> Self {
        let mut levels = vec![Level { side, widths }];
        while let Some(last) = levels.last() {
            if last.side == 1 {
                break;
            }
            levels.push(last.reduced());
        }
        Self { levels }
    }

    pub(super) fn minimum(&self, min: [usize; 2], max: [usize; 2]) -> Option<f64> {
        if (0..2).any(|axis| min[axis] > max[axis] || max[axis] >= self.levels[0].side) {
            return None;
        }
        let [x0, y0] = min;
        let [x1, y1] = max;
        // Fine tile footprints touch a handful of adjacent bins. No tree
        // traversal is needed for these overwhelmingly common queries.
        let width = if (x1 - x0 + 1) * (y1 - y0 + 1) <= 16 {
            let base = &self.levels[0];
            let mut best = f64::INFINITY;
            for y in y0..=y1 {
                for &value in &base.widths[y * base.side + x0..=y * base.side + x1] {
                    best = best.min(value);
                }
            }
            best
        } else {
            self.search(
                self.levels.len() - 1,
                [0, 0],
                Bounds { min, max },
                f64::INFINITY,
            )
        };
        width.is_finite().then_some(width)
    }

    fn search(&self, level: usize, node: [usize; 2], target: Bounds, mut best: f64) -> f64 {
        let width = self.levels[level].at(node[0], node[1]);
        // All descendants are at least this wide.
        if width >= best {
            return best;
        }
        let side = self.levels[0].side;
        let bounds = Bounds {
            min: node.map(|v| v << level),
            max: node.map(|v| (((v + 1) << level) - 1).min(side - 1)),
        };
        if !target.intersects(bounds) {
            return best;
        }
        if level == 0 || target.contains(bounds) {
            return width;
        }
        let next = level - 1;
        let child_side = self.levels[next].side;
        let [x, y] = node.map(|v| v * 2);
        for yy in y..(y + 2).min(child_side) {
            for xx in x..(x + 2).min(child_side) {
                best = self.search(next, [xx, yy], target, best);
                if best <= width {
                    return best;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_grid_rectangles_match_exhaustive_minimum() {
        for side in [1, 3, 17, 33, 257] {
            let values: Vec<_> = (0..side * side)
                .map(|i| {
                    if i % 7 == 0 {
                        f64::from(u32::try_from((i * 37) % 113).unwrap()) + 3.0
                    } else {
                        f64::INFINITY
                    }
                })
                .collect();
            let pyramid = MinWidthPyramid::new(side, values.clone());
            for i in 0..2500 {
                let xa = (i * 13 + 7) % side;
                let xb = (i * 17 + 1) % side;
                let ya = (i * 19 + 3) % side;
                let yb = (i * 23 + 5) % side;
                let min = [xa.min(xb), ya.min(yb)];
                let max = [xa.max(xb), ya.max(yb)];
                let mut expected = f64::INFINITY;
                for y in min[1]..=max[1] {
                    for x in min[0]..=max[0] {
                        expected = expected.min(values[y * side + x]);
                    }
                }
                assert_eq!(
                    pyramid.minimum(min, max),
                    expected.is_finite().then_some(expected)
                );
            }
        }
    }
}
