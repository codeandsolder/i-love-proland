//! Independent target time, submitted physics time, and completed physics
//! time. A renderer may interpolate available snapshots; it never advances
//! simulation time or chooses the integration step.
use std::{io, time::Duration};

const RATE_SCALE: u128 = 1_000_000;
const SECOND: u128 = 1_000_000_000 * RATE_SCALE;

/// One bounded GPU submission; it may contain many fixed integration steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepBatch {
    pub start_tick: u64,
    pub steps: u32,
    pub hz: u32,
}

/// Exact rational target clock. Warp is measured in millionths of real time.
/// Changing warp preserves accumulated fractions and does not change dt.
#[derive(Debug)]
pub struct SimulationClock {
    hz: u32,
    target_credit: u128,
    target_tick: u64,
    submitted: u64,
    complete: u64,
}

impl SimulationClock {
    /// Construct an independent fixed-step timeline.
    ///
    /// # Errors
    /// Rejects a zero integration frequency.
    pub fn new(hz: u32) -> io::Result<Self> {
        Self::at_tick(hz, 0)
    }

    /// Start an independent contact timeline at an absolute tick, for
    /// example when a vessel leaves analytic orbit propagation.
    ///
    /// # Errors
    /// Rejects a zero integration frequency.
    pub fn at_tick(hz: u32, tick: u64) -> io::Result<Self> {
        if hz == 0 {
            return Err(io::Error::other("zero physics frequency"));
        }
        Ok(Self {
            hz,
            target_credit: u128::from(tick) * SECOND,
            target_tick: tick,
            submitted: tick,
            complete: tick,
        })
    }

    /// Accrue requested simulation time independently of GPU availability.
    /// Zero warp pauses the target clock; outstanding work remains valid.
    ///
    /// # Errors
    /// Rejects time overflow.
    pub fn advance(&mut self, elapsed: Duration, warp_millionths: u64) -> io::Result<()> {
        let increment = elapsed
            .as_nanos()
            .checked_mul(u128::from(warp_millionths))
            .and_then(|v| v.checked_mul(u128::from(self.hz)))
            .ok_or_else(|| io::Error::other("warp overflow"))?;
        let next = self
            .target_credit
            .checked_add(increment)
            .ok_or_else(|| io::Error::other("simulation time overflow"))?;
        let whole = next / SECOND;
        self.target_tick = u64::try_from(whole).map_err(io::Error::other)?;
        self.target_credit = next;
        Ok(())
    }

    /// Requested physics tick, including any uncomputed work.
    #[must_use = "Compare target with actual progress"]
    pub const fn target_tick(&self) -> u64 {
        self.target_tick
    }

    /// Requested timestamp, rounded down to a nanosecond. Fractional
    /// nanosecond warp credit remains internal and survives rate changes.
    #[must_use = "Use independent target time for scheduling and display"]
    pub const fn target_nanoseconds(&self) -> u128 {
        self.target_credit / (self.hz as u128 * RATE_SCALE)
    }

    /// Exact tick-to-timestamp conversion, rounded down once, not an
    /// accumulated floating-point dt. Rendering can sample any timestamp.
    #[must_use = "Timestamp the completed state rather than the render frame"]
    pub const fn tick_nanoseconds(&self, tick: u64) -> u128 {
        tick as u128 * 1_000_000_000 / self.hz as u128
    }

    #[must_use = "Use fixed dt independently of rendering and warp"]
    pub fn dt_seconds(&self) -> f64 {
        1.0 / f64::from(self.hz)
    }

    #[must_use = "Use the submitted time when scheduling additional work"]
    pub const fn submitted_tick(&self) -> u64 {
        self.submitted
    }

    #[must_use = "Use completed physics time for authoritative snapshots"]
    pub const fn completed_tick(&self) -> u64 {
        self.complete
    }

    /// Bound both per-dispatch work and the entire in-flight backlog.
    /// No accrued simulation time is discarded when either bound is hit.
    #[must_use = "Submit or retain the pending batch"]
    pub fn plan(&self, max_steps: u32, max_in_flight: u32) -> Option<StepBatch> {
        let available = self.target_tick().saturating_sub(self.submitted);
        let room = u64::from(max_in_flight).saturating_sub(self.submitted - self.complete);
        let count = available.min(room).min(u64::from(max_steps));
        let steps = u32::try_from(count).ok()?;
        (steps != 0).then_some(StepBatch {
            start_tick: self.submitted,
            steps,
            hz: self.hz,
        })
    }

    /// Record a successfully submitted batch, not merely planned work.
    ///
    /// # Errors
    /// Rejects overlapping, out-of-order or future batches.
    pub fn submitted(&mut self, batch: StepBatch) -> io::Result<()> {
        let end = batch
            .start_tick
            .checked_add(u64::from(batch.steps))
            .ok_or_else(|| io::Error::other("batch overflow"))?;
        let contiguous = batch.start_tick == self.submitted;
        let compatible = batch.hz == self.hz && batch.steps > 0;
        if !contiguous || !compatible || end > self.target_tick() {
            return Err(io::Error::other("invalid simulation submission"));
        }
        self.submitted = end;
        Ok(())
    }

    /// Accept an ordered completion notification (possibly several batches).
    ///
    /// # Errors
    /// Rejects time regression or completion of unsubmitted work.
    pub fn completed(&mut self, tick: u64) -> io::Result<()> {
        if !(self.complete..=self.submitted).contains(&tick) {
            return Err(io::Error::other("invalid simulation completion"));
        }
        self.complete = tick;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_cadence_and_batch_sizes_do_not_change_tick_count() -> io::Result<()> {
        let mut clock = SimulationClock::new(240)?;
        for _ in 0..1000 {
            clock.advance(Duration::from_micros(1001), 10_000_000)?;
        }
        assert_eq!(clock.target_tick(), 2402);
        let mut single = SimulationClock::new(240)?;
        single.advance(Duration::from_millis(1001), 10_000_000)?;
        assert_eq!(single.target_tick(), clock.target_tick());
        let batch = clock
            .plan(64, 128)
            .ok_or_else(|| io::Error::other("missing batch"))?;
        clock.submitted(batch)?;
        assert!(clock.submitted(batch).is_err());
        let next = clock
            .plan(1000, 128)
            .ok_or_else(|| io::Error::other("missing second batch"))?;
        clock.submitted(next)?;
        assert!(clock.plan(1000, 128).is_none());
        assert!(clock.completed(129).is_err());
        clock.completed(128)?;
        assert_eq!(clock.plan(1000, 128).map(|b| b.start_tick), Some(128));
        clock.advance(Duration::from_secs(1), 0)?;
        assert_eq!(clock.target_tick(), 2402);
        while let Some(batch) = clock.plan(71, 1000) {
            clock.submitted(batch)?;
            clock.completed(clock.submitted_tick())?;
        }
        assert_eq!(clock.completed_tick(), clock.target_tick());
        Ok(())
    }

    #[test]
    fn high_epoch_retains_exact_tick_and_time_credit() -> io::Result<()> {
        let epoch = u64::from(u32::MAX) - 100;
        let mut clock = SimulationClock::at_tick(240, epoch)?;
        assert_eq!(clock.target_nanoseconds(), clock.tick_nanoseconds(epoch));
        clock.advance(Duration::from_secs(1), 1_000_000)?;
        assert_eq!(clock.target_tick(), epoch + 240);
        let batch = clock
            .plan(300, 300)
            .ok_or_else(|| io::Error::other("missing epoch batch"))?;
        clock.submitted(batch)?;
        clock.completed(epoch + 240)?;
        assert_eq!(clock.completed_tick(), epoch + 240);
        Ok(())
    }

    #[test]
    fn fractional_warp_survives_changes_and_no_work_is_lost() -> io::Result<()> {
        let mut clock = SimulationClock::new(240)?;
        clock.advance(Duration::from_millis(1), 500_000)?;
        assert_eq!(clock.target_tick(), 0);
        clock.advance(Duration::from_millis(19), 500_000)?;
        assert_eq!(clock.target_tick(), 2);
        clock.advance(Duration::from_millis(1), 10_000_000)?;
        assert_eq!(clock.target_tick(), 4);
        assert!(clock.plan(0, 10).is_none());
        assert_eq!(clock.submitted_tick(), 0);
        Ok(())
    }
}
