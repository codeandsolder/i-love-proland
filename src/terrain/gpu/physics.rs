//! Experimental fixed-step GPU rigid-body terrain solver. This is a local
//! chart-space sandbox, not yet a full planetary multi-body physics engine.
mod clock;
use super::{Kernel, RecordedReadback, TerrainGpu, TerrainRequest, invalid, storage};
pub use clock::{SimulationClock, StepBatch};
use std::io;
use wgpu::util::DeviceExt;

pub const STEP_WGSL: &str = concat!(
    include_str!("sample.wgsl"),
    include_str!("physics/body.wgsl"),
    include_str!("physics/step.wgsl")
);

/// A compound collider element: center relative to the body's COM, radius.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

/// A GPU-resident rigid body. Position is tile-relative west/north/up;
/// rotation is a unit quaternion xyzw. Last components encode properties.
///
/// Metadata is [first sphere, sphere count, status, reserved]:
/// status 0 active, 1 terrain coverage exhausted, 2 step too large.
/// The body is retained at its last successfully integrated state on a stop.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigidBody {
    pub position: [f32; 4],
    pub rotation: [f32; 4],
    pub velocity: [f32; 4],
    pub angular: [f32; 4],
    pub inverse_inertia: [f32; 4],
    pub metadata: [u32; 4],
    /// Low/high words of the absolute u64 tick, followed by two reserved words.
    pub clock: [u32; 4],
}

impl RigidBody {
    #[must_use = "Timestamp the actual completed physics state"]
    pub const fn completed_tick(&self) -> u64 {
        (self.clock[1] as u64) << 32 | self.clock[0] as u64
    }
    fn valid(&self, shape_count: usize, initial_tick: u64) -> bool {
        let finite = [
            self.position,
            self.rotation,
            self.velocity,
            self.angular,
            self.inverse_inertia,
        ]
        .into_iter()
        .flatten()
        .all(f32::is_finite);
        let norm = self.rotation.iter().map(|v| v * v).sum::<f32>();
        finite
            && (norm - 1.0).abs() < 1e-4
            && self.position[3] > 0.0
            && self.velocity[3] >= 0.0
            && (0.0..=1.0).contains(&self.angular[3])
            && self.inverse_inertia[..3].iter().all(|&v| v > 0.0)
            && (1..=64).contains(&self.metadata[1])
            && self.metadata[0]
                .checked_add(self.metadata[1])
                .is_some_and(|end| end as usize <= shape_count)
            && self.metadata[2] == 0
            && self.metadata[3] == 0
            && self.completed_tick() == initial_tick
            && self.clock[2..] == [0, 0]
    }
}

/// Bounded integration configuration, independent of render cadence.
#[derive(Clone, Copy, Debug)]
pub struct PhysicsConfig {
    pub initial_tick: u64,
    pub hz: u32,
    pub gravity_m_s2: f32,
    pub contact_slop_m: f32,
    pub solver_iterations: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepConfig {
    side: u32,
    slot: u32,
    count: u32,
    steps: u32,
    spacing: f32,
    dt: f32,
    gravity: f32,
    slop: f32,
    iterations: u32,
    padding: [u32; 3],
}

/// One batch of independent rigid bodies over a pinned terrain slot.
///
/// State stays on the GPU between submissions. The CPU owns only scheduling
/// metadata and optional snapshots. Vessel-vessel contacts and joints are
/// intentionally outside this first terrain-contact solver.
pub struct PhysicsGpu {
    pipeline: Kernel,
    bodies: wgpu::Buffer,
    spheres: wgpu::Buffer,
    terrain: wgpu::Buffer,
    request: TerrainRequest,
    params: StepConfig,
    submitted: u64,
    hz: u32,
}

impl PhysicsGpu {
    /// Upload initial rigid bodies and compound sphere colliders.
    ///
    /// # Errors
    /// Rejects invalid configuration, shapes, sizes or an uninitialized slot.
    pub fn new(
        device: &wgpu::Device,
        terrain: &TerrainGpu,
        slot: u32,
        config: PhysicsConfig,
        bodies: &[RigidBody],
        spheres: &[Sphere],
    ) -> io::Result<Self> {
        if config.hz == 0
            || config.hz > 100_000
            || !(0.0..=1000.0).contains(&config.gravity_m_s2)
            || !(0.0..=1.0).contains(&config.contact_slop_m)
            || !(1..=32).contains(&config.solver_iterations)
            || bodies.is_empty()
            || spheres.is_empty()
            || bodies
                .iter()
                .any(|body| !body.valid(spheres.len(), config.initial_tick))
            || spheres.iter().any(|sphere| {
                !sphere.center.iter().all(|v| v.is_finite())
                    || !(0.01..=1000.0).contains(&sphere.radius)
            })
        {
            return Err(invalid("invalid GPU physics configuration"));
        }
        let count = u32::try_from(bodies.len()).map_err(io::Error::other)?;
        if count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension {
            return Err(invalid("physics body dispatch exceeds device limit"));
        }
        let request = terrain
            .resident
            .get(slot as usize)
            .copied()
            .flatten()
            .ok_or_else(|| invalid("physics terrain slot not initialized"))?;
        let pipeline = Kernel::new(device, "fixed physics steps", STEP_WGSL, "advance");
        let bytes = bytemuck::cast_slice(bodies);
        if u64::try_from(bytes.len()).map_err(io::Error::other)?
            > device
                .limits()
                .max_storage_buffer_binding_size
                .min(device.limits().max_buffer_size)
        {
            return Err(invalid("physics storage exceeds device limit"));
        }
        let states = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU rigid body state"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
        });
        #[expect(
            clippy::cast_precision_loss,
            reason = "bounded local lattice spacing and fixed integration frequency"
        )]
        let params = StepConfig {
            side: terrain.side(),
            slot,
            count,
            steps: 0,
            spacing: request.spacing_ticks as f32 / 16.0,
            dt: 1.0 / config.hz as f32,
            gravity: config.gravity_m_s2,
            slop: config.contact_slop_m,
            iterations: config.solver_iterations,
            padding: [0; 3],
        };
        Ok(Self {
            pipeline,
            bodies: states,
            spheres: storage(
                device,
                "compound sphere shapes",
                bytemuck::cast_slice(spheres),
            )?,
            terrain: terrain.samples().clone(),
            request,
            params,
            submitted: config.initial_tick,
            hz: config.hz,
        })
    }

    /// Bind state directly to a GPU rendering or telemetry kernel.
    #[must_use = "Use the authoritative GPU body state buffer"]
    pub const fn states(&self) -> &wgpu::Buffer {
        &self.bodies
    }

    /// Encode a fixed-step batch without synchronizing with the host.
    /// The pinned terrain address must still occupy its slot. Submit command
    /// encoders in the order they were recorded; do not reuse the pinned slot
    /// while a physics batch using it is in flight.
    ///
    /// # Errors
    /// Rejects stale terrain ownership, invalid clock order or excessive work.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        terrain: &TerrainGpu,
        batch: StepBatch,
    ) -> io::Result<()> {
        if terrain.samples() != &self.terrain
            || terrain
                .resident
                .get(self.request.slot as usize)
                .copied()
                .flatten()
                != Some(self.request)
            || batch.hz != self.hz
            || batch.start_tick != self.submitted
            || !(1..=1024).contains(&batch.steps)
        {
            return Err(invalid("invalid physics batch or stale terrain slot"));
        }
        let end = batch
            .start_tick
            .checked_add(u64::from(batch.steps))
            .ok_or_else(|| invalid("physics tick overflow"))?;
        let params = StepConfig {
            steps: batch.steps,
            ..self.params
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("physics batch parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [&self.terrain, &self.bodies, &self.spheres, &uniform];
        self.pipeline.encode(
            device,
            encoder,
            "advance fixed physics ticks",
            &buffers,
            [self.params.count.div_ceil(64), 1, 1],
        )?;
        self.submitted = end;
        Ok(())
    }

    /// Record an optional asynchronous state snapshot; no terrain download.
    #[must_use = "Submit the encoder and begin the state readback"]
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RecordedReadback<RigidBody> {
        RecordedReadback::copy(device, encoder, &self.bodies, 0, self.bodies.size())
    }
}

mod presentation;
pub use presentation::{POSE_WGSL, PresentationHistory, PresentationTime, RenderPose};
