//! Experimental fixed-step GPU rigid-body terrain solver. This is a local
//! chart-space sandbox, not yet a full planetary multi-body physics engine.
mod clock;
mod contact_grid;
pub use contact_grid::{ContactGrid, PinnedGrid};
mod body_pairs;
mod static_scene;
use super::{
    GridProjection, Kernel, RecordedReadback, TerrainGpu, TerrainRequest, invalid, storage,
};
pub use clock::{SimulationClock, StepBatch};
pub use static_scene::{StaticScene, StaticTriangle};
use std::io;
use wgpu::util::DeviceExt;

pub const COLLIDER_WGSL: &str = include_str!("physics/collider.wgsl");

pub const STEP_WGSL: &str = concat!(
    include_str!("sample.wgsl"),
    include_str!("projection.wgsl"),
    include_str!("physics/body.wgsl"),
    include_str!("physics/body_math.wgsl"),
    include_str!("physics/collider.wgsl"),
    include_str!("physics/step.wgsl"),
    include_str!("physics/static_scene.wgsl")
);

/// Oriented ellipsoid relative to the rigid body's center of mass.
///
/// A sphere is the isotropic special case. Semi-axes are positive metres;
/// rotation is a proper local quaternion (xyzw).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Collider {
    center: [f32; 4],
    axes: [f32; 4],
    rotation: [f32; 4],
}

impl Collider {
    #[must_use = "Use the authored sphere geometry in the compound collider"]
    pub const fn sphere(center: [f32; 3], radius: f32) -> Self {
        Self::ellipsoid(center, [radius; 3], [0.0, 0.0, 0.0, 1.0])
    }

    #[must_use = "Use the authored oriented ellipsoid geometry"]
    pub const fn ellipsoid(center: [f32; 3], axes: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            center: [center[0], center[1], center[2], 0.0],
            axes: [axes[0], axes[1], axes[2], 0.0],
            rotation,
        }
    }

    fn valid(self) -> bool {
        self.center.iter().all(|v| v.is_finite())
            && self.axes[..3].iter().all(|v| (0.01..=1000.0).contains(v))
            && self.rotation.iter().all(|v| v.is_finite())
            && (self.rotation.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-4
    }
}

/// A GPU-resident rigid body. Position is tile-relative west/north/up;
/// rotation is a unit quaternion xyzw. Last components encode properties.
///
/// Metadata is [first collider, collider count, status, contact recorded]:
/// status 0 active, 1 terrain coverage exhausted, 2 step too large.
/// Status 3 is a successful first-contact prediction stop. Other stops retain
/// the last successfully integrated state.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigidBody {
    pub position: [f32; 4],
    pub rotation: [f32; 4],
    pub velocity: [f32; 4],
    pub angular: [f32; 4],
    pub inverse_inertia: [f32; 4],
    pub metadata: [u32; 4],
    /// Absolute completed tick and first-contact tick, each as low/high words.
    pub clock: [u32; 4],
}

impl RigidBody {
    #[must_use = "Timestamp the actual completed physics state"]
    pub const fn completed_tick(&self) -> u64 {
        (self.clock[1] as u64) << 32 | self.clock[0] as u64
    }
    /// First detected solid contact at an integrated tick, not a swept TOI.
    #[must_use = "Use the physical contact timestamp rather than submission time"]
    pub const fn first_contact_tick(&self) -> Option<u64> {
        if self.metadata[3] == 0 {
            None
        } else {
            Some((self.clock[3] as u64) << 32 | self.clock[2] as u64)
        }
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

/// Bounded pressure-dependent compliant layer over the rigid triangles.
///
/// The ellipsoid cross-section at its current indentation provides the
/// loaded area on the local supporting plane.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContactMaterial {
    pub max_sinkage_m: f32,
    pub half_sinkage_pressure_pa: f32,
}

impl ContactMaterial {
    pub const RIGID: Self = Self {
        max_sinkage_m: 0.0,
        half_sinkage_pressure_pa: 1.0,
    };

    fn valid(self) -> bool {
        self.max_sinkage_m.is_finite()
            && self.max_sinkage_m >= 0.0
            && self.half_sinkage_pressure_pa.is_finite()
            && self.half_sinkage_pressure_pa > 0.0
    }
}

/// Optional per-tick contact telemetry.
///
/// Position/normal/orientation use the solver's metric tangent coordinates. A zero material tag is an
/// explicit contact gap. Only completed integrated ticks produce records.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContactEvent {
    /// xyz: actual terrain support point, w: normal force in newtons.
    pub position_force: [f32; 4],
    /// xyz: normal, w: measured indentation relative to rigid triangles.
    pub normal_depth: [f32; 4],
    pub rotation: [f32; 4],
    /// xyz: tangential velocity at the loaded point, w: contact area in m².
    pub slip_area: [f32; 4],
    /// [completed tick low, completed tick high, global shape ID, material index + 1].
    pub metadata: [u32; 4],
    /// Pre-solve closing normal speed (m/s), normal impulse (N s),
    /// friction impulse magnitude (N s), reserved. Never infer crash
    /// severity from post-solve velocity or penetration bias.
    pub impact: [f32; 4],
}

impl ContactEvent {
    #[must_use = "Use the actual completed integration tick"]
    pub const fn tick(&self) -> u64 {
        (self.metadata[1] as u64) << 32 | self.metadata[0] as u64
    }
    /// Index into the exact material palette bound to the solver.
    #[must_use = "Distinguish a loaded contact from a gap"]
    pub const fn material_index(&self) -> Option<u32> {
        self.metadata[3].checked_sub(1)
    }
}

/// Categorical material field for one pinned collision grid.
#[derive(Clone, Copy, Debug)]
pub enum ContactMaterialField<'a> {
    Uniform(ContactMaterial),
    /// Row-major canonical cells, identical indexing to contact triangles.
    /// The profile IDs remain stable for the entire integrated batch.
    Cells {
        profiles: &'a [ContactMaterial],
        indices: &'a [u32],
    },
}

impl ContactMaterialField<'_> {
    fn upload(self, device: &wgpu::Device, side: u32) -> io::Result<(wgpu::Buffer, wgpu::Buffer)> {
        let (uniform, profiles, indices) = match self {
            Self::Uniform(material) => ([material], None, None),
            Self::Cells { profiles, indices } => {
                ([ContactMaterial::RIGID], Some(profiles), Some(indices))
            }
        };
        let profiles = profiles.unwrap_or(&uniform);
        let cells = indices.unwrap_or(&[0]);
        let extent = usize::try_from(u64::from(side - 1).pow(2)).map_err(io::Error::other)?;
        if profiles.is_empty()
            || profiles.len() >= u32::MAX as usize
            || profiles.iter().any(|p| !p.valid())
            || indices.is_some_and(|v| v.len() != extent)
            || cells.iter().any(|&i| i as usize >= profiles.len())
        {
            return Err(invalid("invalid categorical contact material field"));
        }
        Ok((
            storage(
                device,
                "contact material palette",
                bytemuck::cast_slice(profiles),
            )?,
            storage(
                device,
                "contact material cells",
                bytemuck::cast_slice(cells),
            )?,
        ))
    }
}

/// Bounded integration configuration, independent of render cadence.
#[derive(Clone, Copy, Debug)]
pub struct PhysicsConfig<'a> {
    pub initial_tick: u64,
    pub hz: u32,
    pub gravity_m_s2: f32,
    pub contact_slop_m: f32,
    pub solver_iterations: u32,
    pub material: ContactMaterialField<'a>,
    /// Finish a predictive calculation at the first discrete contact.
    pub stop_at_contact: bool,
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
    stop_at_contact: u32,
    record_contacts: u32,
    contact_stride: u32,

    projection: GridProjection,
    first_tick_low: u32,
    padding: [u32; 3],
}

/// One batch of independent rigid bodies over a pinned terrain slot.
///
/// State stays on the GPU between submissions. The CPU owns only scheduling
/// metadata and optional snapshots. Vessel-vessel contacts and joints are
/// intentionally outside this first terrain-contact solver.
pub struct PhysicsGpu {
    pipeline: Kernel,
    static_scene: StaticScene,
    sphere_eligible: bool,
    body_pairs: Option<body_pairs::BodyPairs>,
    bodies: wgpu::Buffer,
    shapes: wgpu::Buffer,
    empty_contacts: wgpu::Buffer,
    materials: wgpu::Buffer,
    material_cells: wgpu::Buffer,
    terrain: wgpu::Buffer,
    request: TerrainRequest,
    params: StepConfig,
    submitted: u64,
    hz: u32,
}

impl PhysicsGpu {
    /// Upload initial rigid bodies and compound ellipsoid colliders.
    ///
    /// # Errors
    /// Rejects invalid configuration, shapes, sizes or an uninitialized slot.
    pub fn new(
        device: &wgpu::Device,
        terrain: &(impl ContactGrid + ?Sized),
        slot: u32,
        config: PhysicsConfig<'_>,
        bodies: &[RigidBody],
        shapes: &[Collider],
    ) -> io::Result<Self> {
        if config.hz == 0
            || config.hz > 100_000
            || !(0.0..=1000.0).contains(&config.gravity_m_s2)
            || !(0.0..=1.0).contains(&config.contact_slop_m)
            || !(1..=32).contains(&config.solver_iterations)
            || bodies.is_empty()
            || shapes.is_empty()
            || bodies
                .iter()
                .any(|body| !body.valid(shapes.len(), config.initial_tick))
            || shapes.iter().any(|shape| !shape.valid())
        {
            return Err(invalid("invalid GPU physics configuration"));
        }
        let count = u32::try_from(bodies.len()).map_err(io::Error::other)?;
        if count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension {
            return Err(invalid("physics body dispatch exceeds device limit"));
        }
        let request = terrain
            .grid_request(slot)
            .ok_or_else(|| invalid("physics terrain slot not initialized"))?;
        let (materials, material_cells) = config.material.upload(device, terrain.side())?;
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
            stop_at_contact: u32::from(config.stop_at_contact),
            record_contacts: 0,
            contact_stride: bodies
                .iter()
                .map(|body| body.metadata[1])
                .max()
                .unwrap_or(1),
            projection: terrain.grid_projection(slot)?,
            first_tick_low: 0,
            padding: [0; 3],
        };
        Ok(Self {
            pipeline,
            static_scene: StaticScene::new(device, &[])?,
            sphere_eligible: body_pairs::eligible(bodies, shapes),
            body_pairs: None,
            bodies: states,
            shapes: storage(
                device,
                "compound ellipsoid shapes",
                bytemuck::cast_slice(shapes),
            )?,
            empty_contacts: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("disabled contact telemetry"),
                size: std::mem::size_of::<ContactEvent>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            materials,
            material_cells,
            terrain: terrain.samples().clone(),
            request,
            params,
            submitted: config.initial_tick,
            hz: config.hz,
        })
    }

    /// Geometry used by both the solver and its terrain draw.
    #[must_use = "Use the same projection for the contact mesh"]
    pub const fn projection(&self) -> GridProjection {
        self.params.projection
    }

    /// Resident terrain slot used by the solver.
    #[must_use = "Use the actual resident slot"]
    pub const fn terrain_slot(&self) -> u32 {
        self.request.slot
    }

    /// Bind state directly to a GPU rendering or telemetry kernel.
    #[must_use = "Use the authoritative GPU body state buffer"]
    pub const fn states(&self) -> &wgpu::Buffer {
        &self.bodies
    }

    /// Share immutable compound ellipsoid data with diagnostics and rendering.
    #[must_use = "Bind the exact collider data without uploading a second copy"]
    pub const fn colliders(&self) -> &wgpu::Buffer {
        &self.shapes
    }

    /// Enable coupled GPU contacts for compound ellipsoids.
    ///
    /// Centered spheres automatically use the specialized local/parallel
    /// solver. General compounds use symmetric Jacobi contact sweeps.
    /// This is discrete contact resolution, not swept collision prediction.
    ///
    /// # Errors
    /// Rejects first-contact stopping, which requires coupled TOI handling.
    pub fn enable_body_pairs(&mut self, device: &wgpu::Device) -> io::Result<()> {
        if self.params.stop_at_contact != 0 {
            return Err(invalid("coupled contacts require normal stepping"));
        }
        self.body_pairs = Some(body_pairs::BodyPairs::new(
            device,
            &self.bodies,
            &self.shapes,
            self.params.count,
            self.sphere_eligible,
        )?);
        Ok(())
    }

    /// Bind a canonical static collision mesh, independent of the render cut.
    ///
    /// # Errors
    /// Rejects material IDs outside the solver's material palette.
    pub fn set_static_scene(&mut self, scene: StaticScene) -> io::Result<()> {
        if u64::from(scene.max_material)
            >= self.materials.size() / std::mem::size_of::<ContactMaterial>() as u64
        {
            return Err(invalid("static scene material outside contact palette"));
        }
        self.static_scene = scene;
        Ok(())
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
        terrain: &(impl ContactGrid + ?Sized),
        batch: StepBatch,
    ) -> io::Result<()> {
        self.encode_batch(device, encoder, terrain, batch, None)
    }

    /// Number of collider entries per body per integrated tick in the
    /// optional observation buffer. Unused entries are explicit gaps.
    #[must_use = "Use the exact telemetry layout when decoding batches"]
    pub const fn contact_stride(&self) -> u32 {
        self.params.contact_stride
    }

    /// Record a bounded batch of authoritative contact observations.
    /// This does not change integration or require synchronizing every tick.
    ///
    /// # Errors
    /// Rejects invalid batch or telemetry allocation beyond adapter limits.
    pub fn encode_contacts(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        terrain: &(impl ContactGrid + ?Sized),
        batch: StepBatch,
    ) -> io::Result<RecordedReadback<ContactEvent>> {
        let size = u64::from(batch.steps)
            .checked_mul(u64::from(self.params.count))
            .and_then(|v| v.checked_mul(u64::from(self.params.contact_stride)))
            .and_then(|v| v.checked_mul(std::mem::size_of::<ContactEvent>() as u64))
            .ok_or_else(|| invalid("contact telemetry size overflow"))?;
        if size == 0
            || size
                > device
                    .limits()
                    .max_storage_buffer_binding_size
                    .min(device.limits().max_buffer_size)
        {
            return Err(invalid(
                "contact telemetry exceeds adapter limit; shorten batch",
            ));
        }
        let contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bounded physics contact observations"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encode_batch(device, encoder, terrain, batch, Some(&contacts))?;
        Ok(RecordedReadback::copy(device, encoder, &contacts, 0, size))
    }

    fn encode_batch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        terrain: &(impl ContactGrid + ?Sized),
        batch: StepBatch,
        contacts: Option<&wgpu::Buffer>,
    ) -> io::Result<()> {
        if terrain.samples() != &self.terrain
            || terrain.grid_request(self.request.slot) != Some(self.request)
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
            steps: if self.body_pairs.is_some() {
                1
            } else {
                batch.steps
            },
            first_tick_low: u32::try_from(batch.start_tick & u64::from(u32::MAX))
                .map_err(io::Error::other)?,
            record_contacts: u32::from(contacts.is_some()),
            ..self.params
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("physics batch parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &self.terrain,
            &self.bodies,
            &self.shapes,
            &uniform,
            contacts.unwrap_or(&self.empty_contacts),
            &self.materials,
            &self.material_cells,
            &self.static_scene.nodes,
            &self.static_scene.triangles,
        ];
        let bind = self.pipeline.bind(device, "physics batch", &buffers)?;
        let groups = [self.params.count.div_ceil(64), 1, 1];
        if let Some(pairs) = &self.body_pairs {
            for _ in 0..batch.steps {
                self.pipeline
                    .dispatch(encoder, "advance one coupled physics tick", &bind, groups);
                pairs.encode(encoder);
            }
        } else {
            self.pipeline
                .dispatch(encoder, "advance fixed physics ticks", &bind, groups);
        }
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
