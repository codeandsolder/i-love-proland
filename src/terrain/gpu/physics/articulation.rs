//! Disjoint articulated rigid-body islands sharing the terrain contact solver.
use super::{Kernel, RigidBody, invalid};
use std::io;
use wgpu::util::DeviceExt;

pub const ARTICULATION_WGSL: &str = include_str!("articulation.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum JointKind {
    Fixed = 0,
    Prismatic = 1,
    Revolute = 2,
    Spherical = 3,
}

/// Local joint pivot and right-handed orientation; X is the free axis.
#[derive(Clone, Copy, Debug)]
pub struct JointFrame {
    pub anchor: [f32; 3],
    pub rotation: [f32; 4],
}

impl JointFrame {
    pub const IDENTITY: Self = Self {
        anchor: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
    };

    fn valid(self) -> bool {
        self.anchor
            .iter()
            .chain(&self.rotation)
            .all(|v| v.is_finite())
            && (self.rotation.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-4
    }
}

/// Absolute force/torque-based motor (SI), not an acceleration gain.
/// Set stiffness to zero for pure velocity control, both gains zero to coast.
#[derive(Clone, Copy, Debug)]
pub struct JointDrive {
    pub target_position: f32,
    pub target_velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
}

impl JointDrive {
    pub const PASSIVE: Self = Self {
        target_position: 0.0,
        target_velocity: 0.0,
        stiffness: 0.0,
        damping: 0.0,
        max_force: 0.0,
    };

    fn valid(self) -> bool {
        [
            self.target_position,
            self.target_velocity,
            self.stiffness,
            self.damping,
            self.max_force,
        ]
        .iter()
        .all(|v| v.is_finite())
            && self.stiffness >= 0.0
            && self.damping >= 0.0
            && self.max_force >= 0.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub bodies: [u32; 2],
    pub kind: JointKind,
    pub frames: [JointFrame; 2],
    /// Metres for a slide, radians for a hinge. None permits free motion.
    pub limits: Option<[f32; 2]>,
    pub drive: JointDrive,
}

impl Joint {
    fn valid(self, first: u32, end: u32) -> bool {
        self.bodies[0] != self.bodies[1]
            && self.bodies.iter().all(|&i| first <= i && i < end)
            && self.frames.iter().all(|frame| frame.valid())
            && self.drive.valid()
            && self
                .limits
                .is_none_or(|[lo, hi]| lo.is_finite() && hi.is_finite() && lo <= hi)
            && match self.kind {
                JointKind::Fixed | JointKind::Spherical => {
                    self.limits.is_none() && self.drive.max_force == 0.0
                }
                JointKind::Prismatic => true,
                JointKind::Revolute => self.limits.is_none_or(|[lo, hi]| {
                    lo >= -std::f32::consts::PI && hi <= std::f32::consts::PI
                }),
            }
    }
}

/// Disjoint contiguous ranges; every body and joint belongs to exactly
/// one island. Loops are permitted, including closed kinematic chains.
#[derive(Clone, Copy, Debug)]
pub struct Island {
    pub first_body: u32,
    pub body_count: u32,
    pub first_joint: u32,
    pub joint_count: u32,
}

#[derive(Clone, Copy)]
pub struct ArticulationLayout<'a> {
    pub islands: &'a [Island],
    pub joints: &'a [Joint],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Record {
    anchor_a: [f32; 4],
    anchor_b: [f32; 4],
    frame_a: [f32; 4],
    frame_b: [f32; 4],
    limits: [f32; 4],
    drive: [f32; 4],
    meta: [u32; 4],
}

const fn padded(v: [f32; 3]) -> [f32; 4] {
    [v[0], v[1], v[2], 0.0]
}

impl Record {
    fn joint(joint: Joint) -> Self {
        let [lo, hi] = joint.limits.unwrap_or([0.0; 2]);
        Self {
            anchor_a: padded(joint.frames[0].anchor),
            anchor_b: padded(joint.frames[1].anchor),
            frame_a: joint.frames[0].rotation,
            frame_b: joint.frames[1].rotation,
            limits: [lo, hi, joint.drive.stiffness, joint.drive.damping],
            drive: [
                joint.drive.target_position,
                joint.drive.target_velocity,
                joint.drive.max_force,
                0.0,
            ],
            meta: [
                joint.bodies[0],
                joint.bodies[1],
                joint.kind as u32,
                u32::from(joint.limits.is_some()),
            ],
        }
    }

    fn header(meta: [u32; 4]) -> Self {
        use bytemuck::Zeroable;
        Self {
            meta,
            ..Self::zeroed()
        }
    }
}

/// Contact and joint scratch is allocated by actual collider and joint
/// counts, not a fixed private-array limit for the entire vessel.
pub(super) struct Articulation {
    pub(super) topology: wgpu::Buffer,
    pub(super) scratch: wgpu::Buffer,
    pub(super) count: u32,
    pub(super) pipeline: Kernel,
    joints: Vec<Joint>,
    pub(super) collision_groups: Vec<u32>,
}

impl Articulation {
    pub(super) fn new(
        device: &wgpu::Device,
        layout: ArticulationLayout<'_>,
        bodies: &[RigidBody],

        stride: u32,
    ) -> io::Result<Self> {
        let count = u32::try_from(layout.islands.len()).map_err(io::Error::other)?;
        let joint_count = u32::try_from(layout.joints.len()).map_err(io::Error::other)?;
        let body_count = u32::try_from(bodies.len()).map_err(io::Error::other)?;
        if count == 0
            || count > device.limits().max_compute_workgroups_per_dimension
            || device.limits().max_storage_buffers_per_shader_stage < 10
        {
            return Err(invalid("articulated dispatch exceeds adapter limits"));
        }
        let mut collision_groups = Vec::with_capacity(bodies.len());
        let mut body_end = 0;
        let mut joint_end = 0;
        let mut records = vec![Record::header([count, body_count, joint_count, 0])];
        for (group, island) in layout.islands.iter().enumerate() {
            if island.first_body != body_end
                || island.first_joint != joint_end
                || island.body_count == 0
            {
                return Err(invalid(
                    "articulated islands must partition bodies and joints",
                ));
            }
            body_end = body_end
                .checked_add(island.body_count)
                .ok_or_else(|| invalid("body range overflow"))?;
            joint_end = joint_end
                .checked_add(island.joint_count)
                .ok_or_else(|| invalid("joint range overflow"))?;
            if body_end > body_count
                || joint_end > joint_count
                || layout.joints[island.first_joint as usize..joint_end as usize]
                    .iter()
                    .any(|joint| !joint.valid(island.first_body, body_end))
            {
                return Err(invalid("joint crosses island or has invalid parameters"));
            }
            collision_groups.extend(std::iter::repeat_n(
                u32::try_from(group).map_err(io::Error::other)?,
                island.body_count as usize,
            ));
            records.push(Record::header([
                island.first_body,
                island.body_count,
                island.first_joint,
                island.joint_count,
            ]));
        }
        if body_end != body_count || joint_end != joint_count {
            return Err(invalid("incomplete articulated island partition"));
        }
        records.extend(layout.joints.iter().copied().map(Record::joint));
        let scratch_count = u64::from(body_count) * u64::from(stride) + u64::from(joint_count);
        let size = scratch_count
            .checked_mul(112)
            .ok_or_else(|| invalid("articulation scratch overflow"))?;
        let bytes = bytemuck::cast_slice(&records);
        let limit = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        if size == 0
            || size > limit
            || u64::try_from(bytes.len()).map_err(io::Error::other)? > limit
        {
            return Err(invalid("articulation storage exceeds device limits"));
        }
        let topology = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("articulation islands and authored joints"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("articulated contact and joint impulses"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let source = format!("{}\n{ARTICULATION_WGSL}", super::STEP_WGSL);
        let pipeline = Kernel::new(
            device,
            "coupled articulated terrain islands",
            &source,
            "advance_island",
        );
        Ok(Self {
            topology,
            scratch,
            count,
            pipeline,
            joints: layout.joints.to_vec(),
            collision_groups,
        })
    }

    /// Encode changes rather than queue-write over in-flight batches.
    pub(super) fn drive(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        index: usize,
        drive: JointDrive,
    ) -> io::Result<()> {
        let joint = self
            .joints
            .get_mut(index)
            .ok_or_else(|| invalid("unknown articulated joint"))?;
        if !drive.valid()
            || matches!(joint.kind, JointKind::Fixed | JointKind::Spherical)
                && drive.max_force > 0.0
        {
            return Err(invalid("invalid joint drive"));
        }
        joint.drive = drive;
        let record = Record::joint(*joint);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("joint drive update"),
            contents: bytemuck::bytes_of(&record),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let stride = std::mem::size_of::<Record>() as u64;
        let offset =
            (1 + u64::from(self.count) + u64::try_from(index).map_err(io::Error::other)?) * stride;
        encoder.copy_buffer_to_buffer(&buffer, 0, &self.topology, offset, stride);
        Ok(())
    }
}
