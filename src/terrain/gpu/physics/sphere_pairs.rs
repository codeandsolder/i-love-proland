//! Coupled spherical rigid bodies share the regular terrain/static-contact
//! solver and advance at the same fixed simulation ticks.
use super::{Collider, Kernel, RigidBody, invalid};
use std::io;
use wgpu::util::DeviceExt;

const SOURCE: &str = concat!(
    include_str!("body.wgsl"),
    include_str!("collider.wgsl"),
    include_str!("body_math.wgsl"),
    include_str!("sphere_pair_math.wgsl"),
    include_str!("sphere_pairs.wgsl"),
);

pub(super) fn eligible(bodies: &[RigidBody], colliders: &[Collider]) -> bool {
    bodies.iter().all(|body| {
        if body.metadata[1] != 1 {
            return false;
        }
        let s = colliders[body.metadata[0] as usize];
        s.center[..3].iter().all(|&v| v == 0.0)
            && s.axes[0].to_bits() == s.axes[1].to_bits()
            && s.axes[0].to_bits() == s.axes[2].to_bits()
    })
}

const LOCAL_SOURCE: &str = concat!(
    include_str!("body.wgsl"),
    include_str!("collider.wgsl"),
    include_str!("body_math.wgsl"),
    include_str!("sphere_pair_math.wgsl"),
    include_str!("sphere_pairs_local.wgsl"),
);

struct Sweep {
    count: wgpu::BindGroup,
    solve: wgpu::BindGroup,
}

pub(super) struct ParallelPairs {
    count: Kernel,
    solve: Kernel,
    sweeps: [Sweep; 2],
    groups: [u32; 3],
}

impl ParallelPairs {
    pub(super) fn new(
        device: &wgpu::Device,
        bodies: &wgpu::Buffer,
        colliders: &wgpu::Buffer,
        count: u32,
    ) -> io::Result<Self> {
        if count == 0 {
            return Err(invalid("empty coupled body set"));
        }
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sphere pair immutable iteration output"),
            size: bodies.size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let degrees = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sphere contact graph degrees"),
            size: u64::from(count) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let count_kernel = Kernel::new(device, "sphere neighbor count", SOURCE, "count_neighbors");
        let solve_kernel = Kernel::new(
            device,
            "sphere pair impulse gather",
            SOURCE,
            "resolve_pairs",
        );
        let sweep =
            |input: &wgpu::Buffer, output: &wgpu::Buffer, restitution: u32| -> io::Result<Sweep> {
                let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sphere pair sweep parameters"),
                    contents: bytemuck::cast_slice(&[restitution, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                Ok(Sweep {
                    count: count_kernel.bind(
                        device,
                        "sphere degree input",
                        &[input, colliders, &degrees],
                    )?,
                    solve: solve_kernel.bind(
                        device,
                        "sphere impulse input/output",
                        &[input, colliders, &degrees, output, &config],
                    )?,
                })
            };
        let sweeps = [sweep(bodies, &scratch, 1)?, sweep(&scratch, bodies, 0)?];
        Ok(Self {
            count: count_kernel,
            solve: solve_kernel,
            sweeps,
            groups: [count.div_ceil(64), 1, 1],
        })
    }

    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        // Two sweeps return the authoritative state to the original buffer.
        for sweep in &self.sweeps {
            self.count
                .dispatch(encoder, "sphere contact degrees", &sweep.count, self.groups);
            self.solve.dispatch(
                encoder,
                "sphere contact impulses",
                &sweep.solve,
                self.groups,
            );
        }
    }
}

pub(super) enum SpherePairs {
    Local {
        kernel: Kernel,
        bind: wgpu::BindGroup,
    },
    Parallel(Box<ParallelPairs>),
}
impl SpherePairs {
    pub(super) fn new(
        device: &wgpu::Device,
        bodies: &wgpu::Buffer,
        colliders: &wgpu::Buffer,
        count: u32,
    ) -> io::Result<Self> {
        if (1..=64).contains(&count) {
            let kernel = Kernel::new(device, "local sphere island", LOCAL_SOURCE, "solve_local");
            let bind = kernel.bind(device, "local sphere island", &[bodies, colliders])?;
            Ok(Self::Local { kernel, bind })
        } else {
            Ok(Self::Parallel(Box::new(ParallelPairs::new(
                device, bodies, colliders, count,
            )?)))
        }
    }
    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Local { kernel, bind } => {
                kernel.dispatch(encoder, "local sphere island", bind, [1, 1, 1])
            }
            Self::Parallel(pairs) => pairs.encode(encoder),
        }
    }
}
