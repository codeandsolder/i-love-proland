//! Sparse, validated replacement of dense resident candidate slots.
use super::{GpuCompactor, PackedInstance, WORKGROUP_SIZE};
use crate::{Error, Result};

/// One full replacement record and its unique target slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CandidatePatch {
    slot: [u32; 4],
    candidate: PackedInstance,
}
impl CandidatePatch {
    #[must_use = "Submit the returned patch to update the intended candidate slot"]
    pub const fn new(slot: u32, candidate: PackedInstance) -> Self {
        Self {
            slot: [slot, 0, 0, 0],
            candidate,
        }
    }
    #[must_use = "Use the destination slot to address the candidate buffer"]
    pub const fn slot(&self) -> u32 {
        self.slot[0]
    }
    #[must_use = "Use the replacement candidate data when applying the patch"]
    pub const fn candidate(&self) -> &PackedInstance {
        &self.candidate
    }
}

#[derive(Clone)]
pub(super) struct PatchPipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}
impl PatchPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("i-love-proland sparse candidate updates"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../gpu_instance.wgsl"),
                    include_str!("../patch_candidates.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("i-love-proland sparse candidate updates"),
            layout: None,
            module: &shader,
            entry_point: Some("patch_candidates"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        Self { pipeline, layout }
    }
}

pub(super) struct PatchStorage {
    data: wgpu::Buffer,
    count: wgpu::Buffer,
    bind: wgpu::BindGroup,
    capacity: u32,
}
impl PatchStorage {
    fn new(
        device: &wgpu::Device,
        pipeline: &PatchPipeline,
        input: &wgpu::Buffer,
        required: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let stride = core::mem::size_of::<CandidatePatch>() as u64;
        let max = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / stride;
        let maximum = u32::try_from(max).unwrap_or(u32::MAX);
        if required > maximum
            || required.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::InvalidConfiguration(
                "GPU candidate patches exceed device limits",
            ));
        }
        let capacity = required
            .checked_next_power_of_two()
            .unwrap_or(required)
            .min(maximum);
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("i-love-proland sparse patch records"),
            size: u64::from(capacity) * stride,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("i-love-proland sparse patch count"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("i-love-proland sparse candidate bindings"),
            layout: &pipeline.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            data,
            count,
            bind,
            capacity,
        })
    }

    fn encode(
        &self,
        pipeline: &PatchPipeline,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        patches: &[CandidatePatch],
        count: u32,
    ) {
        queue.write_buffer(&self.data, 0, bytemuck::cast_slice(patches));
        queue.write_buffer(&self.count, 0, bytemuck::bytes_of(&count));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("i-love-proland sparse candidate writes"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}

fn validate_patches(
    old_count: u32,
    count: u32,
    capacity: u32,
    patches: &[CandidatePatch],
) -> Result<u32> {
    if count > capacity
        || patches.last().is_some_and(|p| p.slot() >= count)
        || !patches
            .windows(2)
            .all(|pair| pair[0].slot() < pair[1].slot())
    {
        return Err(Error::InvalidConfiguration(
            "GPU patch slots must be sorted, unique and within the new dense count",
        ));
    }
    let n = u32::try_from(patches.len())
        .map_err(|_| Error::InvalidConfiguration("too many GPU candidate patches"))?;
    let appended = patches.len() - patches.partition_point(|p| p.slot() < old_count);
    if count > old_count && u32::try_from(appended).ok() != Some(count - old_count) {
        return Err(Error::InvalidConfiguration(
            "GPU patch must initialize every newly exposed candidate slot",
        ));
    }
    Ok(n)
}

impl GpuCompactor {
    /// Apply sorted, unique replacements within the current capacity.
    ///
    /// Newly exposed slots must all be initialized. Shrinking may omit
    /// patches, clearing the visible dense count without touching old bytes.
    /// Submit this encoder before another candidate update; the patch pass
    /// must precede classification and draw in the same submitted frame.
    ///
    /// # Errors
    /// Rejects out-of-range or duplicate/unsorted slots, uninitialized
    /// growth of the dense count, and patches beyond device limits.
    pub fn patch_candidates(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
        patches: &[CandidatePatch],
    ) -> Result<()> {
        let n = validate_patches(self.count, count, self.capacity, patches)?;
        if n != 0 {
            if self.patch_storage.as_ref().is_none_or(|p| p.capacity < n) {
                self.patch_storage = Some(PatchStorage::new(
                    self.pipeline.device(),
                    &self.pipeline.patches,
                    &self.input,
                    n,
                )?);
            }
            self.patch_storage
                .as_ref()
                .ok_or(Error::InvalidConfiguration("missing GPU patch storage"))?
                .encode(&self.pipeline.patches, queue, encoder, patches, n);
        }
        self.count = count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn validates_sparse_dense_count_and_shader_layout() {
        assert_eq!(core::mem::size_of::<CandidatePatch>(), 144);
        assert_eq!(core::mem::offset_of!(CandidatePatch, candidate), 16);
        let patch = |slot| CandidatePatch::new(slot, PackedInstance::zeroed());
        assert_eq!(
            validate_patches(4, 6, 8, &[patch(1), patch(4), patch(5)]).unwrap(),
            3
        );
        assert!(validate_patches(4, 6, 8, &[patch(1), patch(5)]).is_err());
        assert!(validate_patches(4, 6, 8, &[patch(4), patch(4), patch(5)]).is_err());
        assert!(validate_patches(4, 6, 8, &[patch(5), patch(4)]).is_err());
        assert!(validate_patches(4, 6, 8, &[patch(4), patch(5), patch(6)]).is_err());
        assert!(validate_patches(4, 9, 8, &[]).is_err());
        assert!(validate_patches(4, 0, 8, &[patch(0)]).is_err());
        assert_eq!(validate_patches(4, 0, 8, &[]).unwrap(), 0);
        assert_eq!(validate_patches(4, 2, 8, &[]).unwrap(), 0);
    }
}
