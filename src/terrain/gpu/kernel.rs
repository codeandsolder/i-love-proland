//! Ordered compute dispatch with sequential buffer bindings. Terrain and
//! physics use the same device/encoder scheduling and binding mechanics.
pub(super) struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}
impl Kernel {
    pub(super) fn new(device: &wgpu::Device, label: &str, source: &str, entry: &str) -> Self {
        Self::specialized(device, label, source, entry, &[])
    }

    pub(super) fn specialized(
        device: &wgpu::Device,
        label: &str,
        source: &str,
        entry: &str,
        constants: &[(&str, f64)],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        Self { pipeline, layout }
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) -> std::io::Result<()> {
        let bind = self.bind(device, label, buffers)?;
        self.dispatch(encoder, label, &bind, groups);
        Ok(())
    }
    pub(super) fn bind(
        &self,
        device: &wgpu::Device,
        label: &str,
        buffers: &[&wgpu::Buffer],
    ) -> std::io::Result<wgpu::BindGroup> {
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                Ok(wgpu::BindGroupEntry {
                    binding: u32::try_from(index).map_err(std::io::Error::other)?,
                    resource: buffer.as_entire_binding(),
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.layout,
            entries: &entries,
        }))
    }
    pub(super) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        bind: &wgpu::BindGroup,
        groups: [u32; 3],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
}
