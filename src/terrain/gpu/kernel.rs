//! Ordered compute dispatch with sequential buffer bindings. Terrain and
//! physics use the same device/encoder scheduling and binding mechanics.
pub(super) struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}
impl Kernel {
    pub(super) fn new(device: &wgpu::Device, label: &str, source: &str, entry: &str) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
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
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }
}
