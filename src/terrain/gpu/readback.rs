//! Typed asynchronous GPU-to-host snapshots. A recorded copy captures the
//! source at that point in queue order, not at the later time of mapping.
use std::{io, marker::PhantomData, sync::mpsc};

pub struct RecordedReadback<T> {
    staging: wgpu::Buffer,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> RecordedReadback<T> {
    pub(super) fn copy(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> Self {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU state snapshot"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, offset, &staging, 0, size);
        Self {
            staging,
            marker: PhantomData,
        }
    }

    /// Submit the recorded encoder before beginning the asynchronous map.
    #[must_use = "Poll this snapshot when its data is needed"]
    pub fn begin(self) -> PendingReadback<T> {
        let (sender, receiver) = mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        PendingReadback {
            staging: self.staging,
            receiver,
            marker: PhantomData,
        }
    }
}

pub struct PendingReadback<T> {
    staging: wgpu::Buffer,
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> PendingReadback<T> {
    /// Nonblocking completion probe. Poll the device externally.
    ///
    /// # Errors
    /// Reports failed mapping, invalid data or repeated consumption.
    pub fn try_take(&mut self) -> io::Result<Option<Vec<T>>> {
        match self.receiver.try_recv() {
            Ok(result) => result.map_err(io::Error::other)?,
            Err(mpsc::TryRecvError::Empty) => return Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(io::Error::other(
                    "GPU readback disconnected or already consumed",
                ));
            }
        }
        let samples = {
            let mapped = self
                .staging
                .slice(..)
                .get_mapped_range()
                .map_err(io::Error::other)?;
            bytemuck::try_cast_slice::<u8, T>(&mapped)
                .map_err(|error| io::Error::other(error.to_string()))?
                .to_vec()
        };
        self.staging.unmap();
        Ok(Some(samples))
    }
}
