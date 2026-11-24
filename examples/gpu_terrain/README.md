# GPU-authoritative terrain and local rigid-body experiment

Build and run on the actual graphics adapter:

```sh
cargo run --release --features gpu --example gpu_terrain -- /path/to/mountains.plterrain
cargo run --release --features gpu --example gpu_terrain -- --physics
```

The first probe generates integer-addressed chart grids in GPU storage and
checks equal samples across adjacent and nested LOD grids. An optional typed
readback can capture a slot even if the slot is subsequently overwritten.
Only compact source controls and vector drainage reaches are uploaded;
the shader synthesizes local riverbeds and detail.

The second probe consumes the same resident height grid in compute. Each
invocation owns an independent compound sphere rigid body, evaluates
sphere/triangle contacts including edge/vertex cases and applies angular,
normal and friction impulses. Positions are local chart coordinates, not
planet-scale floating point positions. This is a local chart-space
prototype; it does not yet use the final spherical projected vertex buffer,
vessel-vessel collision, joint constraints, swept CCD or moving patch
migration. A speed-per-step limit and explicit missing-coverage status
stop unsupported motion, rather than taking an arbitrarily large step.
A stopped body retains its last valid state and tick.

SimulationClock tracks target physical time, scheduled ticks and
confirmed ticks independently of display cadence. Requested time uses
integer nanosecond × millionth-warp credit. Integration frequency is
constant, GPU submissions batch steps independently of display, and
uncomputed time remains as backlog. The rendering path is intended to
consume GPU state directly; optional host readbacks are for verification,
telemetry, gameplay events and saves, not a per-step solver dependency.
Completed tick means the simulation state really reached that tick, not
just that the command buffer completed. A solver stop must be resolved
before that timeline can advance.

For larger time warp, avoid forcing distant or non-interacting vessels
through all near-ground fixed substeps: a separate analytic orbital/rails
propagator can evaluate requested timestamps and hand off at physical
events. Do not silently increase collision dt with the requested warp.
