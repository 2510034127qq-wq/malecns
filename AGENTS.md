# MaleCNS Project Instructions

## Mission

Build one full-scale embodied MaleCNS v1.0 simulation. The target is the complete published connectome data path: all available native MaleCNS neuron skeletons, all published synaptic partner rows, synapse coordinates, neurotransmitter predictions, morphology-aware multi-compartment simulation in Arbor, and a closed sensory–CNS–motor loop with a MuJoCo fly body and interactive visualization.

Do not replace the target with a reduced connectome, toy subgraph, point-neuron-only final model, or scripted-behavior demo. Small synthetic/sample fixtures are allowed only for unit tests and debugging; they are never the primary implementation or acceptance target.

## Non-negotiable scientific scope

- Dataset: MaleCNS v1.0, native male CNS EM coordinate space.
- Use the official `minconf-0.5` bulk tables as published; do not add additional connectivity thresholds unless explicitly requested.
- Preserve all raw source files unchanged.
- Use all available native SWC skeletons from:
  `gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/`
- Use the published bulk tables in `data/raw/tables/`:
  - body annotations
  - body neurotransmitters
  - body stats
  - connectome weights
  - syn-points
  - syn-partners
  - T-bar neurotransmitter probabilities
- `connectome-weights` is useful for graph-level checks/analysis, but the detailed simulation must preserve explicit synaptic partner information rather than collapsing the final model to one weighted edge per neuron pair.
- Do not download or require the TB-scale raw EM image/segmentation volumes for the simulation. They are source microscopy/segmentation data, not required runtime state for the reconstructed connectome model.

## Coordinate and unit rule

This is a hard correctness requirement.

- Native MaleCNS SWC coordinates are in 8 nm units.
- `syn-points` x/y/z are also voxel coordinates in 8 nm units.
- Arbor morphology geometry is in micrometers.
- Therefore the ingestion conversion is:

  `1 MaleCNS coordinate unit = 8 nm = 0.008 µm`

- Apply the same physical transform to morphology x/y/z/radius and synapse x/y/z before spatial mapping.
- Never pass native 8 nm SWC coordinates directly to Arbor as if they were micrometers.
- Raw files stay unchanged; conversion occurs at the ingestion boundary or in reproducible derived cache data.
- Add tests/assertions that morphology and synapse coordinates share the same coordinate space and units before mapping.

## Current machine/environment

Primary platform: Ubuntu 22.04, not Windows.

Conda environment: `malecns`

Verified core runtime:
- Python 3.11.16
- Arbor 0.12.2, built locally from `vendor/arbor`
- CUDA 13.0 / nvcc 13.0.48
- GPU target: NVIDIA RTX 5060, `CMAKE_CUDA_ARCHITECTURES=120`
- GCC/G++ 12 for Arbor/CUDA compilation
- MPI enabled in the Arbor build
- MuJoCo 3.13.0
- Trimesh 5.1.0
- PyZMQ 27.2.0
- pytest / pytest-xdist
- ruff
- 64 GB system RAM, 8 GB VRAM

Final visualization target:
- Rerun Viewer is the unified end-user visualization/workbench.
- NAVIS/Octarine remains available for specialist morphology inspection/debugging, not as the required final main UI.
- If `rerun-sdk` is not yet installed in `malecns`, installing that single package with `python -m pip install rerun-sdk` is allowed. Do not use this as a reason to recreate or reinstall the environment.

The Arbor v0.12.2 source tree currently contains a CMake 4 compatibility patch in the top-level `CMakeLists.txt`:

`cmake_policy(SET CMP0146 OLD)`

Do not casually restore/revert that patch or reinstall Arbor. It was required because this Arbor release still calls legacy `find_package(CUDA)` under CMake 4.x. If rebuilding Arbor, preserve an equivalent working fix and keep CUDA architecture 120.

The `malecns` Conda environment has project-specific environment variables including:
- `PYTHONNOUSERSITE=1`
- `CC=/usr/bin/gcc-12`
- `CXX=/usr/bin/g++-12`
- `CUDACXX=/usr/local/cuda-13.0/bin/nvcc`
- `CMAKE_CUDA_ARCHITECTURES=120`

Important: the user's bare `pip` command may resolve to a Python 3.10 user installation. Inside this project, always use:

`python -m pip ...`

Never use bare `pip ...` for project dependency changes.

## Repository/data layout

Repository root: `~/malecns`

Expected/target layout; create missing implementation directories as needed:

```text
malecns/
├── AGENTS.md
├── data/
│   ├── raw/
│   │   ├── tables/
│   │   └── skeletons-swc/
│   └── cache/
├── src/
├── scripts/
├── configs/
├── tests/
├── vendor/
│   └── arbor/
└── logs/
```

`data/raw/`, `data/cache/`, logs, Python caches, and build products are intentionally excluded from Git. Never add MaleCNS bulk data to Git.

## Storage policy

The user wants the full dataset but does not want wasteful duplicate copies.

- Prefer Arrow/Feather memory mapping, streaming, and chunked processing for multi-GB tables.
- Do not convert every large source table into another full-size format merely for convenience.
- Do not create multiple full copies of `syn-points` or `syn-partners`.
- Derived caches must be reproducible from raw data.
- Before creating any new derived artifact larger than 2 GB, report why it is necessary and its expected size.
- Keep the normal project footprint roughly in the tens-of-GB range, not 100+ GB, unless a measured requirement proves otherwise.

## Implementation architecture

Use the existing libraries rather than reimplementing solved infrastructure:

- PyArrow/Polars: bulk Feather ingestion and chunked table processing.
- NAVIS: morphology inspection/manipulation where useful.
- Octarine: specialist morphology inspection/debugging.
- Arbor: morphology-aware cable-cell simulation, CV discretization, mechanisms, event delivery, GPU/MPI execution.
- MuJoCo: fly rigid-body/joint/contact physics and experimental environment.
- Trimesh: mesh processing when needed.
- Rerun SDK + Rerun Viewer: final unified visualization, timeline, recording, replay, 3D world/CNS views, eye images, and telemetry.
- ZeroMQ/msgpack: simulation/viewer/physics process communication only if separate processes are actually useful.

Prefer the simplest working process topology. If Arbor, MuJoCo, and Rerun can exchange state directly in one process without compromising throughput or isolation, do that. Add ZeroMQ/shared-memory bridges only when measured needs justify them.

Do not build a custom full PySide/Web GUI when Rerun already provides the required visualization/workbench capability. Add only minimal custom controls/bridges for live-simulation functions that Rerun does not natively control.

Do not write a custom cable-equation solver or custom general-purpose physics engine unless a concrete blocker in the chosen libraries is demonstrated.

## Hardware and performance policy

Full-data semantics do not require the entire simulation to reside on the GPU.

Available hardware includes:
- 64 GB system RAM
- RTX 5060 with 8 GB VRAM
- multicore CPU
- Arbor CUDA / multicore / MPI capabilities

Do not prune neurons, morphology, synapses, or sensory modalities merely to fit 8 GB VRAM.

Measure first:
- model-construction peak RAM
- runtime RAM
- VRAM
- preprocessing/build time
- biological simulation time / wall-clock time
- throughput and bottlenecks

Prefer GPU acceleration where it is actually beneficial. If VRAM is insufficient, use semantics-preserving execution strategies such as Arbor multicore CPU execution, appropriate context/domain decomposition, lazy recipes, compact representations, chunked/cell-centric model construction, and—if Python model-construction overhead becomes the measured blocker—a C++ model-construction path.

GPU OOM alone is not a valid reason to reduce the MaleCNS model.

Only report hardware as a blocker after measuring that the complete model cannot be instantiated or advanced on the available 64 GB RAM / 8 GB VRAM / CPU using reasonable semantics-preserving engineering approaches.

Visualization resource use is separate from neural-model semantics. Rerun must use LOD, static-data reuse, and sensible dynamic logging rates so the viewer does not force the neural model to shrink.

## Morphology and synapse mapping

- Keep the original morphology topology and all available neuron skeletons.
- Construct Arbor cable morphologies only after correct physical unit conversion.
- Explicitly choose and document an Arbor CV discretization policy. Do not silently accept a coarse default merely because it runs.
- Preserve morphology at the source-data level even if numerical CV discretization groups continuous cable into control volumes.
- Map each explicit synaptic partner endpoint to a cable location (`branch`, normalized position) on the correct neuron.
- Prefer nearest-segment/projection geometry over a naive nearest-node mapping when assigning synapses to morphology.
- Record mapping residual distances and summary statistics. Flag pathological/unmapped cases instead of silently dropping them.
- Do not silently discard neurons, synapses, or branches because they are inconvenient. Any unavoidable exclusion must be counted, logged, justified, and surfaced to the user.

## Neurotransmitters and physiology

The connectome is anatomically detailed, but MaleCNS does not provide every electrophysiological parameter needed for a fully measured biophysical replica.

Therefore:
- Never invent undocumented parameters and present them as measured truth.
- Put non-connectomic physiological parameters in explicit config files.
- For each important membrane, channel, synaptic, delay, sensory, muscle, or aerodynamic parameter, record provenance: literature value, derived estimate, calibration value, or explicit modeling assumption.
- Keep uncertainty/assumptions visible in reports.
- Separate anatomical ground truth from modeled physiology in code and documentation.

## Embodiment rule

Behavior must emerge through the closed loop:

`environment -> sensory encoding -> MaleCNS -> motor output -> MuJoCo body -> environment`

Do not implement high-level behavior shortcuts such as:

`if wall_near: turn_left()`

A wall may affect vision/contact sensors; those signals may drive CNS activity; motor output then changes the body. The same principle applies to phototaxis, odor navigation, obstacle avoidance, and escape behavior.

## Environment and sensory systems

The final sandbox should support controlled experiments before decorative complexity:
- visual stimuli / compound-eye sampling
- contact/mechanosensory input
- proprioceptive feedback from joints/contact
- odor concentration fields and antenna sampling when olfaction is enabled
- light sources, obstacles, floor/walls, food/odor sources

MuJoCo handles rigid-body dynamics and contacts. Wing aerodynamics may require custom physically motivated forces; do not assume MuJoCo automatically supplies realistic insect aerodynamics.

## Visualization

Final visualization uses Rerun Viewer as the unified user-facing workbench. Octarine is a specialist morphology debugging/inspection tool and must not be required as a second main window for normal use.

Visualization is not allowed to change or simplify the simulation state.

Use a Rerun Blueprint to organize synchronized views on one simulation timeline. The final workbench should support at least:

- MuJoCo 3D world: fly body, transforms, environment geometry, obstacles/stimuli, and motion.
- MaleCNS 3D: real morphology with activity overlays and useful filtering/selection.
- Left/right eye views: rendered eye input and, where useful, compound-eye sampled representation.
- Time-series/telemetry: sensory activity, DN/motor signals, wing/leg drive, body state, biological time, wall-clock time, and realtime ratio.
- Neuron/synapse inspection: body ID, type, ROI/neuropil, transmitter information, morphology, current simulated state, partners, mapped synapses, and selected synapse position.

Use rendering/logging LOD only:
- whole-CNS view: morphology/activity summaries, not every synapse rendered every frame
- regional view: active cells/connections
- single-neuron view: full morphology, mapped synapses, membrane state
- synapse view: pre/post neuron IDs, location, neuropil, transmitter probabilities, current simulated state
- embodied view: MuJoCo environment + fly + sensory inputs + CNS activity + motor telemetry synchronized in time

Do not resend static morphology or meshes every frame. Log static geometry once or on demand, and log dynamic state at a measured, reasonable cadence. Do not stream all explicit synapses every frame.

Rerun's native timeline/replay/selection capabilities should be used directly where possible. If live simulation pause/resume/speed cannot be controlled directly through Rerun, implement the smallest sufficient control bridge instead of building a separate full GUI framework.

The full data remain in the simulation even when the viewer displays only a level-of-detail subset.

## Full-data acceptance criteria

A task is not complete merely because a small sample works. The project-level acceptance target requires all of the following on the complete locally downloaded MaleCNS v1.0 data:

1. Environment validation passes and Arbor can create a CUDA context on GPU 0.
2. Every raw bulk file is readable and recorded in a dataset manifest with size/schema.
3. All available native SWC skeletons are indexed and validated.
4. Unit conversion is tested and correct.
5. Explicit synaptic partner rows are ingested without an extra ad-hoc connectivity threshold.
6. Synapses are mapped to the correct neuron morphology with mapping-quality statistics and explicit accounting for any failures.
7. An Arbor recipe representing the complete modeled network can be instantiated.
8. The full-network simulation advances simulation time; report wall-clock speed, RAM, VRAM, cell/CV/synapse counts, and any bottleneck.
9. No silent execution-mode change, data dropping, or reduced-network execution is allowed. Explicit multicore CPU or mixed hardware execution is allowed and expected when justified by measured VRAM/RAM/performance constraints; record the chosen Arbor context/domain decomposition and why it was selected.
10. The embodied loop runs through sensory input -> CNS -> motor output -> MuJoCo physics -> sensory feedback.
11. Rerun Viewer output is synchronized with simulation time and can inspect whole-CNS, neuron, synapse, embodied-world, eye-input, and telemetry state without duplicating the raw dataset unnecessarily.

## Testing rules

- Unit tests may use tiny synthetic fixtures for speed.
- Integration tests should use deterministic slices only to diagnose code paths.
- Those tests do not replace full-data acceptance runs.
- Always run relevant tests after changes.
- At minimum before claiming a task complete:

```bash
python -m pip check
pytest -q
ruff check .
git status --short
```

Add project-specific validation commands as they are created.

## Git and change discipline

- Do not commit raw datasets, generated caches, logs, binaries, or large build artifacts.
- Do not rewrite or delete user data.
- Do not modify global/system Python packages for project needs.
- Do not use `sudo pip`.
- Avoid adding Docker/containers unless there is a demonstrated need.
- Keep changes minimal and directly related to the task.
- Before changing the environment/toolchain, inspect the current working configuration and explain why the change is necessary.

## Agent working style

- Work autonomously on ordinary engineering tasks; do not stop for trivial confirmations.
- When a scientific choice is underdetermined by the dataset, surface the evidence, assumptions, and candidate choice instead of fabricating certainty.
- Prefer direct measurement/benchmarking over speculation for performance questions.
- Preserve the full target architecture from the start. Optimize implementation, memory layout, streaming, batching, CV policy, domain decomposition, and GPU/CPU allocation before proposing removal of biological data.
- Keep concise progress notes with concrete counts, timings, memory usage, failures, and fixes.
