# SONIC Inference Replay: Debug Replay System for ML Inference Failures

A debug replay system for ML inference failures in the [SONIC framework](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton) (Services for Optimized Network Inference on Coprocessors). When a model crashes inside Triton's optimized C++ backends (TensorRT, ONNXRuntime, etc.), the client (CMSSW) only receives a generic gRPC exception. This project captures failing inputs server-side as `.npz` files for offline replay and debugging.

## Components

1. **CMSSW SONIC Framework** (`CMSSW_16_1_0_pre2/src/HeterogeneousCore/`) — C++ framework for ML inference within CMS experiment software. Catches gRPC errors and manages the fallback Triton server lifecycle.
2. **Triton Debug Proxy Backend** (`sonic-inference-replay/`) — Python backend on the Triton server that intercepts requests, decodes inputs to NumPy arrays, and serializes them as `.npz` files on failure or on demand.

## Quick Start

### Prerequisites

- CMSSW `16_1_0_pre2` environment (for the C++ side)
- Apptainer (or Docker/Podman) with the `fastml/triton-torchgeo:25.08-py3-geometric` container image
- GPU recommended (CPU fallback supported)

### 1. Build CMSSW modules

```bash
cd CMSSW_16_1_0_pre2/src
cmsenv
scram b -j8
```

### 2. Run the standalone replay server

Edit `sonic-inference-replay/start_server.sh` to set local paths, then:

```bash
cd sonic-inference-replay
bash start_server.sh
```

To dump every request's inputs (not just failures):

```bash
SONIC_DUMP_INPUTS=1 bash start_server.sh
```

### 3. Run the test client

```bash
python sonic-inference-replay/client/run_client.py
```

### 4. Use dumpInputs from CMSSW config

When using the SONIC fallback server, enable input dumping from your Python config:

```python
process.TritonService.fallback.dumpInputs = cms.untracked.bool(True)
```

This passes `SONIC_DUMP_INPUTS=1` into the fallback container automatically.

## Debugging Workflow

1. Model fails during a standard CMSSW production/test run.
2. User reruns the failing event routed to the Python proxy backend (or enables `dumpInputs` in config).
3. The proxy saves inputs to e.g. `/dumps/sonic_req_error_1234567890_1.npz`.
4. User loads the `.npz` locally and feeds those arrays into their model interactively:

```python
import numpy as np
inputs = np.load('sonic_req_error_1234567890_1.npz')
for name in inputs.files:
    print(f"{name}: shape={inputs[name].shape}, dtype={inputs[name].dtype}")
```

## Testing the Implementation

### Test 1: Verify `cmsTriton -e` flag (dry run, no server needed)

The `-D` flag prints container commands without executing them. Verify that `-e` env vars appear in the output:

```bash
# Apptainer (default) — should show --env SONIC_DUMP_INPUTS=1
cd CMSSW_16_1_0_pre2/src
cmsenv
cmsTriton -D -e SONIC_DUMP_INPUTS=1 -m HeterogeneousCore/SonicTriton/data/models/resnet50_netdef start

# Docker — should show -e SONIC_DUMP_INPUTS=1
cmsTriton -D -d docker -e SONIC_DUMP_INPUTS=1 -m HeterogeneousCore/SonicTriton/data/models/resnet50_netdef start

# Podman — should show -e SONIC_DUMP_INPUTS=1
cmsTriton -D -d podman -e SONIC_DUMP_INPUTS=1 -m HeterogeneousCore/SonicTriton/data/models/resnet50_netdef start

# Multiple env vars
cmsTriton -D -e SONIC_DUMP_INPUTS=1 -e MY_VAR=hello -m HeterogeneousCore/SonicTriton/data/models/resnet50_netdef start
```

### Test 2: Verify C++ compiles

```bash
cd CMSSW_16_1_0_pre2/src
cmsenv
scram b -j8
```

### Test 3: Verify the new parameter is accepted

After building, confirm that the `dumpInputs` parameter is recognized without errors:

```bash
python3 -c "
import FWCore.ParameterSet.Config as cms
process = cms.Process('TEST')
process.TritonService = cms.Service('TritonService',
    verbose = cms.untracked.bool(False),
    fallback = cms.PSet(
        enable = cms.untracked.bool(True),
        debug = cms.untracked.bool(False),
        verbose = cms.untracked.bool(False),
        container = cms.untracked.string('apptainer'),
        device = cms.untracked.string('cpu'),
        retries = cms.untracked.int32(-1),
        wait = cms.untracked.int32(-1),
        instanceBaseName = cms.untracked.string('triton_server_instance'),
        instanceName = cms.untracked.string(''),
        tempDir = cms.untracked.string(''),
        imageName = cms.untracked.string(''),
        sandboxDir = cms.untracked.string(''),
        dumpInputs = cms.untracked.bool(True),
    ),
)
print('dumpInputs =', process.TritonService.fallback.dumpInputs.value())
print('Config accepted OK')
"
```

### Test 4: End-to-end with existing SONIC tests

If you have test models available (run `fetch_model.sh` first in the test directory):

```bash
cd CMSSW_16_1_0_pre2/src/HeterogeneousCore/SonicTriton/test

# Fetch test model data
./fetch_model.sh

# Run with dumpInputs enabled (add to the test config or use command-line override)
cmsRun tritonTest_cfg.py --maxEvents 1 --modules TritonImageProducer --models inception_graphdef
```

### Test 5: Standalone server with manual dump

```bash
cd sonic-inference-replay

# Start server with dumping enabled
SONIC_DUMP_INPUTS=1 bash start_server.sh

# In another terminal, send test requests
python client/run_client.py

# Check dump output
ls /path/to/your/dump/folder/
```

## Architecture

```
User Python config                    Triton Container
       |                                     |
       v                                     v
  TritonService.cc                      model.py
  reads dumpInputs ──> cmsTriton -e ──> SONIC_DUMP_INPUTS=1
  from fallback PSet    SONIC_DUMP_INPUTS=1   reads env var,
                                              dumps .npz on
                                              every request
```

## Project Structure

```
ml_hackathon_sonic_replay/
├── CLAUDE.md                           # AI assistant instructions
├── README.md                           # This file
├── CMSSW_16_1_0_pre2/src/HeterogeneousCore/
│   ├── SonicCore/                      # Base SONIC infrastructure
│   └── SonicTriton/
│       ├── interface/TritonService.h   # FallbackOpts with dumpInputs
│       ├── src/TritonService.cc        # Wires dumpInputs → cmsTriton -e
│       ├── scripts/cmsTriton           # Container launcher with -e flag
│       └── test/                       # Test configs and producers
└── sonic-inference-replay/
    ├── start_server.sh                 # Standalone server launcher
    ├── client/run_client.py            # gRPC test client
    └── single_sonic_model/models/
        └── particlenet_AK4_PT/
            ├── config.pbtxt            # Triton model config
            └── 1/model.py              # Python proxy backend
```

## Key Files Modified for dumpInputs

| File | Change |
|------|--------|
| `SonicTriton/scripts/cmsTriton` | Added `-e KEY=VALUE` flag to pass env vars into containers |
| `SonicTriton/interface/TritonService.h` | Added `dumpInputs` bool to `FallbackOpts` |
| `SonicTriton/src/TritonService.cc` | Registered `dumpInputs` param; appends `-e SONIC_DUMP_INPUTS=1` to command |
| `sonic-inference-replay/.../model.py` | Reads `SONIC_DUMP_INPUTS` env var (pre-existing) |
