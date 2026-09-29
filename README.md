# Quadmesh

## Technical Architecture and Implementation

Quadmesh is a from-scratch CAD generation and computer-use system built around a **planner → execution → verification → repair** architecture.

The system combines:

* Multimodal conversational understanding
* CAD planning
* Parametric geometry generation
* Mesh generation
* STEP / B-Rep generation
* Blender computer control
* Vision-Language-Action (VLA)
* Geometry verification
* Printability verification
* ASR and TTS
* Model training and evaluation pipelines

The models are trained from random initialization rather than relying on a pretrained foundation model.

---

# 1. System Architecture

```text
                     ┌─────────────────────────┐
                     │       User Input        │
                     │                         │
                     │ Text / Voice / Image    │
                     │ STL / STEP / 3D File    │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │    Quadmesh Chat Agent  │
                     │                         │
                     │ Multimodal Understanding│
                     │ Tool Selection          │
                     │ Tool Calling             │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │         Planner         │
                     │                         │
                     │ Natural Language        │
                     │        ↓                │
                     │ Verified CAD Operations │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      Plan Checker       │
                     │                         │
                     │ Operation Validation    │
                     │ Constraint Checking     │
                     │ Simulation              │
                     └────────────┬────────────┘
                                  │
                         ┌────────┴────────┐
                         │                 │
                       VALID            INVALID
                         │                 │
                         ▼                 ▼
                ┌────────────────┐   ┌──────────────┐
                │ Geometry Engine│   │    Repair    │
                │                │   │    Loop      │
                │ CSG            │   └──────┬───────┘
                │ Curves         │          │
                │ Loft/Sweep     │          │
                │ Gears/Threads  │          │
                │ Assemblies     │          │
                └───────┬────────┘          │
                        │                   │
                        └────────◄──────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │     CAD / Mesh Layer    │
                     │                         │
                     │ NumPy Geometry          │
                     │ Blender                 │
                     │ CadQuery                │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Verification Stack      │
                     │                         │
                     │ Geometry Checks         │
                     │ Printability Checks     │
                     │ STL Validation           │
                     │ STEP Validation          │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │       Final Output      │
                     │                         │
                     │ STL / STEP / B-Rep      │
                     └─────────────────────────┘
```

---

# 2. Core Architecture

The core Quadmesh pipeline is:

```text
Input
  ↓
Multimodal Understanding
  ↓
Planning
  ↓
Operation Generation
  ↓
Plan Verification
  ↓
Geometry Execution
  ↓
Geometry Verification
  ↓
Printability Verification
  ↓
Export
```

Failed operations can enter the repair loop:

```text
Plan
  ↓
Check
  ↓
FAIL
  ↓
Repair
  ↓
Re-plan
  ↓
Check
  ↓
PASS
```

This separates **generation** from **verification** rather than assuming that a generated CAD operation is automatically correct.

---

# 3. Model Architecture

## 3.1 Quadmesh Transformer

The primary transformer implementation is located in:

```text
quadmesh/model/transformer.py
```

The model contains:

* RMSNorm
* Rotary Positional Embeddings (RoPE)
* SwiGLU
* Transformer blocks
* KV cache
* Vision model components

The model configuration supports multiple parameter tiers through:

```text
quadmesh/config.py
```

The role-specific model configuration is handled through:

```text
quadmesh/role_shapes.py
```

The architecture does not require every role to use the same model size.

---

# 4. Multimodal Chat Model

Implementation:

```text
quadmesh/model/chat.py
```

The chat model is designed to process:

```text
Text
Image
3D File
Voice → ASR → Text
```

The conversational model performs:

1. Input understanding
2. Context processing
3. Tool selection
4. Tool-call generation
5. CAD operation interaction
6. Verification interaction
7. Response generation

The chat training pipeline is implemented in:

```text
quadmesh/train/chat_train.py
```

Training data and tool-call parsing are handled through:

```text
quadmesh/chat_data.py
```

Verified factual responses are stored through:

```text
quadmesh/chat_kb.py
```

---

# 5. CAD Planning

The planner converts engineering instructions into structured CAD operations.

Implementation:

```text
quadmesh/agent/planner.py
quadmesh/model/transformer.py
```

The planner can generate operations corresponding to:

* Primitive creation
* Boolean operations
* Transformations
* Dimensions
* Holes
* Curves
* Lofting
* Sweeping
* Threads
* Gears
* Text
* Mechanisms
* Assemblies

The generated plan is not directly treated as final geometry.

It is passed through the plan checker.

---

# 6. Plan Verification

Implementation:

```text
quadmesh/pipeline/plan_checker.py
```

The plan checker provides:

* Plan simulation
* Operation validation
* Constraint checking
* Invalid-operation detection

The implementation currently covers:

```text
52 operations
```

The checker acts as the primary verifier between planning and geometry execution.

```text
Planner
   ↓
CAD Operation Plan
   ↓
Plan Simulator
   ↓
Checker
   ↓
Validated Plan
```

---

# 7. Geometry Engine

Core geometry implementation:

```text
quadmesh/geometry3d.py
```

The geometry engine uses NumPy-based implementations for geometry generation.

Supported geometry categories include:

### Primitive and CSG Geometry

* Boxes
* Cylinders
* Spheres
* Boolean operations
* Explicit-coordinate CSG

### Curve-Based Geometry

* Curves
* Lathe
* Loft
* Sweep

### Mechanical Geometry

* Gears
* Threads
* Bolts
* Nuts
* Four-bar mechanisms

### Structural Geometry

* Pipes
* Springs
* Vases
* Hexacopter frames
* Robot arms
* Stands

### Text Geometry

3D text generation is supported through the geometry pipeline.

---

# 8. Mesh Processing

Implementation:

```text
quadmesh/mesh_tools.py
```

`mesh_tools.py` acts as the validated entry point for generated geometry.

The mesh layer connects:

```text
Geometry Generator
        ↓
Mesh Validation
        ↓
Printability Gate
        ↓
STL Output
```

---

# 9. STEP / B-Rep Pipeline

Implementation:

```text
quadmesh/step_io.py
```

CadQuery is used for STEP / B-Rep operations.

The pipeline supports:

```text
CAD Plan
   ↓
CadQuery
   ↓
STEP / B-Rep
```

and:

```text
STEP
   ↓
Geometry Processing
   ↓
Mesh
```

STEP functionality is optional and requires CadQuery.

---

# 10. Blender Integration

Blender integration consists of two primary components:

```text
quadmesh/agent/blender_bridge.py
quadmesh/agent/blender_client.py
```

## Blender Bridge

The bridge runs inside Blender.

It provides:

* Blender operation execution
* Socket server
* Approximately 160 supported operations

## Blender Client

The client runs externally and communicates with the Blender bridge.

```text
Quadmesh Agent
      │
      ▼
Blender Client
      │
      │ Socket
      ▼
Blender Bridge
      │
      ▼
Blender
```

---

# 11. Computer-Use Layer

Quadmesh can operate Blender through actual keyboard and mouse interaction.

Relevant modules:

```text
quadmesh/agent/desk.py
quadmesh/agent/desktop.py
quadmesh/agent/gui_recipes.py
quadmesh/agent/visible_ops.py
```

The computer-use layer supports:

* Keyboard input
* Keyboard shortcuts
* Mouse movement
* Mouse dragging
* Rotation
* Orbiting
* Sculpting
* GUI interaction

The full-control mode is enabled with:

```text
--full-control
```

Blender is automatically launched through:

```text
quadmesh/agent/launch.py
```

---

# 12. Vision-Language-Action System

Implementation:

```text
quadmesh/model/vla.py
quadmesh/train/vla_train.py
quadmesh/agent/vla_agent.py
```

The VLA receives:

```text
Screenshot + Text Instruction
```

and produces:

```text
Desktop Action
```

Conceptually:

```text
┌──────────────┐
│ Screenshot   │
└──────┬───────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
 Vision Features   Text Features
       │              │
       └──────┬───────┘
              ▼
          VLA Model
              │
              ▼
        Action Prediction
              │
              ▼
      Keyboard / Mouse
              │
              ▼
           Blender
```

VLA training data is handled through:

```text
quadmesh/data/vla_data.py
```

Real demonstrations can be recorded through:

```text
quadmesh/agent/record_demo.py
```

---

# 13. Vision Processing

Vision-related processing is implemented through:

```text
quadmesh/data/vision.py
```

The vision pipeline processes screenshots and converts visual information into representations usable by the vision roles and VLA system.

---

# 14. Image-to-3D Pipeline

Implementation:

```text
quadmesh/agent/image3d.py
quadmesh/photo3d.py
```

The current image-to-3D system follows two primary approaches.

### Symmetric Objects

For symmetric upright objects:

```text
Image
 ↓
Silhouette
 ↓
Profile
 ↓
Solid of Revolution
```

### General Objects

For other objects:

```text
Image
 ↓
Outline
 ↓
Silhouette Representation
 ↓
Inflated Geometry
```

The depth and hidden geometry are inferred rather than directly observed.

---

# 15. Mechanism Generation

Implementation:

```text
quadmesh/mechanisms.py
```

Supported mechanism functionality includes:

* Gear pairs
* Four-bar linkages
* Motion analysis
* Parameter-based mechanism sizing

---

# 16. Assembly Generation

Implementation:

```text
quadmesh/assemblies.py
```

The assembly system provides parameterized assemblies including:

* Hexacopter
* Robot arm
* Robot-arm stand

Assemblies are generated from structured parameters rather than manually constructed meshes.

---

# 17. Verification Architecture

The verification framework is located at:

```text
quadmesh/verification/
```

Primary implementation:

```text
quadmesh/verification/stack.py
```

The verification architecture is designed to operate after geometry generation.

Current verification includes:

```text
Plan Verification
       ↓
Geometry Verification
       ↓
Printability Verification
       ↓
STL / STEP Validation
```

Advanced engineering verification such as FEA is currently represented only as a verification skeleton and is not implemented.

---

# 18. Printability Verification

Implementation:

```text
quadmesh/agent/printability.py
```

The printability gate combines:

* Blender reports
* STL inspection
* Geometry rules
* Printer-bed constraints

The workflow is:

```text
Generated Geometry
       ↓
Geometry Checks
       ↓
Printability Checks
       ↓
PASS ───────────────► Export
       │
       ▼
      FAIL
       │
       ▼
   Do Not Print
```

A failed design is explicitly prevented from being marked as printable.

---

# 19. Strength Verification

Implementation:

```text
quadmesh/strength.py
```

The current strength system uses hand calculations.

It contains:

* Material definitions
* Strength calculations
* Engineering checks

However, the current implementation uses placeholder material values and does not provide certified mechanical analysis.

Full FEA is not currently implemented.

---

# 20. Speech Architecture

Quadmesh contains independent ASR and TTS components.

Implementation:

```text
quadmesh/model/audio.py
quadmesh/agent/speech.py
```

---

## ASR

The ASR pipeline is:

```text
Microphone
    ↓
Audio
    ↓
ASR Model
    ↓
Text
    ↓
Chat Agent
```

ASR training:

```text
quadmesh/train/io_train.py
```

The ASR model is trained from scratch.

---

## TTS

TTS training:

```text
quadmesh/train/tts_train2.py
```

Synthesis:

```text
quadmesh/tts_synth.py
```

Voice styling:

```text
quadmesh/tts_style.py
```

The synthesis pipeline uses:

```text
Text
 ↓
Acoustic / Mel Representation
 ↓
Griffin-Lim
 ↓
Waveform
```

TTS v2 includes:

* Batched training
* Stop-token loss
* Guided attention

---

# 21. Tokenization

Implementation:

```text
quadmesh/data/tokenizer.py
```

Quadmesh uses a:

```text
32k BPE tokenizer
```

The tokenizer is generated through the training pipeline before model training.

---

# 22. Training Architecture

The training system is located in:

```text
quadmesh/train/
```

Main components:

```text
chat_train.py
io_train.py
loop.py
tts_train2.py
vla_train.py
```

The overall training pipeline includes:

```text
Dataset
   ↓
Tokenizer
   ↓
Role-Specific Training
   ↓
Supervised Fine-Tuning
   ↓
Self-Generation
   ↓
Preference Training
   ↓
Calibration
   ↓
Red-Team Evaluation
   ↓
Evaluation
   ↓
Shadow Testing
```

---

# 23. Training Pipeline Stages

The pipeline contains multiple stages:

```text
Stage 3  → Self Generation
Stage 4  → Preference Training
Stage 5  → Calibration
Stage 6  → Red-Team Evaluation
Stage 7  → Evaluation
Stage 8  → Shadow Testing
```

Implementation:

```text
quadmesh/pipeline/stage3_selfgen.py
quadmesh/pipeline/stage4_preference.py
quadmesh/pipeline/stage5_calibration.py
quadmesh/pipeline/stage6_redteam.py
quadmesh/pipeline/stage7_eval.py
quadmesh/pipeline/stage8_shadow.py
```

---

# 24. Dataset Generation

Engineering data generation:

```text
quadmesh/pipeline/datasets/engineering_gen.py
```

The generator produces:

* Part families
* Free-form CSG examples
* Engineering geometry examples

Planner supervised fine-tuning data:

```text
quadmesh/pipeline/datasets/planner_sft_gen.py
```

---

# 25. Role-Based Architecture

Quadmesh does not require every model role to use an equal parameter count.

Role-specific model sizing is defined through:

```text
quadmesh/role_shapes.py
```

The architecture can therefore allocate different model capacities to different tasks.

Conceptually:

```text
                Quadmesh System
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
   Planner           Chat             VLA
       │               │                │
       ▼               ▼                ▼
   CAD Roles       Language        Vision/Action
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                  Verification
```

---

# 26. KV Cache

The transformer supports KV caching for autoregressive generation.

Implementation:

```text
quadmesh/model/transformer.py
quadmesh/model/planner_model.py
```

The KV cache allows previously computed attention keys and values to be reused during generation.

A dedicated validation script is provided:

```text
check_kv.py
```

The purpose of the test is to verify that KV-cached generation produces the same output as the corresponding non-cached generation.

---

# 27. LoRA

Implementation:

```text
quadmesh/model/lora.py
```

LoRA adapters are provided as a parameter-efficient adaptation mechanism for model training.

The adapters can be used without modifying the complete base model parameter set.

---

# 28. Service Architecture

The service implementation is:

```text
quadmesh/serve.py
```

The service is deployed through:

```text
modal_app.py
```

The architecture separates local computer interaction from the remote model service.

```text
                 Local Machine
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
    Blender        Keyboard        Mouse
       │
       └──────────────┬──────────────┘
                      │
                      ▼
                Quadmesh Agent
                      │
                      ▼
                Remote Service
                      │
                      ▼
                    GPU
```

---

# 29. Modal Training and Deployment

Modal provides the remote GPU execution environment.

The main entry point is:

```text
modal_app.py
```

It contains entry points for:

* Tokenizer generation
* Planner training
* Role training
* Full training
* VLA training
* Chat training
* TTS v2 training
* Pipeline stages
* Deployment

---

# 30. End-to-End Technical Pipeline

The complete system can be represented as:

```text
                         USER
                           │
                           ▼
                ┌─────────────────────┐
                │  Multimodal Input   │
                │                     │
                │ Text                │
                │ Voice               │
                │ Image               │
                │ STL / STEP          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Chat Model       │
                │                     │
                │ Understanding       │
                │ Context             │
                │ Tool Calling        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      Planner        │
                │                     │
                │ Natural Language    │
                │        ↓            │
                │ CAD Operations      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Plan Checker      │
                │                     │
                │ Simulation          │
                │ Constraints         │
                │ Validation          │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                   PASS          FAIL
                    │             │
                    ▼             ▼
             ┌────────────┐   ┌──────────┐
             │  Geometry  │   │  Repair  │
             │  Engine    │   │   Loop   │
             └─────┬──────┘   └────┬─────┘
                   │               │
                   │               │
                   └───────◄───────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Blender / CadQuery  │
                │                     │
                │ Mesh / B-Rep        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Verification Stack  │
                │                     │
                │ Geometry            │
                │ Printability        │
                │ STL / STEP          │
                └──────────┬──────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ STL / STEP  │
                    └─────────────┘
```

---

# 31. Technical Module Map

```text
quadmesh/
│
├── agent/
│   ├── chat_agent.py       → Multimodal agent
│   ├── planner.py          → Planning
│   ├── autopilot.py        → Plan/build/check/repair
│   ├── vla_agent.py        → VLA execution
│   ├── desk.py             → Computer control
│   ├── desktop.py          → Low-level GUI control
│   ├── blender_client.py   → Blender client
│   ├── blender_bridge.py   → Blender server
│   ├── image3d.py          → Image → 3D
│   └── printability.py     → Printability gate
│
├── model/
│   ├── transformer.py      → Transformer architecture
│   ├── chat.py             → Multimodal chat
│   ├── vla.py              → Vision-Language-Action
│   ├── audio.py            → ASR/TTS
│   └── lora.py             → LoRA
│
├── data/
│   ├── tokenizer.py        → 32k BPE
│   ├── vision.py           → Vision processing
│   ├── streaming.py        → Streaming datasets
│   └── vla_data.py         → VLA data
│
├── pipeline/
│   ├── datasets/           → Dataset generation
│   ├── plan_checker.py     → Plan verification
│   ├── stage3_selfgen.py   → Self-generation
│   ├── stage4_preference.py
│   ├── stage5_calibration.py
│   ├── stage6_redteam.py
│   ├── stage7_eval.py
│   └── stage8_shadow.py
│
├── train/
│   ├── chat_train.py       → Chat training
│   ├── io_train.py         → ASR training
│   ├── tts_train2.py       → TTS training
│   └── vla_train.py        → VLA training
│
├── verification/
│   └── stack.py            → Verification framework
│
├── geometry3d.py           → Geometry generation
├── mechanisms.py           → Mechanisms
├── assemblies.py           → Assemblies
├── mesh_tools.py           → Mesh validation
├── step_io.py              → STEP/B-Rep
├── photo3d.py              → Photo → 3D
├── strength.py             → Strength calculations
├── config.py               → Model configuration
└── serve.py                → Service API
```

---

# 32. Technical Design Principle

The central architectural principle of Quadmesh is:

```text
Generation ≠ Verification
```

The model is responsible for generating a proposed solution.

The verification system independently checks the generated operations and geometry.

Therefore:

```text
Natural Language
      ↓
Prediction
      ↓
Verification
      ↓
Execution
      ↓
Verification
      ↓
Export
```

rather than:

```text
Natural Language
      ↓
Prediction
      ↓
Direct Export
```

This architecture is intended to make CAD generation a **closed-loop verified process** rather than a single-shot generative process.
