# Price is not Right: Neuro-Symbolic Methods Outperform VLAs on Structured Long-Horizon Manipulation Tasks with Significantly Lower Energy Consumption

## Setup
```bash
git submodule update --init --recursive
```

## Evaluation

### VLA

#### Download one of our models
Navigate to the checkpoints directory
```bash
cd openpi/checkpoints
```

Clone the model(s) you want to evaluate

End-to-End Model
```bash
git clone https://huggingface.co/tduggan93/pi0-hanoi-end-to-end
```

Planner-Guided Model
```bash
git clone https://huggingface.co/tduggan93/pi0-hanoi-planner-guided
```

#### Set Server Arguments for the pi0 model in config.yml
In openpi/examples/robotsuite/config.yml lines 51 - 52, you will see the following below:
```yml
- SERVER_ARGS=policy:checkpoint --policy.config pi0_hanoi_end_to_end --policy.dir /app/checkpoints/pi0-hanoi-end-to-end
# - SERVER_ARGS=policy:checkpoint --policy.config pi0_hanoi_planner_guided --policy.dir /app/checkpoints/pi0-hanoi-planner-guided
```

Uncomment the model you want to use and comment out the one you don't want to use. Also feel free to add your own model if you train one.

#### Set the evaluation arguments
In openpi/examples/robotsuite/main.py lines 59 - 112, you will see the following below:

The most important important arguments for you will be:
1. env_name - Hanoi or Hanoi4x3
2. env - Hanoi or Hanoi4x3
3. use_sequential_tasks - True for Planner Guided, False for End-to-End
4. random_block_selection - For 3 block hanoi randomly use 3 of the 4 available block types each episode
5. episodes - Number of episodes to run
6. wandb_project_name - the name of your wandb project

```python
@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy and multi-config support"""
    # --- Server Connection ---
    host: str = "127.0.0.1"         # Hostname of the OpenPI policy server
    port: int = 8000                # Port of the OpenPI policy server

    # --- planner ---
    planner:str = "pddl"            # Planner to use: 'pddl' or 'gpt-5'

    # --- Policy Interaction ---
    resize_size: int = 224               # Target size for image resizing (must match model training)
    replan_steps: int = 50               # Number of steps per action chunk from policy server
    use_sequential_tasks: bool =False    # If True, use sequential task prompts; if False, use single prompt
    time_based_progression: bool = False # If True, advance to next task after task_timeout steps regardless of completion
    task_timeout: int = 750              # Number of steps to wait before timing out a task

    # --- Robosuite Environment ---
    env_name: str = "Hanoi4x3" 
    env: str = "Hanoi4x3"                # Environment name for RecordDemos compatibility
    robots: str = "Panda"                # Robot model to use
    controller: str = "OSC_POSE"         # Robosuite controller name
    horizon: int = 7050                  # Max steps per episode
    skip_steps: int = 50                 # Number of initial steps to skip (wait for objects to settle)

    # --- Multi-configuration support ---
    random_block_placement: bool = False   # Place blocks on pegs randomly according to Towers of Hanoi rules
    random_block_selection: bool = False   # Randomly select 3 out of 4 blocks
    cube_init_pos_noise_std: float = 0.01  # Std dev for XY jitter of initial tower position

    # --- Rendering & Video ---
    render_mode: str = "headless"                 # Rendering mode: 'headless' (save video) or 'human' (live view)
    video_out_path: str = "data/robosuite_videos" # Directory to save videos
    camera_names: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Cameras for observation/video
    camera_height: int = 256  # Rendered camera height (before potential resize)
    camera_width: int = 256   # Rendered camera width (before potential resize)
    
    # --- Full Resolution Video Recording ---
    save_full_res_video: bool = True  # Save full resolution videos alongside model observations
    full_res_height: int = 480        # Full resolution video height
    full_res_width: int = 640         # Full resolution video width
    required_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Required cameras for OpenPI preprocessing

    # --- Misc ---
    seed: int = 3           #: Random seed
    episodes: int = 50      #: How many episodes to run back-to-back

    # --- Logging ---
    wandb_project: str = "Your wandb project name"  #: W&B project name
    log_every_n_seconds: float = 0.5                #: W&B system metric sampling interval (seconds)
```

#### Run the experiments
From the openpi directory:

```bash
docker compose -f examples/robosuite/compose.yml up
```

---

## Training Pipeline

This section describes the end-to-end processing pipeline for training the **neuro-symbolic method** — the new approach proposed in this work. The pipeline decomposes long-horizon manipulation tasks into symbolic plans executed by a library of learned skill policies.

### Pipeline Overview

```
Robosuite Simulation Environment
            │
            ▼
  ┌─────────────────────┐
  │  Symbolic Planner   │  (Metric-FF + PDDL domain/problem files)
  │  PDDL → Action Plan │  planning/PDDL/<env>/
  └─────────┬───────────┘
            │ action sequence (reach_pick, grasp, reach_place, drop, …)
            ▼
  ┌─────────────────────┐
  │  Oracle Demo        │  dataset_making/main.py
  │  Collector          │  records (obs, action, keypoint) per skill
  └─────────┬───────────┘
            │ datasets/<exp>/<id>/traces/<skill>.zip  (data.pkl inside)
            ▼
  ┌─────────────────────┐
  │  Data Preprocessing │  neuro_symbolic_method/data_processing/data_to_zarr.py
  │  ZIP → Zarr         │  groups transitions by skill type
  └─────────┬───────────┘
            │ hf_traj/<skill>/keypoint/keypoint.zarr
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   Parallel Model Training                    │
  │                                                              │
  │  ┌─────────────────────┐   ┌────────────────────────────┐   │
  │  │  Object Detection   │   │  Diffusion Skill Policies  │   │
  │  │  (YOLOv8 + Regr.)   │   │  (one per skill primitive) │   │
  │  │                     │   │                            │   │
  │  │  Images → .pt model │   │  Zarr dataset → .ckpt      │   │
  │  │  BBoxes → .pkl regr │   │  (reach_pick, grasp,       │   │
  │  │                     │   │   reach_place, drop)       │   │
  │  └──────────┬──────────┘   └───────────┬────────────────┘   │
  └─────────────┼──────────────────────────┼────────────────────┘
                │                          │
                └──────────┬───────────────┘
                           ▼
              ┌────────────────────────┐
              │  Inference / Evaluation│
              │  (experiments_neurosymbolic.py)
              │                        │
              │  YOLO+Regressor detect │
              │  current state → PDDL  │
              │  plan → skill policies │
              │  execute actions       │
              └────────────────────────┘
```

---

### Step 1 — Collect Demonstrations

Demonstrations are collected automatically by having the PDDL planner generate a plan and an oracle executor carry it out in the Robosuite simulation.  Each successful episode is split by skill primitive and saved as a `.zip` file (containing a `data.pkl` trajectory) under `datasets/`.

```bash
python -m dataset_making.main \
    --env Hanoi \
    --episodes 150 \
    --dir ./datasets \
    --random-block-placement \
    --random-block-selection \
    --cube-init-pos-noise-std 0.01 \
    --noisy-fraction 0.3 \
    --noise-std 0.03
```

Key arguments:

| Argument | Description |
|---|---|
| `--env` | Environment: `Hanoi`, `KitchenEnv`, `NutAssembly`, `CubeSorting` |
| `--episodes` | Number of successful episodes to record |
| `--random-block-placement` | Randomise initial block positions on pegs |
| `--random-block-selection` | Randomly select 3 out of 4 available block types |
| `--noisy-fraction` | Fraction of episodes that inject Gaussian action noise |
| `--noise-std` | Standard deviation scale factor for the injected noise |

**Output:** `datasets/<exp_name>/<exp_id>/traces/<skill>.zip`

Each `.zip` contains a `data.pkl` file with a list of `(obs, action, keypoint)` tuples for that skill primitive.

---

### Step 2 — Process Data into Zarr Format

Convert the collected pickle trajectories to [Zarr](https://zarr.readthedocs.io/) format, which is required by the diffusion policy training framework.

```bash
python -m neuro_symbolic_method.data_processing.data_to_zarr \
    --data_dir ./datasets/<experiment_name>/<experiment_id>
```

**What this script does:**

1. Reads all `.zip` trace files from `<data_dir>/traces/`.
2. Groups transitions by skill type (e.g. `reach_pick`, `grasp`, `reach_place`, `drop`).
3. For each skill, creates a `TrajectoryWithKeypoint` object with:
   - `obs` — robot proprioceptive observations
   - `acts` — end-effector delta actions
   - `keypoint` — 3-D object keypoint positions
4. Writes per-skill Zarr stores to `hf_traj/<skill>/keypoint/keypoint.zarr`.

**Zarr store layout per skill:**

```
keypoint.zarr/
├── data/
│   ├── action      (T, action_dim)   float64
│   ├── keypoint    (T, n_obj, kp_dim) float64
│   └── state       (T, obs_dim)      float64
└── meta/
    └── episode_ends (n_episodes,)    int64
```

---

### Step 3 — Train Object Detection and Pose Regression Models

#### 3a. YOLOv8 Object Detection

A YOLOv8 model is fine-tuned to detect task-relevant objects (blocks and pegs) from robot camera images.

1. Annotate images using [Roboflow](https://roboflow.com) and export in YOLOv8 format.
2. Train using the provided notebook or the CLI:

```bash
yolo task=detect mode=train \
    model=yolov8s.pt \
    data=<path_to_dataset>/data.yaml \
    epochs=25 \
    imgsz=800
```

Notebook location:
```
neuro_symbolic_method/objects_detection/train_yolov8_object_detection_on_custom_dataset.ipynb
```

Save the best weights to:
```
neuro_symbolic_method/models/yolo/<env>_yolo.pt
```

#### 3b. Pose Regression Model

A lightweight regressor (e.g. scikit-learn) is trained to map YOLO bounding-box features to 3-D object positions.  Once trained, save the model to:

```
neuro_symbolic_method/models/regressors/<env>_regressor.pkl
```

---

### Step 4 — Train Diffusion Skill Policies

Each primitive skill (`reach_pick`, `grasp`, `reach_place`, `drop`) is trained as a separate diffusion policy using the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) framework (submodule at `neuro_symbolic_method/diffusion_policy`).

Initialise the submodule if you have not already:

```bash
git submodule update --init --recursive
```

Train each skill policy pointing at the corresponding Zarr dataset:

```bash
cd neuro_symbolic_method/diffusion_policy
python train.py --config-name=train_diffusion_unet_lowdim_workspace \
    task.dataset_path=../../hf_traj/<skill>/keypoint/keypoint.zarr
```

Repeat for each of the four skills: `reach_pick`, `grasp`, `reach_place` (reach-to-drop), and `drop`.

After training, update the checkpoint paths in the environment config (e.g. `neuro_symbolic_method/config/hanoi.yaml`):

```yaml
policies:
  grasp:       policies/noisy_30/grasp.ckpt
  drop:        policies/noisy_30/drop.ckpt
  reach_pick:  policies/noisy_30/reach_pick.ckpt
  reach_place: policies/noisy_30/reach_drop.ckpt
```

---

### Step 5 — (Optional) VLA Fine-tuning via pi0

To fine-tune the pi0 VLA model on a new task:

1. **Collect demonstrations** using `dataset_making/main.py` (VLA-friendly formatting is enforced automatically).
2. **Convert to RLDS** using the [rlds_dataset_builder](rlds_dataset_builder/) submodule.
3. **Fine-tune pi0** following the [openpi](openpi/) training instructions (see `openpi/` submodule, branch `ICRA2026`).

After training, place the new checkpoint inside `openpi/checkpoints/` and update `openpi/examples/robosuite/config.yml` to point to it.