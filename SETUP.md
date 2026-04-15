# Setup Guide (MPE Environment)

Tested on Python 3.13, Windows 11, NVIDIA RTX 5060 (Blackwell sm_120), April 2025.

## 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\activate

# Git Bash / Linux / macOS
source venv/Scripts/activate
```

## 2. Install PyTorch

**RTX 50xx (Blackwell, sm_120) — requires CUDA 12.8 build:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**RTX 30xx / 40xx or older:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 3. Install dependencies

```bash
pip install -r requirements_mpe.txt
```

## 4. Install the onpolicy package

```bash
pip install -e .
```

## 5. Run MPE simple_spread (quick test)

```bash
python ./onpolicy/scripts/train/train_mpe.py \
  --env_name MPE --algorithm_name rmappo --experiment_name check \
  --scenario_name simple_spread --num_agents 3 --num_landmarks 3 --seed 1 \
  --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 25 --num_env_steps 20000000 \
  --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
  --use_wandb
```

- `--use_wandb` disables wandb and uses TensorBoard instead
- Increase `--n_rollout_threads` for faster training (128 recommended on Linux/Colab, keep low on Windows)
- Results saved to `onpolicy/scripts/results/`
- View with: `tensorboard --logdir onpolicy/scripts/results`

## Notes

- The original `requirements.txt` targets Python 3.6 / PyTorch 1.5 and is incompatible with modern Python — use `requirements_mpe.txt` instead
- On Windows, spawning many rollout threads is resource-intensive; 4–16 is safe, 128 works best on Linux/Colab
- Full 20M timestep run takes ~9 hours locally with 4 threads, ~1 hour on Colab with 128 threads
