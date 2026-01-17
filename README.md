# CleanRL PPO for Gymnasium (with video capture)

This repository provides a CleanRL-style PPO training script for Gymnasium environments.
It supports optional video recording during training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_ppo.py --env-id CartPole-v1 --total-timesteps 250000
```

## Capture video

```bash
python train_ppo.py --env-id CartPole-v1 --capture-video --video-dir videos
```

Videos will be saved under the `videos/` directory. You can change the capture frequency with
`--video-step-trigger`.

## Notes for Atari-like environments

If you want to train on Atari (e.g. `ALE/Breakout-v5`), install the Atari extras and ROMs:

```bash
pip install "gymnasium[atari,accept-rom-license]"
```

Then run:

```bash
python train_ppo.py --env-id ALE/Breakout-v5 --capture-video
```
