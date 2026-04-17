# PPO Agent - Ping pong

Training a Proximal Policy Optimization (PPO) agent to play Pong using Stable-Baselines3 and Gymnasium.

**Algorithm:** PPO - model-free, on-policy, policy gradient  
**Network:** ActorCriticCnnPolicy on raw pixels with 4-frame stacking  
**Environment:** ALE/Pong-v5 - score range -21 to +21

## Results

| Metric | Value |
|---|---|
| Training steps | 2,000,000 |
| Best reward | +9.2 |
| Eval mean (20 eps) | +2.4 ± 6.8 |
| Best episode | +13.0 |
| Worst episode | −11.0 |
| Win rate | 65% |
| Avg episode length | 4118 steps |

## Setup

```bash
pip install gymnasium[atari,accept-rom-license] stable-baselines3[extra] ale-py shimmy opencv-python moviepy
```

## Usage

```bash
python train.py          # train — saves model to models/
python evaluate.py       # evaluate best model, print stats
python record_video.py   # record gameplay video to videos/
python plot_results.py   # generate results dashboard
```

## File Structure

```
├── train.py             # PPO training, 8 parallel envs, checkpoints every 100k steps
├── evaluate.py          # 20-episode evaluation with charts
├── record_video.py      # custom visual renderer, FPS=5
├── plot_results.py      # training curve + learning phases dashboard
├── models/              # saved checkpoints + best_model.zip
├── plots/               # training_curve.png, evaluation_report.png, results_dashboard.png
├── videos/              # episode_1.mp4, episode_2.mp4, ...
└── logs/                # TensorBoard logs
```

## Hyperparameters

| Parameter | Value |
|---|---|
| n_steps | 128 |
| batch_size | 256 |
| n_epochs | 4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.1 |
| learning_rate | 2.5e-4 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |


## Contributors

- Tariq Aziz
- Matin Moradi


<img width="824" height="539" alt="image" src="https://github.com/user-attachments/assets/b2f55b90-f20b-490d-94b7-4713073753c9" />



