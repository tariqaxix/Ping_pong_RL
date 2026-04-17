import os
import sys
import glob
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ale_py
import gymnasium
gymnasium.register_envs(ale_py)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

sys.stdout.reconfigure(encoding="utf-8")

ENV_ID     = "ALE/Pong-v5"
MODEL_DIR  = "models/"
PLOT_DIR   = "plots/"
N_STACK    = 4
N_EPISODES = 20

os.makedirs(PLOT_DIR, exist_ok=True)


def pick_model() -> str:
    best = os.path.join(MODEL_DIR, "best_model.zip")
    if os.path.exists(best):
        return best
    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "*.zip")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No model found in {MODEL_DIR}. Run train.py first.")


def rating(score: float) -> str:
    if score >= 15:  return "Excellent"
    if score >= 5:   return "Great"
    if score >= 0:   return "Good"
    if score >= -10: return "Still learning"
    return "Needs more training"


def run_episodes(model, env, n: int):
    rewards, lengths = [], []
    obs = env.reset()
    done_count = 0

    while done_count < n:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        for d, inf in zip(done, info):
            if d and "episode" in inf:
                rewards.append(inf["episode"]["r"])
                lengths.append(inf["episode"]["l"])
                done_count += 1
                if done_count >= n:
                    break

    return np.array(rewards[:n]), np.array(lengths[:n])


def save_evaluation_chart(rewards: np.ndarray, path: str):
    fig = plt.figure(figsize=(14, 6), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#161b22")
    colours = ["#00ff88" if r > 0 else ("#ffa500" if r > -10 else "#ff4444") for r in rewards]
    ax1.bar(range(1, len(rewards) + 1), rewards, color=colours, edgecolor="none", width=0.7)
    ax1.axhline(rewards.mean(), color="#ffffff", linestyle="--", linewidth=1.5,
                label=f"Mean {rewards.mean():+.1f}")
    ax1.axhline(0,   color="#888",    linestyle=":", linewidth=1)
    ax1.axhline(21,  color="#ffd700", linestyle=":", linewidth=1, alpha=0.5, label="Max (+21)")
    ax1.axhline(-21, color="#ff4444", linestyle=":", linewidth=1, alpha=0.5, label="Min (-21)")
    ax1.set_xlabel("Episode", color="white", fontsize=12)
    ax1.set_ylabel("Reward",  color="white", fontsize=12)
    ax1.set_title("Per-Episode Rewards", color="white", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")
    ax1.legend(facecolor="#1e2530", labelcolor="white", fontsize=10)
    ax1.grid(axis="y", color="#333", linestyle="--", linewidth=0.5, alpha=0.6)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#161b22")
    ax2.hist(rewards, bins=15, color="#30a2da", edgecolor="#0d1117",
             linewidth=0.5, orientation="horizontal")
    ax2.axhline(rewards.mean(), color="#00ff88", linestyle="--", linewidth=1.5,
                label=f"Mean {rewards.mean():+.1f}")
    ax2.set_xlabel("Count",  color="white", fontsize=12)
    ax2.set_ylabel("Reward", color="white", fontsize=12)
    ax2.set_title("Reward Distribution", color="white", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")
    ax2.legend(facecolor="#1e2530", labelcolor="white", fontsize=10)
    ax2.grid(axis="x", color="#333", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle("PPO Agent -- Atari Pong -- Evaluation Report",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def main():
    print(f"\nPPO agent evaluation -- Atari Pong")

    model_path = pick_model()
    print(f"  model:    {model_path}")
    print(f"  episodes: {N_EPISODES}\n")

    env   = make_atari_env(ENV_ID, n_envs=4, seed=0)
    env   = VecFrameStack(env, n_stack=N_STACK)
    model = PPO.load(model_path, env=env)

    print(f"  running {N_EPISODES} episodes ...\n")
    t0 = time.time()
    rewards, lengths = run_episodes(model, env, N_EPISODES)
    elapsed = time.time() - t0

    mean_r = rewards.mean()
    std_r  = rewards.std()
    max_r  = rewards.max()
    min_r  = rewards.min()
    win_r  = (rewards > 0).mean() * 100

    lr      = model.learning_rate if not callable(model.learning_rate) else model.learning_rate(1)
    hparams = {
        "Algorithm":        "PPO (Proximal Policy Optimization)",
        "Category":         "Model-free | On-policy | Policy Gradient",
        "Policy":           str(model.policy.__class__.__name__),
        "Environment":      ENV_ID,
        "Frame stack":      str(N_STACK),
        "Learning rate":    f"{lr:.2e}",
        "Discount (gamma)": f"{model.gamma}",
        "GAE lambda":       f"{model.gae_lambda}",
        "Clip range":       f"{model.clip_range(1) if callable(model.clip_range) else model.clip_range}",
        "Entropy coef":     f"{model.ent_coef}",
        "VF coef":          f"{model.vf_coef}",
        "Batch size":       f"{model.batch_size}",
        "n_steps":          f"{model.n_steps}",
        "n_epochs":         f"{model.n_epochs}",
        "Max grad norm":    f"{model.max_grad_norm}",
    }

    print("  Hyperparameters:")
    for k, v in hparams.items():
        print(f"    {k}: {v}")

    print("\n  Results:")
    print(f"    episodes:    {N_EPISODES}")
    print(f"    mean reward: {mean_r:+.1f} +/- {std_r:.1f}")
    print(f"    best:        {max_r:+.1f}")
    print(f"    worst:       {min_r:+.1f}")
    print(f"    win rate:    {win_r:.1f}%")
    print(f"    avg length:  {lengths.mean():.0f} steps")
    print(f"    eval time:   {elapsed:.1f}s")
    print(f"    rating:      {rating(mean_r)}\n")

    chart_path = os.path.join(PLOT_DIR, "evaluation_report.png")
    save_evaluation_chart(rewards, chart_path)
    print(f"  chart saved -> {chart_path}")
    print(f"  next: python record_video.py\n")

    env.close()


if __name__ == "__main__":
    main()
