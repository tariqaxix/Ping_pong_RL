import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ale_py
import gymnasium
gymnasium.register_envs(ale_py)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.evaluation import evaluate_policy

sys.stdout.reconfigure(encoding="utf-8")

ENV_ID          = "ALE/Pong-v5"
N_ENVS          = 8
TOTAL_TIMESTEPS = 2_000_000
N_STACK         = 4
CHECKPOINT_FREQ = 100_000
LOG_DIR         = "logs/"
MODEL_DIR       = "models/"
PLOT_DIR        = "plots/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


class ProgressCallback(BaseCallback):

    BAR_WIDTH = 40

    def __init__(self, total_timesteps: int, log_every: int = 10_000):
        super().__init__()
        self.total      = total_timesteps
        self.log_every  = log_every
        self.ep_rewards: list = []
        self.reward_log: list = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_every == 0:
            self._print_progress()

        return True

    def _print_progress(self):
        elapsed   = time.time() - self.start_time
        fraction  = self.num_timesteps / self.total
        filled    = int(self.BAR_WIDTH * fraction)
        bar       = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        pct       = fraction * 100
        fps       = self.num_timesteps / max(elapsed, 1)
        remaining = (self.total - self.num_timesteps) / max(fps, 1)

        mean_r = np.mean(self.ep_rewards[-50:]) if self.ep_rewards else float("nan")
        if not np.isnan(mean_r):
            self.reward_log.append((self.num_timesteps, mean_r))

        print(
            f"\r[{bar}] {pct:5.1f}%  "
            f"step={self.num_timesteps:>7,}  "
            f"reward={mean_r:+6.1f}  "
            f"fps={fps:>5.0f}  "
            f"eta={remaining/60:4.1f}m   ",
            end="", flush=True
        )

    def _on_training_end(self):
        print()
        self._save_curve()

    def _save_curve(self):
        if len(self.reward_log) < 2:
            return

        steps, rewards = zip(*self.reward_log)
        rewards = np.array(rewards)
        window  = max(1, len(rewards) // 15)
        smooth  = np.convolve(rewards, np.ones(window) / window, mode="same")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        ax.plot(steps, rewards, color="#30a2da", alpha=0.3, linewidth=1, label="Episode reward")
        ax.plot(steps, smooth,  color="#00ff88", linewidth=2.5, label=f"Smoothed (w={window})")
        ax.axhline(0,   color="#ff6b6b", linestyle="--", linewidth=1, alpha=0.7, label="Score = 0")
        ax.axhline(21,  color="#ffd700", linestyle=":",  linewidth=1, alpha=0.6, label="Perfect (+21)")
        ax.axhline(-21, color="#ff4444", linestyle=":",  linewidth=1, alpha=0.6, label="Worst (-21)")

        ax.set_xlabel("Timesteps", color="white", fontsize=12)
        ax.set_ylabel("Mean Episode Reward", color="white", fontsize=12)
        ax.set_title("PPO on Atari Pong -- Training Curve", color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(facecolor="#1e2530", labelcolor="white", fontsize=10)
        ax.grid(color="#333", linestyle="--", linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "training_curve.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  training curve saved -> {path}")


def make_env(n_envs=N_ENVS, seed=42):
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed,
                         env_kwargs={"render_mode": None})
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def main():
    print(f"\nPPO training -- Atari Pong")
    print(f"  environment:   {ENV_ID}")
    print(f"  parallel envs: {N_ENVS}")
    print(f"  frame stack:   {N_STACK}")
    print(f"  timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"  checkpoint:    every {CHECKPOINT_FREQ:,} steps\n")

    env      = make_env(n_envs=N_ENVS)
    eval_env = make_env(n_envs=4, seed=999)

    model = PPO(
        policy          = "CnnPolicy",
        env             = env,
        n_steps         = 128,
        batch_size      = 256,
        n_epochs        = 4,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.1,
        ent_coef        = 0.01,
        learning_rate   = 2.5e-4,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        tensorboard_log = LOG_DIR,
        verbose         = 0,
    )

    progress_cb   = ProgressCallback(TOTAL_TIMESTEPS, log_every=10_000)
    checkpoint_cb = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ // N_ENVS,
        save_path   = MODEL_DIR,
        name_prefix = "ppo_pong",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = 50_000 // N_ENVS,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0,
    )

    print(f"  training started ...  (Ctrl+C to stop and save)\n")
    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = [progress_cb, checkpoint_cb, eval_cb],
            tb_log_name     = "PPO_Pong",
        )
    except KeyboardInterrupt:
        print(f"\n  interrupted -- saving current model ...")

    final_path = os.path.join(MODEL_DIR, "ppo_pong_final")
    model.save(final_path)
    print(f"  final model saved -> {final_path}.zip")

    print(f"\n  evaluating final model ...")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"  mean reward: {mean_r:+.1f} +/- {std_r:.1f}")

    env.close()
    eval_env.close()

    print(f"\n  tensorboard: tensorboard --logdir {LOG_DIR}")
    print(f"  next: python evaluate.py   or   python record_video.py\n")


if __name__ == "__main__":
    main()
