# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

from PIL import Image, ImageDraw, ImageFont
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

sys.stdout.reconfigure(encoding="utf-8")

ENV_ID     = "ALE/Pong-v5"
MODEL_DIR  = "models/"
VIDEO_DIR  = "videos/"
N_EPISODES = 3
FPS        = 5

OUT_W, OUT_H = 800, 600

PLAY_TOP, PLAY_BOT = 34, 194
PLAY_H = PLAY_BOT - PLAY_TOP
ARI_W  = 160

TABLE_LEFT  = 60
TABLE_RIGHT = OUT_W - 60
TABLE_TOP   = 80
TABLE_BOT   = OUT_H - 55
TABLE_W     = TABLE_RIGHT - TABLE_LEFT
TABLE_H     = TABLE_BOT   - TABLE_TOP

PADDLE_W = 20
PADDLE_H = 80
BALL_R   = 11

C_BG          = (15,  15,  30)
C_TABLE_DARK  = (20,  110,  55)
C_TABLE_LIGHT = (28,  145,  72)
C_LINE        = (255, 255, 255)
C_NET         = (255, 255, 200)
C_AGENT       = (0,   220, 255)
C_OPPONENT    = (255, 140,  30)
C_BALL        = (255, 245,  80)
C_HUD_BG      = (10,  10,   20)

os.makedirs(VIDEO_DIR, exist_ok=True)


def pick_model() -> str:
    best = os.path.join(MODEL_DIR, "best_model.zip")
    if os.path.exists(best):
        return best
    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "*.zip")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No model in {MODEL_DIR}. Run train.py first.")


def detect_elements(frame: np.ndarray):
    play = frame[PLAY_TOP:PLAY_BOT, :, :]
    gray = play.mean(axis=2)

    lstrip = gray[:, 14:22]
    hot_l  = lstrip > 105
    if hot_l.any():
        rows = np.where(hot_l.any(axis=1))[0]
        opp_y = float(np.clip((rows.min() + rows.max()) / 2 / PLAY_H, 0.05, 0.95))
    else:
        opp_y = 0.5

    rstrip = gray[12:, 138:147]
    hot_r  = rstrip > 105
    if hot_r.any():
        rows = np.where(hot_r.any(axis=1))[0]
        agent_y = float(np.clip((rows.mean() + 12) / PLAY_H, 0.05, 0.95))
    else:
        agent_y = 0.5

    mid   = gray[12:, 20:140]
    hot_b = mid > 200
    if hot_b.any():
        rows, cols = np.where(hot_b)
        raw_y  = (rows.mean() + 12) / PLAY_H
        raw_x  = cols.mean()
        ball_x = float(np.clip(raw_x / 120.0, 0.0, 1.0))
        ball_y = float(np.clip(raw_y, 0.05, 0.95))
    else:
        ball_x, ball_y = 0.5, 0.5

    return agent_y, opp_y, ball_x, ball_y


def draw_table(draw: ImageDraw.ImageDraw):
    for row in range(TABLE_TOP, TABLE_BOT, 18):
        c = C_TABLE_DARK if (row // 18) % 2 == 0 else C_TABLE_LIGHT
        draw.rectangle([TABLE_LEFT, row, TABLE_RIGHT, min(row + 18, TABLE_BOT)], fill=c)

    draw.rectangle([TABLE_LEFT, TABLE_TOP, TABLE_RIGHT, TABLE_BOT],
                   outline=C_LINE, width=3)

    cx = (TABLE_LEFT + TABLE_RIGHT) // 2
    for y in range(TABLE_TOP + 6, TABLE_BOT, 14):
        draw.line([(cx, y), (cx, min(y + 7, TABLE_BOT))], fill=C_LINE, width=2)

    nw = 10
    draw.rectangle([cx - nw//2, TABLE_TOP - 12, cx + nw//2, TABLE_BOT + 12], fill=C_NET)
    for y in range(TABLE_TOP - 12, TABLE_BOT + 12, 10):
        draw.line([(cx - nw//2, y), (cx + nw//2, y)], fill=(190, 190, 140), width=1)


def draw_paddle(img: Image.Image, draw: ImageDraw.ImageDraw,
                cx: int, y_frac: float, colour: tuple, side: str):
    cy = TABLE_TOP + int(y_frac * TABLE_H)
    x0, x1 = cx - PADDLE_W//2, cx + PADDLE_W//2
    y0, y1 = cy - PADDLE_H//2, cy + PADDLE_H//2

    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    gd.rounded_rectangle([x0-7, y0-7, x1+7, y1+7], radius=12, fill=(*colour, 70))
    img.alpha_composite(glow)

    draw.rounded_rectangle([x0, y0, x1, y1], radius=7, fill=colour)

    face = tuple(max(0, c - 70) for c in colour)
    draw.rounded_rectangle([x0+3, y0+5, x1-3, y1-5], radius=5, fill=face)

    mx = (x0 + x1) // 2
    if side == "agent":
        draw.rectangle([mx-3, y1, mx+3, y1+22], fill=colour)
        draw.ellipse([mx-5, y1+18, mx+5, y1+28], fill=colour)
    else:
        draw.rectangle([mx-3, y0-22, mx+3, y0], fill=colour)
        draw.ellipse([mx-5, y0-28, mx+5, y0-18], fill=colour)


def draw_ball(img: Image.Image, draw: ImageDraw.ImageDraw, bx: int, by: int):
    draw.ellipse([bx-BALL_R+5, by-BALL_R+5, bx+BALL_R+5, by+BALL_R+5],
                 fill=(0, 0, 0, 100))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    gd.ellipse([bx-BALL_R-9, by-BALL_R-9, bx+BALL_R+9, by+BALL_R+9],
               fill=(*C_BALL, 55))
    img.alpha_composite(glow)
    draw.ellipse([bx-BALL_R, by-BALL_R, bx+BALL_R, by+BALL_R], fill=C_BALL)
    draw.ellipse([bx-BALL_R+3, by-BALL_R+3, bx-BALL_R+8, by-BALL_R+8],
                 fill=(255, 255, 255, 210))


def draw_hud(draw: ImageDraw.ImageDraw, episode: int, step: int, reward: float):
    draw.rectangle([0, 0, OUT_W, TABLE_TOP - 4], fill=C_HUD_BG)
    try:
        fb = ImageFont.truetype("arial.ttf", 26)
        fs = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        fb = fs = ImageFont.load_default()

    draw.text((OUT_W//2, 12), "PPO Agent -- Atari Pong",
              fill=(180, 180, 180), font=fs, anchor="mt")
    draw.text((TABLE_LEFT+10, 34),  "OPPONENT",   fill=C_OPPONENT, font=fs, anchor="lt")
    draw.text((TABLE_RIGHT-10, 34), "AGENT (PPO)", fill=C_AGENT,  font=fs, anchor="rt")
    draw.text((OUT_W//2, 34), f"Episode {episode+1}   Step {step}",
              fill=(140, 140, 140), font=fs, anchor="mt")
    col = (0, 210, 110) if reward >= 0 else (255, 70, 70)
    draw.text((OUT_W//2, 52), f"Score: {reward:+.0f}", fill=col, font=fb, anchor="mt")

    draw.rectangle([0, OUT_H-28, OUT_W, OUT_H], fill=C_HUD_BG)
    draw.text((10, OUT_H-14),
              "Algorithm: PPO (Proximal Policy Optimization)  |  "
              "Model-free · On-policy · Policy Gradient  |  CnnPolicy",
              fill=(90, 90, 110), font=fs, anchor="lm")


def render_frame(raw_frame: np.ndarray,
                 episode: int, step: int, reward: float) -> np.ndarray:
    agent_y, opp_y, ball_x, ball_y = detect_elements(raw_frame)

    img  = Image.new("RGBA", (OUT_W, OUT_H), (*C_BG, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    draw_table(draw)
    draw_hud(draw, episode, step, reward)

    opp_cx   = TABLE_LEFT  + 32
    agent_cx = TABLE_RIGHT - 32
    draw_paddle(img, draw, opp_cx,   opp_y,   C_OPPONENT, "opp")
    draw_paddle(img, draw, agent_cx, agent_y, C_AGENT,    "agent")

    play_x0 = opp_cx   + PADDLE_W // 2 + 2
    play_x1 = agent_cx - PADDLE_W // 2 - 2
    bx = int(play_x0 + ball_x * (play_x1 - play_x0))
    by = int(TABLE_TOP + ball_y * TABLE_H)
    draw_ball(img, draw, bx, by)

    return np.array(img.convert("RGB"))


def main():
    print("\nPPO agent -- Atari Pong -- recording video")

    model_path = pick_model()
    print(f"  model:    {model_path}")
    print(f"  episodes: {N_EPISODES}")
    print(f"  fps:      {FPS}  (slowed down)")
    print(f"  output:   {VIDEO_DIR}\n")

    vec_env = make_atari_env(ENV_ID, n_envs=1, seed=42,
                              env_kwargs={"render_mode": "rgb_array"})
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model   = PPO.load(model_path, env=vec_env)

    for ep in range(N_EPISODES):
        out_path = os.path.join(VIDEO_DIR, f"episode_{ep+1}.mp4")
        writer   = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8)

        obs       = vec_env.reset()
        ep_reward = 0.0
        step      = 0
        done      = False

        print(f"  recording episode {ep+1}/{N_EPISODES} ...", end="", flush=True)

        while not done:
            raw_frames = vec_env.env_method("render")
            raw_frame  = raw_frames[0]

            canvas = render_frame(raw_frame, ep, step, ep_reward)
            writer.append_data(canvas)

            action, _   = model.predict(obs, deterministic=True)
            obs, reward, dones, info = vec_env.step(action)
            ep_reward  += float(reward[0])
            step       += 1
            done        = bool(dones[0])

        writer.close()
        print(f"  score={ep_reward:+.0f}  ({step} steps)  -> {out_path}")

    vec_env.close()

    print("\nDone. Videos saved:")
    for f in sorted(glob.glob(os.path.join(VIDEO_DIR, "episode_*.mp4"))):
        print(f"  {f}")
    print()


if __name__ == "__main__":
    main()
