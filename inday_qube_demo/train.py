"""PPO (colored noise) training on the simulated Quanser Qube swingup.

Based on cnppo/experiments/learn.py, plus one recorded/rendered episode per evaluation.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import stable_baselines3.common.utils
from cnppo.cnpolicy import ColoredNoiseActorCriticPolicy
from gymnasium.wrappers import HumanRendering
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from inday_qube_demo.wrappers import RenderCache, StreamingRecordVideo

# -----
"""
VERSION 1.1: add SDE support as a baseline
VERSION 1.2: add n_steps configuration support
VERSION 1.3: changed backward version support.
"""
# -----

VERSION = 1.3
SUPPORTED_VERSIONS = {1.1, 1.2, VERSION}

ENV_ID = "gymqube:QubeSwingupSimEnv-v0"


class RenderEpisodeCallback(BaseCallback):
    """Run one deterministic episode, recorded as video and/or shown (human).

    Used as `EvalCallback(callback_after_eval=...)`: one episode per evaluation.
    """

    def __init__(self, env_id: str, mode: str, video_folder: str, total_timesteps: int):
        super().__init__()
        self.mode = mode
        self.recorder = None
        if mode == "human":
            self.env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
            self.recorder = StreamingRecordVideo(
                RenderCache(env), video_folder, episode_trigger=lambda _: False
            )
            if mode == "both":
                # SDL's default GL (GLX) framebuffer clashes with the EGL context
                # of the rgb_array renderer (X BadAccess); use a software framebuffer.
                os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")
            self.env = (
                HumanRendering(self.recorder) if mode == "both" else self.recorder
            )
        self._width = len(str(int(total_timesteps)))

    def _on_step(self) -> bool:
        obs, _ = self.env.reset()
        if self.recorder is not None:
            self.recorder.start_recording(f"t{self.num_timesteps:0{self._width}d}")
        done, ret = False, 0.0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            if self.mode == "human":
                self.env.render()
            ret += reward
            done = terminated or truncated
        if self.recorder is not None:
            self.recorder.stop_recording()
        print(f"[{self.mode}] t={self.num_timesteps} return={ret:.1f}")
        return True


def run_experiment(
    env_id=ENV_ID,
    seed=1234,
    n_envs=50,
    n_eval_envs=10,
    clip_range=0.2,
    lr=1e-4,
    ent_coef=0.0,
    gamma=0.99,
    gae_lambda=0.95,
    batch_size=64,
    n_epochs=10,
    noise_color=0.0,
    eval_episodes=50,
    cn_policy=True,
    total_timesteps=1e7,
    eval_freq: int = 10_240,
    n_steps: int = 2048,
    use_sde=False,
    render: str = "video",
    video_folder: str = "videos",
):
    print(f"using seed {seed}")
    stable_baselines3.common.utils.set_random_seed(seed)
    # Create the environment
    # FlattenObservation wrapper is necessary because dm-control environments
    # (with dm2gym) return dict-based space.
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=gym.wrappers.FlattenObservation,
    )
    eval_env = make_vec_env(
        env_id,
        n_envs=n_eval_envs,
        seed=seed,
        wrapper_class=gym.wrappers.FlattenObservation,
    )

    policy_kwargs = {}
    if cn_policy:
        policy = ColoredNoiseActorCriticPolicy
        policy_kwargs["noise_color_beta"] = noise_color
        policy_kwargs["noise_rng"] = np.random.default_rng(seed=seed)
    else:
        policy = "MlpPolicy"
    agent = PPO(
        env=env,
        policy=policy,
        policy_kwargs=policy_kwargs,
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_epochs=n_epochs,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        learning_rate=lr,
        clip_range=clip_range,
        tensorboard_log=f"tblogs/nc{noise_color}",
        verbose=1,
        use_sde=use_sde,  # --- V1.1
    )
    render_cb = None
    if render != "none":
        render_cb = RenderEpisodeCallback(env_id, render, video_folder, total_timesteps)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=eval_episodes,
        eval_freq=max(eval_freq // n_envs, 1),
        callback_after_eval=render_cb,
    )
    agent.learn(
        total_timesteps=int(total_timesteps),
        progress_bar=True,
        callback=[eval_cb],
    )
    if render_cb is not None:
        render_cb.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-eval-envs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.5e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=10_240)
    parser.add_argument("--no-cn-policy", action="store_true")
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--total-timesteps", type=float, default=2_048_010)
    parser.add_argument(
        "--noise-color", type=float, default=0.5, help="0=white 1=pink 2=red"
    )
    # 2048 is the stable-baselines default
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument(
        "--render", choices=["video", "human", "both", "none"], default="video"
    )
    parser.add_argument("--video-folder", type=str, default="videos")
    # ----------
    parser.add_argument("--version", type=float, default=VERSION)

    args = parser.parse_args()
    if args.use_sde and not args.no_cn_policy:
        parser.error(
            "Use of state-dependent-exploration SDE can only be used with "
            "--no-cn-policy."
        )
    if args.version not in SUPPORTED_VERSIONS:
        raise AssertionError(
            f"ERROR: you requested version {args.version}, "
            f"however the code is version {VERSION}, "
            f"only {SUPPORTED_VERSIONS} are supported."
        )
    print(f"step: 0\nVERSION: {VERSION}")
    run_experiment(
        env_id=args.env_id,
        seed=args.seed,
        n_envs=args.n_envs,
        n_eval_envs=args.n_eval_envs,
        clip_range=args.clip_range,
        lr=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_episodes=args.eval_episodes,
        cn_policy=not args.no_cn_policy,
        noise_color=args.noise_color,
        total_timesteps=int(args.total_timesteps),
        eval_freq=args.eval_freq,
        use_sde=args.use_sde,
        n_steps=args.n_steps,
        render=args.render,
        video_folder=args.video_folder,
    )


if __name__ == "__main__":
    main()
