import os
import sys
import argparse
import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

# =========================
# 1) Add project root to sys.path (no absolute path)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# =========================
# 2) Your imports
# =========================
from continuous_env.make_env import make_env
from algo.epigraph_pinn.epi_pinn_agent import epi_agent_new
from algo.utils import *  # consider replacing with explicit imports for camera-ready cleanliness

# =========================
# 3) Device (your original code uses "device" but did not define it)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WandbLogger:
    def __init__(self, args):
        self.enabled = bool(args.use_wandb)
        self.wandb = None
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it or run without --use-wandb."
            ) from exc

        self.wandb = wandb
        self.wandb.init(
            project=args.project_name,
            name=args.run_name,
            dir=args.run_dir,
            config=vars(args),
        )

    def log(self, metrics, step=None):
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def video(self, path, fps):
        if not self.enabled or path is None:
            return None
        return self.wandb.Video(path, fps=fps, format="mp4")

    def finish(self):
        if self.enabled:
            self.wandb.finish()


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _summarize_array(prefix, values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        f"{prefix}_sum": float(np.sum(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _safe_path_part(value):
    return str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")


def prepare_run_dirs(args):
    run_name = (
        f"{_safe_path_part(args.algo)}_"
        f"{_safe_path_part(args.scenario)}_"
        f"seed{_safe_path_part(args.seed)}_"
        f"{_safe_path_part(args.log_dir)}"
    )
    run_dir = Path("runs") / args.project_name / run_name
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    args.run_name = run_name
    args.run_dir = str(run_dir)
    args.checkpoint_dir = str(checkpoint_dir)
    args.eval_dir = str(eval_dir)
    return run_dir, checkpoint_dir, eval_dir


def compute_discounted_returns(rewards, dts, gamma=0.99):
    """
    Compute discounted returns for a single episode trajectory with irregular time steps.
    Args:
        rewards: list of tensors, each is [n_agents]
        dts:     list of scalar tensors or floats, each is dt for that step
        gamma:   discount factor; using gamma**dt to approximate exp(-rho*dt)
    Returns:
        returns: tensor [T, n_agents]
    """
    rewards = torch.stack(rewards).to(device)  # [T, n_agents]
    T, n_agents = rewards.shape[0], rewards.shape[1]

    if isinstance(dts[0], torch.Tensor):
        dts_t = torch.stack(dts).to(device).view(-1)  # [T]
    else:
        dts_t = torch.tensor(dts, dtype=torch.float32, device=device).view(-1)  # [T]

    returns = torch.zeros_like(rewards)
    future_return = torch.zeros(n_agents, device=device)

    for t in reversed(range(T)):
        discount = gamma ** dts_t[t]
        future_return = rewards[t] * dts_t[t] + discount * future_return
        returns[t] = future_return

    return returns


def compute_time_to_go_sequence(delta_ts, T_total):
    """
    Compute time-to-go (remaining time) for each step.
    Args:
        delta_ts: [num_steps] array of dt
        T_total:  total horizon
    Returns:
        time_to_go: [num_steps] array
    """
    cumulative_time = np.cumsum(delta_ts)
    time_to_go = T_total - cumulative_time
    return time_to_go


def sample_irregular_dts(num_steps, T_total, rng: np.random.Generator):
    """
    Sample an episode-specific irregular dt sequence with sum(dt)=T_total.
    Using Dirichlet ensures dt_i > 0 and total duration is fixed.
    """
    delta_ts = rng.dirichlet(np.ones(num_steps, dtype=np.float32)) * T_total
    time_to_go = compute_time_to_go_sequence(delta_ts, T_total)
    return delta_ts.astype(np.float32), time_to_go.astype(np.float32)


def _capture_mpe_frame(env):
    frames = env.render(mode="rgb_array")
    if isinstance(frames, list):
        if not frames:
            return None
        frame = frames[0]
    else:
        frame = frames
    return np.asarray(frame, dtype=np.uint8)


def _write_eval_video(video_path, frames, fps):
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed; skipping eval video export.")
        return None

    for old_path in Path(video_path).parent.glob("*.mp4"):
        old_path.unlink()
    try:
        imageio.mimsave(video_path, frames, fps=fps)
    except Exception as exc:
        print(f"Eval video export failed: {exc}")
        return None
    return video_path


def _set_policy_mode(model, training):
    for policy_net in model.policy_nets:
        policy_net.train(training)


def run_evaluation(model, args, episode, T_total, num_steps):
    eval_env = make_env(args.scenario, args.seed + episode)
    eval_rng = np.random.default_rng(args.seed + episode)
    prev_mode = args.mode
    args.mode = "eval"
    _set_policy_mode(model, training=False)

    episode_rewards = []
    con_reward_values = []
    dis_reward_values = []
    combined_reward_values = []
    constraint_values = []
    episode_violation_flags = []
    frames = []

    try:
        with torch.no_grad():
            for eval_episode in range(args.eval_episodes):
                delta_ts, _ = sample_irregular_dts(num_steps, T_total, eval_rng)
                state = eval_env.reset()
                episode_reward = 0.0
                episode_constraints = []

                if eval_episode == 0 and args.eval_video:
                    try:
                        frame = _capture_mpe_frame(eval_env)
                        if frame is not None:
                            frames.append(frame)
                    except Exception as exc:
                        print(f"Eval video capture failed at reset: {exc}")
                        frames = []

                for step in range(num_steps):
                    action = model.choose_action(state, float(delta_ts[step]))
                    next_state, reward, done, con_constraints = eval_env.step_con(
                        deepcopy(action),
                        float(delta_ts[step]),
                    )
                    con_reward, dis_reward = reward
                    con_reward = np.asarray(con_reward, dtype=np.float32).reshape(-1)
                    dis_reward = np.asarray(dis_reward, dtype=np.float32).reshape(-1)
                    combined_reward = con_reward + dis_reward

                    con_reward_values.extend(con_reward.tolist())
                    dis_reward_values.extend(dis_reward.tolist())
                    combined_reward_values.extend(combined_reward.tolist())
                    episode_reward += float(np.sum(combined_reward))

                    if con_constraints is not None:
                        current_constraints = np.asarray(
                            con_constraints, dtype=np.float32
                        ).reshape(-1)
                        constraint_values.extend(current_constraints.tolist())
                        episode_constraints.extend(current_constraints.tolist())

                    state = next_state

                    if eval_episode == 0 and args.eval_video and frames is not None:
                        try:
                            frame = _capture_mpe_frame(eval_env)
                            if frame is not None:
                                frames.append(frame)
                        except Exception as exc:
                            print(f"Eval video capture failed at step {step}: {exc}")
                            frames = []

                    if True in done:
                        break

                episode_rewards.append(episode_reward)
                episode_violation_flags.append(
                    float(
                        np.any(np.asarray(episode_constraints, dtype=np.float32) > 0.0)
                    )
                    if episode_constraints
                    else 0.0
                )
    finally:
        args.mode = prev_mode
        _set_policy_mode(model, training=True)

    constraint_arr = np.asarray(constraint_values, dtype=np.float32).reshape(-1)
    video_path = None
    if args.eval_video and frames:
        video_path = str(Path(args.eval_dir) / f"eval_episode_{episode}.mp4")
        video_path = _write_eval_video(video_path, frames, args.eval_video_fps)

    metrics = {
        "eval/episode_reward_total_mean": (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        ),
        "eval/episode_reward_total_min": (
            float(np.min(episode_rewards)) if episode_rewards else 0.0
        ),
        "eval/episode_reward_total_max": (
            float(np.max(episode_rewards)) if episode_rewards else 0.0
        ),
        "eval/episodes": args.eval_episodes,
        "eval/violation_rate": (
            float(np.mean(episode_violation_flags)) if episode_violation_flags else 0.0
        ),
        "eval/violation_steps": (
            int(np.sum(constraint_arr > 0.0)) if constraint_arr.size else 0
        ),
    }
    metrics.update(_summarize_array("eval/con_reward", con_reward_values))
    metrics.update(_summarize_array("eval/dis_reward", dis_reward_values))
    metrics.update(_summarize_array("eval/combined_reward", combined_reward_values))
    metrics.update(_summarize_array("eval/constraint", constraint_values))

    return metrics, video_path


def main(args):
    # ---- Environment & directories ----
    run_dir, checkpoint_dir, eval_dir = prepare_run_dirs(args)
    print(args)
    wandb_logger = WandbLogger(args)

    env = make_env(args.scenario, args.seed)

    # Some envs expose env.world.seed; guard to avoid attribute errors
    if hasattr(env, "world") and hasattr(env.world, "seed"):
        env.world.seed = args.seed

    # ---- Seeding ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(args.seed)

    # ---- Dimensions ----
    n_agents = env.n
    n_actions = env.world.dim_p
    args.n_agents = n_agents
    n_states = env.observation_space[0].shape[0]  # assume all agents share obs_dim

    # ---- Model ----
    if args.algo == "epi":
        model = epi_agent_new(n_states, n_actions, n_agents, args, env)

    model.model_dir = str(checkpoint_dir)
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Eval directory: {eval_dir}")
    print(model)

    # ---- z annealing ----
    initial_z_max = float(args.return_factor)
    final_z_min = float(args.z_lowerbound)
    z_decay_start = 0
    z_decay_end = 5000

    denom = max(1, (z_decay_end - z_decay_start))
    decay_rate = (initial_z_max - final_z_min) / denom
    z_range = initial_z_max

    # ---- Training settings ----
    T_total = 5.0
    model.T = T_total
    num_steps = int(args.episode_length)

    episode = 0

    try:
        while episode < args.max_episodes:
            # Re-sample irregular dt every episode (more sensible than fixing dt for all episodes)
            delta_ts, time_to_go = sample_irregular_dts(num_steps, T_total, rng)

            state = env.reset()

            # Trajectory buffers
            trajectory_obs = []
            trajectory_next_obs = []
            trajectory_actions = []
            trajectory_rewards = []
            trajectory_dts = []
            trajectory_constraints = []
            trajectory_z = []

            # Episode metric buffers, kept aggregated across agents.
            con_reward_values = []
            dis_reward_values = []
            combined_reward_values = []
            constraint_values = []
            z_values = []
            action_norm_values = []

            # Anneal z_range within window
            if z_decay_start <= episode <= z_decay_end:
                z_range = max(final_z_min, z_range - decay_rate)

            # Sample z once per episode
            sampled_z = float(rng.uniform(low=0.0, high=max(1e-8, z_range)))
            z_initial = sampled_z

            episode += 1
            step = 0

            # Episode logging
            accum_reward_total = 0.0

            # Rollout
            while True:
                if args.mode != "train":
                    raise NotImplementedError("This cleaned script currently supports train only. Eval can be added if needed.")

                # Guard against dt index overflow
                if step >= num_steps:
                    done = [True] * n_agents
                    con_constraints = None
                    break

                # Choose action
                action = model.choose_action(state, float(delta_ts[step]))

                # Env step
                next_state, reward, done, con_constraints = env.step_con(deepcopy(action), float(delta_ts[step]))
                con_reward, dis_reward = reward

                con_reward = np.asarray(con_reward, dtype=np.float32).reshape(-1)  # [n_agents]
                dis_reward = np.asarray(dis_reward, dtype=np.float32).reshape(-1)  # [n_agents]
                combined_reward = con_reward + dis_reward

                con_reward_values.extend(con_reward.tolist())
                dis_reward_values.extend(dis_reward.tolist())
                combined_reward_values.extend(combined_reward.tolist())
                if con_constraints is not None:
                    constraint_values.extend(np.asarray(con_constraints, dtype=np.float32).reshape(-1).tolist())
                z_values.append(sampled_z)
                action_norm_values.append(float(np.linalg.norm(np.asarray(action, dtype=np.float32))))

                accum_reward_total += float(np.sum(combined_reward))

                # z update (keep your formula, but ensure numeric stability)
                dt_now = float(delta_ts[step])
                z_next = sampled_z - (-np.sum(con_reward) / args.normal_factor + np.log(args.gamma) * sampled_z) * dt_now
                z_next = float(np.clip(z_next, args.z_min, args.z_max))

                # Store trajectory (epi)
                obs = torch.from_numpy(np.stack(state)).float().to(device)         # [n_agents, obs_dim]
                obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)   # [n_agents, obs_dim]

                # Keep your learning signal: normalized continuous reward only
                con_r_t = torch.from_numpy(con_reward / args.normal_factor).float().to(device)  # [n_agents]

                ac_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)         # [n_agents, act_dim]
                dt_tensor = torch.tensor(dt_now, dtype=torch.float32, device=device)            # scalar
                constraint_tensor = (
                    torch.tensor(con_constraints, dtype=torch.float32, device=device)
                    if con_constraints is not None
                    else torch.zeros(1, device=device)
                )
                z_tensor = torch.tensor(sampled_z, dtype=torch.float32, device=device)          # scalar

                trajectory_obs.append(obs)
                trajectory_next_obs.append(obs_)
                trajectory_actions.append(ac_tensor)
                trajectory_rewards.append(con_r_t)
                trajectory_dts.append(dt_tensor)
                trajectory_constraints.append(constraint_tensor)
                trajectory_z.append(z_tensor)

                # Advance
                state = next_state
                sampled_z = z_next
                step += 1

                # Termination
                if (step >= num_steps) or (True in done):
                    break

            # Update at end of episode
            returns_tensor = compute_discounted_returns(
                rewards=trajectory_rewards,
                dts=trajectory_dts,
                gamma=args.gamma
            )  # [T, n_agents]

            batch = [
                trajectory_obs,
                trajectory_actions,
                trajectory_next_obs,
                trajectory_rewards,
                trajectory_dts,
                returns_tensor,
                trajectory_constraints,
                trajectory_z,
            ]

            losses = model.update(batch)

            # Use .get to avoid KeyError if some losses are disabled by ablations
            d_loss      = losses.get("dynamics_loss", None)
            r_loss      = losses.get("reward_loss", None)
            v_loss      = losses.get("value_loss", None)
            c_loss      = losses.get("cost_loss", None)
            a_loss      = losses.get("policy_loss", None)
            tildeV_loss = losses.get("tilde_value_loss", None)
            Q_loss      = losses.get("Q_loss", None)
            vgi_loss    = losses.get("vgi_loss", None)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[Episode {episode:05d}] total_reward={accum_reward_total:.4f}")

            # Save model
            saved_checkpoint = int((episode % args.save_interval == 0) and (args.mode == "train"))
            if saved_checkpoint:
                model.save_model(episode)

            return_values = returns_tensor.detach().cpu().numpy()
            constraint_arr = np.asarray(constraint_values, dtype=np.float32).reshape(-1)
            violation_steps = int(np.sum(constraint_arr > 0.0)) if constraint_arr.size else 0
            violation_rate = float(np.mean(constraint_arr > 0.0)) if constraint_arr.size else 0.0

            metrics = {
                "episode": episode,
                "rollout/episode_reward_total": accum_reward_total,
                "rollout/episode_steps": step,
                "rollout/action_norm_mean": float(np.mean(action_norm_values)) if action_norm_values else 0.0,
                "rollout/action_norm_max": float(np.max(action_norm_values)) if action_norm_values else 0.0,
                "train/dynamics_loss": _finite_or_none(d_loss),
                "train/reward_loss": _finite_or_none(r_loss),
                "train/return_value_loss": _finite_or_none(v_loss),
                "train/constraint_value_loss": _finite_or_none(c_loss),
                "train/policy_loss": _finite_or_none(a_loss),
                "train/tilde_value_loss": _finite_or_none(tildeV_loss),
                "train/constraint_model_loss": _finite_or_none(Q_loss),
                "train/vgi_loss": _finite_or_none(vgi_loss),
                "train/num_agents": n_agents,
                "train/saved_checkpoint": saved_checkpoint,
                "train/exploration_sigma": float(getattr(model, "_sigma_scale", 0.0)),
                "epi/z_initial": z_initial,
                "epi/z_final": sampled_z,
                "epi/z_range": z_range,
                "time/dt_mean": float(np.mean(delta_ts[:step])) if step > 0 else 0.0,
                "time/dt_min": float(np.min(delta_ts[:step])) if step > 0 else 0.0,
                "time/dt_max": float(np.max(delta_ts[:step])) if step > 0 else 0.0,
                "time/dt_std": float(np.std(delta_ts[:step])) if step > 0 else 0.0,
                "time/episode_horizon": float(np.sum(delta_ts[:step])) if step > 0 else 0.0,
                "return/discounted_return_mean": float(np.mean(return_values)),
                "return/discounted_return_sum": float(np.sum(return_values)),
                "return/discounted_return_start": float(np.mean(return_values[0])) if return_values.size else 0.0,
                "return/discounted_return_min": float(np.min(return_values)),
                "return/discounted_return_max": float(np.max(return_values)),
                "safety/violation_steps": violation_steps,
                "safety/violation_rate": violation_rate,
            }
            metrics.update(_summarize_array("rollout/con_reward", con_reward_values))
            metrics.update(_summarize_array("rollout/dis_reward", dis_reward_values))
            metrics.update(_summarize_array("rollout/combined_reward", combined_reward_values))
            metrics.update(_summarize_array("safety/constraint", constraint_values))
            metrics.update(_summarize_array("epi/z", z_values))

            wandb_logger.log(metrics, step=episode)

            should_eval = (
                args.eval_interval > 0
                and episode % args.eval_interval == 0
                and args.mode == "train"
            )
            if should_eval:
                eval_metrics, eval_video_path = run_evaluation(
                    model=model,
                    args=args,
                    episode=episode,
                    T_total=T_total,
                    num_steps=num_steps,
                )
                eval_video = wandb_logger.video(
                    eval_video_path, fps=args.eval_video_fps
                )
                if eval_video is not None:
                    eval_metrics["eval/video"] = eval_video
                wandb_logger.log(eval_metrics, step=episode)
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="formation", type=str,
                        help="simple_spread/line/corridor/formation/simple_tag/target")
    parser.add_argument("--max_episodes", default=30000, type=int)
    parser.add_argument("--algo", default="epi", type=str, help="cleaned for epi entrypoint")
    parser.add_argument("--mode", default="train", type=str, help="train/eval")
    parser.add_argument("--episode_length", default=50, type=int)

    parser.add_argument("--memory_length", default=int(1e4), type=int)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--seed", default=120, type=int)

    parser.add_argument("--a_lr", default=0.0001, type=float)
    parser.add_argument("--c_lr", default=0.001, type=float)

    parser.add_argument("--lr_dynamics", default=0.001, type=float)
    parser.add_argument("--lr_reward", default=0.001, type=float)
    parser.add_argument("--lr_cost", default=0.001, type=float)

    parser.add_argument("--return_factor", default=15, type=float)
    parser.add_argument("--z_lowerbound", default=0, type=float)
    parser.add_argument("--z_min", default=0.0, type=float)
    parser.add_argument("--z_max", default=50.0, type=float)
    parser.add_argument("--noise_level", default=0.1, type=float)

    parser.add_argument("--plot_frequency", default=1000, type=int)
    parser.add_argument("--normal_factor", default=2, type=float)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--rnn_hidden_size", default=64, type=int)

    parser.add_argument("--ablation_hjb", default=False, type=bool)
    parser.add_argument("--ablation_target", default=False, type=bool)
    parser.add_argument("--ablation_vgi", default=False, type=bool)

    parser.add_argument("--render_flag", default=False, type=bool)
    parser.add_argument("--use-wandb", default=False, action="store_true")
    parser.add_argument("--project-name", default="EPI-CT-MARL", type=str)
    parser.add_argument("--exploration_steps", default=1000, type=int)

    parser.add_argument("--ou_theta", default=0.15, type=float)
    parser.add_argument("--ou_mu", default=0.0, type=float)
    parser.add_argument("--ou_sigma", default=0.2, type=float)
    parser.add_argument("--z_bias", default=0.2, type=float)

    parser.add_argument("--epsilon_decay", default=10000, type=int)

    parser.add_argument("--tensorboard", default=False, action="store_true")
    parser.add_argument("--ablation", default=False, action="store_true")
    parser.add_argument("--relu", default=False, action="store_true")

    parser.add_argument("--save_interval", default=3000, type=int)
    parser.add_argument(
        "--eval-interval",
        default=1000,
        type=int,
        help="Run eval every N training episodes; <=0 disables periodic eval.",
    )
    parser.add_argument("--eval-episodes", default=10, type=int)
    parser.add_argument(
        "--eval-video", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--eval-video-fps", default=15, type=int)
    parser.add_argument("--model_episode", default=300000, type=int)
    parser.add_argument("--episode_before_train", default=10, type=int)
    parser.add_argument("--log_dir", default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    args = parser.parse_args()

    main(args)
