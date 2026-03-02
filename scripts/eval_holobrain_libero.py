#!/usr/bin/env python3
"""
Evaluate HoloBrain Policy on LIBERO-10 benchmark.

Loads a trained checkpoint, runs rollouts in LIBERO simulation, reports
per-task and aggregate success rates.

Usage:
  source /.venv/bin/activate
  MUJOCO_GL=egl python3 eval_holobrain_libero.py \
      --checkpoint /share/holobrain_libero_100k/checkpoint-100000/policy.pt \
      --num_episodes 20

Environment variables:
  MUJOCO_GL=egl   (headless rendering, required on servers)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HoloBrain on LIBERO-10")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to policy.pt checkpoint",
    )
    parser.add_argument(
        "--dataset_repo_id", type=str, default="lerobot/libero_10_image",
        help="HuggingFace dataset repo_id (for normalization stats)",
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="Comma-separated task indices to evaluate (e.g. '7' or '0,3,7')")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Max environment steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results JSON")
    # HoloBrain model params (must match training)
    parser.add_argument("--embed_dims", type=int, default=256)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--pred_steps", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--single_camera", action="store_true")
    parser.add_argument("--train_task_id", type=int, default=None,
                        help="Dataset task_index used for training (for single-task normalization stats)")
    return parser.parse_args()


def _find_task_episodes(dataset, task_id):
    """Find episode indices for a specific task_index in the dataset."""
    import json as json_mod
    from pathlib import Path

    root = dataset.root if hasattr(dataset, 'root') else None
    task_name = None

    if root:
        tasks_file = Path(root) / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    obj = json_mod.loads(line)
                    if obj.get("task_index") == task_id:
                        task_name = obj["task"]
                        break

    if task_name is None:
        return []

    episode_indices = []
    if root:
        episodes_file = Path(root) / "meta" / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    obj = json_mod.loads(line)
                    if task_name in obj.get("tasks", []):
                        episode_indices.append(obj["episode_index"])

    return sorted(episode_indices)


def _compute_single_task_stats(dataset, episodes):
    """Compute normalization stats for a subset of episodes using parquet."""
    from pathlib import Path

    root = dataset.root if hasattr(dataset, 'root') else None
    if root is None:
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    root = Path(root)
    episode_set = set(episodes)
    state_data = []
    action_data = []

    data_dir = root / "data"
    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pq_file in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(pq_file, columns=["episode_index", "state", "actions"])
            ep_col = table.column("episode_index").to_pylist()
            for i, ep_idx in enumerate(ep_col):
                if ep_idx in episode_set:
                    state_data.append(torch.tensor(table.column("state")[i].as_py(), dtype=torch.float32))
                    action_data.append(torch.tensor(table.column("actions")[i].as_py(), dtype=torch.float32))

    if not state_data:
        return None

    state_t = torch.stack(state_data)
    action_t = torch.stack(action_data)

    stats = {}
    for key, tensor in [("state", state_t), ("actions", action_t)]:
        flat = tensor.reshape(-1, tensor.shape[-1])
        stats[key] = {
            "mean": flat.mean(dim=0),
            "std": flat.std(dim=0).clamp(min=1e-8),
            "min": flat.min(dim=0).values,
            "max": flat.max(dim=0).values,
        }

    global_stats = dataset.meta.stats if hasattr(dataset, "meta") else {}
    for k, v in global_stats.items():
        if k not in stats:
            stats[k] = v

    return stats


def build_policy(args, device):
    """Build HoloBrain policy and load checkpoint weights."""
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    except ImportError:
        from lerobot.common.policies.utils import PolicyFeature, FeatureType
        from lerobot.common.policies.normalize import NormalizationMode

    from lerobot.common.policies.holobrain.configuration_holobrain import HoloBrainConfig
    from lerobot.common.policies.holobrain.modeling_holobrain import HoloBrainPolicy

    image_shape = (3, 256, 256)
    state_shape = (8,)
    action_shape = (7,)

    input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
    }
    if not args.single_camera:
        input_features["observation.images.wrist_image"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=image_shape,
        )

    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape),
    }

    normalization_mapping = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }

    # Load dataset stats for normalization
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_kwargs = {"repo_id": args.dataset_repo_id}
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root

    dataset = LeRobotDataset(**ds_kwargs)
    dataset_stats = dataset.meta.stats if hasattr(dataset, "meta") else None

    # Single-task normalization: match what training script does
    if args.train_task_id is not None:
        from pathlib import Path
        import json as json_mod

        task_episodes = _find_task_episodes(dataset, args.train_task_id)
        if task_episodes:
            single_stats = _compute_single_task_stats(dataset, task_episodes)
            if single_stats:
                logging.info(f"Using single-task stats for train_task_id={args.train_task_id} "
                             f"({len(task_episodes)} episodes)")
                dataset_stats = single_stats
            else:
                logging.warning("Failed to compute single-task stats, using global stats")
        else:
            logging.warning(f"No episodes found for train_task_id={args.train_task_id}")

    # Key remapping for LIBERO v2.0 short names
    feat_keys = set(dataset.meta.features.keys())
    if "image" in feat_keys and "observation.images.image" not in feat_keys:
        batch_key_remap = {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
            "state": "observation.state",
            "actions": "action",
        }
        if dataset_stats:
            remapped_stats = {}
            for ds_key, std_key in batch_key_remap.items():
                if ds_key in dataset_stats:
                    remapped_stats[std_key] = dataset_stats[ds_key]
            for k, v in dataset_stats.items():
                if k not in batch_key_remap:
                    remapped_stats[k] = v
            dataset_stats = remapped_stats
    del dataset

    config = HoloBrainConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=normalization_mapping,
        embed_dims=args.embed_dims,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        action_dim=7,
        state_dim=8,
        pred_steps=args.pred_steps,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        device=device,
    )
    config.validate_features()

    policy = HoloBrainPolicy(config=config, dataset_stats=dataset_stats)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    num_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"Policy loaded: {num_params:,} params from {args.checkpoint}")

    return policy


def obs_to_batch(obs, device, single_camera=False):
    """Convert LIBERO observation dict to policy batch dict.

    IMPORTANT: LIBERO dataset state format is:
        eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8D
    NOT quaternion(4) + gripper(1). See 6a1 Loop 29 (Task 25/26) for details.
    """
    # LIBERO obs keys: agentview_image, robot0_eye_in_hand_image, robot0_eef_pos,
    # robot0_eef_quat, robot0_gripper_qpos, robot0_joint_pos, etc.

    # Image: (H, W, C) uint8 -> (1, C, H, W) float32 [0, 1]
    agentview = obs["agentview_image"]
    img = torch.from_numpy(agentview).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    batch = {"observation.images.image": img.to(device)}

    if not single_camera:
        wrist = obs["robot0_eye_in_hand_image"]
        wrist_t = torch.from_numpy(wrist).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        batch["observation.images.wrist_image"] = wrist_t.to(device)

    # Robot state: eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8D
    from scipy.spatial.transform import Rotation
    eef_pos = obs["robot0_eef_pos"]         # (3,)
    eef_quat = obs["robot0_eef_quat"]       # (4,) in wxyz format from robosuite
    gripper = obs["robot0_gripper_qpos"]     # (2,)

    # Convert quaternion to axis-angle using robosuite's convention
    # robosuite 1.4 uses xyzw format throughout, and quat2axisangle
    # produces non-negative angles (matching the LIBERO dataset convention)
    from robosuite.utils.transform_utils import quat2axisangle
    eef_axisangle = quat2axisangle(eef_quat.copy())  # .copy() to avoid modifying obs

    state = np.concatenate([eef_pos, eef_axisangle, gripper[:2]])  # (8,) = 3+3+2

    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    batch["observation.state"] = state_t

    return batch


def run_evaluation(policy, args, device):
    """Run LIBERO evaluation and return results."""
    from libero.libero import benchmark as libero_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bm = libero_benchmark.get_benchmark(args.benchmark)()
    num_tasks = bm.get_num_tasks()
    task_names = bm.get_task_names()

    logging.info(f"Benchmark: {args.benchmark}, {num_tasks} tasks")
    logging.info(f"Episodes per task: {args.num_episodes}, max steps: {args.max_steps}")

    results = {}
    all_successes = []

    # Filter to specific tasks if requested
    if args.task_ids:
        eval_task_indices = [int(x.strip()) for x in args.task_ids.split(",")]
        logging.info(f"Evaluating specific tasks: {eval_task_indices}")
    else:
        eval_task_indices = list(range(num_tasks))

    for task_idx in eval_task_indices:
        task_name = task_names[task_idx]
        bddl_file = bm.get_task_bddl_file_path(task_idx)
        init_states = bm.get_task_init_states(task_idx)

        logging.info(f"\n--- Task {task_idx}/{num_tasks}: {task_name} ---")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=args.camera_height,
            camera_widths=args.camera_width,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )

        task_successes = []
        task_steps = []

        for ep in range(args.num_episodes):
            # Reset env with specific init state
            init_idx = ep % len(init_states)
            env.reset()
            env.set_init_state(init_states[init_idx])
            obs = env.env._get_observations()

            policy.reset()
            done = False
            success = False

            for step in range(args.max_steps):
                batch = obs_to_batch(obs, device, args.single_camera)

                with torch.no_grad():
                    action = policy.select_action(batch)

                # action shape: (1, action_dim) or (action_dim,)
                if action.dim() > 1:
                    action = action.squeeze(0)
                action_np = action.cpu().numpy()

                obs, reward, done, info = env.step(action_np)

                success = env.check_success()
                if success:
                    break

            task_successes.append(int(success))
            task_steps.append(step + 1)

            status = "SUCCESS" if success else "FAIL"
            logging.info(f"  Ep {ep+1}/{args.num_episodes}: {status} ({step+1} steps)")

        env.close()

        sr = np.mean(task_successes)
        avg_steps = np.mean(task_steps)
        results[task_name] = {
            "success_rate": float(sr),
            "avg_steps": float(avg_steps),
            "successes": task_successes,
        }
        all_successes.extend(task_successes)

        logging.info(f"  Task SR: {sr:.1%} ({sum(task_successes)}/{len(task_successes)}), "
                     f"avg steps: {avg_steps:.0f}")

    overall_sr = np.mean(all_successes) if all_successes else 0.0
    results["_overall"] = {
        "success_rate": float(overall_sr),
        "num_tasks": len(eval_task_indices),
        "num_episodes_per_task": args.num_episodes,
        "total_episodes": len(all_successes),
        "total_successes": int(sum(all_successes)),
    }

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    logging.info("=" * 60)
    logging.info("  HoloBrain Policy Evaluation on LIBERO")
    logging.info("=" * 60)
    logging.info(f"Args: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    # Build policy
    policy = build_policy(args, device)

    # Run evaluation
    t0 = time.time()
    results = run_evaluation(policy, args, device)
    elapsed = time.time() - t0

    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("  EVALUATION RESULTS")
    logging.info("=" * 60)

    for task_name, task_res in results.items():
        if task_name.startswith("_"):
            continue
        short_name = task_name.split("_", 2)[-1][:60]
        logging.info(f"  {short_name:60s} SR={task_res['success_rate']:.1%}")

    overall = results["_overall"]
    logging.info("-" * 60)
    logging.info(f"  {'OVERALL':60s} SR={overall['success_rate']:.1%} "
                 f"({overall['total_successes']}/{overall['total_episodes']})")
    logging.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "eval_results.json"
    results["_meta"] = {
        "checkpoint": args.checkpoint,
        "benchmark": args.benchmark,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "elapsed_seconds": elapsed,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
