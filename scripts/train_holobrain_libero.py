#!/usr/bin/env python3
"""
HoloBrain Policy training on LIBERO dataset for LeRobot 0.1.0 (pi0-fast:v1.0 image).

Based on train_dp_libero.py, adapted for HoloBrainConfig/Policy.

Usage:
  source /.venv/bin/activate
  python3 train_holobrain_libero.py
  python3 train_holobrain_libero.py --steps 10000 --batch_size 16 --output_dir /share/holobrain_libero_output

Environment variables:
  HF_ENDPOINT=https://hf-mirror.com   (for China mainland)
  MUJOCO_GL=egl                        (headless rendering)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Environment setup (before any torch import)
# ---------------------------------------------------------------------------
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HoloBrain Policy on LIBERO")
    parser.add_argument(
        "--dataset_repo_id", type=str,
        default="lerobot/libero_10_image",
        help="HuggingFace dataset repo_id",
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=None,
                        help="If set, train for N epochs (overrides --steps)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--save_freq_epochs", type=int, default=None,
                        help="Save checkpoint every N epochs (overrides --save_freq)")
    parser.add_argument("--log_freq", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--single_camera", action="store_true")
    parser.add_argument("--pretrained_backbone", action="store_true")
    # Single-task training
    parser.add_argument("--task_id", type=int, default=None,
                        help="Train on single task by dataset task_index")
    # HoloBrain-specific
    parser.add_argument("--pred_steps", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--embed_dims", type=int, default=256)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="mse",
                        help="Loss type: mse or smooth_l1")
    parser.add_argument("--state_noise_std", type=float, default=0.0,
                        help="Gaussian noise std for state augmentation")
    parser.add_argument("--use_urdf", action="store_true",
                        help="Enable URDF-based joint graph attention (Phase 4)")
    parser.add_argument("--num_joints", type=int, default=7,
                        help="Number of joints for URDF mode")
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="holobrain-libero")
    parser.add_argument("--feishu_notify", action="store_true")
    parser.add_argument("--feishu_app_id", type=str, default=None)
    parser.add_argument("--feishu_app_secret", type=str, default=None)
    parser.add_argument(
        "--feishu_chat_id", type=str,
        default="oc_7a337b608f93a7fab2448667151ee208",
    )
    return parser.parse_args()


def find_task_episodes(dataset, task_id):
    """Find episode indices for a specific task_index in the dataset.

    Uses metadata files (tasks.jsonl + episodes.jsonl) for fast lookup
    instead of iterating over all frames in hf_dataset.
    """
    import json

    # Step 1: Get task name from tasks.jsonl
    root = dataset.root if hasattr(dataset, 'root') else None
    task_name = None

    if root:
        tasks_file = Path(root) / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("task_index") == task_id:
                        task_name = obj["task"]
                        break

    if task_name is None:
        # Fallback: scan hf_dataset (slow)
        logging.warning(f"tasks.jsonl not found, scanning hf_dataset for task_id={task_id}...")
        hf_dataset = dataset.hf_dataset
        episode_indices = set()
        for row in hf_dataset:
            if row.get("task_index") == task_id:
                episode_indices.add(row["episode_index"])
        return sorted(episode_indices)

    logging.info(f"Task {task_id}: '{task_name}'")

    # Step 2: Find episodes with matching task name from episodes.jsonl
    episode_indices = []
    if root:
        episodes_file = Path(root) / "meta" / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    obj = json.loads(line)
                    tasks = obj.get("tasks", [])
                    if task_name in tasks:
                        episode_indices.append(obj["episode_index"])

    if not episode_indices:
        # Fallback: scan hf_dataset
        logging.warning("episodes.jsonl match failed, scanning hf_dataset...")
        hf_dataset = dataset.hf_dataset
        ep_set = set()
        for row in hf_dataset:
            if row.get("task_index") == task_id:
                ep_set.add(row["episode_index"])
        episode_indices = sorted(ep_set)

    return sorted(episode_indices)


def compute_single_task_stats(dataset, episodes):
    """Compute normalization stats for a subset of episodes using parquet.

    Reads state/action columns directly from parquet files for speed.
    Images use global stats (MEAN_STD normalization is task-independent).
    """
    import torch
    root = dataset.root if hasattr(dataset, 'root') else None
    if root is None:
        logging.warning("No dataset root, using global stats")
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logging.warning("pyarrow not available, using global stats")
        return None

    root = Path(root)
    episode_set = set(episodes)
    state_data = []
    action_data = []

    # Read parquet files directly (much faster than hf_dataset iteration)
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
        logging.warning("No data found for specified episodes")
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

    # Copy global image stats from dataset
    global_stats = dataset.meta.stats if hasattr(dataset, "meta") else {}
    for k, v in global_stats.items():
        if k not in stats:
            stats[k] = v

    return stats


def send_feishu_notification(app_id, app_secret, chat_id, message):
    """Send a notification to Feishu group chat."""
    import json
    import urllib.request

    token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    token_data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
    token_req = urllib.request.Request(
        token_url, data=token_data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(token_req, timeout=10) as resp:
            access_token = json.loads(resp.read()).get("tenant_access_token", "")
    except Exception as e:
        logging.warning(f"Failed to get Feishu token: {e}")
        return

    msg_url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    msg_data = json.dumps({
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": message}),
    }).encode()
    msg_req = urllib.request.Request(
        msg_url, data=msg_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
    )
    try:
        with urllib.request.urlopen(msg_req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 0:
                logging.info("Feishu notification sent")
            else:
                logging.warning(f"Feishu notification failed: {result}")
    except Exception as e:
        logging.warning(f"Failed to send Feishu notification: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    logging.info("=" * 60)
    logging.info("  HoloBrain Policy Training on LIBERO")
    logging.info("=" * 60)
    logging.info(f"Args: {vars(args)}")

    import torch

    # Import types - try multiple paths
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    except ImportError:
        from lerobot.common.policies.normalize import NormalizationMode
        from lerobot.common.policies.utils import PolicyFeature, FeatureType

    # Import HoloBrain
    from lerobot.common.policies.holobrain.configuration_holobrain import HoloBrainConfig
    from lerobot.common.policies.holobrain.modeling_holobrain import HoloBrainPolicy

    logging.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # 1. Detect dataset features & build key remap
    # -----------------------------------------------------------------------
    image_shape = (3, 256, 256)
    state_shape = (8,)
    action_shape = (7,)
    batch_key_remap = {}

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_kwargs = {"repo_id": args.dataset_repo_id}
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root

    _tmp_ds = LeRobotDataset(**ds_kwargs)
    _feat_keys = set(_tmp_ds.meta.features.keys())
    logging.info(f"Dataset feature keys: {_feat_keys}")

    if "image" in _feat_keys and "observation.images.image" not in _feat_keys:
        batch_key_remap = {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
            "state": "observation.state",
            "actions": "action",
            "actions_is_pad": "action_is_pad",
            "image_is_pad": "observation.images.image_is_pad",
            "wrist_image_is_pad": "observation.images.wrist_image_is_pad",
            "state_is_pad": "observation.state_is_pad",
        }
        logging.info(f"Will remap batch keys: {batch_key_remap}")

    # Single-task episode filtering
    task_episodes = None
    if args.task_id is not None:
        task_episodes = find_task_episodes(_tmp_ds, args.task_id)
        logging.info(f"Single-task mode: task_id={args.task_id}, "
                     f"found {len(task_episodes)} episodes")
        if not task_episodes:
            raise ValueError(f"No episodes found for task_id={args.task_id}")

    _fps = _tmp_ds.fps
    _dt = 1.0 / _fps
    logging.info(f"Dataset FPS: {_fps}")

    # -----------------------------------------------------------------------
    # 2. Build config
    # -----------------------------------------------------------------------
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

    config = HoloBrainConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=normalization_mapping,
        # Architecture
        embed_dims=args.embed_dims,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        # Action space
        action_dim=7,
        state_dim=8,
        pred_steps=args.pred_steps,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        # Vision
        pretrained_backbone_weights="IMAGENET1K_V1" if args.pretrained_backbone else None,
        use_group_norm=not args.pretrained_backbone,
        # Training
        lr=args.lr,
        device=device,
        loss_type=args.loss_type,
        use_urdf=args.use_urdf,
        num_joints=args.num_joints,
    )
    config.validate_features()

    logging.info(f"HoloBrainConfig created")
    logging.info(f"  embed_dims={config.embed_dims}, decoder_layers={config.num_decoder_layers}")
    logging.info(f"  pred_steps={config.pred_steps}, chunk_size={config.chunk_size}")
    logging.info(f"  n_action_steps={config.n_action_steps}")
    logging.info(f"  image_features: {list(config.image_features.keys())}")

    # -----------------------------------------------------------------------
    # 3. Load dataset with delta_timestamps
    # -----------------------------------------------------------------------
    # HoloBrain uses n_obs_steps=1, action horizon = pred_steps
    n_obs_steps = config.n_obs_steps  # 1
    horizon = config.pred_steps       # 64

    obs_ts = [i * _dt for i in range(1 - n_obs_steps, 1)]  # [0.0] for n_obs_steps=1
    action_ts = [i * _dt for i in range(horizon)]           # [0.0, 0.1, ..., 6.3]

    delta_timestamps = {}
    for k in _tmp_ds.meta.features.keys():
        feat = _tmp_ds.meta.features[k]
        if feat.get("dtype") == "image":
            delta_timestamps[k] = obs_ts
        elif k in ("state", "observation.state"):
            delta_timestamps[k] = obs_ts
        elif k in ("actions", "action"):
            delta_timestamps[k] = action_ts

    logging.info(f"delta_timestamps keys: {list(delta_timestamps.keys())}")

    # Compute single-task stats before deleting _tmp_ds
    single_task_stats = None
    if task_episodes is not None:
        logging.info("Computing single-task normalization stats (parquet)...")
        single_task_stats = compute_single_task_stats(_tmp_ds, task_episodes)
        if single_task_stats:
            logging.info(f"Single-task stats keys: {list(single_task_stats.keys())}")
        else:
            logging.info("Using global dataset stats (single-task stats unavailable)")

    del _tmp_ds

    ds_kwargs["delta_timestamps"] = delta_timestamps
    if task_episodes is not None:
        ds_kwargs["episodes"] = task_episodes
    dataset = LeRobotDataset(**ds_kwargs)

    # Patch episode remap for single-task episode filtering
    # (fixes IndexError when episodes= is used with LeRobotDataset)
    if task_episodes is not None and hasattr(dataset, 'episode_data_index'):
        # The filtered dataset has episode_data_index with 0-based indices,
        # but hf_dataset rows retain original episode_index values (e.g., 388).
        # Patch _get_query_indices to remap ep_idx before array lookup.
        ep_remap = {ep: i for i, ep in enumerate(task_episodes)}
        _orig_get_query_indices = dataset._get_query_indices

        def _patched_get_query_indices(idx, ep_idx):
            mapped = ep_remap.get(ep_idx, ep_idx)
            return _orig_get_query_indices(idx, mapped)

        dataset._get_query_indices = _patched_get_query_indices
        logging.info(f"Patched _get_query_indices with episode remap: "
                     f"{len(ep_remap)} episodes")

    logging.info(f"Dataset: {dataset.num_frames} frames, {dataset.num_episodes} episodes")

    # -----------------------------------------------------------------------
    # 4. Create policy
    # -----------------------------------------------------------------------
    dataset_stats = dataset.meta.stats if hasattr(dataset, "meta") else None

    # Use single-task stats if available (overrides dataset-level mixed stats)
    if single_task_stats is not None:
        dataset_stats = single_task_stats
        logging.info("Using single-task normalization stats")

    if dataset_stats and batch_key_remap:
        remapped_stats = {}
        for ds_key, std_key in batch_key_remap.items():
            if ds_key in dataset_stats:
                remapped_stats[std_key] = dataset_stats[ds_key]
        for k, v in dataset_stats.items():
            if k not in batch_key_remap:
                remapped_stats[k] = v
        dataset_stats = remapped_stats
        logging.info(f"Remapped stats keys: {list(dataset_stats.keys())}")

    policy = HoloBrainPolicy(config=config, dataset_stats=dataset_stats)
    policy.to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"Policy: {num_params:,} total params, {num_trainable:,} trainable")

    # -----------------------------------------------------------------------
    # 5. Optimizer and scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.95, 0.999),
    )

    # -----------------------------------------------------------------------
    # 6. DataLoader
    # -----------------------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    logging.info(f"DataLoader: batch_size={args.batch_size}, num_workers={args.num_workers}")

    # -----------------------------------------------------------------------
    # 7. Output directory
    # -----------------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        import datetime
        now = datetime.datetime.now()
        output_dir = Path(f"outputs/train/{now:%Y-%m-%d}/{now:%H-%M-%S}_holobrain_libero")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # -----------------------------------------------------------------------
    # 8. WandB
    # -----------------------------------------------------------------------
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            logging.info("WandB initialized")
        except Exception as e:
            logging.warning(f"WandB init failed: {e}")
            args.wandb = False

    # -----------------------------------------------------------------------
    # 9. Training loop
    # -----------------------------------------------------------------------
    # Compute total steps
    if args.epochs is not None:
        steps_per_epoch = len(dataloader)
        total_steps = args.epochs * steps_per_epoch
        logging.info(f"Epoch mode: {args.epochs} epochs × {steps_per_epoch} steps/epoch "
                     f"= {total_steps} total steps")
    else:
        total_steps = args.steps
        steps_per_epoch = len(dataloader)

    # Recompute lr scheduler with correct total steps
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    logging.info(f"Starting training: {total_steps} steps")

    def cycle(dl):
        while True:
            for batch in dl:
                yield batch

    dl_iter = cycle(dataloader)
    policy.train()

    best_loss = float("inf")
    start_time = time.time()
    loss_sum = 0.0
    loss_count = 0

    for step in range(1, total_steps + 1):
        batch = next(dl_iter)

        # Move to device
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        # Remap batch keys
        if batch_key_remap:
            remapped = {}
            for k, v in batch.items():
                remapped[batch_key_remap.get(k, k)] = v
            batch = remapped

        # Ensure action_is_pad exists
        if "action_is_pad" not in batch and "action" in batch:
            batch["action_is_pad"] = torch.zeros(
                batch["action"].shape[:-1], dtype=torch.bool, device=device,
            )

        # State augmentation: add Gaussian noise for robustness
        if args.state_noise_std > 0 and "observation.state" in batch:
            state_noise = torch.randn_like(batch["observation.state"]) * args.state_noise_std
            batch["observation.state"] = batch["observation.state"] + state_noise

        # Forward
        loss, _ = policy.forward(batch)

        # Backward
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), config.grad_clip_max_norm
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        loss_val = loss.item()
        loss_sum += loss_val
        loss_count += 1

        # Logging
        if step % args.log_freq == 0:
            avg_loss = loss_sum / loss_count
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            lr_cur = optimizer.param_groups[0]["lr"]
            logging.info(
                f"step:{step:>6d} | loss:{avg_loss:.4f} | "
                f"grad:{grad_norm:.3f} | lr:{lr_cur:.1e} | "
                f"speed:{sps:.1f} steps/s | elapsed:{elapsed:.0f}s"
            )
            if args.wandb:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                    "train/lr": lr_cur,
                }, step=step)
            loss_sum = 0.0
            loss_count = 0

        # Save checkpoint
        save_by_step = (args.save_freq_epochs is None and
                        step % args.save_freq == 0)
        save_by_epoch = (args.save_freq_epochs is not None and
                         steps_per_epoch > 0 and
                         step % steps_per_epoch == 0 and
                         (step // steps_per_epoch) % args.save_freq_epochs == 0)
        save_final = (step == total_steps)

        if save_by_step or save_by_epoch or save_final:
            epoch_num = step // steps_per_epoch if steps_per_epoch > 0 else 0
            ckpt_dir = output_dir / f"checkpoint-{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), ckpt_dir / "policy.pt")
            torch.save({
                "step": step,
                "epoch": epoch_num,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }, ckpt_dir / "training_state.pt")
            logging.info(f"Checkpoint saved: {ckpt_dir} (epoch {epoch_num})")

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - start_time
    task_info = f"task_id={args.task_id}" if args.task_id is not None else "all tasks"
    epoch_info = f"{args.epochs} epochs" if args.epochs else f"{total_steps} steps"
    summary = (
        f"Training completed!\n"
        f"  Mode: {task_info}, {epoch_info}\n"
        f"  Steps: {total_steps}\n"
        f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)\n"
        f"  Avg speed: {total_steps/total_time:.1f} steps/s\n"
        f"  Output: {output_dir}\n"
        f"  Dataset: {args.dataset_repo_id}\n"
        f"  Batch size: {args.batch_size}\n"
        f"  Parameters: {num_trainable:,}\n"
        f"  Architecture: HoloBrain (embed={config.embed_dims}, "
        f"dec_layers={config.num_decoder_layers}, pred_steps={config.pred_steps})"
    )
    logging.info(summary)

    if args.wandb:
        import wandb
        wandb.finish()

    if args.feishu_notify and args.feishu_app_id and args.feishu_app_secret:
        send_feishu_notification(
            args.feishu_app_id, args.feishu_app_secret, args.feishu_chat_id,
            f"[HoloBrain LIBERO] {summary}",
        )

    return output_dir


if __name__ == "__main__":
    main()
