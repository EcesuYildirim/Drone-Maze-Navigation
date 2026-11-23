import os
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env import DroneMazeEnv


def _load_model(model_path: str):
    """PPO veya RecurrentPPO'yu otomatik algıla ve yükle."""
    try:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(model_path)
    except Exception:
        return PPO.load(model_path)


def load_config(path: str = "config/ppo_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(
    model_path: str,
    layout: str = "medium",
    episodes: int = 100,
    gui: bool = False,
    render_delay: float = 0.0,
    env_cfg: dict = None,
) -> dict:
    """
    Kaydedilen modeli yükler ve `episodes` bölüm çalıştırır.
    Döndürdüğü dict:
      success_rate, collision_rate, timeout_rate,
      mean_reward, std_reward, mean_ep_length
    """
    cfg = env_cfg or {}
    model = _load_model(model_path)
    env   = DroneMazeEnv(
        layout                 = layout,
        random_goal            = False,
        gui                    = gui,
        approach_scale         = cfg.get("approach_scale"),
        altitude_penalty_scale = cfg.get("altitude_penalty_scale"),
        hover_bonus_scale      = cfg.get("hover_bonus_scale"),
        dist_penalty_scale     = cfg.get("dist_penalty_scale"),
        collision_penalty      = cfg.get("collision_penalty"),
        goal_reward            = cfg.get("goal_reward"),
        step_penalty           = cfg.get("step_penalty"),
        max_steps              = cfg.get("max_steps"),
    )

    rewards, lengths = [], []
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    is_recurrent = hasattr(model, "policy") and hasattr(model.policy, "lstm_actor")

    for ep in range(episodes):
        obs, _ = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_len     = 0
        terminated = False
        truncated  = False
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_start,
                    deterministic=True)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len    += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(ep_len)

        # Öncelik: hedef > çarpışma > timeout
        dist = info.get("distance_to_goal", np.inf)
        if dist < env.GOAL_RADIUS:
            outcomes["success"] += 1
        elif terminated:
            outcomes["collision"] += 1
        else:
            outcomes["timeout"] += 1

        if (ep + 1) % 10 == 0:
            print(f"  Bölüm {ep+1:>3}/{episodes}  reward={ep_reward:7.2f}  len={ep_len}")

    env.close()

    n = episodes
    results = {
        "success_rate":   outcomes["success"]   / n,
        "collision_rate": outcomes["collision"]  / n,
        "timeout_rate":   outcomes["timeout"]    / n,
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
    }
    return results


def compare_seeds(cfg: dict, episodes: int = 100, model_tag: str = None) -> None:
    """Seed bazlı model karşılaştırması. model_tag verilirse curriculum isimlerini arar."""
    seeds     = cfg["ppo"]["seeds"]
    model_dir = cfg["training"]["model_dir"]
    env_cfg   = cfg.get("environment", {})

    all_results = {}
    for seed in seeds:
        tag  = f"{model_tag}_seed{seed}" if model_tag else f"seed{seed}"
        path = os.path.join(model_dir, f"best_model_{tag}")
        if not os.path.exists(path + ".zip"):
            print(f"[UYARI] Model bulunamadi: {path}.zip -- atlaniyor")
            continue
        print(f"\n--- {tag} degerlendiriliyor ---")
        all_results[tag] = evaluate_model(path, episodes=episodes, env_cfg=env_cfg)

    if not all_results:
        print("Değerlendirilecek model yok.")
        return

    _print_table(all_results)
    _plot_results(all_results)


def _print_table(results: dict) -> None:
    header = f"{'Seed':>6}  {'Başarı':>8}  {'Çarpışma':>10}  {'Timeout':>8}  {'Ort.Ödül':>10}  {'Std':>7}  {'Ort.Uzunluk':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for seed, r in results.items():
        print(
            f"{seed:>6}  "
            f"{r['success_rate']:>7.1%}  "
            f"{r['collision_rate']:>9.1%}  "
            f"{r['timeout_rate']:>7.1%}  "
            f"{r['mean_reward']:>10.2f}  "
            f"{r['std_reward']:>7.2f}  "
            f"{r['mean_ep_length']:>12.1f}"
        )
    print("=" * len(header))


def _plot_results(results: dict) -> None:
    seeds        = list(results.keys())
    success      = [results[s]["success_rate"]   * 100 for s in seeds]
    mean_rewards = [results[s]["mean_reward"]         for s in seeds]
    std_rewards  = [results[s]["std_reward"]          for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar([str(s) for s in seeds], success, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Başarı Oranı (%)")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 100)

    axes[1].bar([str(s) for s in seeds], mean_rewards,
                yerr=std_rewards, capsize=5,
                color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("Ortalama Episodik Ödül (± std)")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Ödül")

    plt.tight_layout()
    out_path = "results/evaluation_comparison.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nGrafik kaydedildi: {out_path}")
    plt.show()


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="İHA Labirent — Model Değerlendirme")
    parser.add_argument("--model",     type=str,  default=None,     help="Tek model dosya yolu (.zip olmadan)")
    parser.add_argument("--episodes",  type=int,  default=100,      help="Değerlendirme bölümü sayısı")
    parser.add_argument("--gui",       action="store_true",         help="PyBullet GUI aç")
    parser.add_argument("--compare",   action="store_true",         help="Seed karşılaştırması")
    parser.add_argument("--model-tag", type=str,  default=None,     help="compare modunda model etiketi (örn: modelA)")
    parser.add_argument("--layout",    type=str,  default="medium", help="Labirent tipi: easy veya medium (varsayılan: medium)")
    parser.add_argument("--config",    type=str,  default="config/ppo_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.compare:
        compare_seeds(cfg, episodes=args.episodes, model_tag=args.model_tag)
    elif args.model:
        env_cfg = cfg.get("environment", {})
        print(f"\nModel degerlendiriliyor: {args.model}  (layout={args.layout})")
        r = evaluate_model(args.model, layout=args.layout, episodes=args.episodes, gui=args.gui, env_cfg=env_cfg)
        _print_table({args.model.split("/")[-1]: r})
    else:
        parser.print_help()
