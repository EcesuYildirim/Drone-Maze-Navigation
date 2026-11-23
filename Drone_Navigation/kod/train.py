import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env import DroneMazeEnv
from utils import EvalAndSaveCallback
from utils.callbacks import CURRICULUM_STAGES


def _get_model_class(policy_name: str):
    """MlpLstmPolicy → RecurrentPPO, diğerleri → PPO."""
    if policy_name == "MlpLstmPolicy":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    return PPO


def load_config(path: str = "config/ppo_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(layout: str = "medium", random_goal: bool = True,
             env_cfg: dict = None, monitor_path: str = None) -> Monitor:
    cfg = env_cfg or {}
    env = DroneMazeEnv(
        layout                 = layout,
        random_goal            = random_goal,
        gui                    = False,
        approach_scale         = cfg.get("approach_scale"),
        altitude_penalty_scale = cfg.get("altitude_penalty_scale"),
        hover_bonus_scale      = cfg.get("hover_bonus_scale"),
        dist_penalty_scale     = cfg.get("dist_penalty_scale"),
        collision_penalty      = cfg.get("collision_penalty"),
        goal_reward            = cfg.get("goal_reward"),
        step_penalty           = cfg.get("step_penalty"),
        max_steps              = cfg.get("max_steps"),
    )
    return Monitor(env, filename=monitor_path)


def train_single(cfg: dict, seed: int) -> None:
    ppo     = cfg["ppo"]
    tr      = cfg["training"]
    env_cfg = cfg.get("environment", {})

    log_dir   = os.path.join(tr["log_dir"],   f"seed_{seed}")
    model_dir = tr["model_dir"]
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Eğitim başlıyor — seed={seed}")
    print(f"{'='*55}")

    train_env = make_env(random_goal=True,  env_cfg=env_cfg,
                         monitor_path=os.path.join(log_dir, "monitor_train"))
    eval_env  = make_env(random_goal=False, env_cfg=env_cfg,
                         monitor_path=os.path.join(log_dir, "monitor_eval"))

    callback = EvalAndSaveCallback(
        eval_env      = eval_env,
        eval_freq     = tr["eval_freq"],
        eval_episodes = tr["eval_episodes"],
        save_path     = model_dir,
        seed          = seed,
        model_tag     = f"seed{seed}",
        verbose       = 1,
    )

    model = PPO(
        policy           = "MlpPolicy",
        env              = train_env,
        learning_rate    = ppo["learning_rate"],
        n_steps          = ppo["n_steps"],
        batch_size       = ppo["batch_size"],
        n_epochs         = ppo["n_epochs"],
        gamma            = ppo["gamma"],
        gae_lambda       = ppo["gae_lambda"],
        clip_range       = ppo["clip_range"],
        ent_coef         = ppo["entropy_coef"],
        vf_coef          = ppo["vf_coef"],
        max_grad_norm    = ppo["max_grad_norm"],
        tensorboard_log  = log_dir,
        seed             = seed,
        verbose          = 1,
    )

    model.learn(
        total_timesteps  = tr["total_timesteps"],
        callback         = callback,
        tb_log_name      = f"PPO_seed{seed}",
        reset_num_timesteps = True,
    )

    final_path = os.path.join(model_dir, f"final_model_seed{seed}")
    model.save(final_path)
    print(f"\n  Final model kaydedildi: {final_path}.zip")

    train_env.close()
    eval_env.close()


def train_curriculum(cfg: dict, seed: int, stages: list = None, model_tag: str = "curriculum") -> None:
    """
    Curriculum Learning ile eğitim.

    model_tag  : kayıt dosyası öneki (örn. 'modelA', 'modelB')
    """
    ppo  = cfg["ppo"]
    tr   = cfg["training"]

    log_dir   = os.path.join(tr["log_dir"],   f"{model_tag}_seed_{seed}")
    model_dir = tr["model_dir"]
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    curriculum = stages or CURRICULUM_STAGES
    first_stage = curriculum[0]

    print(f"\n{'='*55}")
    print(f"  Curriculum eğitim başlıyor — tag={model_tag}, seed={seed}")
    print(f"  {len(curriculum)} aşama: {[s['layout'] for s in curriculum]}")
    print(f"{'='*55}")

    env_cfg     = cfg.get("environment", {})
    policy_name = ppo.get("policy", "MlpPolicy")
    ModelClass  = _get_model_class(policy_name)

    def _make_env(layout: str, monitor_path: str = None) -> Monitor:
        return Monitor(DroneMazeEnv(
            layout                 = layout,
            random_goal            = False,
            gui                    = False,
            approach_scale         = env_cfg.get("approach_scale"),
            altitude_penalty_scale = env_cfg.get("altitude_penalty_scale"),
            hover_bonus_scale      = env_cfg.get("hover_bonus_scale"),
            dist_penalty_scale     = env_cfg.get("dist_penalty_scale"),
            collision_penalty      = env_cfg.get("collision_penalty"),
            goal_reward            = env_cfg.get("goal_reward"),
            step_penalty           = env_cfg.get("step_penalty"),
            max_steps              = env_cfg.get("max_steps"),
        ), filename=monitor_path)

    train_env = _make_env(first_stage["layout"],
                          monitor_path=os.path.join(log_dir, "monitor_train"))
    eval_env  = _make_env(first_stage["layout"],
                          monitor_path=os.path.join(log_dir, "monitor_eval"))

    callback = EvalAndSaveCallback(
        eval_env                     = eval_env,
        eval_freq                    = tr["eval_freq"],
        eval_episodes                = tr.get("eval_episodes", 20),
        save_path                    = model_dir,
        seed                         = seed,
        verbose                      = 1,
        model_tag                    = f"{model_tag}_seed{seed}",
        curriculum_stages            = curriculum,
        curriculum_success_threshold = cfg.get("curriculum", {}).get("success_threshold", 0.5),
    )

    policy_kwargs = {}
    if "lstm_hidden_size" in ppo:
        policy_kwargs["lstm_hidden_size"] = ppo["lstm_hidden_size"]

    model = ModelClass(
        policy           = policy_name,
        env              = train_env,
        learning_rate    = ppo["learning_rate"],
        n_steps          = ppo["n_steps"],
        batch_size       = ppo["batch_size"],
        n_epochs         = ppo["n_epochs"],
        gamma            = ppo["gamma"],
        gae_lambda       = ppo["gae_lambda"],
        clip_range       = ppo["clip_range"],
        ent_coef         = ppo["entropy_coef"],
        vf_coef          = ppo["vf_coef"],
        max_grad_norm    = ppo["max_grad_norm"],
        tensorboard_log  = log_dir,
        seed             = seed,
        verbose          = 1,
        **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
    )

    model.learn(
        total_timesteps     = tr["total_timesteps"],
        callback            = callback,
        tb_log_name         = f"PPO_{model_tag}_seed{seed}",
        reset_num_timesteps = True,
    )

    final_path = os.path.join(model_dir, f"{model_tag}_final_seed{seed}")
    model.save(final_path)
    print(f"\n  Final model kaydedildi: {final_path}.zip")

    train_env.close()
    eval_env.close()


def train_lstm(cfg: dict, seed: int, model_tag: str = "modelLSTM") -> None:
    """
    RecurrentPPO (MlpLstmPolicy) ile eğitim.
    Policy adımlar arası hidden state taşır — labirent hafızası.
    """
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        raise ImportError("pip install sb3-contrib")

    ppo = cfg["ppo"]
    tr  = cfg["training"]

    log_dir   = os.path.join(tr["log_dir"], f"{model_tag}_seed_{seed}")
    model_dir = tr["model_dir"]
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    curriculum = CURRICULUM_STAGES
    first_stage = curriculum[0]

    print(f"\n{'='*55}")
    print(f"  RecurrentPPO eğitim — tag={model_tag}, seed={seed}")
    print(f"  LSTM hidden_size={ppo.get('lstm_hidden_size', 256)}")
    print(f"{'='*55}")

    train_env = Monitor(DroneMazeEnv(layout=first_stage["layout"], random_goal=False, gui=False))
    eval_env  = Monitor(DroneMazeEnv(layout=first_stage["layout"], random_goal=False, gui=False))

    callback = EvalAndSaveCallback(
        eval_env                     = eval_env,
        eval_freq                    = tr["eval_freq"],
        eval_episodes                = tr.get("eval_episodes", 20),
        save_path                    = model_dir,
        seed                         = seed,
        model_tag                    = model_tag,
        verbose                      = 1,
        curriculum_stages            = curriculum,
        curriculum_success_threshold = cfg.get("curriculum", {}).get("success_threshold", 0.5),
    )

    model = RecurrentPPO(
        policy       = "MlpLstmPolicy",
        env          = train_env,
        learning_rate = ppo["learning_rate"],
        n_steps      = ppo["n_steps"],
        batch_size   = ppo["batch_size"],
        n_epochs     = ppo["n_epochs"],
        gamma        = ppo["gamma"],
        gae_lambda   = ppo["gae_lambda"],
        clip_range   = ppo["clip_range"],
        ent_coef     = ppo["entropy_coef"],
        vf_coef      = ppo["vf_coef"],
        max_grad_norm = ppo["max_grad_norm"],
        policy_kwargs = {"lstm_hidden_size": ppo.get("lstm_hidden_size", 256),
                         "n_lstm_layers": 1},
        tensorboard_log = log_dir,
        seed         = seed,
        verbose      = 1,
    )

    model.learn(
        total_timesteps     = tr["total_timesteps"],
        callback            = callback,
        tb_log_name         = f"RecurrentPPO_{model_tag}_seed{seed}",
        reset_num_timesteps = True,
    )

    final_path = os.path.join(model_dir, f"{model_tag}_final_seed{seed}")
    model.save(final_path)
    print(f"\n  LSTM final model kaydedildi: {final_path}.zip")

    train_env.close()
    eval_env.close()


def train_comparison(
    config_a: str = "config/ppo_config_A.yaml",
    config_b: str = "config/ppo_config_B.yaml",
) -> None:
    """
    Model A (RL Zoo Dengeli) ve Model B (Yüksek Keşif) sırayla eğitir.
    Her iki model de seed=42, 3M adım, curriculum learning.
    Sonuçlar TensorBoard'da karşılaştırılabilir.
    """
    cfg_a = load_config(config_a)
    cfg_b = load_config(config_b)

    seed = cfg_a["ppo"]["seeds"][0]   # her iki config da [42] kullanır

    print("\n" + "=" * 55)
    print("  KARŞILAŞTIRMA EĞİTİMİ")
    print(f"  Model A: {config_a}")
    print(f"  Model B: {config_b}")
    print(f"  Seed: {seed}  |  3M adım  |  Curriculum: 6 aşama")
    print("=" * 55)

    print("\n>>> Model A başlıyor (RL Zoo Dengeli)...")
    train_curriculum(cfg_a, seed, model_tag="modelA")

    print("\n>>> Model B başlıyor (Yüksek Keşif)...")
    train_curriculum(cfg_b, seed, model_tag="modelB")

    log_dir = cfg_a["training"]["log_dir"]
    print("\n" + "=" * 55)
    print("  Her iki model tamamlandı.")
    print("  TensorBoard karşılaştırması için:")
    print(f"  tensorboard --logdir {log_dir}")
    print("=" * 55)


def train_all(config_path: str = "config/ppo_config.yaml") -> None:
    cfg   = load_config(config_path)
    seeds = cfg["ppo"]["seeds"]

    print(f"Toplam {len(seeds)} seed için eğitim başlıyor: {seeds}")
    for seed in seeds:
        train_single(cfg, seed)

    print("\nTüm eğitimler tamamlandı.")
    print("TensorBoard görselleştirmek için:")
    print(f"  tensorboard --logdir {cfg['training']['log_dir']}")


def train_all_curriculum(config_path: str = "config/ppo_config.yaml") -> None:
    cfg   = load_config(config_path)
    seeds = cfg["ppo"]["seeds"]

    print(f"Curriculum eğitim — {len(seeds)} seed: {seeds}")
    for seed in seeds:
        train_curriculum(cfg, seed)

    print("\nTüm curriculum eğitimler tamamlandı.")
    print("TensorBoard görselleştirmek için:")
    print(f"  tensorboard --logdir {cfg['training']['log_dir']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["A", "B", "C", "D", "E", "both"],
        default="both",
        help="Hangi model eğitilsin: A, B, C, D, E ya da both (varsayılan: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Tek seed çalıştır (paralel eğitim için). Verilmezse config'deki tüm seedler sırayla.",
    )
    args = parser.parse_args()

    MODEL_CONFIGS = {
        "A": ("config/ppo_config_A.yaml", "modelA"),
        "B": ("config/ppo_config_B.yaml", "modelB"),
        "C": ("config/ppo_config_C.yaml", "modelC"),
        "D": ("config/ppo_config_D.yaml", "modelD"),
        "E": ("config/ppo_config_E.yaml", "modelE"),
    }

    if args.model in MODEL_CONFIGS:
        config_path, tag = MODEL_CONFIGS[args.model]
        cfg = load_config(config_path)
        seeds = [args.seed] if args.seed is not None else cfg["ppo"]["seeds"]
        for seed in seeds:
            train_curriculum(cfg, seed=seed, model_tag=tag)
    else:
        train_comparison()
