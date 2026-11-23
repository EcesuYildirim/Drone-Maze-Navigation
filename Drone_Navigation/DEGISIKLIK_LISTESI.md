# DEGISIKLIK_LISTESI.md

> İlk teslim

## Genel Açıklama

PPO tabanlı İHA labirent navigasyonu; gym-pybullet-drones + stable-baselines3.

---

## Değişiklikler (Önceki → Bu Sürüm)

Bu belge ilk teslim olduğu için "önceki sürüm" yoktur.  
Tüm geliştirmeler oturum içinde tamamlanmış ve tek teslimde gönderilmektedir.

### 1. Ortam (env/drone_maze_env.py)
- `_preprocessAction`: Aksiyon [0,1] → hover RPM merkezli dönüşüm eklendi.  
  Önceden drone açılışta düşüyor (sıfır adım itkisi), şimdi `action=0.5` → hover RPM.
- `DIST_PENALTY_SCALE`: Mesafe cezası sabiti eklendi; hover lokal optimumunu kırar.
- `_computeReward`: Mesafe cezası (`-scale * curr_dist`) ve önbellek mantığı eklendi.
- `_computeTerminated`: `_last_collision` önbelleğine geçildi (çift hesaplama kaldırıldı).

### 2. Callback (utils/callbacks.py)
- `model_tag` parametresi eklendi → model adına göre kayıt.
- Tüm train env'leri için curriculum döngüsü düzeltildi.
- Outcome önceliği: hedef > çarpışma > timeout.
- **Yeni**: Her eval adımında `eval_log_{model_tag}.csv` CSV'ye yazıyor.  
  → plot_results.py için veri kaynağı.
- LSTM (`lstm_actor`) tespiti: RecurrentPPO için `lstm_states` yönetimi eklendi.

### 3. Eğitim (train.py)
- `_get_model_class()`: `MlpLstmPolicy` → `RecurrentPPO` otomatik algılama.
- `make_env()` / `_make_env()`: Monitor CSV kaydı için `monitor_path` parametresi.
- `policy_kwargs` desteği: `lstm_hidden_size` YAML'dan okunup modele geçiriliyor.
- Multi-seed döngüsü: `ppo.seeds` listesindeki tüm seed'ler sırayla eğitiliyor.
- CLI: Choices [A, B, C, D, E]; her model tüm seed'lerini çalıştırıyor.

### 4. Değerlendirme (evaluate.py)
- `_load_model()`: RecurrentPPO → PPO fallback ile otomatik model yükleme.
- LSTM `lstm_states` ve `episode_start` yönetimi eklendi.

### 5. Yeni Konfigürasyonlar
- `config/ppo_config_C.yaml`: MlpPolicy + dist_penalty_scale=0.01
- `config/ppo_config_D.yaml`: MlpLstmPolicy + dist_penalty_scale=0.02 (LSTM)
- `config/ppo_config_E.yaml`: Literatür optimalli LSTM  
  (lr=3e-4, n_steps=512, entropy_coef=0.01, clip_range=0.15, lstm_hidden=128, 5 seed)

### 6. Yeni Betikler
- `plot_results.py`: 5 zorunlu grafik (Monitor CSV + TensorBoard + eval CSV)
- `baseline_eval.py`: Rastgele ve hover politika baseline değerlendirmesi
- `sensitivity_train.py`: Hiperparametre duyarlılığı (lr × entropy, 6 konfigürasyon)
- `run_all.sh`: Tam pipeline (eğitim → baseline → duyarlılık → grafik)

### 7. Teslim Dosyaları
- `requirements.txt` (== sabitlenmiş)
- `seeds.txt` (5 seed: 42, 0, 1, 2, 3)
- `DEGISIKLIK_LISTESI.md` (bu dosya)

---

## Ödül Formülü (Kodla Birebir Aynı)

Aşağıdaki LaTeX formülü `env/drone_maze_env.py:185-214` satırlarındaki kodla birebir aynıdır:

$$
r_t = \underbrace{\alpha (d_{t-1} - d_t)}_{\text{yaklaşma}} 
    + \underbrace{c_{\text{adım}}}_{\text{adım cezası}}
    - \underbrace{\lambda \cdot d_t}_{\text{mesafe cezası}}
    + \begin{cases} +100 & d_t < r_{\text{hedef}} \\ -100 & \text{çarpışma} \end{cases}
$$

- $\alpha = 8.0$ (yaklaşma ölçeği, `APPROACH_SCALE`)
- $d_t$ = drone ile hedef arasındaki Öklid mesafesi
- $c_{\text{adım}} = -0.001$ (`REWARD_STEP`)
- $\lambda = 0.02$ (`DIST_PENALTY_SCALE`, Model E)
- $r_{\text{hedef}} = 0.3\,\text{m}$ (`GOAL_RADIUS`)
