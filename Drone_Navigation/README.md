# Drone Maze Navigation — PPO ile Labirent Navigasyonu

## Gereksinimler

```bash
pip install -r requirements.txt
```

## Çalıştırma

Tüm pipeline (eğitim + grafik + baseline):
```bash
bash run_all.sh
```

Sadece Model E eğitimi (tek seed):
```bash
python train.py --model E --seed 42
```

Tüm seedler sırayla:
```bash
for seed in 42 0 1 2 3; do python train.py --model E --seed $seed; done
```

## Değerlendirme

```bash
python evaluate.py --model-path results/models/best_model_modelE_seed42.zip --episodes 20
```

## Grafik Üretimi

```bash
python plot_results.py --model-tag modelE --seeds 42 0 1 2 3 --output sunum/grafikler
```

## Hiperparametre Duyarlılığı (Grafik 4)

```bash
python sensitivity_train.py
```

## Baseline Karşılaştırma (Grafik 5)

```bash
python baseline_eval.py
```

## Proje Yapısı

```
env/                  ← Gym ortamı (DroneMazeEnv, MazeGenerator)
utils/callbacks.py    ← EvalAndSaveCallback (curriculum + CSV log)
config/               ← Model konfigürasyonları (YAML)
train.py              ← PPO/RecurrentPPO eğitim scripti
evaluate.py           ← Eğitilmiş model değerlendirme
plot_results.py       ← 5 zorunlu grafik üreticisi
sensitivity_train.py  ← Grafik 4 için hiper-param tarama
baseline_eval.py      ← Grafik 5 için baseline ölçümü
seeds.txt             ← Kullanılan seed'ler: 42 0 1 2 3
config.yaml           ← Model E konfigürasyonu (varsayılan)
```

## Seed Listesi

```
42, 0, 1, 2, 3
```
(bkz. `seeds.txt`)

## Ortam Detayları

- **Platform**: gym-pybullet-drones (CF2X quadrotor)
- **Labirent**: easy (11×11) ve medium (13×13) grid
- **Gözlem**: 39 boyut (konum, hız, açısal hız, RPY, LIDAR-24, hedef vektörü)
- **Eylem**: [0,1]^4, hover RPM'e göre normalize
- **Curriculum**: 6 aşama (easy 1.5m → medium serbest)
