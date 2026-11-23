#!/usr/bin/env bash
# run_all.sh — Tam Pipeline: Eğitim → Baseline → Duyarlılık → Grafik
# Kullanım: bash run_all.sh
# Not: Model E, 5 seed × 3M adım = ~15M adım; GPU'suz makinede birkaç gün sürebilir.

set -e

echo "======================================================"
echo "  Drone RL v2 — Tam Çalıştırma Pipeline"
echo "======================================================"

# Dizinler
mkdir -p results/logs results/models results/sensitivity sonuclar/loglar sunum/grafikler

# ── 1. Model E Eğitimi (5 seed) ───────────────────────────
echo ""
echo ">>> Adım 1/4: Model E eğitimi başlıyor (5 seed × 3M adım)..."
python train.py --model E

# ── 2. Baseline Değerlendirmesi ───────────────────────────
echo ""
echo ">>> Adım 2/4: Baseline değerlendirmesi (rastgele + hover)..."
python baseline_eval.py

# ── 3. Hiperparametre Duyarlılığı ─────────────────────────
echo ""
echo ">>> Adım 3/4: Hiperparametre duyarlılığı eğitimi (6 konfigürasyon × 500K adım)..."
python sensitivity_train.py

# ── 4. Grafik Üretimi ─────────────────────────────────────
echo ""
echo ">>> Adım 4/4: 5 grafik üretiliyor..."
python plot_results.py --model-tag modelE --seeds 42 0 1 2 3

# ── Log'ları teslim dizinine kopyala ──────────────────────
echo ""
echo ">>> Log'lar sonuclar/loglar/ altına kopyalanıyor..."
cp -r results/logs/* sonuclar/loglar/ 2>/dev/null || true
cp -r results/sensitivity/* sonuclar/loglar/ 2>/dev/null || true

echo ""
echo "======================================================"
echo "  Pipeline tamamlandı!"
echo "  Grafikler : sunum/grafikler/"
echo "  Loglar    : sonuclar/loglar/"
echo "  Modeller  : results/models/"
echo ""
echo "  TensorBoard için:"
echo "  tensorboard --logdir results/logs"
echo "======================================================"
