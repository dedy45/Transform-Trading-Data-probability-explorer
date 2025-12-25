# 📊 Transform Trading DATA Probability Explorer

> **"Transform Trading Data into Actionable Intelligence"**

Aplikasi analisis probabilitas trading berbasis Dash Plotly yang mengubah data historis trading menjadi insight probabilistik untuk pengambilan keputusan yang lebih baik.
Demo APP https://analis.bamsbung.id/
---

## 🎯 Filosofi & Nilai Inti

### Filosofi: "Probability Over Prediction"

Trading bukan tentang memprediksi masa depan dengan pasti, tetapi tentang **memahami probabilitas** dan **mengelola risiko**. Aplikasi ini dibangun dengan filosofi:

1. **📊 Data-Driven Decision Making**
   - Setiap keputusan harus didukung oleh data historis
   - Probabilitas kondisional lebih akurat dari prediksi absolut
   - Pattern recognition melalui analisis statistik

2. **🎲 Embrace Uncertainty**
   - Tidak ada strategi yang menang 100%
   - Yang penting adalah **positive expectancy** dalam jangka panjang
   - Risk management lebih penting dari win rate

3. **🔬 Scientific Approach**
   - Hypothesis testing dengan data
   - Validasi model dengan calibration
   - Continuous improvement melalui feedback loop

4. **⚖️ Risk-Reward Balance**
   - Fokus pada R-multiple, bukan profit absolut
   - Optimize expectancy, bukan win rate
   - Position sizing berdasarkan Kelly Criterion

### Nilai Berharga yang Ditawarkan

#### 1. 💡 **Clarity in Chaos**
Trading penuh dengan noise dan emosi. Aplikasi ini memberikan **clarity** melalui:
- Visualisasi probabilitas yang mudah dipahami
- Identifikasi kondisi pasar terbaik untuk trading
- Deteksi pattern yang tidak terlihat dengan mata telanjang

#### 2. 🎯 **Actionable Insights**
Bukan hanya angka dan grafik, tetapi **rekomendasi konkret**:
- "Trade only when composite score > 70"
- "Optimal SL at 0.8R based on MAE analysis"
- "Use 0.5% risk per trade (Half Kelly) for this strategy"

#### 3. 🛡️ **Risk Awareness**
Memahami **worst-case scenario** sebelum terjadi:
- Monte Carlo simulation menunjukkan kemungkinan drawdown maksimal
- Risk of ruin calculator mencegah over-leverage
- Expectancy analysis mengidentifikasi strategi negatif

#### 4. 🚀 **Continuous Improvement**
Framework untuk **iterasi dan optimasi**:
- Backtest berbagai threshold dan parameter
- Compare multiple scenarios side-by-side
- Track improvement over time

---

# 📊 Trading Probability Explorer v5.1

> **"Transform Trading Data into Actionable Intelligence"**

Aplikasi analisis probabilitas trading berbasis Dash Plotly dengan **18 halaman analisis komprehensif** untuk mengubah data historis trading menjadi insight probabilistik.

---

## 💎 Nilai & Manfaat

| Nilai | Manfaat Konkret |
|-------|-----------------|
| 💡 **Clarity** | Visualisasi probabilitas yang mudah dipahami |
| 🎯 **Actionable** | Rekomendasi konkret untuk trading |
| 🛡️ **Risk Awareness** | Monte Carlo simulation, risk of ruin analysis |
| 🚀 **Efficiency** | Dari 3 hari analisis manual → 5 menit otomatis |

---

## ✨ 18 Halaman Analisis

| # | Halaman | Fungsi Utama |
|---|---------|--------------|
| 1 | **Trade Analysis Dashboard** | Overview performa, expectancy, equity curve |
| 2 | **Probability Explorer** | Probabilitas kondisional, composite score |
| 3 | **Sequential Analysis** | Markov chain, streak analysis |
| 4 | **Calibration Lab** | Reliability diagram, Brier score |
| 5 | **Regime Explorer** | Performa per kondisi pasar |
| 6 | **What-If Scenarios** | Monte Carlo, MAE/MFE optimizer |
| 7 | **Auto Feature Selection** ⭐ | ML-based feature ranking (30 fitur → 8 optimal) |
| 8 | **Market Condition Scoring** ⭐ | Dual mode: v1 Trade Regime + v2 Market State |
| 9 | **ML Prediction Engine** ⭐ | Calibrated probability + Conformal Prediction + Decision Engine |
| 10 | **Combination Probability Analyzer** ⭐ | Bayesian analysis + SQA Optimizer + Cross-validation |
| 11 | **Decision Tree Rules** ⭐ | Extract IF-THEN rules untuk MQL5 EA |
| 12 | **Walk-Forward Validation** ⭐ | Time-series cross-validation |
| 13 | **Ensemble Voting** ⭐ | Multi-method consensus voting |
| 14 | **Optuna Hyperparameter Optimizer** ⭐ | Bayesian optimization untuk ML pipeline |
| 15 | **Isolation Forest Analyzer** | Deteksi positive outliers dalam winning trades |
| 16 | **NGBoost Probabilistic Engine** | Native uncertainty estimation dengan distribusi probabilitas |
| 17 | **Risk Lab** | Risk OS, policy builder, position sizing, gating rules |
| 18 | **Neural Network Lab** 🆕 | MLP, LSTM, Transformer + ONNX Export untuk MT5 |

---

## 🆕 What's New in v5.1

| Feature | Deskripsi |
|---------|-----------|
| 🔧 **Server DataFrame Migration** | Final validation complete, 253,327x speedup |
| 🐛 **Market Scoring Bug Fixes** | Fixed n_samples=0 dan KeyError: 'score' |
| 🔗 **Send to Walk-Forward** 🆕 | Tombol baru di Page 9 untuk kirim prediksi ke Page 12 |
| 📊 **Performance Benchmark** | Script benchmark untuk migration metrics |

## 🆕 What's New in v5.0

| Feature | Deskripsi |
|---------|-----------|
| 🧠 **Neural Network Lab (Page 18)** 🆕 | Deep learning dengan MLP, LSTM, Transformer |
| 🔄 **ONNX Export** 🆕 | Export neural network models ke ONNX untuk MT5 EA |
| 📊 **Ensemble NN** 🆕 | Kombinasi multiple NN models dengan weighted averaging |
| 🎯 **Uncertainty Estimation** 🆕 | MC Dropout untuk confidence levels |
| 📈 **Batch Prediction** 🆕 | Process semua trades sekaligus dengan histogram |
| 🛡️ **Risk Lab (Page 17)** | Risk OS dengan policy builder, hard gating, soft throttling |
| 🎯 **NGBoost Engine (Page 16)** | Native probabilistic predictions dengan distribusi uncertainty |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run application
python app.py

# 3. Open browser
http://127.0.0.1:8050
```

---

## 📈 Workflow Rekomendasi

```
Load Data → Auto Feature Selection (Page 7) → Market Condition Scoring (Page 8)
    ↓
Optuna Optimizer (Page 14) → Optimize hyperparameters & features
    ↓
Isolation Forest (Page 15) → Deteksi positive outliers → Enrich ke Page 9
    ↓
Combination Analyzer + SQA (Page 10) → Decision Tree Rules (Page 11)
    ↓
Walk-Forward Validation (Page 12) → Ensemble Voting (Page 13)
    ↓
ML Prediction Engine (Page 9) ←→ NGBoost Engine (Page 16)
    ↓
Neural Network Lab (Page 18) → Train MLP/LSTM/Transformer → ONNX Export 🆕
    ↓
Risk Lab (Page 17) → Risk policy, sizing, gating → EA Export ✅
```

---

## 📚 Dokumentasi

| Dokumen | Deskripsi |
|---------|-----------|
| [README_FULL.md](README_FULL.md) | Dokumentasi lengkap & komprehensif |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Arsitektur sistem & data flow |
| [PANDUAN_LENGKAP.md](PANDUAN_LENGKAP.md) | Panduan penggunaan setiap halaman |
| [Version.md](Version.md) | Version history & roadmap |

---

## 🛠️ Tech Stack

- **Frontend:** Dash, Plotly, Bootstrap
- **Backend:** Python, Pandas, NumPy
- **ML:** LightGBM, NGBoost, Scikit-learn, SHAP
- **Deep Learning:** TensorFlow/Keras, ONNX 🆕
- **Optimization:** Optuna (Bayesian), SQA (Quantum Annealing)
- **Tracking:** MLflow
- **Testing:** Pytest, Hypothesis (Property-Based Testing)

---

**Built with ❤️ for traders who believe in data-driven decisions.**

*Version 5.1 | 26 December 2025*


### 📞 Support
dedy@bamsbung.id
---

**ML Prediction Engine Status:** ✅ **PRODUCTION READY**  

**Ready to predict. Ready to win.** 🎯
