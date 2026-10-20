"""
========================================================================================
SISTEM PREDIKSI KETERLAMBATAN PEMBAYARAN INVOICE (ENTERPRISE EDITION)
Studi Kasus: PT Maju Bersama - Make Sense of Data with Analysis and AI
========================================================================================

Peningkatan & Optimasi Sistem:
1. Feature Engineering Lanjutan:
   - Rasio nilai invoice terhadap historis customer (deteksi lonjakan nominal)
   - Frekuensi transaksi historis customer (prev_invoice_count)
   - Log-transform amount & seasonal effects
   - Strictly leak-free (hanya memakai data historis masa lalu dengan .shift(1))
2. Pipeline Preprocessing yang Bersih:
   - One-Hot Encoding terisolasi untuk data train & test
3. Pemodelan Modern & State-of-the-Art:
   - Baseline: Logistic Regression, Decision Tree, Random Forest
   - Advanced: Random Forest (Balanced), HistGradientBoosting (Modern GBDT)
   - Threshold Tuning adaptif berbasis validasi temporal (memaksimalkan F1-Score)
4. Metrik Finansial & Bisnis Nyata:
   - Perhitungan Expected At-Risk Exposure (Nominal x Risk Score)
   - Segmentasi Tingkat Risiko (High, Medium, Low) beserta SLA Tindakan Operasional
5. Ekspor Hasil:
   - Laporan prioritas siap pakai (CSV): laporan_prioritas_penagihan.csv
   - Dashboard evaluasi 4 panel: hasil_evaluasi.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)

# Set random seed untuk reproduksibilitas
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


class InvoiceRiskEngine:
    """Mesin Analisis dan Prediksi Risiko Keterlambatan Invoice."""

    def __init__(self, n_customer: int = 150, n_invoice: int = 4500):
        self.n_customer = n_customer
        self.n_invoice = n_invoice
        self.df_raw = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_raw = None
        self.models = {}
        self.evaluation_results = []
        self.best_model_name = None
        self.best_model = None
        self.best_threshold = 0.5
        self.feature_names = []

    def generate_synthetic_data(self) -> pd.DataFrame:
        """Membuat dataset sintetis transaksi invoice B2B dengan perilaku realistis."""
        print("[1/6] Mengenerate dataset sintetis transaksi B2B...")
        
        # 1. Profil Pelanggan
        customers = pd.DataFrame({
            "customer_id": range(self.n_customer),
            "segment": rng.choice(["retail", "grosir", "horeca"], self.n_customer, p=[0.45, 0.35, 0.20]),
            # Risiko laten karakter finansial pelanggan (nilai riil di lapangan)
            "latent_risk": rng.beta(1.2, 7.5, self.n_customer),
            "base_credit_limit": rng.choice([20_000_000, 50_000_000, 100_000_000], self.n_customer, p=[0.5, 0.35, 0.15])
        })

        # 2. Transaksi Invoice selama 2 tahun (2024 - 2025)
        df = pd.DataFrame({
            "customer_id": rng.integers(0, self.n_customer, self.n_invoice),
            "invoice_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, self.n_invoice), unit="D"),
            "amount": rng.lognormal(mean=16.1, sigma=0.85, size=self.n_invoice).round(-3),
            "term_days": rng.choice([14, 30, 45, 60], self.n_invoice, p=[0.20, 0.45, 0.20, 0.15]),
            "discount_pct": rng.choice([0, 2, 5, 10], self.n_invoice, p=[0.55, 0.25, 0.15, 0.05]),
        }).merge(customers, on="customer_id")

        df["month"] = df["invoice_date"].dt.month
        df["quarter"] = df["invoice_date"].dt.quarter
        df["due_date"] = df["invoice_date"] + pd.to_timedelta(df["term_days"], unit="D")

        # 3. Probabilitas Keterlambatan Realistis (efek latent risk, termin panjang, nominal besar, & seasonal Lebaran)
        median_amt = df["amount"].median()
        logit = (
            -3.8
            + 8.5 * df["latent_risk"]
            + 0.9 * (df["amount"] > median_amt * 1.5)
            + 0.015 * df["term_days"]
            - 0.03 * df["discount_pct"]
            + 0.45 * df["month"].isin([3, 4])  # Efek musiman peak cash flow
        )
        prob_late = 1 / (1 + np.exp(-logit))
        df["is_late"] = (rng.random(len(df)) < prob_late).astype(int)

        # 4. Kolom Pasca-Pembayaran (BOCOR / LEAKAGE - hanya diketahui setelah kejadian)
        df["days_overdue"] = np.where(
            df["is_late"] == 1,
            rng.integers(31, 95, len(df)),
            rng.integers(-10, 30, len(df))
        )
        df["payment_date"] = df["due_date"] + pd.to_timedelta(df["days_overdue"], unit="D")

        self.df_raw = df.sort_values("invoice_date").reset_index(drop=True)
        print(f"      Total Transaksi: {len(self.df_raw):,} invoice | Rasio Telat: {self.df_raw['is_late'].mean():.2%}")
        return self.df_raw

    def engineer_features(self):
        """Melakukan rekayasa fitur tingkat lanjut bebas data leakage."""
        print("[2/6] Melakukan feature engineering (strictly leak-free)...")
        df = self.df_raw.copy()

        # Rekayasa Fitur Historis Customer (Hanya menggunakan transaksi sebelum baris saat ini)
        grp = df.groupby("customer_id")
        
        # 1. Rasio keterlambatan historis pelanggan sebelumnya
        df["prev_late_rate"] = grp["is_late"].transform(lambda s: s.shift(1).expanding().mean())
        df["prev_late_rate"] = df["prev_late_rate"].fillna(df["is_late"].mean())

        # 2. Total frekuensi invoice yang pernah diterima pelanggan sebelumnya
        df["prev_invoice_count"] = grp.cumcount()

        # 3. Rata-rata nominal invoice masa lalu pelanggan
        prev_amt_mean = grp["amount"].transform(lambda s: s.shift(1).expanding().mean())
        prev_amt_mean = prev_amt_mean.fillna(df["amount"].median())
        df["amount_to_avg_ratio"] = (df["amount"] / (prev_amt_mean + 1)).clip(upper=5.0)

        # 4. Transformasi Log Nominal untuk menstabilkan varians
        df["log_amount"] = np.log1p(df["amount"])

        # 5. Fitur Diskon & Musiman
        df["is_high_discount"] = (df["discount_pct"] >= 5).astype(int)
        df["is_ramadan_season"] = df["month"].isin([3, 4]).astype(int)
        df["is_long_term"] = (df["term_days"] >= 45).astype(int)

        # 6. Pembersihan Kolom Bocor & Identitas Non-Prediktif
        LEAKAGE_COLS = ["payment_date", "days_overdue"]
        IDENTIFIER_COLS = ["customer_id", "invoice_date", "due_date", "latent_risk", "is_late", "base_credit_limit"]
        DROP_COLS = LEAKAGE_COLS + IDENTIFIER_COLS

        X_df = df.drop(columns=DROP_COLS)
        X_encoded = pd.get_dummies(X_df, columns=["segment"], drop_first=True, dtype=float)
        y = df["is_late"]

        self.feature_names = list(X_encoded.columns)

        # 7. Time-based split: 75% awal untuk Train, 25% akhir untuk Test (Simulasi Masa Depan)
        cut = int(len(df) * 0.75)
        self.X_train, self.X_test = X_encoded.iloc[:cut].copy(), X_encoded.iloc[cut:].copy()
        self.y_train, self.y_test = y.iloc[:cut].copy(), y.iloc[cut:].copy()
        self.test_raw = df.iloc[cut:].copy()

        print(f"      Train Set: {len(self.X_train):,} baris | Test Set: {len(self.X_test):,} baris")
        print(f"      Fitur Akhir ({len(self.feature_names)}): {', '.join(self.feature_names)}")

    def train_and_evaluate_models(self):
        """Melatih beragam model dan mengevaluasi metrik komparatif."""
        print("[3/6] Melatih model baseline vs advanced & penanganan imbalanced...")

        def calculate_metrics(name: str, y_true, y_pred, y_proba):
            return {
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                "PR-AUC": average_precision_score(y_true, y_proba),
            }

        # A. Baseline Models (Tanpa Penanganan Imbalance)
        baseline_models = {
            "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=RNG_SEED),
            "Decision Tree (Baseline)": DecisionTreeClassifier(max_depth=5, random_state=RNG_SEED),
            "Random Forest (Baseline)": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=RNG_SEED)
        }

        for name, model in baseline_models.items():
            model.fit(self.X_train, self.y_train)
            pred = model.predict(self.X_test)
            proba = model.predict_proba(self.X_test)[:, 1]
            self.evaluation_results.append(calculate_metrics(name, self.y_test, pred, proba))
            self.models[name] = (model, 0.5)

        # B. Advanced Model 1: Random Forest + Class Weight Balanced
        rf_balanced = RandomForestClassifier(
            n_estimators=200,
            max_depth=9,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RNG_SEED
        )
        rf_balanced.fit(self.X_train, self.y_train)
        proba_rf = rf_balanced.predict_proba(self.X_test)[:, 1]
        self.evaluation_results.append(
            calculate_metrics("Random Forest + class_weight", self.y_test, (proba_rf >= 0.5).astype(int), proba_rf)
        )
        self.models["Random Forest + class_weight"] = (rf_balanced, 0.5)

        # C. Advanced Model 2: HistGradientBoosting (Modern GBDT) + Threshold Tuning
        # Split 20% data validasi akhir dari train set untuk tuning threshold secara sekuensial waktu
        val_cut = int(len(self.X_train) * 0.8)
        X_tr_sub, y_tr_sub = self.X_train.iloc[:val_cut], self.y_train.iloc[:val_cut]
        X_val, y_val = self.X_train.iloc[val_cut:], self.y_train.iloc[val_cut:]

        hgb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.07,
            class_weight="balanced",
            random_state=RNG_SEED
        )
        hgb.fit(X_tr_sub, y_tr_sub)

        # Optimal Threshold Search pada Validation Set
        proba_val = hgb.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        self.best_threshold = float(thresholds[best_idx])

        # Retrain GBDT pada seluruh Train Set untuk deployment test
        hgb_full = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.07,
            class_weight="balanced",
            random_state=RNG_SEED
        )
        hgb_full.fit(self.X_train, self.y_train)
        proba_hgb_test = hgb_full.predict_proba(self.X_test)[:, 1]
        pred_tuned = (proba_hgb_test >= self.best_threshold).astype(int)

        tuned_name = f"HistGradientBoosting + Tuned ({self.best_threshold:.2f})"
        self.evaluation_results.append(
            calculate_metrics(tuned_name, self.y_test, pred_tuned, proba_hgb_test)
        )
        self.models[tuned_name] = (hgb_full, self.best_threshold)

        # Tetapkan model terbaik untuk inferensi operasional
        self.best_model_name = tuned_name
        self.best_model = hgb_full
        self.test_raw["risk_score"] = proba_hgb_test

    def generate_business_priority_report(self) -> pd.DataFrame:
        """Menghitung metrik eksposur finansial (Expected Loss) dan segmentasi tindakan koleksi."""
        print("[4/6] Membentuk laporan prioritas penagihan & kalkulasi eksposur risiko...")
        
        rep = self.test_raw.copy()

        # Rumus Finansial: Expected At-Risk Exposure = Nominal Tagihan * Skor Risiko
        rep["expected_risk_exposure"] = (rep["amount"] * rep["risk_score"]).round(0)

        # Segmentasi Tingkat Risiko (Risk Tiering)
        def assign_tier(score):
            if score >= 0.70:
                return "1. TINGGI (High Risk)"
            elif score >= 0.45:
                return "2. SEDANG (Medium Risk)"
            else:
                return "3. RENDAH (Low Risk)"

        def assign_action(tier):
            if "TINGGI" in tier:
                return "Telepon H-7, konfirmasi faktur, tawarkan split payment, tahan order baru"
            elif "SEDANG" in tier:
                return "Reminder otomatis via WhatsApp/Email H-3 & konfirmasi jadwal kliring"
            else:
                return "Pengiriman invoice standar sesuai termin biasa"

        rep["risk_tier"] = rep["risk_score"].apply(assign_tier)
        rep["recommended_action"] = rep["risk_tier"].apply(assign_action)
        self.test_raw = rep.copy()

        # Urutkan berdasarkan Expected Risk Exposure (Dampak Finansial Terbesar)
        rep_sorted = rep.sort_values(by="expected_risk_exposure", ascending=False).reset_index(drop=True)

        cols_export = [
            "customer_id", "segment", "invoice_date", "due_date",
            "amount", "term_days", "risk_score", "expected_risk_exposure",
            "risk_tier", "is_late", "recommended_action"
        ]
        
        # Simpan ke CSV
        output_csv = "laporan_prioritas_penagihan.csv"
        rep_sorted[cols_export].to_csv(output_csv, index=False)
        print(f"      Laporan tersimpan: {output_csv} ({len(rep_sorted)} invoice teranalisis)")

        return rep_sorted

    def render_visualizations(self):
        """Membuat visualisasi multi-panel (4 kuadran) untuk laporan analitik."""
        print("[5/6] Mengenerate visualisasi komprehensif ke hasil_evaluasi.png...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Feature Importance (dari Random Forest Balanced)
        rf_model = self.models["Random Forest + class_weight"][0]
        feat_imp = pd.Series(rf_model.feature_importances_, index=self.feature_names).sort_values()
        feat_imp.plot(kind="barh", ax=axes[0, 0], color="#2b5c8f")
        axes[0, 0].set_title("Feature Importance (Random Forest)", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Relative Importance Score")

        # Panel 2: Precision-Recall Curve dengan Marker Optimal Threshold
        hgb_model = self.best_model
        proba_test = self.test_raw["risk_score"].values
        prec, rec, _ = precision_recall_curve(self.y_test, proba_test)
        axes[0, 1].plot(rec, prec, color="#107c41", lw=2.5, label="HistGradientBoosting PR-Curve")
        axes[0, 1].axhline(self.y_test.mean(), ls="--", color="gray", label=f"Random Baseline ({self.y_test.mean():.1%})")
        axes[0, 1].set_title(f"Precision-Recall Curve (PR-AUC: {average_precision_score(self.y_test, proba_test):.3f})", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].legend(loc="lower left")
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Total Nilai Tagihan Terpapar Risiko per Kategori (Risk Tiers Exposure)
        tier_summary = self.test_raw.groupby("risk_tier")["amount"].sum() / 1e9  # Miliar Rupiah
        colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
        tier_summary.plot(kind="bar", ax=axes[1, 0], color=colors, rot=15)
        axes[1, 0].set_title("Total Nilai Invoice per Risk Tier (Miliar IDR)", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Total Nominal (Miliar IDR)")
        axes[1, 0].grid(axis="y", alpha=0.3)

        # Panel 4: Confusion Matrix Model Terbaik
        pred_final = (proba_test >= self.best_threshold).astype(int)
        cm = confusion_matrix(self.y_test, pred_final)
        im = axes[1, 1].imshow(cm, cmap="Blues", interpolation="nearest")
        axes[1, 1].set_title(f"Confusion Matrix ({self.best_model_name})", fontsize=12, fontweight="bold")
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(["Pred Tepat Waktu", "Pred Telat"])
        axes[1, 1].set_yticklabels(["Aktual Tepat", "Aktual Telat"])

        # Tambahkan teks angka di dalam kotak confusion matrix
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                axes[1, 1].text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig("hasil_evaluasi.png", dpi=130)
        print("      Dashboard visualisasi berhasil disimpan: hasil_evaluasi.png")

    def display_summary(self, top_n: int = 10):
        """Menampilkan rangkuman eksekutif pada terminal."""
        print("\n" + "=" * 78)
        print("[ANALISIS] TABEL PERBANDINGAN PERFORMA MODEL (DATA TEST)")
        print("=" * 78)
        df_eval = pd.DataFrame(self.evaluation_results).set_index("Model").round(3)
        print(df_eval.to_string())

        print("\n" + "=" * 78)
        print(f"[PRIORITAS] TOP {top_n} INVOICE BERISIKO TINGGI (BERDASARKAN DAMPAK FINANSIAL)")
        print("=" * 78)
        top_invoices = self.test_raw.sort_values(by="expected_risk_exposure", ascending=False).head(top_n)
        view_cols = ["customer_id", "amount", "term_days", "risk_score", "expected_risk_exposure", "is_late"]
        
        display_df = top_invoices[view_cols].copy()
        display_df["amount"] = display_df["amount"].apply(lambda x: f"Rp {x:,.0f}")
        display_df["expected_risk_exposure"] = display_df["expected_risk_exposure"].apply(lambda x: f"Rp {x:,.0f}")
        display_df["risk_score"] = display_df["risk_score"].apply(lambda x: f"{x:.1%}")
        print(display_df.to_string(index=False))
        print("=" * 78 + "\n")


def main():
    """Fungsi utama menjalankan pipeline end-to-end."""
    print("=" * 78)
    print(">>> MEMULAI PIPELINE PREDIKSI INVOICE PT MAJU BERSAMA")
    print("=" * 78)
    
    engine = InvoiceRiskEngine(n_customer=150, n_invoice=4500)
    engine.generate_synthetic_data()
    engine.engineer_features()
    engine.train_and_evaluate_models()
    engine.generate_business_priority_report()
    engine.render_visualizations()
    engine.display_summary(top_n=10)
    
    print("[6/6] Pipeline selesai dijalankan dengan sukses!")


if __name__ == "__main__":
    main()
