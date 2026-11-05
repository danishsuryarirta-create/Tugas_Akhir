"""
Aplikasi Streamlit: Klasifikasi Risiko Osteoporosis
Penelitian: perbandingan baseline vs hyperparameter tuning (Optuna)
pada Random Forest, XGBoost, dan LightGBM.

File yang dibutuhkan (satu folder dengan app.py):
  - model_osteoporosis.pkl        (wajib)
  - results_baseline.csv          (halaman Hasil Penelitian)
  - results_tuned_optuna.csv      (halaman Hasil Penelitian)
  - best_params_optuna.json       (halaman Hasil Penelitian)
  - permutation_importance.csv    (opsional, halaman Feature Importance)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Osteoporosis", page_icon="🦴", layout="wide")

BASE = Path(__file__).parent
LOKASI_CARI = [BASE, BASE / "artefak_streamlit", BASE / "data"]
NAMA_MODEL = "model_osteoporosis.pkl"

URUTAN_INPUT = [
    "Age", "Gender", "Hormonal Changes", "Family History", "Race/Ethnicity",
    "Body Weight", "Calcium Intake", "Vitamin D Intake", "Physical Activity",
    "Smoking", "Alcohol Consumption", "Medical Conditions", "Medications",
    "Prior Fractures",
]
# Teks untuk pengguna awam. Nilai asli dataset (bahasa Inggris) tetap dipakai di balik layar,
# hanya tampilannya yang diterjemahkan.
LABEL_AWAM = {
    "Age": ("Usia (tahun)", "Usia Anda saat ini."),
    "Gender": ("Jenis kelamin", None),
    "Hormonal Changes": (
        "Apakah sudah menopause?",
        "Menopause adalah berhentinya haid secara permanen. Setelah menopause, kadar hormon "
        "estrogen menurun dan kepadatan tulang bisa berkurang lebih cepat.",
    ),
    "Family History": (
        "Riwayat keluarga",
        "Apakah ada anggota keluarga (misalnya orang tua) yang pernah terkena osteoporosis "
        "atau tulang keropos?",
    ),
    "Race/Ethnicity": ("Kelompok etnis", None),
    "Body Weight": (
        "Kondisi berat badan",
        "Berat badan yang terlalu rendah dapat membuat tulang lebih mudah rapuh.",
    ),
    "Calcium Intake": (
        "Asupan kalsium sehari-hari",
        "Kalsium banyak terdapat pada susu, keju, yoghurt, ikan teri, dan sayuran hijau.",
    ),
    "Vitamin D Intake": (
        "Asupan vitamin D",
        "Vitamin D didapat dari sinar matahari pagi, ikan berlemak, telur, dan makanan "
        "yang diperkaya vitamin D. Vitamin D membantu tubuh menyerap kalsium.",
    ),
    "Physical Activity": ("Kebiasaan bergerak atau berolahraga", None),
    "Smoking": ("Apakah merokok?", None),
    "Alcohol Consumption": ("Kebiasaan minum alkohol", None),
    "Medical Conditions": (
        "Penyakit yang dimiliki",
        "Gangguan kelenjar tiroid yang terlalu aktif dan radang sendi autoimun (rematik) "
        "diketahui dapat memengaruhi kesehatan tulang.",
    ),
    "Medications": (
        "Obat yang rutin diminum",
        "Pemakaian obat steroid dalam jangka panjang dapat mengurangi kepadatan tulang.",
    ),
    "Prior Fractures": ("Pernah patah tulang sebelumnya?", "Pernah mengalami patah tulang."),
}

OPSI_AWAM = {
    "Gender": {"Female": "Perempuan", "Male": "Laki-laki"},
    "Hormonal Changes": {"Normal": "Normal", "Postmenopausal": "Postmenopausal (Tidak lagi mengalami menstruasi selama 12 bulan berturut-turut)"},
    "Family History": {"Yes": "Ya", "No": "Tidak"},
    "Race/Ethnicity": {"African American": "Keturunan Afrika-Amerika", "Asian": "Asia",
                       "Caucasian": "Caucasian (Kulit putih)"},
    "Body Weight": {"Normal": "Normal", "Underweight": "Kurus"},
    "Calcium Intake": {"Adequate": "Cukup", "Low": "Kurang"},
    "Vitamin D Intake": {"Sufficient": "Cukup", "Insufficient": "Kurang"},
    "Physical Activity": {"Active": "Aktif bergerak / berolahraga", "Sedentary": "Jarang bergerak (banyak duduk)"},
    "Smoking": {"Yes": "Ya, merokok", "No": "Tidak merokok"},
    "Alcohol Consumption": {"Moderate": "Kadang-kadang", "None": "Tidak minum alkohol"},
    "Medical Conditions": {"Hyperthyroidism": "Tiroid terlalu aktif (hipertiroid)",
                           "Rheumatoid Arthritis": "Radang sendi kronis (rematik)",
                           "None": "Tidak ada"},
    "Medications": {"Corticosteroids": "Corticosteroids (Obat antiradang atau anti-bengkak yang kuat dari dokter)", "None": "Tidak ada"},
    "Prior Fractures": {"Yes": "Ya, pernah", "No": "Belum pernah"},
}


def tampil_opsi(kolom, nilai):
    return OPSI_AWAM.get(kolom, {}).get(nilai, nilai)


UMUR_MIN, UMUR_MAX, UMUR_DEFAULT = 18, 90, 45


# ----------------------------------------------------------------------
# Pemuatan file
# ----------------------------------------------------------------------
def cari_file(nama):
    for folder in LOKASI_CARI:
        path = folder / nama
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    path = cari_file(NAMA_MODEL)
    if path is None:
        raise FileNotFoundError(
            f"File '{NAMA_MODEL}' tidak ditemukan. Taruh file itu satu folder dengan app.py."
        )
    return joblib.load(path)


@st.cache_data
def muat_csv(nama):
    path = cari_file(nama)
    return None if path is None else pd.read_csv(path)


@st.cache_data
def muat_json(nama):
    path = cari_file(nama)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Preprocessing (harus sama persis dengan saat training)
# ----------------------------------------------------------------------
def opsi_kategori(art):
    """Pilihan tiap kolom kategorikal diambil dari encoder, jadi selalu sama dengan data latih."""
    opsi = {}
    for kolom, le in art["label_encoders"].items():
        opsi[kolom] = [str(x) for x in le.classes_]
    for kolom, kategori in zip(art["ohe_cols"], art["ohe"].categories_):
        opsi[kolom] = [str(x) for x in kategori]
    return opsi


def siapkan_fitur(art, data):
    X = pd.DataFrame([data])
    for kolom, le in art["label_encoders"].items():
        X[kolom] = le.transform(X[kolom])

    ohe_df = pd.DataFrame(
        art["ohe"].transform(X[art["ohe_cols"]]),
        columns=art["ohe"].get_feature_names_out(art["ohe_cols"]),
        index=X.index,
    )
    X = pd.concat([X.drop(columns=art["ohe_cols"]), ohe_df], axis=1)
    X.columns = X.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    return X[art["feature_names"]]


def prediksi(art, data):
    X = siapkan_fitur(art, data)
    model = art["model"]
    proba = float(model.predict_proba(X)[0, 1])
    label = int(model.predict(X)[0])
    return proba, label, X


# ----------------------------------------------------------------------
# Halaman: Prediksi
# ----------------------------------------------------------------------
def teks_persen(persen):
    """Hindari klaim 0% atau 100% yang terdengar pasti."""
    if persen >= 99.5:
        return ">99%", "lebih dari 99%"
    if persen < 0.5:
        return "<1%", "kurang dari 1%"
    return f"{persen:.0f}%", f"sekitar {persen:.0f}%"


def halaman_prediksi(art):
    st.title("🦴 Cek Risiko Osteoporosis")
    st.markdown(
        "**Osteoporosis** adalah kondisi tulang keropos: tulang menjadi rapuh dan mudah patah. "
        "Isi data diri di bawah ini, lalu tekan **Cek sekarang** untuk melihat perkiraan peluangnya."
    )
    st.caption(
        "Alat ini hanya perkiraan awal dari sebuah penelitian, **bukan diagnosis dokter**. "
        "Untuk kepastian, periksakan diri ke tenaga kesehatan."
    )

    opsi = opsi_kategori(art)
    kolom_input = [k for k in URUTAN_INPUT if k == "Age" or k in opsi]
    kolom_input += [k for k in opsi if k not in kolom_input]   # jaga-jaga kalau ada kolom lain

    with st.form("form_prediksi"):
        kiri, kanan = st.columns(2)
        data = {}
        for i, kolom in enumerate(kolom_input):
            wadah = kiri if i % 2 == 0 else kanan
            label, bantuan = LABEL_AWAM.get(kolom, (kolom, None))
            if kolom == "Age":
                data[kolom] = wadah.slider(label, UMUR_MIN, UMUR_MAX, UMUR_DEFAULT, help=bantuan)
            else:
                data[kolom] = wadah.selectbox(
                    label, opsi[kolom],
                    format_func=lambda v, k=kolom: tampil_opsi(k, v),
                    help=bantuan,
                )
        kirim = st.form_submit_button("Cek sekarang", type="primary")

    if not kirim:
        return

    try:
        proba, label, X = prediksi(art, data)
    except Exception as e:
        st.error(f"Maaf, data belum bisa diproses: {e}")
        return

    persen = proba * 100
    singkat, frasa = teks_persen(persen)
    st.divider()
    st.subheader("Hasil perkiraan")
    kol_a, kol_b = st.columns([1, 2])
    kol_a.metric("Perkiraan peluang osteoporosis", singkat)
    with kol_b:
        st.progress(min(max(proba, 0.0), 1.0))
        if label == 1:
            st.warning("**Ada kemungkinan terkena osteoporosis.** Sebaiknya periksakan diri ke dokter.")
        else:
            st.success("**Kemungkinan osteoporosis kecil** berdasarkan data yang Anda isi.")

    st.markdown("#### Apa artinya?")
    if label == 1:
        st.write(
            f"Berdasarkan data yang Anda isi, sistem memperkirakan peluang osteoporosis "
            f"**{frasa}**. Ini bukan vonis. Hasil ini hanya tanda untuk lebih waspada dan "
            f"berkonsultasi dengan dokter. Dokter dapat menyarankan pemeriksaan kepadatan tulang "
            f"untuk memastikan."
        )
    else:
        st.write(
            f"Berdasarkan data yang Anda isi, sistem memperkirakan peluang osteoporosis "
            f"**{frasa}**. Hasil ini tidak menjamin Anda pasti aman. Tetap jaga kesehatan "
            f"tulang dan periksakan diri jika ada keluhan."
        )

    st.markdown("#### Tips umum menjaga tulang tetap kuat")
    st.markdown(
        "- Cukupi kalsium dari makanan (susu, keju, ikan teri, sayuran hijau).\n"
        "- Dapatkan vitamin D, misalnya dengan berjemur pagi hari.\n"
        "- Rutin bergerak atau berolahraga, terutama yang menumpu berat badan seperti jalan kaki.\n"
        "- Hindari merokok dan batasi alkohol."
    )

    st.info(
        "**Perlu diketahui:** pada data yang dipakai untuk melatih sistem ini, **usia** sangat "
        "menentukan hasil. Mengubah isian lain biasanya hampir tidak mengubah perkiraan. "
        "Karena itu hasil ini tidak boleh dipakai sebagai satu-satunya dasar keputusan kesehatan."
    )

    with st.expander("Ringkasan data yang Anda isi"):
        ringkasan = []
        for kolom in kolom_input:
            label_k = LABEL_AWAM.get(kolom, (kolom, None))[0]
            nilai = data[kolom] if kolom == "Age" else tampil_opsi(kolom, data[kolom])
            ringkasan.append({"Pertanyaan": label_k, "Jawaban Anda": str(nilai)})
        st.dataframe(pd.DataFrame(ringkasan), hide_index=True)

    with st.expander("Detail teknis (untuk peneliti)"):
        st.caption("Data setelah diubah ke bentuk angka yang dibaca model.")
        st.dataframe(X)


# ----------------------------------------------------------------------
# Halaman: Hasil Penelitian
# ----------------------------------------------------------------------
METRIK_URUT = ["accuracy", "precision", "recall", "f1", "roc_auc"]
NAMA_METRIK = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
               "f1": "F1-Score", "roc_auc": "ROC-AUC"}


def tabel_perbandingan(base, tuned, metrik):
    gabung = pd.concat([base.assign(tahap="Baseline"), tuned.assign(tahap="Tuned")])
    gabung["algoritma"] = (gabung["model"].str.replace(r"\s*\[.*\]", "", regex=True).str.strip())

    baris = []
    for algo in gabung["algoritma"].unique():
        b = gabung[(gabung["algoritma"] == algo) & (gabung["tahap"] == "Baseline")]
        t = gabung[(gabung["algoritma"] == algo) & (gabung["tahap"] == "Tuned")]
        if b.empty or t.empty:
            continue
        nb, nt = b.iloc[0][metrik].astype(float), t.iloc[0][metrik].astype(float)
        baris.append({"Algoritma": algo, "Tahap": "Baseline", **nb.to_dict()})
        baris.append({"Algoritma": algo, "Tahap": "Tuned", **nt.to_dict()})
        baris.append({"Algoritma": algo, "Tahap": "Selisih", **(nt - nb).to_dict()})
    return pd.DataFrame(baris)


def grafik_perbandingan(tabel, metrik):
    algoritma = list(tabel["Algoritma"].unique())
    nb = [tabel[(tabel.Algoritma == a) & (tabel.Tahap == "Baseline")][metrik].iloc[0] for a in algoritma]
    nt = [tabel[(tabel.Algoritma == a) & (tabel.Tahap == "Tuned")][metrik].iloc[0] for a in algoritma]

    x = np.arange(len(algoritma))
    lebar = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    r1 = ax.bar(x - lebar / 2, nb, lebar, label="Baseline", color="#8ecae6")
    r2 = ax.bar(x + lebar / 2, nt, lebar, label="Tuned (Optuna)", color="#ffb703")
    for r in list(r1) + list(r2):
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(algoritma)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel(NAMA_METRIK.get(metrik, metrik))
    ax.set_title(f"{NAMA_METRIK.get(metrik, metrik)}: Baseline vs Tuned (data test)", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def halaman_hasil():
    st.title("📊 Hasil Penelitian")
    st.caption("Perbandingan model baseline dan model hasil tuning Optuna pada data test.")

    base = muat_csv("results_baseline.csv")
    tuned = muat_csv("results_tuned_optuna.csv")
    params = muat_json("best_params_optuna.json")

    if base is None or tuned is None:
        st.warning("File `results_baseline.csv` dan/atau `results_tuned_optuna.csv` tidak ditemukan "
                   "di folder aplikasi.")
    else:
        metrik = [m for m in METRIK_URUT if m in base.columns and m in tuned.columns]
        tabel = tabel_perbandingan(base, tuned, metrik)

        st.subheader("Perbandingan Baseline vs Tuned")
        tampil = tabel.set_index(["Algoritma", "Tahap"]).rename(columns=NAMA_METRIK)
        st.dataframe(tampil.round(4))
        st.caption("Selisih = Tuned − Baseline.")

        pilih = st.selectbox("Metrik untuk grafik", metrik, format_func=lambda m: NAMA_METRIK.get(m, m))
        st.pyplot(grafik_perbandingan(tabel, pilih))

        with st.expander("Tabel hasil per tahap"):
            st.markdown("**Baseline**")
            st.dataframe(base.set_index("model").round(4))
            st.markdown("**Tuned (Optuna)**")
            st.dataframe(tuned.set_index("model").round(4))

    st.subheader("Hyperparameter terbaik (Optuna)")
    if params is None:
        st.warning("File `best_params_optuna.json` tidak ditemukan di folder aplikasi.")
    else:
        pilihan = st.selectbox("Pilih model", list(params.keys()))
        df_param = pd.DataFrame({"Parameter": list(params[pilihan].keys()),
                                 "Nilai terbaik": [str(v) for v in params[pilihan].values()]})
        st.dataframe(df_param, hide_index=True)


# ----------------------------------------------------------------------
# Halaman: Feature Importance
# ----------------------------------------------------------------------
def grafik_importance(seri, judul, warna, n):
    top = seri.sort_values(ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(top) + 1.5))
    ax.barh(top.index, top.values, color=warna)
    ax.set_title(judul, fontweight="bold")
    fig.tight_layout()
    return fig


def halaman_importance(art):
    st.title("🔍 Feature Importance")
    st.caption(f"Fitur yang paling berpengaruh pada model terbaik: **{art['model_name']}**.")

    model = art["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=art["feature_names"], dtype=float)
        imp = imp / imp.sum()
        n = st.slider("Jumlah fitur yang ditampilkan", 5, len(imp), min(10, len(imp)))

        st.subheader("Importance bawaan model")
        st.pyplot(grafik_importance(imp, "Importance bawaan model (dinormalkan)", "steelblue", n))
        st.caption("Importance bawaan menunjukkan seberapa sering/besar fitur dipakai model saat "
                   "membentuk tree, sehingga fitur berisik pun bisa mendapat nilai kecil.")
        with st.expander("Tabel importance"):
            st.dataframe(imp.sort_values(ascending=False).rename("importance").to_frame().round(4))
    else:
        st.info("Model ini tidak menyediakan importance bawaan.")

    perm = muat_csv("permutation_importance.csv")
    st.subheader("Permutation importance")
    if perm is None or not {"fitur", "penurunan_akurasi"}.issubset(perm.columns):
        st.caption("Belum ada file `permutation_importance.csv`. Simpan dari notebook "
                   "(`perm_df.to_csv('permutation_importance.csv', index=False)`) lalu taruh "
                   "satu folder dengan app.py untuk menampilkannya di sini.")
    else:
        seri = perm.set_index("fitur")["penurunan_akurasi"]
        st.pyplot(grafik_importance(seri, "Penurunan akurasi saat fitur diacak (data test)", "darkorange",
                                    min(10, len(seri))))
        st.caption("Nilai mendekati 0 berarti model tidak memakai fitur tersebut.")
        with st.expander("Tabel permutation importance"):
            st.dataframe(perm.round(4), hide_index=True)


# ----------------------------------------------------------------------
# Halaman: Tentang & Limitasi
# ----------------------------------------------------------------------
def halaman_tentang(art):
    st.title("ℹ️ Tentang Penelitian & Limitasi")

    st.subheader("Tentang penelitian")
    st.markdown(
        f"""
Penelitian ini mengklasifikasikan osteoporosis dari data demografi, gaya hidup, dan riwayat kesehatan,
serta membandingkan **model baseline** dengan **model hasil hyperparameter tuning (Optuna)** pada
Random Forest, XGBoost, dan LightGBM.

**Alur singkat:** data understanding → split 80/20 (stratified) → preprocessing (Label Encoding dan
One-Hot Encoding) → model baseline → tuning Optuna dengan 5-fold cross-validation pada data latih →
evaluasi di data test → pemilihan model terbaik.

**Model yang dipakai di aplikasi:** {art['model_name']}
"""
    )

    st.subheader("Data")
    st.markdown(
        """
Dataset osteoporosis dari Kaggle: 1.958 baris dengan 14 fitur (setelah kolom `Id` dibuang) dan satu
target (`Osteoporosis`), dengan proporsi kelas seimbang.
"""
    )

    st.subheader("Limitasi")
    st.warning(
        """
- **Bukan alat diagnosis.** Hasil aplikasi tidak menggantikan pemeriksaan medis oleh tenaga kesehatan.
- **Sumber data tidak jelas.** Dataset Kaggle ini tidak menyertakan pengukuran klinis seperti kepadatan
  tulang (BMD), dan asal pengumpulan datanya tidak diketahui. Pola data yang sangat bersih
  mengindikasikan dataset kemungkinan bersifat sintetis.
- **Sinyal terkonsentrasi pada usia.** Pada dataset ini hampir seluruh daya prediksi berasal dari fitur
  usia; fitur lain tidak menambah daya prediksi yang berarti. Ini tidak berarti faktor lain tidak
  berpengaruh pada osteoporosis di dunia nyata.
- **Belum ada validasi eksternal.** Model hanya dievaluasi pada data test dari dataset yang sama, dan
  performanya belum tentu berlaku pada data klinis sungguhan.
"""
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    try:
        art = muat_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(
            "Model gagal dimuat. Penyebab paling umum: versi library di aplikasi berbeda dengan versi "
            "saat model disimpan (scikit-learn, xgboost, atau lightgbm). Samakan versinya di "
            "`requirements.txt`."
        )
        st.exception(e)
        st.stop()

    with st.sidebar:
        st.header("🦴 Osteoporosis")
        halaman = st.radio(
            "Menu",
            ["Prediksi", "Hasil Penelitian", "Feature Importance", "Tentang & Limitasi"],
        )
        st.divider()
        st.caption(f"Model: {art['model_name']}")
        st.caption("Prototipe penelitian, bukan alat diagnosis medis.")

    if halaman == "Prediksi":
        halaman_prediksi(art)
    elif halaman == "Hasil Penelitian":
        halaman_hasil()
    elif halaman == "Feature Importance":
        halaman_importance(art)
    else:
        halaman_tentang(art)


main()
