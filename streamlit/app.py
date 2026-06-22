import streamlit as st
import numpy as np
import pickle
import pandas as pd
from catboost import CatBoostClassifier
import pywt
from scipy.stats import skew, kurtosis
from scipy.fft import fftn
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Knee OA Classification System",
    page_icon="🦴",
    layout="wide"
)

# =====================================================
# Model File Names — edit if needed
# =====================================================
MODEL_1_FILE  = "model_binary_clinical.cbm"
SCALER_1_FILE = "scaler_binary_clinical.pkl"

MODEL_2_FILE  = "model_lr_kl2.pkl"
SCALER_2_FILE = "scaler_kl2.pkl"

MODEL_3_FILE  = "model_catboost_3class.cbm"
SCALER_3_FILE = "scaler_3class.pkl"

MODEL_4_FILE  = "model_catboost_5class.cbm"
SCALER_4_FILE = "scaler_5class.pkl"

REGIONS = ["femoral", "tibia_med", "tibia_lat", "med_meniscus", "lat_meniscus"]

# =====================================================
# Feature Extraction — exact same as training
# =====================================================
def extract_region_features(img, region_mask):
    vox = img[region_mask > 0]
    if len(vox) < 20:
        return np.zeros(60, dtype=np.float32)
    feats = []
    feats.extend([
        np.mean(vox), np.std(vox), np.min(vox), np.max(vox), np.median(vox),
        np.percentile(vox,5), np.percentile(vox,10), np.percentile(vox,25),
        np.percentile(vox,75), np.percentile(vox,90), np.percentile(vox,95),
        skew(vox), kurtosis(vox),
        np.percentile(vox,75) - np.percentile(vox,25)
    ])
    coords = np.argwhere(region_mask > 0)
    voxel_count = len(coords)
    feats.append(voxel_count)
    feats.append(voxel_count / region_mask.size)
    mins = coords.min(axis=0); maxs = coords.max(axis=0)
    dz, dy, dx = maxs - mins + 1
    feats.extend([dx, dy, dz, dx/(dy+1e-8), dx/(dz+1e-8), dy/(dz+1e-8)])
    centroid = coords.mean(axis=0)
    feats.extend([centroid[0], centroid[1], centroid[2]])
    feats.append(voxel_count / (dx*dy*dz + 1e-8))
    thickness_cols = []
    for x in np.unique(coords[:, 2]):
        col = coords[coords[:, 2] == x]
        if len(col) > 1:
            thickness_cols.append(col[:, 0].max() - col[:, 0].min() + 1)
    if len(thickness_cols) > 0:
        tc = np.array(thickness_cols)
        feats.extend([np.mean(tc), np.min(tc), np.max(tc), np.std(tc),
                      np.percentile(tc,25), np.percentile(tc,50), np.percentile(tc,75)])
    else:
        feats.extend([0]*7)
    spec = np.abs(fftn(region_mask.astype(np.float32)))
    feats.extend([spec.mean(), spec.std(), spec.max(), np.sum(spec**2)])
    p = spec.flatten(); p = p/(p.sum()+1e-8)
    feats.append(-np.sum(p*np.log2(p+1e-12)))
    mid = img.shape[0] // 2
    sl = img[mid]
    coeffs = pywt.dwt2(sl, "haar")
    LL,(LH,HL,HH) = coeffs
    for arr in [LL,LH,HL,HH]:
        feats.append(np.mean(arr)); feats.append(np.std(arr))
        feats.append(np.sum(arr**2))
        p = np.abs(arr).flatten(); p = p/(p.sum()+1e-8)
        feats.append(-np.sum(p*np.log2(p+1e-12)))
    sl8 = ((sl-sl.min())/(sl.max()-sl.min()+1e-8)*255).astype(np.uint8)
    glcm = graycomatrix(sl8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    feats.extend([
        graycoprops(glcm,'contrast')[0,0], graycoprops(glcm,'homogeneity')[0,0],
        graycoprops(glcm,'energy')[0,0],   graycoprops(glcm,'correlation')[0,0],
        graycoprops(glcm,'dissimilarity')[0,0], graycoprops(glcm,'ASM')[0,0]
    ])
    return np.array(feats, dtype=np.float32)


def extract_all_features(data, age=60.0, bmi=28.0, gender=1.0):
    """Extract features from all 5 regions + clinical (303 total)."""
    img = data["image"]
    all_feats = []
    for region in REGIONS:
        all_feats.extend(extract_region_features(img, data[region]))
    # Pad/trim to 300 imaging features
    all_feats = all_feats[:300]
    while len(all_feats) < 300:
        all_feats.append(0.0)
    # Add clinical features (303 total)
    all_feats.extend([age, bmi, gender])
    return np.array(all_feats, dtype=np.float32)


def preprocess(features, scaler, expected_size):
    f = features[:expected_size]
    while len(f) < expected_size:
        f = np.append(f, 0.0)
    f = f.reshape(1, -1)
    return scaler.transform(f) if scaler is not None else f


# =====================================================
# Load Models
# =====================================================
@st.cache_resource
def load_all_models():
    configs = {
        "Model 1: Binary (KL0-1 vs KL3-4)": {
            "model_file": MODEL_1_FILE, "scaler_file": SCALER_1_FILE,
            "model_type": "catboost",
            "class_names": ["No OA (KL0-1)", "Severe OA (KL3-4)"],
            "description": "Distinguishes healthy/minimal OA from severe OA",
            "needs_clinical": True, "order": 1
        },
        "Model 2: KL2 Detector (KL2 vs Rest)": {
            "model_file": MODEL_2_FILE, "scaler_file": SCALER_2_FILE,
            "model_type": "sklearn",
            "class_names": ["Not KL2", "KL2 (Moderate OA)"],
            "description": "Specifically detects the transitional KL2 stage",
            "needs_clinical": True, "order": 2
        },
        "Model 3: 3-Class (KL0-1 / KL2 / KL3-4)": {
            "model_file": MODEL_3_FILE, "scaler_file": SCALER_3_FILE,
            "model_type": "catboost",
            "class_names": ["No OA (KL0-1)", "Moderate OA (KL2)", "Severe OA (KL3-4)"],
            "description": "Three-class OA severity prediction",
            "needs_clinical": True, "order": 3
        },
        "Model 4: 5-Class (KL0 → KL4)": {
            "model_file": MODEL_4_FILE, "scaler_file": SCALER_4_FILE,
            "model_type": "catboost",
            "class_names": ["KL0 (No OA)", "KL1 (Doubtful OA)",
                            "KL2 (Moderate OA)", "KL3 (Severe OA)", "KL4 (End-stage OA)"],
            "description": "Full KL grading from 0 to 4",
            "needs_clinical": True, "order": 4
        },
    }

    models, scalers = {}, {}
    for name, cfg in configs.items():
        try:
            if not os.path.exists(cfg["model_file"]):
                st.warning(f"⚠️ Not found: {cfg['model_file']}")
                continue
            if cfg["model_type"] == "catboost":
                clf = CatBoostClassifier()
                clf.load_model(cfg["model_file"])
            else:
                with open(cfg["model_file"], "rb") as f:
                    clf = pickle.load(f)
            if os.path.exists(cfg["scaler_file"]):
                with open(cfg["scaler_file"], "rb") as f:
                    scalers[name] = pickle.load(f)
                n = scalers[name].n_features_in_ if hasattr(scalers[name], 'n_features_in_') else 303
                cfg["feature_size"] = n
            else:
                scalers[name] = None
                cfg["feature_size"] = 303
            clf._cfg = cfg
            models[name] = clf
        except Exception as e:
            st.error(f"❌ {name}: {e}")
    return models, scalers


# =====================================================
# Helper: predict one model
# =====================================================
def run_model(model, scaler, features, cfg):
    fs = preprocess(features, scaler, cfg["feature_size"])
    pred = int(model.predict(fs).ravel()[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(fs)[0]
    else:
        probs = np.zeros(len(cfg["class_names"])); probs[pred] = 1.0
    return pred, probs


# =====================================================
# Visualise MRI + masks
# =====================================================
def show_mri(data):
    from matplotlib.patches import Patch

    img = data["image"]
    D, H, W = img.shape
    mid_z = D // 2   # axial
    mid_y = H // 2   # coronal
    mid_x = W // 2   # sagittal

    region_colors = ["red", "blue", "green", "yellow", "magenta"]
    region_labels = ["Femoral", "Tibia Med", "Tibia Lat", "Med Meniscus", "Lat Meniscus"]
    cmap = mcolors.ListedColormap([(0,0,0,0)] + region_colors)

    def build_overlay(slicer):
        combined = np.zeros(slicer(img).shape, dtype=np.int32)
        for i, r in enumerate(REGIONS, start=1):
            combined[slicer(data[r]) > 0] = i
        return combined

    views = [
        ("Axial (mid)",    img[mid_z],    build_overlay(lambda x, z=mid_z: x[z])),
        ("Coronal (mid)",  img[:,mid_y,:], build_overlay(lambda x, y=mid_y: x[:,y,:])),
        ("Sagittal (mid)", img[:,:,mid_x], build_overlay(lambda x, xv=mid_x: x[:,:,xv])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('#1a1a2e')

    for col, (title, mri_slice, overlay) in enumerate(views):
        for row, (show_overlay, row_title) in enumerate([
            (False, f"{title} — MRI"),
            (True,  f"{title} — Overlay")
        ]):
            ax = axes[row][col]
            ax.imshow(mri_slice, cmap="gray")
            if show_overlay:
                ax.imshow(overlay, cmap=cmap, alpha=0.55, vmin=0, vmax=5)
            ax.set_title(row_title, color="white", fontsize=9, pad=4)
            ax.axis("off")

    # Legend bottom right
    patches = [Patch(color=c, label=l)
               for c, l in zip(region_colors, region_labels)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    st.pyplot(fig)


# =====================================================
# Main UI
# =====================================================
st.title("🦴 Knee Osteoarthritis Multi-Model Classification")
st.markdown("**Automated OA severity assessment using 3D MRI segmentation features**")

with st.spinner("Loading models..."):
    models, scalers = load_all_models()

if not models:
    st.error("No models loaded. Please place model files in the same folder as app.py")
    st.stop()

# Sidebar — clinical info + model status
with st.sidebar:
    st.markdown("## 👤 Patient Clinical Information")
    age    = st.number_input("Age (years)", min_value=18, max_value=100, value=60)
    bmi    = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=55.0, value=28.0, step=0.1)
    gender = st.radio("Gender", ["Male", "Female"])
    gender_val = 1.0 if gender == "Male" else 0.0

    st.markdown("---")
    st.markdown("## ✅ Loaded Models")
    for name in models:
        st.success(f"✓ {name.split(':')[0]}")

    st.markdown("---")
    st.markdown("## 📋 KL Grade Scale")
    st.markdown("""
| Grade | Description |
|---|---|
| KL0 | No OA |
| KL1 | Doubtful OA |
| KL2 | Moderate OA |
| KL3 | Severe OA |
| KL4 | End-stage OA |
""")

# Tabs
tab1, tab2 = st.tabs(["🎯 Single Model", "📊 Compare All Models"])

# =====================================================
# Tab 1 — Single Model
# =====================================================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        chosen = st.selectbox("Select Model:", list(models.keys()))
        npz_file = st.file_uploader("Upload .npz file", type=["npz"], key="single")
    with col2:
        if chosen:
            cfg = models[chosen]._cfg
            st.info(f"""
**{chosen}**
- {cfg['description']}
- Classes: {' | '.join(cfg['class_names'])}
""")

    if npz_file and chosen:
        try:
            data = np.load(npz_file)
            model  = models[chosen]
            scaler = scalers.get(chosen)
            cfg    = model._cfg

            with st.spinner("Extracting features and predicting..."):
                features = extract_all_features(data, age, bmi, gender_val)
                pred, probs = run_model(model, scaler, features, cfg)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("🏥 Prediction",  cfg['class_names'][pred])
            c2.metric("📊 Confidence",  f"{max(probs)*100:.1f}%")
            c3.metric("🔢 Model",       f"{len(cfg['class_names'])}-Class")

            prob_df = pd.DataFrame({
                "Class": cfg['class_names'],
                "Probability": (probs * 100).round(1)
            })
            st.bar_chart(prob_df.set_index("Class"))

            st.markdown("### MRI Visualization")
            show_mri(data)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# =====================================================
# Tab 2 — Compare All Models
# =====================================================
with tab2:
    npz_all = st.file_uploader("Upload .npz file", type=["npz"], key="compare")

    if npz_all:
        try:
            data = np.load(npz_all)
            results = {}
            prog = st.progress(0)

            with st.spinner("Running all models..."):
                features = extract_all_features(data, age, bmi, gender_val)
                for i, (name, model) in enumerate(models.items()):
                    scaler = scalers.get(name)
                    cfg    = model._cfg
                    pred, probs = run_model(model, scaler, features, cfg)
                    results[name] = {
                        "pred": pred,
                        "label": cfg['class_names'][pred],
                        "probs": probs,
                        "class_names": cfg['class_names'],
                        "confidence": float(max(probs)),
                        "order": cfg['order']
                    }
                    prog.progress((i+1)/len(models))

            # Summary table
            st.markdown("### 📊 Results Summary")
            rows = sorted(results.items(), key=lambda x: x[1]['order'])
            df = pd.DataFrame([{
                "Model":      n.split(':')[0],
                "Prediction": r['label'],
                "Confidence": f"{r['confidence']*100:.1f}%"
            } for n,r in rows])
            st.dataframe(df, use_container_width=True)

            # Probability bars
            st.markdown("### 📈 Probability Distribution")
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
            if len(results) == 1:
                axes = [axes]
            for ax, (name, r) in zip(axes, rows):
                colors = ['#2196F3' if i != r['pred'] else '#F44336'
                          for i in range(len(r['probs']))]
                ax.barh(r['class_names'], r['probs'], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title(name.split(':')[0], fontsize=9)
                for i, p in enumerate(r['probs']):
                    ax.text(p+0.01, i, f"{p*100:.1f}%", va='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            # Clinical recommendation
            st.markdown("### 💡 Clinical Recommendation")
            if "Model 4: 5-Class (KL0 → KL4)" in results:
                label = results["Model 4: 5-Class (KL0 → KL4)"]["label"]
                if "KL0" in label or "KL1" in label:
                    st.success(f"✅ **{label}** — Routine monitoring recommended")
                elif "KL2" in label:
                    st.warning(f"⚠️ **{label}** — Conservative treatment and regular follow-up")
                elif "KL3" in label:
                    st.error(f"🔴 **{label}** — Medical treatment, consider specialist referral")
                else:
                    st.error(f"🔴 **{label}** — Surgical consultation recommended")

            # MRI
            st.markdown("### 🖼️ MRI Visualization")
            show_mri(data)

            # Download JSON
            export = {n: {"prediction": r['label'], "confidence": r['confidence'],
                          "probabilities": {c: float(p)
                          for c,p in zip(r['class_names'], r['probs'])}}
                      for n,r in results.items()}
            st.download_button(
                "📥 Download Results (JSON)",
                data=json.dumps(export, indent=2),
                file_name="oa_predictions.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

st.markdown("---")
st.caption("🦴 Knee OA Multi-Model Classification System | Kahla Chiraz | 2025/2026")