"""
=============================================================
  Stanford Dogs — Aplicación Streamlit de Inferencia
  Autor: Jorge - UNAB - Bases de Datos II
  Uso local   : streamlit run app.py
  Deploy      : https://streamlit.io/cloud
=============================================================
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA (debe ir primero)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title = "DogBreed AI · Clasificador de Razas",
    page_icon  = "🐾",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
#  CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0d0d0f; color: #e8e3d9; }

  [data-testid="stSidebar"] {
    background: #111115 !important;
    border-right: 1px solid #252530 !important;
  }

  .hero-wrap  { padding: 0.5rem 0 1.8rem 0; }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 700;
    background: linear-gradient(130deg, #f5c518 0%, #e8964a 55%, #d44a4a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.05;
    margin: 0 0 0.35rem 0;
  }
  .hero-sub {
    font-size: 0.85rem;
    color: #52504a;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .panel-label {
    font-size: 0.72rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #52504a;
    margin-bottom: 0.8rem;
  }
  [data-testid="stFileUploadDropzone"] {
    background: #0d0d0f !important;
    border: 2px dashed #2a2a35 !important;
    border-radius: 12px !important;
  }

  .stButton > button {
    background: linear-gradient(120deg, #f5c518, #e8964a) !important;
    color: #0d0d0f !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    letter-spacing: 0.03em;
    width: 100%;
    transition: opacity 0.2s, transform 0.15s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
  .stButton > button:disabled { opacity: 0.3 !important; }

  .result-card {
    background: linear-gradient(145deg, #161619 0%, #1e1e26 100%);
    border: 1px solid #2a2a38;
    border-radius: 18px;
    padding: 1.6rem 1.8rem 1.4rem;
    box-shadow: 0 12px 48px rgba(0,0,0,0.55);
  }
  .badge-tag {
    display: inline-block;
    background: rgba(245,197,24,0.10);
    color: #f5c518;
    border: 1px solid rgba(245,197,24,0.28);
    border-radius: 30px;
    padding: 0.18rem 0.85rem;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
  }
  .winner-name {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.5rem, 3vw, 2.1rem);
    color: #f5c518;
    margin: 0.15rem 0 0.5rem;
    line-height: 1.15;
  }
  .winner-pct {
    font-size: clamp(2.8rem, 6vw, 4rem);
    font-weight: 700;
    color: #e8964a;
    line-height: 1;
    letter-spacing: -0.02em;
  }
  .conf-bar-bg {
    background: #1a1a22;
    border-radius: 99px;
    height: 8px;
    margin-top: 1rem;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #f5c518, #e8964a);
  }
  .conf-label { color: #3e3e50; font-size: 0.78rem; margin-top: 0.45rem; }

  .hr { border-top: 1px solid #1e1e28; margin: 1.4rem 0; }

  .empty-state {
    background: #0f0f13;
    border: 1.5px dashed #1e1e28;
    border-radius: 18px;
    padding: 4rem 2rem;
    text-align: center;
  }
  .empty-icon  { font-size: 3.2rem; line-height: 1; margin-bottom: 0.8rem; }
  .empty-title { font-family: 'Playfair Display', serif; font-size: 1.15rem; color: #2a2a38; margin-bottom: 0.4rem; }
  .empty-sub   { font-size: 0.82rem; color: #242432; }

  .sb-heading { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #f5c518; }
  .sb-stat { display:flex; justify-content:space-between; padding:0.45rem 0; border-bottom:1px solid #1e1e28; font-size:0.82rem; }
  .sb-stat-key { color: #52504a; }
  .sb-stat-val { color: #c8c0b0; font-weight: 500; }
  .sb-step { display:flex; gap:0.75rem; align-items:flex-start; padding:0.55rem 0; font-size:0.83rem; color:#706860; }
  .sb-step-num { background:rgba(245,197,24,0.12); color:#f5c518; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:700; flex-shrink:0; }
  .sb-footer { font-size:0.72rem; color:#2a2a38; text-align:center; padding-top:0.5rem; }

  [data-testid="stExpander"] { background:#111115 !important; border:1px solid #1e1e28 !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────
MODEL_PATH  = "model/dog_breed_mobilenetv2.h5"
LABELS_PATH = "model/class_labels.json"
IMG_SIZE    = (224, 224)
TOP_K       = 5

# ─────────────────────────────────────────────
#  CARGA DEL MODELO (cacheado)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")

    labels = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            raw = json.load(f)
        for k, v in raw.items():
            clean = v.split("-")[-1].replace("_", " ").title()
            labels[int(k)] = clean

    return model, labels

def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def build_bar_chart(breeds: list, probs: list) -> go.Figure:
    colors = ["#f5c518" if i == 0 else "#26262e" for i in range(len(breeds))]
    fig = go.Figure(go.Bar(
        x=probs, y=breeds, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#c8c0b0", size=12, family="DM Sans"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0, max(probs) * 1.28]),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed",
                   tickfont=dict(color="#a09890", size=12.5, family="DM Sans")),
        margin=dict(l=0, r=64, t=6, b=6), height=230, bargap=0.38,
    )
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sb-heading'>🐾 DogBreed AI</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1e1e28;margin:0.8rem 0 1rem'></div>", unsafe_allow_html=True)

    for key, val in [("Modelo","MobileNetV2"),("Técnica","Transfer Learning"),
                     ("Dataset","Stanford Dogs"),("Razas","120 clases"),
                     ("Imágenes","~20 000"),("Framework","TensorFlow/Keras")]:
        st.markdown(f"<div class='sb-stat'><span class='sb-stat-key'>{key}</span><span class='sb-stat-val'>{val}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-label'>¿Cómo usar?</div>", unsafe_allow_html=True)

    for i, paso in enumerate(["Sube una foto de perro (JPG, PNG, WEBP)",
                               "Haz clic en <b>Clasificar raza</b>",
                               "Revisa las predicciones y probabilidades"], 1):
        st.markdown(f"<div class='sb-step'><div class='sb-step-num'>{i}</div><div>{paso}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#1e1e28;margin:1.2rem 0 1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-footer'>Proyecto académico · UNAB<br>Bases de Datos II</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CARGAR MODELO
# ─────────────────────────────────────────────
model, labels = load_model_and_labels()
model_ready   = model is not None and len(labels) > 0

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-title'>DogBreed AI</div>
  <div class='hero-sub'>Clasificador de razas · Stanford Dogs · 120 clases</div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.warning(
        "**Modelo no encontrado.** Coloca `model/dog_breed_mobilenetv2.h5` y "
        "`model/class_labels.json` en la misma carpeta que `app.py`, "
        "luego entrena con `python train.py`.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────
#  LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.15], gap="large")

# ── COLUMNA IZQUIERDA: Upload ──────────────────
with col_left:
    st.markdown("<div class='panel-label'>📤 Subir imagen</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("foto", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True,
                 caption=f"📷 {uploaded.name}  •  {img.width}×{img.height} px")
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        classify_btn = st.button("🔍  Clasificar raza", use_container_width=True,
                                 disabled=not model_ready)
    else:
        st.markdown("""
        <div style='border:2px dashed #1e1e28;border-radius:14px;padding:3.5rem 1.5rem;
                    text-align:center;color:#28282e;margin-bottom:0.8rem;'>
          <div style='font-size:2.8rem;margin-bottom:0.6rem;'>🐕</div>
          <div style='font-size:0.9rem;'>Arrastra una foto aquí<br>o usa el botón de arriba</div>
        </div>""", unsafe_allow_html=True)
        classify_btn = False

# ── COLUMNA DERECHA: Resultados ────────────────
with col_right:
    st.markdown("<div class='panel-label'>🏆 Resultado de la clasificación</div>",
                unsafe_allow_html=True)

    if uploaded and classify_btn:
        with st.spinner("Analizando imagen con el modelo…"):
            preds = model.predict(preprocess(img), verbose=0)[0]

        top_idx    = np.argsort(preds)[::-1][:TOP_K]
        top_breeds = [labels.get(i, f"Clase {i}") for i in top_idx]
        top_probs  = [float(preds[i]) * 100 for i in top_idx]

        w_breed = top_breeds[0]
        w_prob  = top_probs[0]

        # Card resultado principal
        st.markdown(f"""
<div class='result-card'>
  <div class='badge-tag'>✨ Raza detectada</div>
  <div class='winner-name'>{w_breed}</div>
  <div class='winner-pct'>{w_prob:.1f}%</div>
  <div class='conf-bar-bg'>
    <div class='conf-bar-fill' style='width:{min(w_prob,100):.1f}%'></div>
  </div>
  <div class='conf-label'>Nivel de confianza del modelo</div>
</div>
""", unsafe_allow_html=True)

        # Gráfica Top 5
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>Top 5 predicciones</div>", unsafe_allow_html=True)
        st.plotly_chart(build_bar_chart(top_breeds, top_probs),
                        use_container_width=True,
                        config={"displayModeBar": False})

        # Tabla expandible
        with st.expander("Ver tabla completa de probabilidades"):
            import pandas as pd
            df = pd.DataFrame({
                "Posición"        : [f"#{i+1}" for i in range(TOP_K)],
                "Raza"            : top_breeds,
                "Probabilidad (%)": [f"{p:.2f}" for p in top_probs],
                "Barra"           : top_probs,
            })
            st.dataframe(
                df.style.bar(subset=["Barra"], color="#f5c518", vmin=0, vmax=100),
                hide_index=True, use_container_width=True,
            )

    elif uploaded and not classify_btn:
        st.markdown("""
<div class='empty-state'>
  <div class='empty-icon'>👆</div>
  <div class='empty-title'>Listo para clasificar</div>
  <div class='empty-sub'>Haz clic en <b style='color:#e8964a'>Clasificar raza</b></div>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div class='empty-state'>
  <div class='empty-icon'>🐾</div>
  <div class='empty-title'>Las predicciones aparecerán aquí</div>
  <div class='empty-sub'>Sube una imagen a la izquierda para comenzar</div>
</div>""", unsafe_allow_html=True)
