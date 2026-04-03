import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="RespireAI Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
.stApp { background: #050810; color: #c8d6e5; font-family: 'Syne', sans-serif; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
.stApp > header { display: none; }

.hero { position: relative; padding: 3rem 0 1.5rem; text-align: center; }
.hero::before {
    content: ''; position: absolute; top: -40px; left: 50%;
    transform: translateX(-50%); width: 500px; height: 260px;
    background: radial-gradient(ellipse, rgba(56,189,248,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    letter-spacing: 4px; color: #38bdf8; margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: center; gap: 12px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: ''; width: 40px; height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8);
}
.hero-eyebrow::after { background: linear-gradient(90deg, #38bdf8, transparent); }
.hero-title {
    font-size: clamp(3.2rem, 7vw, 5.5rem); font-weight: 800; line-height: 1; letter-spacing: -2px;
    background: linear-gradient(135deg, #e2e8f0 0%, #38bdf8 50%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.8rem;
}
.hero-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #475569; letter-spacing: 1px; }

.stat-strip {
    display: flex; justify-content: center; gap: 3rem; margin: 1.5rem 0;
    padding: 1.2rem 0; border-top: 1px solid rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stat-item { text-align: center; }
.stat-num { font-size: 1.7rem; font-weight: 800; color: #38bdf8; line-height: 1; }
.stat-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #334155; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

.sec-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 3px; color: #38bdf8;
    padding-bottom: 0.7rem; border-bottom: 1px solid rgba(56,189,248,0.1);
    margin-bottom: 1rem; display: block;
}

.metrics-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 1.2rem; }
.metric-card { background: #080f1c; border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; padding: 1.1rem; border-top: 2px solid; }
.metric-val { font-size: 2.4rem; font-weight: 800; line-height: 1; letter-spacing: -1px; }
.metric-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 2px; color: #475569; margin-top: 6px; }

.disease-item { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.025); }
.disease-item:last-child { border-bottom: none; }
.disease-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.disease-name { font-size: 0.95rem; font-weight: 500; flex: 1; color: #cbd5e1; }
.bar-bg { width: 75px; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; flex-shrink: 0; }
.disease-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; min-width: 45px; text-align: right; flex-shrink: 0; }

.gcam-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; text-align: center; color: #475569; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(56,189,248,0.1), transparent); margin: 1.4rem 0; }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 300px; gap: 0.8rem; }
.empty-icon  { font-size: 3.2rem; opacity: 0.12; filter: grayscale(1); }
.empty-text  { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #1e293b; text-align: center; letter-spacing: 1px; line-height: 2; }

.stButton > button {
    width: 100%; background: linear-gradient(135deg,#0ea5e9,#6366f1) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.75rem !important; font-family: 'Syne',sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
}
.stDownloadButton > button {
    background: transparent !important; color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.2) !important; border-radius: 10px !important;
    padding: 0.75rem 2rem !important; font-family: 'JetBrains Mono',monospace !important;
    font-size: 0.75rem !important; letter-spacing: 2px !important; text-transform: uppercase !important;
}
div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid rgba(56,189,248,0.07); }
section[data-testid="stFileUploadDropzone"] {
    background: rgba(15,23,42,0.4) !important;
    border: 1px dashed rgba(56,189,248,0.12) !important; border-radius: 12px !important;
}

/* ── REPORT (self-contained single block) ── */
.report-wrap {
    background: #060c18;
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-top: 0.5rem;
}
.rp-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem; font-weight: 500; color: #38bdf8;
    letter-spacing: 3px; padding-bottom: 1rem; margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(56,189,248,0.15);
}
.rp-sec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: 3px;
    color: #38bdf8; opacity: 0.55; margin: 1.2rem 0 0.6rem;
}
.rp-row {
    display: flex; font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; line-height: 2.1; color: #64748b;
}
.rp-k { min-width: 150px; color: #475569; }
.rp-v { color: #94a3b8; }
.rp-div { height: 1px; background: rgba(56,189,248,0.07); margin: 0.8rem 0; }
.rp-finding {
    font-family: 'JetBrains Mono', monospace;
    padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.03);
    line-height: 1.7;
}
.rp-fn  { font-size: 0.9rem; color: #7dd3fc; font-weight: 500; }
.rp-fp  { font-size: 0.85rem; color: #94a3b8; }
.rp-fd  { font-size: 0.75rem; color: #475569; display: block; margin-top: 2px; }
.rp-nof { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #22c55e; padding: 0.5rem 0; }
.rp-disc { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #334155; line-height: 1.9; }
</style>
""", unsafe_allow_html=True)

LABELS = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pneumonia','Pneumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
]
DISEASE_INFO = {
    'Atelectasis':        'Partial or complete lung collapse',
    'Cardiomegaly':       'Enlargement of the heart',
    'Effusion':           'Excess fluid around the lungs',
    'Infiltration':       'Dense substance in lung parenchyma',
    'Mass':               'Lesion greater than 3cm diameter',
    'Nodule':             'Small rounded lesion under 3cm',
    'Pneumonia':          'Infection causing lung inflammation',
    'Pneumothorax':       'Air leaking around the lungs',
    'Consolidation':      'Airspace filled with liquid',
    'Edema':              'Excess fluid in lung tissue',
    'Emphysema':          'Damaged alveoli in the lungs',
    'Fibrosis':           'Scarring of lung tissue',
    'Pleural_Thickening': 'Scarring of the pleural lining',
    'Hernia':             'Organ protruding through chest wall'
}
MODEL_PATH     = "../best_model.pth"
THRESHOLD_PATH = "../best_thresholds.json"

@st.cache_resource
def load_thresholds():
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f: return json.load(f)
    return {l: 0.5 for l in LABELS}

@st.cache_resource
def load_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 14)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval(); return m
    return None

class GradCAM:
    def __init__(self, model, layer):
        self.model = model; self.grads = self.acts = None
        layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o))
        layer.register_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0]))
    def generate(self, t, idx):
        out = self.model(t); self.model.zero_grad(); out[0,idx].backward()
        g = self.grads[0].cpu().data.numpy(); a = self.acts[0].cpu().data.numpy()
        w = np.mean(g, axis=(1,2))
        cam = sum(w[i]*a[i] for i in range(len(w)))
        cam = np.maximum(cam,0); cam = cv2.resize(cam,(224,224))
        cam -= cam.min()
        if cam.max()>0: cam /= cam.max()
        return cam

def make_overlay(img, cam):
    h = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_INFERNO)
    o = 0.55*np.float32(h)/255 + 0.45*np.float32(img)/255
    return (o/o.max()*255).astype(np.uint8)

tfm = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def sev(pct):
    if pct<10:  return "Mild",     "#22c55e"
    if pct<30:  return "Moderate", "#eab308"
    if pct<60:  return "Severe",   "#f97316"
    return "Critical", "#ef4444"

def build_report_html(detected, dmg, sv, sv_col, now):
    findings = ""
    if not detected:
        findings = '<div class="rp-nof">No significant findings detected.</div>'
    else:
        for d, p in detected:
            findings += f"""<div class="rp-finding">
                <span class="rp-fn">{d}</span>
                <span class="rp-fp"> &nbsp;—&nbsp; {p*100:.1f}%</span>
                <span class="rp-fd">{DISEASE_INFO.get(d,'')}</span>
            </div>"""
    return f"""
    <div class="report-wrap">
        <div class="rp-head">CHEST X-RAY ANALYSIS REPORT</div>
        <div class="rp-sec">Report Info</div>
        <div class="rp-row"><span class="rp-k">Date</span><span class="rp-v">{now}</span></div>
        <div class="rp-row"><span class="rp-k">Model</span><span class="rp-v">DenseNet-121</span></div>
        <div class="rp-row"><span class="rp-k">Dataset</span><span class="rp-v">NIH ChestX-ray14</span></div>
        <div class="rp-row"><span class="rp-k">Classes</span><span class="rp-v">14 thoracic pathologies</span></div>
        <div class="rp-div"></div>
        <div class="rp-sec">Findings</div>
        {findings}
        <div class="rp-div"></div>
        <div class="rp-sec">Quantitative Analysis</div>
        <div class="rp-row"><span class="rp-k">Affected area</span><span class="rp-v">{dmg:.1f}%</span></div>
        <div class="rp-row"><span class="rp-k">Severity</span><span class="rp-v" style="color:{sv_col}">{sv}</span></div>
        <div class="rp-row"><span class="rp-k">Findings</span><span class="rp-v">{len(detected)} of 14 classes</span></div>
        <div class="rp-div"></div>
        <div class="rp-disc">AI-generated report not for clinical use.<br>Always consult a qualified radiologist for medical decisions.</div>
    </div>"""

def plain_report(detected, dmg, sv):
    now = datetime.now().strftime("%d %b %Y %H:%M")
    lines = ["CHEST X-RAY ANALYSIS REPORT","="*42,
             f"Date     : {now}","Model    : DenseNet-121",
             "Dataset  : NIH ChestX-ray14","Classes  : 14 thoracic pathologies",
             "","FINDINGS","-"*42]
    if not detected:
        lines.append("No significant findings detected.")
    else:
        for d,p in detected:
            lines += [f"[+] {d:<22} {p*100:5.1f}%",f"    {DISEASE_INFO.get(d,'')}"]
    lines += ["","QUANTITATIVE ANALYSIS","-"*42,
              f"Affected area  : {dmg:.1f}%",f"Severity       : {sv}",
              f"Findings       : {len(detected)} / 14",
              "","DISCLAIMER","-"*42,
              "AI-generated. Not for clinical use.",
              "Consult a qualified radiologist.",""]
    return "\n".join(lines)
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Radiology</div>
    <div class="hero-title">RespireAI</div>
    <div class="hero-sub">CHEST X-RAY DISEASE DETECTION &nbsp;·&nbsp; DENSENET-121 &nbsp;·&nbsp; 14 PATHOLOGIES</div>
</div>
<div class="stat-strip">
    <div class="stat-item"><div class="stat-num">14</div><div class="stat-label">Diseases</div></div>
    <div class="stat-item"><div class="stat-num">112K</div><div class="stat-label">Images</div></div>
    <div class="stat-item"><div class="stat-num">NIH</div><div class="stat-label">Dataset</div></div>
    <div class="stat-item"><div class="stat-num">DenseNet</div><div class="stat-label">Architecture</div></div>
</div>
""", unsafe_allow_html=True)

model      = load_model()
thresholds = load_thresholds()
if model is None:
    st.error("Model not found at `../best_model.pth`")
    st.stop()

left, right = st.columns([1,1], gap="large")

with left:
    st.markdown('<span class="sec-title">Upload X-Ray</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("xray", type=["png","jpg","jpeg"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("⚡  RUN ANALYSIS")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🫁</div>
            <div class="empty-text">DROP CHEST X-RAY HERE<br>PNG &nbsp;·&nbsp; JPG &nbsp;·&nbsp; JPEG</div>
        </div>""", unsafe_allow_html=True)
        run = False

with right:
    st.markdown('<span class="sec-title">Results</span>', unsafe_allow_html=True)
    if not uploaded or not run:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-text">UPLOAD AN X-RAY AND<br>CLICK RUN ANALYSIS</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing..."):
            img_arr = np.array(image.resize((224,224)))
            tensor  = tfm(image).unsqueeze(0)
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor)).cpu().numpy()[0]
            detected = sorted(
                [(LABELS[i], float(probs[i])) for i in range(14)
                 if probs[i] > thresholds.get(LABELS[i], 0.5)], 
                key=lambda x: x[1], reverse=True
            )
            gc   = GradCAM(model, model.features[-1])
            top3 = np.argsort(probs)[::-1][:3]
            cams = [(LABELS[i], float(probs[i]),
                     gc.generate(tfm(image).unsqueeze(0).requires_grad_(True), i))
                    for i in top3]
            combined = np.max([c for _,_,c in cams], axis=0)
            dmg_pct  = (combined > 0.5).sum() / combined.size * 100
            sv, sv_col = sev(dmg_pct)

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card" style="border-top-color:#38bdf8">
                <div class="metric-val" style="color:#38bdf8">{len(detected)}</div>
                <div class="metric-lbl">Detected</div>
            </div>
            <div class="metric-card" style="border-top-color:#f472b6">
                <div class="metric-val" style="color:#f472b6">{dmg_pct:.0f}%</div>
                <div class="metric-lbl">Affected Area</div>
            </div>
            <div class="metric-card" style="border-top-color:{sv_col}">
                <div class="metric-val" style="color:{sv_col};font-size:1.9rem">{sv}</div>
                <div class="metric-lbl">Severity</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<span class="sec-title">All 14 Pathologies</span>', unsafe_allow_html=True)
        for i, label in enumerate(LABELS):
            p   = float(probs[i])
            hit = p > thresholds.get(label, 0.5)
            col = "#ef4444" if hit else "#22c55e"
            st.markdown(f"""
            <div class="disease-item">
                <div class="disease-dot" style="background:{col};box-shadow:0 0 5px {col}55"></div>
                <div class="disease-name">{label}</div>
                <div class="bar-bg"><div style="background:{col};width:{min(p*100,100):.1f}%;height:4px;border-radius:2px"></div></div>
                <div class="disease-pct" style="color:{'#fca5a5' if hit else '#4ade80'}">{p*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<span class="sec-title">Grad-CAM Activation Maps</span>', unsafe_allow_html=True)
        g1,g2,g3 = st.columns(3)
        for gcol,(lbl,prob,cam) in zip([g1,g2,g3], cams):
            with gcol:
                st.image(make_overlay(img_arr,cam), use_container_width=True)
                st.markdown(f'<div class="gcam-lbl">{lbl}<br>{prob*100:.1f}%</div>', unsafe_allow_html=True)

if uploaded and run and 'detected' in dir():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sec-title">Generated Report</span>', unsafe_allow_html=True)

    now = datetime.now().strftime("%d %b %Y  %H:%M")

    st.markdown(build_report_html(detected, dmg_pct, sv, sv_col, now), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, dl_col, _ = st.columns([1,1,1])
    with dl_col:
        st.download_button(
            "↓  DOWNLOAD REPORT",
            data=plain_report(detected, dmg_pct, sv),
            file_name=f"respireai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )