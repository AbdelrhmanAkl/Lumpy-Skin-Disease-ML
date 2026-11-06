# ============================================================
# LSD Risk Studio - Streamlit (v3)
# ============================================================
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LSD Risk Studio", page_icon="🐄", layout="wide",
                   initial_sidebar_state="collapsed")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "lumpy_skin_model.joblib"
METRICS_PATH = BASE / "artifacts" / "metrics.json"
FEATURES_PATH = BASE / "artifacts" / "feature_names.json"
GITHUB = "https://github.com/AbdelrhmanAkl/lumpy-skin-disease-ml"

# label, min, max, step, default
META = {
    "x": ("Longitude", -180.0, 180.0, 0.01, 31.0),
    "y": ("Latitude", -90.0, 90.0, 0.01, 30.8),
    "elevation": ("Elevation (m)", -100.0, 6000.0, 1.0, 10.0),
    "dominant_land_cover": ("Land cover class", 0.0, 255.0, 1.0, 12.0),
    "tmp": ("Mean temperature (°C)", -50.0, 50.0, 0.1, 21.0),
    "tmn": ("Min temperature (°C)", -50.0, 45.0, 0.1, 14.0),
    "tmx": ("Max temperature (°C)", -40.0, 60.0, 0.1, 28.0),
    "dtr": ("Day–night range (°C)", 0.0, 30.0, 0.1, 13.0),
    "pre": ("Rain (mm/month)", 0.0, 2000.0, 1.0, 15.0),
    "wet": ("Wet days", 0.0, 31.0, 0.1, 2.0),
    "cld": ("Cloud cover (%)", 0.0, 100.0, 0.1, 20.0),
    "vap": ("Vapour pressure (hPa)", 0.0, 50.0, 0.1, 14.0),
    "pet": ("Evapotranspiration (mm/day)", 0.0, 15.0, 0.1, 3.5),
    "frs": ("Frost days", 0.0, 31.0, 0.1, 0.0),
    "X5_Ct_2010_Da": ("Cattle (head/km²)", 0.0, 5000.0, 1.0, 150.0),
    "X5_Bf_2010_Da": ("Buffalo (head/km²)", 0.0, 5000.0, 1.0, 40.0),
}
GROUPS = {
    "Location": ["x", "y", "elevation", "dominant_land_cover"],
    "Climate": ["tmp", "tmn", "tmx", "dtr", "pre", "wet", "cld", "vap", "pet", "frs"],
    "Livestock": ["X5_Ct_2010_Da", "X5_Bf_2010_Da"],
}
# Illustrative examples, replace with real rows from your dataset
PRESETS = {
    "Nile Delta farmland": dict(x=31.0, y=30.8, elevation=10, dominant_land_cover=12, tmp=21, tmn=14, tmx=28,
                                dtr=13, pre=15, wet=2, cld=20, vap=14, pet=3.5, frs=0, X5_Ct_2010_Da=150, X5_Bf_2010_Da=40),
    "Cool highland": dict(x=38.7, y=9.0, elevation=2300, dominant_land_cover=12, tmp=16, tmn=9, tmx=23,
                          dtr=14, pre=40, wet=4, cld=45, vap=11, pet=3.0, frs=0, X5_Ct_2010_Da=60, X5_Bf_2010_Da=0),
    "Hot humid lowland": dict(x=90.4, y=23.8, elevation=8, dominant_land_cover=14, tmp=29, tmn=25, tmx=33,
                              dtr=8, pre=300, wet=18, cld=70, vap=30, pet=4.2, frs=0, X5_Ct_2010_Da=900, X5_Bf_2010_Da=120),
}

# ------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--ink:#101828;--mute:#667085;--line:#dde3ec;--brand:#2f4bff;--low:#0e9f8e;--mod:#e8a317;--high:#e5484d;}
html,body,.stApp,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink);}
.stApp{background:#eef1f6;}
h1,h2,h3,h4{font-family:'Sora',sans-serif!important;letter-spacing:-.02em;color:var(--ink);}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent;height:0;}
.block-container{max-width:1200px;padding-top:1.4rem;padding-bottom:3rem;}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;}
.brand{display:flex;gap:10px;align-items:center;font:700 1.15rem 'Sora';}
.logo{width:36px;height:36px;border-radius:11px;background:var(--ink);display:grid;place-items:center;font-size:19px;}
.topbar a{color:var(--ink);text-decoration:none;font-weight:700;font-size:.85rem;border:1px solid var(--ink);padding:6px 14px;border-radius:99px;}
.topbar a:hover{background:var(--ink);color:#fff;}
.hero{background:#101828;border-radius:24px;padding:2.4rem 2.6rem;margin-bottom:1rem;position:relative;overflow:hidden;}
.hero:after{content:"";position:absolute;right:-80px;top:-80px;width:340px;height:340px;border-radius:50%;
 background:radial-gradient(circle,#2f4bff88,transparent 70%);}
.hero h1{color:#fff;font-size:2.5rem;font-weight:800;line-height:1.1;margin:0 0 .7rem;max-width:640px;}
.hero p{color:#b8c2d6;max-width:560px;margin:0;font-size:1.02rem;}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.2rem;}
.kpi{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.9rem 1.1rem;}
.kpi b{display:block;font:800 1.55rem 'Sora';}
.kpi span{color:var(--mute);font-size:.82rem;}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:18px!important;border-color:var(--line)!important;}
button[data-baseweb="tab"]{font-weight:700;}
div[data-baseweb="tab-highlight"]{background:var(--brand)!important;}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--brand);border:0;border-radius:12px;font-weight:700;padding:.65rem 1rem;}
.res{display:grid;grid-template-columns:200px 1fr;gap:18px;align-items:center;}
.res svg{width:200px;}
.badge{display:inline-block;color:#fff;border-radius:99px;padding:4px 14px;font-weight:700;font-size:.85rem;}
.res p{color:var(--mute);margin:.6rem 0 0;font-size:.93rem;}
.fac{display:grid;grid-template-columns:170px 1fr 28px;gap:8px;align-items:center;font-size:.85rem;margin:7px 0;}
.trk{height:8px;background:#f0f2f6;border-radius:9px;position:relative;}
.trk i{position:absolute;top:0;height:8px;border-radius:9px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------ artifacts
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


model = load_model()
metrics = load_json(str(METRICS_PATH))
FEATURES = load_json(str(FEATURES_PATH))
if not FEATURES:
    st.error("feature_names.json not found.")
    st.stop()

test = metrics.get("test", {})
acc, f1, auc = (float(test.get(k, 0)) for k in ("accuracy", "f1", "roc_auc"))
model_name = metrics.get("model", type(model).__name__)
COL = {"low": "#0e9f8e", "moderate": "#e8a317", "high": "#e5484d"}
TXT = {
    "low": ("Low risk", "Conditions look unfavourable for LSD spread."),
    "moderate": ("Moderate risk", "Mixed signals. Monitor the herd and re-check with fresh data."),
    "high": ("High risk", "Conditions resemble past outbreaks. Consider a vet inspection and a vaccination review."),
}


def band(p):
    return "low" if p < 0.30 else "moderate" if p < 0.60 else "high"


def contributions(df):
    try:
        import xgboost as xgb
        if hasattr(model, "steps") or not hasattr(model, "get_booster"):
            return None
        c = model.get_booster().predict(xgb.DMatrix(df), pred_contribs=True)[0]
        return dict(zip(FEATURES, map(float, c[:-1])))
    except Exception:
        return None


def gauge(p):
    b = band(p)
    return f"""<div class="res"><svg viewBox="0 0 200 118">
<path d="M10 100A90 90 0 0 1 190 100" fill="none" stroke="#eef1f6" stroke-width="18" stroke-linecap="round"/>
<path d="M10 100A90 90 0 0 1 190 100" fill="none" stroke="{COL[b]}" stroke-width="18" stroke-linecap="round"
 stroke-dasharray="{max(p, .005) * 283:.1f} 283"/>
<text x="100" y="98" text-anchor="middle" font-family="Sora" font-weight="800" font-size="32" fill="#101828">{p*100:.1f}%</text></svg>
<div><span class="badge" style="background:{COL[b]}">{TXT[b][0]}</span><p>{TXT[b][1]}</p></div></div>"""


# ------------------------------------------------------------ header
st.markdown(f"""
<div class="topbar"><div class="brand"><div class="logo">🐄</div>LSD Risk Studio</div>
<a href="{GITHUB}" target="_blank">GitHub</a></div>
<div class="hero"><h1>Check lumpy skin disease risk for any location.</h1>
<p>Set the climate and livestock conditions, and get an instant risk score from a trained {model_name} model.</p></div>
<div class="kpis">
<div class="kpi"><b>{acc*100:.1f}%</b><span>Accuracy on unseen data</span></div>
<div class="kpi"><b>{f1*100:.1f}%</b><span>F1 score</span></div>
<div class="kpi"><b>{auc*100:.1f}%</b><span>ROC-AUC</span></div>
<div class="kpi"><b>{len(FEATURES)}</b><span>Input features</span></div></div>
""", unsafe_allow_html=True)

tab_one, tab_batch, tab_about = st.tabs(["Check a location", "Check a file", "How it works"])

# ------------------------------------------------------------ single
with tab_one:
    for f in FEATURES:
        st.session_state.setdefault(f"in_{f}", float(META[f][4]) if f in META else 0.0)

    def apply_preset():
        name = st.session_state["preset"]
        if name in PRESETS:
            for k, v in PRESETS[name].items():
                st.session_state[f"in_{k}"] = float(v)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.selectbox("Start from an example", ["Custom values"] + list(PRESETS),
                         key="preset", on_change=apply_preset,
                         help="Examples are illustrative. Edit any value afterwards.")
            for group, feats in GROUPS.items():
                feats = [f for f in feats if f in FEATURES]
                if not feats:
                    continue
                st.markdown(f"#### {group}")
                cols = st.columns(2)
                for i, f in enumerate(feats):
                    lab, lo, hi, step, _ = META[f]
                    with cols[i % 2]:
                        if group == "Location":
                            st.number_input(lab, lo, hi, step=step, key=f"in_{f}")
                        else:
                            st.slider(lab, lo, hi, step=step, key=f"in_{f}")

    values = {f: float(st.session_state[f"in_{f}"]) for f in FEATURES}
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    with right:
        try:
            p = float(model.predict_proba(row)[0, 1])
        except Exception as e:
            st.error("The model could not score these values.")
            st.exception(e)
            st.stop()

        with st.container(border=True):
            st.markdown("#### Estimated risk")
            st.markdown(gauge(p), unsafe_allow_html=True)

        contrib = contributions(row)
        if contrib:
            with st.container(border=True):
                st.markdown("#### What pushes this score")
                top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
                mx = max(abs(v) for _, v in top) or 1
                html = ""
                for k, v in top:
                    w = abs(v) / mx * 50
                    up = v > 0
                    pos = "left:50%" if up else f"left:{50 - w}%"
                    html += (f'<div class="fac"><span>{META.get(k, (k,))[0]}</span>'
                             f'<div class="trk"><i style="background:{COL["high"] if up else COL["low"]};width:{w}%;{pos}"></i></div>'
                             f'<span style="color:{COL["high"] if up else COL["low"]}">{"▲" if up else "▼"}</span></div>')
                st.markdown(html + '<div style="font-size:.78rem;color:#667085">▲ raises risk · ▼ lowers risk</div>',
                            unsafe_allow_html=True)

        if {"x", "y"} <= set(FEATURES):
            with st.container(border=True):
                st.markdown("#### Location")
                st.map(pd.DataFrame({"lat": [values["y"]], "lon": [values["x"]]}), zoom=4, height=230)

# ------------------------------------------------------------ batch
with tab_batch:
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.markdown("#### Score many locations at once")
        c1.caption("Upload a CSV with one row per location.")
        tpl = pd.DataFrame([PRESETS["Nile Delta farmland"]]).reindex(columns=FEATURES, fill_value=0)
        c2.download_button("Download template", tpl.to_csv(index=False).encode(), "lsd_template.csv",
                           "text/csv", use_container_width=True)
        up = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")

    if up:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error("This file could not be read as CSV.")
            st.exception(e)
            st.stop()
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
            bad = int(X.isna().any(axis=1).sum())
            if bad:
                st.error(f"{bad} row(s) have empty or non-numeric values. Fix them and upload again.")
            else:
                prob = model.predict_proba(X)[:, 1]
                out = df.copy()
                out["risk_percent"] = (prob * 100).round(2)
                out["risk_level"] = [band(float(v)) for v in prob]
                n = len(out)
                cnt = out["risk_level"].value_counts()
                m = st.columns(4)
                m[0].metric("Locations", n)
                m[1].metric("High", int(cnt.get("high", 0)))
                m[2].metric("Moderate", int(cnt.get("moderate", 0)))
                m[3].metric("Low", int(cnt.get("low", 0)))
                if {"x", "y"} <= set(out.columns):
                    st.map(out.rename(columns={"x": "lon", "y": "lat"})[["lat", "lon"]], height=300)
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("Download results", out.to_csv(index=False).encode(),
                                   "lsd_predictions.csv", "text/csv", type="primary", use_container_width=True)

# ------------------------------------------------------------ about
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("#### How it works")
            st.markdown(f"1. You provide **{len(FEATURES)} inputs**: location, climate, terrain and livestock density.\n"
                        f"2. A trained **{model_name}** model scores how closely they resemble past LSD records.\n"
                        "3. You get a **risk percentage** and a plain level.")
            st.markdown("**Levels:** 🟢 low below 30% · 🟠 moderate 30–60% · 🔴 high above 60%")
    with b:
        with st.container(border=True):
            st.markdown("#### Good to know")
            st.markdown("- The tool scores **environmental conditions**. It does not examine animals.\n"
                        "- It is a screening aid. Confirm suspected cases with a veterinarian.\n"
                        "- Accuracy was measured on held-out data and may be lower far from the training regions.")
            st.link_button("View code on GitHub", GITHUB)

st.caption("LSD Risk Studio · Built by Abdelrhman Akl")
