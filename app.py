# ============================================================
# Lumpy Skin Disease Risk Studio  (v2 - redesigned)
# ============================================================
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LSD Risk Studio",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "lumpy_skin_model.joblib"
METRICS_PATH = BASE / "artifacts" / "metrics.json"
FEATURES_PATH = BASE / "artifacts" / "feature_names.json"
GITHUB = "https://github.com/AbdelrhmanAkl/lumpy-skin-disease-ml"

# ------------------------------------------------------------
# Feature metadata: label, min, max, default, step, help
# ------------------------------------------------------------
META = {
    "x": ("Longitude", -180.0, 180.0, 31.0, 0.01, "Decimal degrees, east is positive."),
    "y": ("Latitude", -90.0, 90.0, 30.8, 0.01, "Decimal degrees, north is positive."),
    "elevation": ("Elevation (m)", -100.0, 6000.0, 10.0, 1.0, "Metres above sea level."),
    "dominant_land_cover": ("Land cover class", 0.0, 255.0, 12.0, 1.0, "Land-cover code used in the training data."),
    "cld": ("Cloud cover (%)", 0.0, 100.0, 20.0, 0.1, "Monthly mean cloud cover."),
    "dtr": ("Day–night temp. range (°C)", 0.0, 30.0, 13.0, 0.1, "Diurnal temperature range."),
    "frs": ("Frost days", 0.0, 31.0, 0.0, 0.1, "Days per month with ground frost."),
    "pet": ("Evapotranspiration (mm/day)", 0.0, 15.0, 3.5, 0.1, "Potential evapotranspiration."),
    "pre": ("Precipitation (mm/month)", 0.0, 2000.0, 15.0, 1.0, "Monthly precipitation."),
    "tmn": ("Min. temperature (°C)", -50.0, 45.0, 14.0, 0.1, "Monthly mean of daily minimum."),
    "tmp": ("Mean temperature (°C)", -50.0, 50.0, 21.0, 0.1, "Monthly mean temperature."),
    "tmx": ("Max. temperature (°C)", -40.0, 60.0, 28.0, 0.1, "Monthly mean of daily maximum."),
    "vap": ("Vapour pressure (hPa)", 0.0, 50.0, 14.0, 0.1, "Monthly mean vapour pressure."),
    "wet": ("Wet days", 0.0, 31.0, 2.0, 0.1, "Days per month with rainfall."),
    "X5_Ct_2010_Da": ("Cattle density (head/km²)", 0.0, 5000.0, 150.0, 1.0, "Gridded cattle density, 2010."),
    "X5_Bf_2010_Da": ("Buffalo density (head/km²)", 0.0, 5000.0, 40.0, 1.0, "Gridded buffalo density, 2010."),
}
GROUPS = {
    "📍 Location": ["x", "y", "elevation", "dominant_land_cover"],
    "🌦️ Climate": ["tmp", "tmn", "tmx", "dtr", "pre", "wet", "cld", "vap", "pet", "frs"],
    "🐃 Livestock": ["X5_Ct_2010_Da", "X5_Bf_2010_Da"],
}
# Illustrative starting points (not real records) – replace with rows from your dataset.
PRESETS = {
    "Cool & dry highland": dict(x=38.7, y=9.0, elevation=2300, dominant_land_cover=12, tmp=16, tmn=9, tmx=23,
                                dtr=14, pre=40, wet=4, cld=45, vap=11, pet=3.0, frs=0, X5_Ct_2010_Da=60, X5_Bf_2010_Da=0),
    "Hot & humid lowland": dict(x=90.4, y=23.8, elevation=8, dominant_land_cover=14, tmp=29, tmn=25, tmx=33,
                                dtr=8, pre=300, wet=18, cld=70, vap=30, pet=4.2, frs=0, X5_Ct_2010_Da=900, X5_Bf_2010_Da=120),
    "Nile Delta farmland": dict(x=31.0, y=30.8, elevation=10, dominant_land_cover=12, tmp=21, tmn=14, tmx=28,
                                dtr=13, pre=15, wet=2, cld=20, vap=14, pet=3.5, frs=0, X5_Ct_2010_Da=150, X5_Bf_2010_Da=40),
}

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Public+Sans:wght@400;500;600&display=swap');
:root{--ink:#0e1f1a;--moss:#14532d;--leaf:#16a34a;--amber:#f59e0b;--rose:#e11d48;--mist:#f2f5f3;--line:#dbe3de;}
html,body,[class*="css"],.stApp{font-family:'Public Sans',sans-serif;color:var(--ink);}
.stApp{background:var(--mist);}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif!important;letter-spacing:-.02em;}
.block-container{max-width:1180px;padding-top:1.2rem;padding-bottom:3rem;}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:var(--ink);}
section[data-testid="stSidebar"] *{color:#e6efe9;}
.hero{background:radial-gradient(900px 300px at 85% -20%,#22c55e55,transparent),linear-gradient(135deg,#0e1f1a,#14532d);
 border-radius:24px;padding:2.6rem 2.8rem;color:#fff;margin-bottom:1.4rem;}
.hero h1{color:#fff;font-size:2.7rem;font-weight:800;margin:0 0 .5rem;line-height:1.05;}
.hero p{color:#cfe3d6;font-size:1.05rem;max-width:640px;margin:0 0 1.2rem;}
.pill{display:inline-block;background:#ffffff1c;border:1px solid #ffffff30;color:#fff;border-radius:999px;
 padding:.28rem .8rem;font-size:.82rem;margin:0 .4rem .4rem 0;}
.stat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.2rem;}
.stat b{display:block;font-family:'Bricolage Grotesque';font-size:1.9rem;line-height:1.1;}
.stat span{color:#5b6b63;font-size:.85rem;}
.result{border-radius:20px;padding:1.6rem 1.8rem;color:#fff;margin:1rem 0;}
.result h2{color:#fff;margin:0;font-size:2rem;}
.result p{margin:.3rem 0 0;color:#ffffffe0;}
.bar{height:14px;border-radius:99px;background:linear-gradient(90deg,#16a34a,#f59e0b 55%,#e11d48);position:relative;margin-top:1rem;}
.bar i{position:absolute;top:-5px;width:4px;height:24px;background:#fff;border-radius:3px;box-shadow:0 0 0 2px #0e1f1a55;}
div[data-testid="stTabs"] button{font-weight:600;}
div[data-testid="stExpander"],div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important;background:#fff;}
.stButton>button[kind="primary"]{background:var(--moss);border:0;border-radius:12px;padding:.7rem 1rem;font-weight:700;}
.stButton>button[kind="primary"]:hover{background:var(--leaf);}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


model = load_model()
metrics = load_json(METRICS_PATH)
FEATURES = load_json(FEATURES_PATH)
if not FEATURES:
    st.error("feature_names.json not found.")
    st.stop()

test = metrics.get("test", {})
acc, f1, auc = (float(test.get(k, 0)) for k in ("accuracy", "f1", "roc_auc"))
model_name = metrics.get("model", type(model).__name__)


def risk_band(p):
    if p < 0.30:
        return "Low risk", "#15803d", "Conditions look unfavourable for LSD spread."
    if p < 0.60:
        return "Moderate risk", "#b45309", "Mixed signals. Monitor the herd and re-check with fresh data."
    return "High risk", "#be123c", "Conditions resemble past LSD outbreaks. Consider a vet inspection and vaccination review."


def predict(df):
    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    return pred, prob


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("🐄 LSD Risk Studio")
    st.caption("Machine-learning risk screening for lumpy skin disease.")
    st.divider()
    st.markdown(f"**Model:** {model_name}")
    st.markdown(f"**Inputs:** {len(FEATURES)} features")
    st.markdown(f"[Source code on GitHub]({GITHUB})")
    st.divider()
    st.caption("A screening aid, not a veterinary diagnosis.")

# ------------------------------------------------------------
# Hero + KPIs
# ------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Know your lumpy skin disease risk<br>before it spreads.</h1>
  <p>Enter a location's climate and livestock data, or upload a whole file, and get an instant risk score from a trained XGBoost model.</p>
  <span class="pill">⚡ Results in seconds</span><span class="pill">📁 Batch CSV support</span><span class="pill">🗺️ Map view</span>
</div>
""",
    unsafe_allow_html=True,
)

k = st.columns(4)
for col, (val, lab) in zip(
    k,
    [(f"{acc*100:.1f}%", "Accuracy on unseen data"), (f"{f1*100:.1f}%", "F1 score"),
     (f"{auc*100:.1f}%", "ROC-AUC"), (model_name, "Model")],
):
    col.markdown(f'<div class="stat"><b>{val}</b><span>{lab}</span></div>', unsafe_allow_html=True)

st.write("")
tab_one, tab_batch, tab_about = st.tabs(["🔍 Check a location", "📁 Check a file", "ℹ️ How it works"])

# ------------------------------------------------------------
# Single prediction
# ------------------------------------------------------------
with tab_one:
    for f in FEATURES:
        st.session_state.setdefault(f"in_{f}", float(META[f][3]) if f in META else 0.0)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("1 · Choose a starting point")
        chosen = st.radio("Starting point", list(PRESETS) + ["Start from my own values"],
                          horizontal=True, label_visibility="collapsed")
        if chosen in PRESETS and st.session_state.get("_preset") != chosen:
            for f, v in PRESETS[chosen].items():
                st.session_state[f"in_{f}"] = float(v)
            st.session_state["_preset"] = chosen
        elif chosen not in PRESETS:
            st.session_state["_preset"] = chosen
        st.caption("Presets are illustrative examples. Edit any value below.")

        st.subheader("2 · Adjust the values")
        for group, feats in GROUPS.items():
            with st.expander(group, expanded=(group.startswith("📍"))):
                cols = st.columns(2)
                for i, f in enumerate([f for f in feats if f in FEATURES]):
                    lab, lo, hi, _, step, hlp = META[f]
                    cols[i % 2].number_input(lab, min_value=lo, max_value=hi, step=step,
                                             key=f"in_{f}", help=hlp)

        go = st.button("Calculate risk", type="primary", use_container_width=True)

    with right:
        st.subheader("3 · Your result")
        if go:
            row = pd.DataFrame([[st.session_state[f"in_{f}"] for f in FEATURES]], columns=FEATURES)
            try:
                pred, prob = predict(row)
            except Exception as e:
                st.error("The model could not score these values.")
                st.exception(e)
                st.stop()
            p = float(prob[0]) if prob is not None else float(pred[0])
            title, color, advice = risk_band(p)
            st.markdown(
                f'<div class="result" style="background:{color}"><p>Estimated risk</p>'
                f'<h2>{p*100:.1f}% · {title}</h2><p>{advice}</p>'
                f'<div class="bar"><i style="left:calc({p*100:.1f}% - 2px)"></i></div></div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Healthy probability", f"{(1-p)*100:.1f}%")
            c2.metric("LSD probability", f"{p*100:.1f}%")
            st.map(pd.DataFrame({"lat": [row["y"][0]], "lon": [row["x"][0]]}) if {"x", "y"} <= set(FEATURES) else None,
                   zoom=4, height=240)
        else:
            st.info("Pick a starting point, adjust the values, then press **Calculate risk**.")

        if hasattr(model, "feature_importances_"):
            with st.expander("What drives the model overall?"):
                imp = pd.Series(model.feature_importances_, index=[META.get(f, (f,))[0] for f in FEATURES])
                st.bar_chart(imp.sort_values(), horizontal=True)

# ------------------------------------------------------------
# Batch
# ------------------------------------------------------------
with tab_batch:
    st.subheader("Score many locations at once")
    c1, c2 = st.columns([2, 1])
    c1.write("Upload a CSV with one row per location. Not sure about the format? Download the template.")
    template = pd.DataFrame([PRESETS["Nile Delta farmland"]]).reindex(columns=FEATURES)
    c2.download_button("⬇️ CSV template", template.to_csv(index=False).encode(), "lsd_template.csv",
                       "text/csv", use_container_width=True)

    up = st.file_uploader("Drop your CSV here", type=["csv"])
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
                pred, prob = predict(X)
                out = df.copy()
                if prob is not None:
                    out["risk_percent"] = (prob * 100).round(2)
                    out["risk_level"] = [risk_band(p)[0] for p in prob]
                out["prediction"] = pred
                n = len(out)
                hi = int((pred == 1).sum())
                m = st.columns(3)
                m[0].metric("Locations", n)
                m[1].metric("Flagged as LSD", hi)
                m[2].metric("Flagged share", f"{hi/n*100:.1f}%")
                if {"x", "y"} <= set(out.columns):
                    st.map(out.rename(columns={"x": "lon", "y": "lat"})[["lat", "lon"]], height=300)
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download results", out.to_csv(index=False).encode(),
                                   "lsd_predictions.csv", "text/csv", use_container_width=True, type="primary")

# ------------------------------------------------------------
# About
# ------------------------------------------------------------
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("How it works")
        st.markdown(
            f"""
1. You provide **{len(FEATURES)} inputs**: location, climate, terrain and livestock density.
2. A **{model_name}** classifier trained on historical records scores the conditions.
3. You get a **risk percentage** and a plain-language level: low, moderate or high.
"""
        )
        st.subheader("Risk levels")
        st.success("Low: below 30%")
        st.warning("Moderate: 30% to 60%")
        st.error("High: above 60%")
    with b:
        st.subheader("Good to know")
        st.markdown(
            """
- The tool scores **environmental conditions**. It does not examine animals.
- Results are a screening aid. Confirm suspected cases with a veterinarian.
- Accuracy is measured on held-out data and may differ in regions far from the training data.
"""
        )
        st.subheader("Tech stack")
        st.markdown("Python · scikit-learn · XGBoost · pandas · Streamlit")
        st.link_button("View the code on GitHub", GITHUB)

st.divider()
st.caption("LSD Risk Studio · Built by Abdelrahman Akl")
