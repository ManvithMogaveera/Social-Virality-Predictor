import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
 
st.set_page_config(
    page_title="ViraLens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #08080f;
    color: #dddaf0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 5rem 3rem; max-width: 1100px; }
 
.vl-hero {
    padding: 2.8rem 0 2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.5rem;
}
.vl-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: #6c5ce7;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.vl-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.6rem 0;
    line-height: 1.1;
}
.vl-hero p {
    font-size: 0.95rem;
    color: #7b78a8;
    max-width: 520px;
    line-height: 1.75;
    margin: 0;
    font-weight: 300;
}
.vl-section-head {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.vl-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    text-transform: uppercase;
}
.vl-badge-post {
    background: rgba(108,92,231,0.15);
    color: #a89cff;
    border: 1px solid rgba(108,92,231,0.3);
}
.vl-badge-comment {
    background: rgba(0,184,148,0.12);
    color: #5df0c8;
    border: 1px solid rgba(0,184,148,0.25);
}
.vl-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8e5ff;
    margin: 0;
}
.vl-result-box {
    background: #13132a;
    border-radius: 14px;
    border: 1px solid rgba(108,92,231,0.25);
    padding: 1.8rem;
    text-align: center;
    margin-top: 0.4rem;
}
.vl-score-big {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}
.vl-score-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: #4a4870;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.vl-tier {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.28rem 1rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
}
.vl-bars { margin-top: 1.4rem; text-align: left; }
.vl-bar-row { margin-bottom: 0.55rem; }
.vl-bar-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #7b78a8;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.22rem;
}
.vl-bar-bg {
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    overflow: hidden;
}
.vl-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.vl-divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 2rem 0;
}
.vl-tip {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4a4870;
    letter-spacing: 0.06em;
    margin-top: 1.2rem;
    text-align: center;
}
.vl-empty {
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: #3a3860;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
textarea {
    background: #0a0a18 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #dddaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
textarea:focus {
    border-color: rgba(108,92,231,0.5) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.08) !important;
}
label, .stSlider label, div[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    color: #7b78a8 !important;
    font-weight: 400 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0a0a18 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #dddaf0 !important;
}
</style>
""", unsafe_allow_html=True)
 

@st.cache_resource
def load_model():
    if os.path.exists("social_virality_xgb_model.pkl") and os.path.exists("virality_tfidf_vectorizer.pkl"):
        import xgboost as xgb
        m = joblib.load("social_virality_xgb_model.pkl")
        m.set_params(device="cpu")   # force CPU to kill the CUDA warning
        t = joblib.load("virality_tfidf_vectorizer.pkl")
        return m, t
    return None, None
 
model, tfidf = load_model()
model_loaded = model is not None
 
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
PEAK_HOURS = {12, 13, 14, 15, 20, 21, 22}
 
def tier_style(score):
    if score < 5:
        return "Minimal reach", "#1c1c30", "#6b66a0"
    elif score < 30:
        return "Gaining traction", "#0f1f18", "#00b894"
    elif score < 200:
        return "High engagement", "#1a1430", "#a89cff"
    else:
        return "Viral territory", "#2a1810", "#ff9f43"
 
def bar(label, val, color):
    pct = max(0, min(100, int(val * 100)))
    return f"""
    <div class="vl-bar-row">
      <div class="vl-bar-lbl"><span>{label}</span><span style="color:{color}">{pct}%</span></div>
      <div class="vl-bar-bg"><div class="vl-bar-fill" style="width:{pct}%;background:{color};"></div></div>
    </div>"""
 
 
# ── Hero ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="vl-hero">
  <div class="vl-tag">ViraLens · Reddit Engagement Intelligence</div>
  <h1>Predict before you post.</h1>
  <p>Two-layer virality analysis — check if your <em>post</em> has the right ingredients to trend,
     then see if your <em>comment</em> will rise to the top.</p>
</div>
""", unsafe_allow_html=True)
 
 

st.markdown("""
<div class="vl-section-head">
  <span class="vl-badge vl-badge-post">01 · Post</span>
  <span class="vl-section-title">Post virality checker</span>
</div>
""", unsafe_allow_html=True)
 
pcol_in, pcol_out = st.columns([1.15, 0.85], gap="large")
 
with pcol_in:
    p_title = st.text_input("Post title", placeholder="e.g. What's the most underrated Python library?", key="p_title")
 
    pc1, pc2 = st.columns(2)
    with pc1:
        p_hour = st.slider("Hour posted (UTC)", 0, 23, 14, key="p_hour")
    with pc2:
        p_day = st.selectbox("Day of week", DAYS, index=2, key="p_day")
        p_dow = DAYS.index(p_day)
 
    pc3, pc4 = st.columns(2)
    with pc3:
        p_body_len = st.number_input("Post body length (chars)", 0, 40000, 0, 100, key="p_body")
    with pc4:
        p_sub_size = st.number_input("Subreddit size (approx)", 0, 50000000, 500000, 50000, key="p_sub")
 
with pcol_out:
    with st.container():
        if p_title.strip():
            is_q      = int("?" in p_title)
            title_len = len(p_title)
            is_peak   = int(p_hour in PEAK_HOURS)
            is_wknd   = int(p_dow >= 5)
            has_body  = int(p_body_len > 50)
 
            
            title_sc  = min(title_len / 80,        1.0)
            length_sc = min(title_len / 120,        1.0)
            timing_sc = 0.9 if is_peak else (0.5 if is_wknd else 0.2)
            q_sc      = 0.85 if is_q else 0.2
            body_sc   = 0.8  if has_body else 0.1
            reach_sc  = min(p_sub_size / 5_000_000, 1.0)
 
            virality = (
                title_sc  * 0.15 +
                length_sc * 0.10 +
                timing_sc * 0.25 +
                q_sc      * 0.20 +
                body_sc   * 0.15 +
                reach_sc  * 0.15
            )
            # virality is always 0.0–1.0, pred always 1–2000
            pred = max(1, min(2000, int(virality * 2000)))
 
            tier_name, tier_bg, tier_col = tier_style(pred)
 
            st.markdown(f"""
            <div class="vl-result-box">
              <div class="vl-score-big">{pred:,}</div>
              <div class="vl-score-lbl">estimated upvotes</div>
              <div class="vl-tier" style="background:{tier_bg};color:{tier_col};border:1px solid {tier_col}40;">{tier_name}</div>
              <div class="vl-bars">
                {bar("Title length & clarity", title_sc, "#a89cff")}
                {bar("Title word count signal", length_sc, "#a89cff")}
                {bar("Posting time advantage", timing_sc, "#a89cff")}
                {bar("Question framing", q_sc, "#a89cff")}
                {bar("Body content depth", body_sc, "#a89cff")}
                {bar("Subreddit reach", reach_sc, "#a89cff")}
              </div>
              <div class="vl-tip">{'✓ Peak hour — strong timing window' if is_peak else '↑ Off-peak — try posting at 12–15h or 20–22h UTC'}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="vl-empty">Enter a title to see prediction</div>', unsafe_allow_html=True)
 
 
# ── Divider ──────────────────────────────────────────────────────────
st.markdown('<div class="vl-divider"></div>', unsafe_allow_html=True)
 
 

st.markdown("""
<div class="vl-section-head">
  <span class="vl-badge vl-badge-comment">02 · Comment</span>
  <span class="vl-section-title">Comment virality checker</span>
</div>
""", unsafe_allow_html=True)
 
ccol_in, ccol_out = st.columns([1.15, 0.85], gap="large")
 
with ccol_in:
    c_body = st.text_area("Your comment", placeholder="Write or paste your Reddit comment here...", height=120, key="c_body")
 
    cc1, cc2 = st.columns(2)
    with cc1:
        c_post_score = st.number_input("Parent post score", 0, 500000, 1200, 100, key="c_post_score")
    with cc2:
        c_sentiment = st.slider("Sentiment (−1 negative → +1 positive)", -1.0, 1.0, 0.1, 0.05, key="c_sent")
 
    cc3, cc4 = st.columns(2)
    with cc3:
        c_hour = st.slider("Hour posted (UTC)", 0, 23, 14, key="c_hour")
    with cc4:
        c_day = st.selectbox("Day of week", DAYS, index=2, key="c_day")
        c_dow = DAYS.index(c_day)
 
    c_post_title = st.text_input(
        "Parent post title (optional — improves ML accuracy)",
        placeholder="e.g. Best way to learn ML?", key="c_ptitle"
    )
 
with ccol_out:
    with st.container():
        if c_body.strip():
            word_count  = len(c_body.split())
            char_count  = len(c_body)
            is_short    = int(word_count < 5)
            is_peak     = int(c_hour in PEAK_HOURS)
            is_wknd     = int(c_dow >= 5)
            sent_intens = abs(c_sentiment)
            log_ps      = np.log1p(max(c_post_score, 0))
            is_q_reply  = int("?" in (c_post_title or ""))
 
            
            parent_sc = min(c_post_score / 50_000, 1.0)
            depth_sc  = min(word_count / 60,       1.0)
            timing_sc = 0.9 if is_peak else (0.5 if is_wknd else 0.2)
            sent_sc   = min(sent_intens,            1.0)
            length_sc = 0.1 if is_short else min(char_count / 400, 1.0)
 
            if model_loaded:
                try:
                    title_text = c_post_title.strip() if c_post_title.strip() else "untitled"
                    tfidf_vec  = tfidf.transform([title_text])
                    tfidf_cols = {
                        f"title_word_{w}": float(tfidf_vec[0, i])
                        for i, w in enumerate(tfidf.get_feature_names_out())
                    }
                    row = {
                        'sentiment':           c_sentiment,
                        'sentiment_intensity': sent_intens,
                        'hour':                c_hour,
                        'day_of_week':         c_dow,
                        'word_count':          word_count,
                        'post_score':          c_post_score,
                        'post_title_len':      len(c_post_title),
                        'post_body_len':       0,
                        'post_is_question':    is_q_reply,
                        'char_count':          char_count,
                        'is_short_comment':    is_short,
                        'log_post_score':      log_ps,
                        'is_weekend':          is_wknd,
                        'is_peak_hour':        is_peak,
                        'comment_depth':       1,
                        **tfidf_cols
                    }
                    X        = pd.DataFrame([row])
                    log_pred = model.predict(X)[0]
                    pred_c   = max(1, min(10000, int(np.expm1(log_pred))))
                except Exception as e:
                    st.warning(f"Model error: {e} — falling back to heuristic.")
                    model_loaded = False
                    pred_c = None
 
                if pred_c is None:
                    model_loaded = False
 
            if not model_loaded:
                
                virality = (
                    parent_sc * 0.35 +
                    depth_sc  * 0.25 +
                    timing_sc * 0.20 +
                    sent_sc   * 0.10 +
                    length_sc * 0.10
                )
                pred_c = max(1, min(500, int(virality * 500)))
 
            tier_name, tier_bg, tier_col = tier_style(pred_c)
 
            st.markdown(f"""
            <div class="vl-result-box" style="border-color:rgba(0,184,148,0.25);">
              <div class="vl-score-big">{pred_c:,}</div>
              <div class="vl-score-lbl">predicted upvotes</div>
              <div class="vl-tier" style="background:{tier_bg};color:{tier_col};border:1px solid {tier_col}40;">{tier_name}</div>
              <div class="vl-bars">
                {bar("Parent post momentum", parent_sc, "#5df0c8")}
                {bar("Comment depth & detail", depth_sc, "#5df0c8")}
                {bar("Sentiment strength", sent_sc, "#5df0c8")}
                {bar("Timing advantage", timing_sc, "#5df0c8")}
                {bar("Response length", length_sc, "#5df0c8")}
              </div>
              <div class="vl-tip">{'✓ ML model active' if model_loaded else '○ Demo mode — run train.py to activate full model'}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="vl-empty">Enter a comment to see prediction</div>', unsafe_allow_html=True)