import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="RingSentinel — Case Files", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# DESIGN SYSTEM — case-file / evidence-dossier aesthetic
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #0B1220;
  --panel: #131B2C;
  --panel-border: #263049;
  --paper: #EDE7D8;
  --paper-line: #C9BFA5;
  --stamp-red: #A9342A;
  --stamp-green: #4C7A5E;
  --gold: #C9A227;
  --muted: #8B93A7;
  --ink-text: #E4E7EE;
}

.stApp { background-color: var(--ink); color: var(--ink-text); }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 500 !important; letter-spacing: 0.01em; }

/* kill default streamlit chrome that screams "template" */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* wordmark header */
.case-header { border-bottom: 1px solid var(--panel-border); padding-bottom: 1.1rem; margin-bottom: 1.6rem; }
.case-eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
  color: var(--gold); text-transform: uppercase; margin-bottom: 0.35rem;
}
.case-title { font-size: 2.05rem; margin: 0; color: var(--ink-text); }
.case-sub { color: var(--muted); font-size: 0.94rem; margin-top: 0.3rem; }

/* ledger stat row, replaces st.metric look */
.ledger-row { display: flex; gap: 0; border: 1px solid var(--panel-border); margin-bottom: 1.8rem; }
.ledger-cell {
  flex: 1; padding: 0.95rem 1.1rem; border-right: 1px solid var(--panel-border);
  background: var(--panel);
}
.ledger-cell:last-child { border-right: none; }
.ledger-label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem;
}
.ledger-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; color: var(--ink-text); font-weight: 500; }
.ledger-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--stamp-green); margin-top: 0.15rem; }

/* section label — dossier tab style */
.tab-label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--gold); border-left: 3px solid var(--gold);
  padding-left: 0.6rem; margin: 1.9rem 0 0.9rem 0;
}

/* stamp badge */
.stamp {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-weight: 700;
  font-size: 0.78rem; letter-spacing: 0.08em; padding: 0.28rem 0.7rem;
  border: 2px solid currentColor; border-radius: 2px; transform: rotate(-2deg);
  text-transform: uppercase;
}
.stamp-flagged { color: var(--stamp-red); }
.stamp-clear { color: var(--stamp-green); }

/* case card for selected ring */
.case-card {
  background: var(--panel); border: 1px solid var(--panel-border);
  padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
.case-number {
  font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: 0.8rem;
  letter-spacing: 0.06em; margin-bottom: 0.6rem;
}

/* evidence report styled as an actual paper document */
.evidence-paper {
  background: var(--paper); color: #1a1a1a; padding: 1.8rem 2rem;
  border: 1px solid var(--paper-line); font-family: 'IBM Plex Sans', sans-serif;
  line-height: 1.65; position: relative;
}
.evidence-paper .stamp { position: absolute; top: 1.4rem; right: 1.6rem; }
.evidence-paper h4 {
  font-family: 'Fraunces', serif; font-weight: 500; margin-top: 0; margin-bottom: 0.9rem;
  border-bottom: 1px solid var(--paper-line); padding-bottom: 0.5rem;
}
.evidence-paper .field-label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.06em;
  color: #5c5540; text-transform: uppercase; display: inline-block; width: 190px;
}
.evidence-paper .field-value { font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; }
.evidence-note {
  margin-top: 1rem; padding-top: 0.9rem; border-top: 1px dashed var(--paper-line);
  font-size: 0.85rem; color: #4a4534; font-style: italic;
}

/* dataframe restyle */
[data-testid="stDataFrame"] { border: 1px solid var(--panel-border) !important; }

/* footer strip */
.floor-note {
  font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--muted);
  border-top: 1px solid var(--panel-border); padding-top: 0.9rem; margin-top: 2.2rem;
  letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return lgb.Booster(model_file='reports/model_b_graph_augmented.txt')

@st.cache_data
def load_data():
    df = pd.read_parquet('data/baseline_features.parquet')
    rings = pd.read_parquet('data/device_ring_assignments.parquet')
    rings = rings.rename(columns={'device_ring_id': 'ring_id', 'device_ring_size': 'ring_size'})
    return df.merge(rings, on='TransactionID', how='left')

@st.cache_data
def load_summaries():
    with open('reports/ablation_summary.json') as f:
        ablation = json.load(f)
    with open('reports/cost_summary.json') as f:
        cost = json.load(f)
    return ablation, cost

model = load_model()
df = load_data()
ablation, cost = load_summaries()
base_rate = df['isFraud'].mean()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="case-header">
  <div class="case-eyebrow">Razorpay AI Buildathon · Track 02 · AI Risk Manager</div>
  <p class="case-title">RingSentinel</p>
  <p class="case-sub">Case files for transactions that don't add up — flagging fraud rings by what they share, not what they claim.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LEDGER — headline metrics, replaces st.metric row
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="ledger-row">
  <div class="ledger-cell">
    <div class="ledger-label">Baseline model · PR-AUC</div>
    <div class="ledger-value">{ablation['model_a_pr_auc']:.4f}</div>
  </div>
  <div class="ledger-cell">
    <div class="ledger-label">Ring-augmented · PR-AUC</div>
    <div class="ledger-value">{ablation['model_b_pr_auc']:.4f}</div>
    <div class="ledger-delta">+{ablation['improvement']:.4f}</div>
  </div>
  <div class="ledger-cell">
    <div class="ledger-label">Modeled savings, test set</div>
    <div class="ledger-value">₹{cost['savings_absolute']:,.0f}</div>
  </div>
  <div class="ledger-cell">
    <div class="ledger-label">Per transaction</div>
    <div class="ledger-value">₹{cost['savings_per_transaction']:.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# RING TABLE
# ----------------------------------------------------------------------------
st.markdown('<div class="tab-label">Open Cases — Rings by Fraud Concentration</div>', unsafe_allow_html=True)

ring_stats = df.dropna(subset=['ring_id']).groupby('ring_id').agg(
    size=('TransactionID', 'count'),
    fraud_rate=('isFraud', 'mean'),
    total_amount=('TransactionAmt', 'sum')
).reset_index()
ring_stats = ring_stats[ring_stats['size'] >= 3].sort_values('fraud_rate', ascending=False)

min_size = st.slider("Minimum ring size", 3, 50, 5, label_visibility="visible")
filtered = ring_stats[ring_stats['size'] >= min_size].head(20).copy()
filtered['vs_baseline'] = filtered['fraud_rate'] / base_rate

st.dataframe(
    filtered.rename(columns={
        'ring_id': 'Case No.', 'size': 'Members', 'fraud_rate': 'Internal Fraud Rate',
        'total_amount': 'Total Value', 'vs_baseline': 'x Baseline'
    }).style.format({
        'Internal Fraud Rate': '{:.1%}', 'Total Value': '₹{:,.0f}', 'x Baseline': '{:.1f}x'
    }),
    use_container_width=True, hide_index=True
)

# ----------------------------------------------------------------------------
# CASE EXPLORER
# ----------------------------------------------------------------------------
st.markdown('<div class="tab-label">Case File — Select for Detail</div>', unsafe_allow_html=True)

selected_ring = st.selectbox("Case number", filtered['ring_id'].tolist(), label_visibility="collapsed")

if selected_ring is not None:
    ring_txns = df[df['ring_id'] == selected_ring]
    meta = ring_stats[ring_stats['ring_id'] == selected_ring].iloc[0]
    ratio = meta['fraud_rate'] / base_rate
    is_flagged = meta['fraud_rate'] > base_rate * 1.5

    stamp_html = (
        '<span class="stamp stamp-flagged">Flagged</span>' if is_flagged
        else '<span class="stamp stamp-clear">Within Range</span>'
    )

    st.markdown(f"""
    <div class="case-card">
      <div class="case-number">CASE No. {int(selected_ring):06d} — {int(meta['size'])} linked transactions</div>
      {stamp_html}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="ledger-label">Internal Fraud Rate</div><div class="ledger-value">{meta["fraud_rate"]:.1%}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="ledger-label">vs. Platform Baseline</div><div class="ledger-value">{ratio:.1f}x</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="ledger-label">Total Value</div><div class="ledger-value">₹{meta["total_amount"]:,.0f}</div>', unsafe_allow_html=True)

    st.write("")

    net = Network(height="380px", width="100%", bgcolor="#131B2C", font_color="#E4E7EE")
    for _, row in ring_txns.iterrows():
        color = "#A9342A" if row['isFraud'] == 1 else "#4C7A5E"
        net.add_node(int(row['TransactionID']), label=str(int(row['TransactionID']))[-4:],
                     color=color, title=f"₹{row['TransactionAmt']:.0f}")
    tx_list = ring_txns['TransactionID'].tolist()
    for i in range(len(tx_list) - 1):
        net.add_edge(int(tx_list[i]), int(tx_list[i + 1]), color="#C9A227")
    net.save_graph('reports/_ring_viz.html')
    with open('reports/_ring_viz.html', 'r', encoding='utf-8') as f:
        components.html(f.read(), height=400)
    st.caption("Red — confirmed fraud in training data · Green — legitimate · Gold — shared-device link")

    # ---- evidence report, styled as an actual document ----
    st.markdown('<div class="tab-label">Evidence Report</div>', unsafe_allow_html=True)

    stamp_class = "stamp-flagged" if is_flagged else "stamp-clear"
    stamp_word = "REVIEW" if is_flagged else "CLEAR"

    report_html = f"""
    <div class="evidence-paper">
      <span class="stamp {stamp_class}">{stamp_word}</span>
      <h4>Case No. {int(selected_ring):06d}</h4>
      <p><span class="field-label">Ring size</span><span class="field-value">{int(meta['size'])} transactions</span></p>
      <p><span class="field-label">Internal fraud rate</span><span class="field-value">{meta['fraud_rate']:.1%} (baseline {base_rate:.1%})</span></p>
      <p><span class="field-label">Total value</span><span class="field-value">₹{meta['total_amount']:,.0f}</span></p>
      <p><span class="field-label">Linking signal</span><span class="field-value">device + browser + resolution + email domain</span></p>
      <p><span class="field-label">Recommendation</span><span class="field-value">{"Flag for manual review — " + f"{ratio:.1f}x baseline fraud rate" if is_flagged else "No action — within normal range"}</span></p>
      <div class="evidence-note">This is a decision-support output for a human reviewer. It does not auto-block or auto-deny any transaction.</div>
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)

    plain_report = (
        f"Case No. {int(selected_ring):06d}\n"
        f"Ring size: {int(meta['size'])} transactions\n"
        f"Internal fraud rate: {meta['fraud_rate']:.1%} (baseline {base_rate:.1%})\n"
        f"Total value: Rs {meta['total_amount']:,.0f}\n"
        f"Linking signal: device + browser + resolution + email domain\n"
        f"Recommendation: {'Flag for manual review' if is_flagged else 'No action needed'}\n"
        f"Note: decision-support only, no automated blocking.\n"
    )
    st.download_button("Download case file", plain_report, file_name=f"case_{int(selected_ring):06d}.txt")

st.markdown(
    '<div class="floor-note">Detection tool only. Every recommendation routes to a human reviewer — '
    'no automated blocking or denial. Metrics measured on a held-out, time-based test split.</div>',
    unsafe_allow_html=True
)