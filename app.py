"""
Fast Weights as Memory
Interactive Simulator of Associative Learning & Synaptic Plasticity

Core idea
---------
A neural system can store short-term memory in *dynamically updated
connection weights* (fast weights) instead of modifying long-term
parameters. This app lets you build, query, and visualize that memory
matrix in real time, and see learning, interference, plasticity and
forgetting emerge from one Hebbian update rule.

Run with:
    streamlit run app.py
"""

import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------
# 1. CORE MATH  (pure functions, no Streamlit dependency)
# --------------------------------------------------------------------------

LETTERS = list(string.ascii_uppercase)
N = len(LETTERS)  # 26
LETTER_IDX = {c: i for i, c in enumerate(LETTERS)}

# Illustrative word labels used only for the guided research demo.
# The underlying model still only ever sees single A-Z symbols.
DEMO_LABELS = {"F": "Apple 🍎 (Fruit)", "I": "Ant 🐜 (Insect)"}


def one_hot(letter: str) -> np.ndarray:
    """Encode a single A-Z symbol as an N-dim one-hot vector."""
    v = np.zeros(N)
    v[LETTER_IDX[letter]] = 1.0
    return v


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def store_association(W: np.ndarray, key: str, value: str, eta: float, lam: float) -> np.ndarray:
    """
    Hebbian fast-weight update:
        W <- (1 - lambda) * W + eta * (k (x) v)

    Every update globally decays the whole matrix by (1 - lambda) --
    this is what produces gradual forgetting of *everything* over time --
    and then reinforces the new key->value connection by eta.
    """
    k = one_hot(key)
    v = one_hot(value)
    return (1.0 - lam) * W + eta * np.outer(k, v)


def query_memory(W: np.ndarray, key: str):
    """
    Retrieval rule:
        output = softmax(W^T k)
        prediction = argmax(output), confidence = max(output)
    Returns (output_vector, predicted_letter, confidence, top3_list)
    """
    k = one_hot(key)
    logits = W.T @ k
    output = softmax(logits)
    pred_idx = int(np.argmax(output))
    confidence = float(np.max(output))
    top3_idx = np.argsort(output)[-3:][::-1]
    top3 = [(LETTERS[i], float(output[i])) for i in top3_idx]
    return output, LETTERS[pred_idx], confidence, top3


# --------------------------------------------------------------------------
# 2. VISUALIZATION
# --------------------------------------------------------------------------

def draw_heatmap(W: np.ndarray, title: str, highlight_top_n: int = 5, figsize=(5.2, 5.2)):
    """Draw a normalized [0,1] heatmap of the fast-weight matrix W,
    with the strongest few connections highlighted."""
    w_min, w_max = W.min(), W.max()
    if w_max - w_min > 1e-9:
        normalized = (W - w_min) / (w_max - w_min)
    else:
        normalized = np.zeros_like(W)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(N))
    ax.set_xticklabels(LETTERS, fontsize=6)
    ax.set_yticks(range(N))
    ax.set_yticklabels(LETTERS, fontsize=6)
    ax.set_xlabel("Value (v)", fontsize=9)
    ax.set_ylabel("Key (k)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized weight", fontsize=8)

    # Highlight the strongest real (non-zero) connections
    flat = W.flatten()
    if np.max(np.abs(flat)) > 1e-9:
        top_idx = np.argsort(np.abs(flat))[::-1][:highlight_top_n]
        for idx in top_idx:
            if abs(flat[idx]) < 1e-9:
                continue
            row, col = divmod(idx, N)
            ax.add_patch(
                plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=1.6)
            )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------------------------------

def init_state():
    defaults = {
        "W": np.zeros((N, N)),
        "W_before_last": None,
        "last_action": None,          # ("store", key, value) or ("query", key)
        "key_history": {},            # key -> last stored value
        "insights": [],               # rolling feed of auto-generated insights
        "last_query": None,           # dict with output/pred/conf/top3
        "demo_result": None,          # dict populated after Run Research Demo
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_memory():
    st.session_state.W = np.zeros((N, N))
    st.session_state.W_before_last = None
    st.session_state.last_action = None
    st.session_state.key_history = {}
    st.session_state.insights = []
    st.session_state.last_query = None
    st.session_state.demo_result = None


def log_insight(text: str):
    st.session_state.insights.insert(0, text)
    st.session_state.insights = st.session_state.insights[:8]


def do_store(key: str, value: str, eta: float, lam: float):
    st.session_state.W_before_last = st.session_state.W.copy()
    prev_value = st.session_state.key_history.get(key)
    st.session_state.W = store_association(st.session_state.W, key, value, eta, lam)
    st.session_state.key_history[key] = value
    st.session_state.last_action = ("store", key, value)

    if prev_value is None:
        log_insight(f"🌱 **New memory formed:** `{key} → {value}` stored as a synaptic connection.")
    elif prev_value == value:
        log_insight(f"🔁 **Reinforcement:** repeating `{key} → {value}` strengthens the existing connection (like rehearsal).")
    else:
        log_insight(
            f"⚡ **Interference detected:** `{key}` was mapped to `{prev_value}`, now competing with `{value}`. "
            f"Interference occurs because both memories share the same key representation."
        )
    log_insight(f"📉 Every update also decays *all* connections by λ={lam:.2f} — this is background forgetting, not interference.")


def do_query(key: str):
    output, pred, conf, top3 = query_memory(st.session_state.W, key)
    st.session_state.last_query = {"key": key, "pred": pred, "conf": conf, "top3": top3}
    st.session_state.last_action = ("query", key)
    expected = st.session_state.key_history.get(key)
    if expected is not None and pred == expected:
        log_insight(f"🔍 Queried `{key}` → correctly recalled `{pred}` (confidence {conf:.1%}).")
    elif expected is not None:
        log_insight(
            f"🔍 Queried `{key}` → predicted `{pred}` instead of the most recently stored `{expected}` "
            f"(confidence {conf:.1%}). Interference or decay has shifted the memory."
        )
    else:
        log_insight(f"🔍 Queried `{key}` → predicted `{pred}` (confidence {conf:.1%}); no association was ever stored for this key.")


def run_research_demo(eta: float, lam: float):
    """Guided 2-step walkthrough: A -> Apple, query, A -> Ant, query again."""
    reset_memory()
    do_store("A", "F", eta, lam)          # A -> Apple (Fruit)
    do_query("A")
    result_1 = dict(st.session_state.last_query)
    W_after_first = st.session_state.W.copy()

    do_store("A", "I", eta, lam)          # A -> Ant (Insect) : interference
    do_query("A")
    result_2 = dict(st.session_state.last_query)
    W_after_second = st.session_state.W.copy()

    st.session_state.demo_result = {
        "W_after_first": W_after_first,
        "W_after_second": W_after_second,
        "result_1": result_1,
        "result_2": result_2,
        "eta": eta,
        "lam": lam,
    }


# --------------------------------------------------------------------------
# 4. STREAMLIT APP
# --------------------------------------------------------------------------

st.set_page_config(page_title="Fast Weights as Memory", page_icon="🧠", layout="wide")
init_state()

# ---- Header ---------------------------------------------------------------
st.title("🧠 Fast Weights as Memory")
st.markdown(
    "#### Interactive Simulator of Associative Learning & Synaptic Plasticity\n"
    "> **Memory is stored in connections, not parameters.** "
    "This system never changes any long-term weight — every association you "
    "teach it lives entirely inside a fast, dynamically updated matrix `W`."
)
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Learning**\n\nOne-shot recall after a single association.")
    c2.markdown("**2. Interference**\n\nRemapping a key overwrites its old value.")
    c3.markdown("**3. Plasticity**\n\nHigh η = fast but unstable; low η = slow but stable.")
    c4.markdown("**4. Forgetting**\n\nλ decays every stored memory over time.")

if st.button("🚀 Run Research Demo  (A → Apple, then A → Ant)", type="primary", use_container_width=True):
    run_research_demo(st.session_state.get("eta_slider", 0.6), st.session_state.get("lam_slider", 0.05))

# ---- Sidebar: control panel -------------------------------------------------
st.sidebar.header("⚙️ Control Panel")
eta = st.sidebar.slider("Plasticity (η) — learning rate", 0.0, 1.0, 0.6, 0.01, key="eta_slider")
if eta >= 0.7:
    st.sidebar.caption("⚡ High plasticity: learns almost instantly, but new memories overwrite old ones easily.")
elif eta <= 0.3:
    st.sidebar.caption("🐢 Low plasticity: learns gradually, but memories resist being overwritten.")
else:
    st.sidebar.caption("⚖️ Balanced plasticity.")

lam = st.sidebar.slider("Forgetting (λ) — decay rate", 0.0, 0.3, 0.05, 0.01, key="lam_slider")
if lam >= 0.2:
    st.sidebar.caption("🌪️ High forgetting: all memories decay quickly with every new update.")
elif lam <= 0.05:
    st.sidebar.caption("🧊 Low forgetting: memories are long-lasting.")
else:
    st.sidebar.caption("🌤️ Moderate forgetting.")

st.sidebar.divider()
if st.sidebar.button("♻️ Reset Memory", use_container_width=True):
    reset_memory()

nonzero = int(np.count_nonzero(np.abs(st.session_state.W) > 1e-6))
st.sidebar.metric("Active connections", f"{nonzero} / {N * N}")

# ---- Main: Memory Builder + Query System -----------------------------------
col_build, col_query = st.columns(2)

with col_build:
    st.subheader("📌 Memory Builder")
    key_in = st.selectbox("Key (A–Z)", LETTERS, index=0, key="build_key")
    val_in = st.selectbox("Value (A–Z)", LETTERS, index=1, key="build_val")
    if st.button("💾 Store Association", use_container_width=True):
        do_store(key_in, val_in, eta, lam)

with col_query:
    st.subheader("🔍 Query System")
    q_key = st.selectbox("Query Key (A–Z)", LETTERS, index=0, key="query_key_select")
    if st.button("🔎 Query Memory", use_container_width=True):
        do_query(q_key)

    lq = st.session_state.last_query
    if lq:
        m1, m2 = st.columns(2)
        m1.metric("Prediction", lq["pred"])
        m2.metric("Confidence", f"{lq['conf']:.1%}")
        st.caption("Top-3 predictions:")
        for letter, prob in lq["top3"]:
            st.progress(min(1.0, prob / max(1e-9, lq["top3"][0][1])), text=f"{letter} — {prob:.1%}")
    else:
        st.info("No query yet — pick a key and click **Query Memory**.")

# ---- Visualization panel ----------------------------------------------------
st.divider()
st.subheader("🔥 Synaptic Weight Matrix (W)")

if st.session_state.last_action and st.session_state.last_action[0] == "store" and st.session_state.W_before_last is not None:
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.pyplot(draw_heatmap(st.session_state.W_before_last, "Before last update"), clear_figure=True)
    with vcol2:
        st.pyplot(draw_heatmap(st.session_state.W, "After last update"), clear_figure=True)
else:
    st.pyplot(draw_heatmap(st.session_state.W, "Current fast-weight matrix W"), clear_figure=True)

# ---- Live insights feed ------------------------------------------------------
st.divider()
st.subheader("🧠 Auto-Generated Insights")
with st.container(border=True):
    st.markdown(
        "- Fast weights act as **temporary synaptic storage**, separate from any trained parameters.\n"
        "- Interference occurs when **overlapping key representations** compete for the same connection.\n"
        "- Decay (λ) simulates **biological forgetting** of unused memories."
    )
if st.session_state.insights:
    st.caption("Live feed (most recent first):")
    for text in st.session_state.insights:
        st.markdown(f"- {text}")

# ---- Research demo results ---------------------------------------------------
if st.session_state.demo_result:
    st.divider()
    st.subheader("🔬 Research Demo Results")
    dr = st.session_state.demo_result
    st.markdown(
        f"Ran with **η={dr['eta']:.2f}**, **λ={dr['lam']:.2f}**: stored `A → Apple 🍎`, queried `A`, "
        f"then stored `A → Ant 🐜` (same key, new value) and queried `A` again."
    )

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown(f"**Step 1 — after `A → Apple`**")
        st.pyplot(draw_heatmap(dr["W_after_first"], "After A → Apple"), clear_figure=True)
        r1 = dr["result_1"]
        st.metric("Query A → prediction", f"{DEMO_LABELS.get(r1['pred'], r1['pred'])}", f"{r1['conf']:.1%} confidence")
    with dcol2:
        st.markdown(f"**Step 2 — after `A → Ant` (interference)**")
        st.pyplot(draw_heatmap(dr["W_after_second"], "After A → Ant"), clear_figure=True)
        r2 = dr["result_2"]
        st.metric("Query A → prediction", f"{DEMO_LABELS.get(r2['pred'], r2['pred'])}", f"{r2['conf']:.1%} confidence")

    st.success(
        "**Interference, made visible:** the second association competes with the first for the same key `A`. "
        "With higher λ the old memory (Apple) fades faster; with higher η the new memory (Ant) dominates faster. "
        "Try dragging the sliders and re-running the demo to feel the plasticity–stability tradeoff."
    )

# ---- Theory connection --------------------------------------------------------
st.divider()
with st.expander("🌐 Theory Connection"):
    st.markdown(
        """
**Biological link**
- Hebbian learning: *"neurons that fire together, wire together."*
- The fast-weight matrix `W` plays the role of **synaptic potentiation** — a temporary strengthening of a connection.
- The decay term (λ) mirrors **biological forgetting** of a synapse that isn't reinforced.

**AI link**
- This mirrors how **attention mechanisms in transformers** compute associations dynamically at inference time, rather than only through static trained parameters.
- More broadly, it reflects a family of *fast-weight* / associative-memory models where **memory is stored in dynamic weights**, not in the frozen weights of the network.
"""
    )

st.caption("Pure NumPy + Matplotlib implementation — no deep learning frameworks used.")