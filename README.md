# 🧠 Fast Weights as Memory

**Interactive Simulator of Associative Learning & Synaptic Plasticity**

> Memory is stored in connections, not parameters.

## Why I built this

I've always found it strange that most neural nets only "remember" through
slow, offline weight updates — nothing changes while the model is actually
running. Fast weights flip that: the network updates its own connections
*in real time* as it sees new associations, then lets them decay, without
touching any trained parameters.

I wanted to actually *see* that happen instead of just reading the equations,
so I built this as a small, self-contained simulator: pick a key and a value,
store the association, query it back, and watch a live heatmap of the
weight matrix change. No PyTorch, no TensorFlow, no pretrained anything —
just NumPy, a Hebbian update rule, and Matplotlib.

## What it demonstrates

- **Learning** — store a key → value pair once, query it, get correct recall instantly.
- **Interference** — remap an existing key to a new value and watch the old memory get overwritten.
- **Plasticity vs. stability** — high η (plasticity) learns fast but is unstable; low η is slower but more durable.
- **Forgetting** — λ (decay) gradually fades every stored connection over time.

## Try it live

🔗 https://fastweightdemo-ewcquqnhfbrh3qrxtkusda.streamlit.app/

## Running it locally

```bash
git clone <this-repo-url>
cd <this-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## How to use it

1. **Control Panel (sidebar)** — set Plasticity (η) and Forgetting (λ), or hit **Reset Memory**.
2. **Memory Builder** — pick a Key and a Value (A–Z) and click **Store Association**.
3. **Query System** — pick a key and click **Query Memory** to see the prediction, confidence, and top-3 candidates.
4. **Run Research Demo** — a one-click walkthrough: stores `A → Apple`, queries it, stores `A → Ant` (same key, new value), queries again, and shows the interference side by side.
5. **Visualization Panel** — a live heatmap of the weight matrix `W`, normalized to `[0, 1]`, with the strongest connections outlined in red.

## The math

Vocabulary: A–Z, one-hot encoded.

**Update rule (Hebbian plasticity)** for association `k → v`:

```
W ← (1 − λ) · W + η · (k ⊗ v)
```

**Retrieval** for query key `k`:

```
output      = softmax(Wᵀ k)
prediction  = argmax(output)
confidence  = max(output)
```

No temperature scaling, so confidence numbers can look modest (5–20%) even
on a correct recall — that's expected. The top-3 comparison is usually the
clearer signal of which memory is currently winning.

## What I learned

Building this made the "interference vs. forgetting" distinction click for
me in a way the math alone hadn't: interference is *local* (two memories
competing for the same key), while forgetting from λ is *global* (every
connection decays a little on every single update, whether it's related or
not). Watching both happen on the same heatmap made that obvious.

## Ideas for later

- Multi-symbol / sequence associations instead of single letters
- A small RNN-style controller that reads and writes the fast-weight matrix itself
- Side-by-side comparison of two η/λ settings running at once

## Feedback

