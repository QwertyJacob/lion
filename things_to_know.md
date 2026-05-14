## 1. What is the `label_A` plot?

Each episode, `labels_bought[0]` is either `True` (label A was purchased at some point during the episode) or `False`. That boolean is logged as 0/1. The plot shows the **rolling mean** of that binary across episodes — so the Y-axis is simply "fraction of recent episodes in which label A was purchased". It resets every episode (the env calls `labels_bought = [False]*4` on reset), so buying A once per episode is the maximum. The curve goes up as the agent learns to prioritise it.

---

## 2. How can DAI-P know something will be surprising before it experiences it?

It can't — and that's the precise answer. **DAI-P does not have foreknowledge.** The epistemic gain is computed from the transition network's *current* prediction error on transitions that have already been stored in the replay buffer. The sequence across episodes is:

1. **Early exploration** — the agent tries buy-label-A by chance (or Boltzmann noise). The transition network is poorly trained and makes a large prediction error on the resulting proprioceptive state flip. This error gets stored in the replay buffer tied to that action.
2. **EFE target update** — the large MSE augments the EFE target for buy-label-A, pushing its intrinsic value up.
3. **Next episodes** — buy-label-A now has a higher EFE value → Boltzmann sampling makes it more likely → the agent does it more often → more replay data confirms the pattern.

So yes, you're right: this is **learned across episodes from data**, not innate foreknowledge. And yes, it is **completely tied to the specific data distributions** — the reason DAI-P picks up label A first is precisely because A causes the largest state change *in this environment*. Change the geometry of the clusters and the epistemic priority ranking changes with it. That's actually a desirable property for a paper argument: DAI-P discovers the most informative label automatically, without being told which one it is.

---

## 3. Does this confirm the DDQN / DAI-P timing difference?

Yes. Both agents need to *experience* buying A before they can learn from it. The difference is the **temporal credit assignment**:

- **DDQN** must wait for the downstream reward signal to propagate back through many subsequent steps of blocked malicious flows before the Q-value for buy-label-A rises enough to be selected consistently.
- **DAI-P** gets an *immediate* intrinsic signal — the very transition in which A is bought produces the large MSE — so the EFE for that action is updated after just one experience. Faster signal, faster learning.

---

## 4. Explaining the five plots (those generated at the end of the experiment, not the wandb ones)

| Plot | X axis | Y axis | What it tells you |
|---|---|---|---|
| **Episode cumulative reward** | episode | sum of all rewards within the episode | Overall "how well did the agent play the budget game" |
| **Win rate** | episode | rolling fraction of episodes that ended with budget ≥ 40 | The main success metric |
| **Labels purchased / episode** | episode | mean number of CTI labels bought per episode | Are agents buying strategically (few, targeted) or impulsively? |
| **Label-A purchased (frac.)** | episode | rolling fraction of episodes in which label A was bought | Did the agent learn to prioritise the critical label? |
| **Final budget** | episode | budget value at episode end (win=40+, lose=0) | Finer-grained than win/lose — shows how comfortable the wins are |

For **Label-A**: within a single episode, label A is either bought or not — it's binary. The Y-axis is the smoothed average of that binary over a sliding window of episodes, so it's a fraction in [0, 1]. You cannot buy label A "more than once" per episode because after the first purchase `labels_bought[0]` stays `True` for the rest of the episode and subsequent buy attempts get the invalid-action penalty.

---

## 5. How does the inference module distinguish known from unknown?

It doesn't have access to ground-truth labels — it uses **distance to known prototypes** as a proxy:

```python
min_dist   = dists_to_known.min(-1).values
is_anomaly = min_dist > anomaly_threshold   # 1.8 in hidden space
```

A flow is declared *unknown/anomalous* if it is farther than `anomaly_threshold` from every known prototype. Below the threshold → classified as the nearest known class.

The clustering of unknowns then happens separately via the EMA unknown-cluster prototypes — but only for flows already deemed anomalous.

**The fundamental limitation** — and the deliberate trap — is that **Unknown A lives below that threshold** (it's close to K0). So it never even reaches the unknown-clustering step. It gets classified as K0 (benign) with high confidence, accepted, and the agent loses 7 per flow. There's no way for the inference module to detect A as unknown without either (a) buying the label, or (b) lowering the threshold so much that it starts flagging legitimate K0 flows as anomalous too. This is a principled approximation of the real-world problem: some attack traffic genuinely looks identical to normal traffic in your feature space.

---

## 6. Per-cluster proprioceptive state

PROPRIO_DIM 6→9, STATE_DIM 11→14. The proprioceptive state is waht the transitionnet predicts.  proprio[0:4] = [conf_A, conf_B, conf_C, conf_D] drawn from the existing unk_conf EMA buffers. The transition network can learn the exact pattern: buy-label-A → conf_A → 0, others unchanged. That's the sharpest possible signal for the epistemic gain. unk_conf[uid] is also zeroed in step() the moment a label is bought.