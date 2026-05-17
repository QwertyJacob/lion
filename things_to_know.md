## 1. What is the `label_A` plot?

Each episode, `labels_bought[0]` is either `True` (label A was purchased at some point during the episode) or `False`. That boolean is logged as 0/1. The plot shows the **rolling mean** of that binary across episodes — so the Y-axis is simply "fraction of recent episodes in which label A was purchased". It resets every episode (the env calls `labels_bought = [False]*4` on reset). The curve reflects how consistently each agent chooses to acquire that particular label.

---

## 2. How can DAI-P know something will be surprising before it experiences it?

It can't — and that's the precise answer. **DAI-P does not have foreknowledge.** The epistemic gain is computed from the transition network's *current* prediction error on transitions that have already been stored in the replay buffer. The sequence across episodes is:

1. **Early exploration** — the agent tries a label purchase by chance (or Boltzmann sampling). The transition network is poorly trained and makes a large prediction error on the resulting proprioceptive state change. This error is stored in the replay buffer tied to that action.
2. **EFE target update** — the large MSE augments the EFE target for that purchase, pushing its intrinsic value up.
3. **Next episodes** — that label purchase now has a higher EFE value → Boltzmann sampling makes it more likely → the agent does it more often → more replay data confirms the pattern.

So yes: this is **learned across episodes from data**, not innate foreknowledge. The reason DAI-P prioritises certain labels is precisely because they cause the largest proprioceptive state changes *in this environment*. Change the geometry of the clusters and the epistemic priority ranking changes with it. That's a desirable property for a paper argument: DAI-P discovers the most informative label automatically, without being told which one it is.

---

## 3. Does this confirm the DDQN / DAI-P timing difference?

Yes. Both agents need to *experience* a label purchase before they can learn from it. The difference is **temporal credit assignment**:

- **DDQN** must wait for the downstream reward signal to propagate back through many subsequent steps before the Q-value for that purchase rises enough to be selected consistently.
- **DAI-P** gets an *immediate* intrinsic signal — the very transition in which the label is bought produces the large MSE — so the EFE for that action is updated after just one experience. Faster signal, faster learning.

---

## 4. Explaining the five plots (those generated at the end of the experiment, not the wandb ones)

| Plot | X axis | Y axis | What it tells you |
|---|---|---|---|
| **Episode cumulative reward** | episode | sum of all rewards within the episode | Overall "how well did the agent play the budget game" |
| **Win rate** | episode | rolling fraction of episodes that ended with budget ≥ 40 | The main success metric |
| **Labels purchased / episode** | episode | mean number of CTI labels bought per episode | Are agents buying strategically (few, targeted) or impulsively? |
| **Label-A purchased (frac.)** | episode | rolling fraction of episodes in which label A was bought | How consistently does the agent acquire this label? |
| **Final budget** | episode | budget value at episode end (win=40+, lose=0) | Finer-grained than win/lose — shows how comfortable the wins are |

For **Label-A**: within a single episode, label A is either bought or not — it's binary. The Y-axis is the smoothed average of that binary over a sliding window of episodes. You cannot buy label A "more than once" per episode because after the first purchase `labels_bought[0]` stays `True` for the rest of the episode and subsequent buy attempts get the invalid-action penalty.

---

## 5. How does the inference module distinguish known from unknown?

It doesn't have access to ground-truth labels — it uses **distance to known prototypes** as a proxy:

```python
min_dist   = dists_to_known.min(-1).values
is_anomaly = min_dist > anomaly_threshold   # 1.8 in hidden space
```

A flow is declared *unknown/anomalous* if it is farther than `anomaly_threshold` from every known prototype. Below the threshold → classified as the nearest known class.

The clustering of unknowns then happens separately via the EMA unknown-cluster prototypes — but only for flows already deemed anomalous.

Unknown A lives below that threshold (it's close to K0 in feature space), so it never reaches the unknown-clustering step. It gets classified as K0 (benign) with high confidence until its label is purchased. This is a principled approximation of the real-world problem: some attack traffic genuinely looks identical to normal traffic in your feature space.

---

## 6. Per-cluster proprioceptive state

PROPRIO_DIM 6→9, STATE_DIM 11→14. The proprioceptive state is what the TransitionNet predicts. `proprio[0:4] = [conf_A, conf_B, conf_C, conf_D]` drawn from the existing `unk_conf` EMA buffers. The transition network can learn the exact pattern: buying a label zeroes that cluster's confidence slot while leaving the others unchanged — a clean, cluster-specific signal. `unk_conf[uid]` is also zeroed in `step()` the moment a label is bought.
