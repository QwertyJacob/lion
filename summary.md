## The Big Picture

You're simulating a **network security operator** that must decide, in real time, what to do with incoming traffic flows. The twist: some attack types are *unknown* — the system has never seen them before and doesn't know if they're dangerous. Buying a **Cyber Threat Intelligence (CTI) label** reveals the truth about an unknown type, but labels cost money and your budget is limited.

---

## The World: 7 Traffic Classes

Picture a 2D feature space (two measurements per flow, like packet size × frequency). Seven Gaussian clusters live in this space:

```
                    UB (malicious, clearly separate)
                    ●

    K0 (benign)        K1 (benign)
      ●          UA●      ●
                 (malicious, overlaps K0)

                    K2 (malicious, known)
                    ●

              UD ●            UC ●
         (benign, near K2)   (benign, far away)
```

**Known classes** (K0, K1, K2) — the system was pretrained on these. It knows exactly what they are.

**Unknown classes** — the system has *never seen these before*:
| Name | True type | Where it lives |
|---|---|---|
| **A** | malicious | overlaps K0 (benign) — misclassified as benign until labelled |
| B | malicious | far from known classes — clearly anomalous |
| C | benign | far from known classes — looks anomalous, but harmless |
| D | benign | near K2 (malicious) — looks dangerous, but it's fine |

---

## The Budget Game

Each episode starts with **budget = 15**. Every step, one flow arrives and the agent picks one of three actions:

- **Block (0)** — reject the flow
- **Accept (1)** — let it through
- **Buy label (2)** — spend budget to learn the true type of an unknown cluster

Win condition: budget reaches **40**. Lose condition: budget hits **0** or time runs out (50 steps).

Total label cost if you bought everything: **5 + 6 + 4 + 4 = 19 > 15**. So the agent *cannot afford all labels* — it must choose which ones to buy and in what order. This ordering decision is where the agents diverge.

---

## The Agents

### Break-Even (rule-based heuristic)
Uses anomaly score (distance of a cluster's prototype to the nearest known class) as a proxy for danger. Highest anomaly score → most suspicious → buy first.

**Why it struggles:** Anomaly score is a proxy that doesn't account for reward impact or budget consequences. It tends to prioritise visually conspicuous clusters over strategically critical ones.

### DDQN (pure reward-driven RL)
Standard Double DQN. Learns entirely from reward signals through trial and error: the reward structure gradually shapes the agent's label-acquisition strategy, but temporal credit assignment across many steps is slow.

### DAI-P (Deep Active Inference — the protagonist)
Same critic architecture as DDQN, but adds a **perceptive epistemic gain** term to the target:

```
EFE target = reward  +  ε × ||next_proprio − TransitionNet(s, a)||²  +  γ × EFE(s')
```

The transition network predicts the next **proprioceptive state** — the agent's internal readings: budget fraction, number of labels bought, current cluster confidence, etc.

**The key insight:** when the agent buys a label that causes a large proprioceptive state change, the transition network incurs high residual prediction error — **maximum epistemic gain** — making that purchase intrinsically valuable before the downstream reward signal alone would justify it. This immediate, model-based signal provides faster credit assignment than multi-step reward propagation.

---

## What the Experiment Measures

Running N episodes × M seeds, you compare:

| Phase | What you're watching |
|---|---|
| **Early episodes (1–50)** | How quickly do agents learn to allocate their label budget strategically? |
| **Late episodes (250–300)** | Which agents converge to a stable winning policy? Break-Even never improves. |

The **label purchase patterns** reveal how each agent prioritises its budget: which clusters it buys labels for, in what order, and whether it converges to a consistent strategy.

---

## The Inference Module Under the Hood

The inference module is a **prototypical classifier**: it learns a hidden embedding space where each known class lives near a learnable prototype vector. Classification = find the nearest prototype. No softmax over raw logits — distances in embedding space.

For unknown clusters, it maintains soft prototypes updated via **exponential moving average** (no K-means, fully differentiable). When a label is bought, that cluster's EMA prototype is *promoted* to the known-class list. Future flows from that cluster immediately get classified correctly — producing a large, sharp change in the proprioceptive state that the transition network reacts to.
