## The Big Picture

You're simulating a **network security operator** that must decide, in real time, what to do with incoming traffic flows. The twist: some attack types are *unknown* — the system has never seen them before and doesn't know if they're dangerous. Buying a **Cyber Threat Intelligence (CTI) label** reveals the truth about an unknown type, but labels cost money and your budget is limited.

---

## The World: 7 Traffic Classes

Picture a 2D feature space (two measurements per flow, like packet size × frequency). Seven Gaussian clusters live in this space:

```
                    UB (malicious, clearly separate)
                    ●  ← easy to spot as anomalous

    K0 (benign)        K1 (benign)
      ●          UA●      ●
                 ← malicious, but overlaps K0!
                 The system thinks it's benign.

                    K2 (malicious, known)
                    ●

              UD ●            UC ●
         (benign, near K2)   (benign, far away)
```

**Known classes** (K0, K1, K2) — the system was pretrained on these. It knows exactly what they are.

**Unknown classes** — the system has *never seen these before*:
| Name | True type | Where it lives | The problem |
|---|---|---|---|
| **A** | **malicious** | overlaps K0 (benign) | System thinks it's benign → **accepts it → bleeds budget** |
| B | malicious | far from known classes | Clearly anomalous, easy to spot |
| C | benign | far from known classes | Looks anomalous, but harmless |
| D | benign | near K2 (malicious) | Looks dangerous, but it's fine |

**Unknown A is the trap.** It sits so close to a benign cluster that the inference module confidently misclassifies every A-flow as benign — and the agent accepts it, taking a −7 reward hit each time.

---

## The Budget Game

Each episode starts with **budget = 15**. Every step, one flow arrives and the agent picks one of three actions:

- **Block (0)** — reject the flow
- **Accept (1)** — let it through
- **Buy label (2)** — spend budget to learn the true type of an unknown cluster

Win condition: budget reaches **40**. Lose condition: budget hits **0** or time runs out (150 steps).

Total label cost if you bought everything: **5 + 6 + 4 + 4 = 19 > 15**. So the agent *cannot afford all labels* — it must choose which ones to buy and in what order. This ordering decision is where the three agents diverge.

---

## The Three Agents

### Break-Even (rule-based heuristic)
Uses anomaly score (distance of a cluster's prototype to the nearest known class) as a proxy for danger. Highest anomaly score → most suspicious → buy first.

**Why it fails:** Unknown A has a *low* anomaly score (it overlaps K0). So Break-Even always buys B first (most anomalous-looking), then C or D. By the time it gets to A, the budget is gone. Meanwhile it keeps accepting A-flows and haemorrhaging −7 each time.

### DDQN (pure reward-driven RL)
Standard Double DQN with Boltzmann exploration. Learns by trial-and-error: eventually figures out that buying A early is the right move, but it takes many painful episodes of accepting A-flows before the reward signal teaches it this.

### DAI-P (Deep Active Inference — the protagonist)
Same critic architecture as DDQN, but adds a **perceptive epistemic gain** term to the target:

```
EFE target = reward  +  ε × ||next_proprio − TransitionNet(s, a)||²  +  γ × EFE(s')
```

The transition network predicts the next **proprioceptive state** — the agent's internal readings: budget fraction, number of labels bought, current cluster confidence, etc.

**The key insight:** when the agent buys label A, the proprioceptive state *lurches*. The cluster that the system had been confidently calling "benign K0" suddenly becomes "malicious A". That's the largest possible one-step state change in the whole environment. The transition network — still early in training — completely fails to predict this jump, so its residual error is huge → **maximum epistemic gain → DAI-P is intrinsically motivated to buy label A early**, before any reward signal even tells it this is the right thing to do.

In other words: DAI-P is curious about the most *surprising* action. And buying label A is the most surprising thing that can happen in this world.

---

## What the Experiment Measures

Running N episodes × M seeds, you compare:

| Phase | What you're watching |
|---|---|
| **Early episodes (1–50)** | Who figures out to buy A first before convergence? DAI-P's epistemic drive gives it a head start. |
| **Late episodes (250–300)** | All learned agents converge. Break-Even stays stuck because it never changes its heuristic. |

The **label_A_rate** metric is the smoking gun: does the agent buy label A in a given episode? DAI-P buys it earlier and more consistently in the early phase. The `[BUY]` lines in the console log let you watch this happen step by step.

---

## The Inference Module Under the Hood

The inference module is a **prototypical classifier**: it learns a hidden embedding space where each known class lives near a learnable prototype vector. Classification = find the nearest prototype. No softmax over raw logits — distances in embedding space.

For unknown clusters, it maintains soft prototypes updated via **exponential moving average** (no K-means, fully differentiable). When a label is bought, that cluster's EMA prototype is *promoted* to the known-class list. Future flows from that cluster immediately get classified correctly — which is exactly the large state flip that DAI-P's transition model reacts to.