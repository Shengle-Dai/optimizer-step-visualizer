---
title: "Watch an LSTM Remember — Demo Rationale"
geometry: margin=1in
fontsize: 11pt
---

**What the demo shows.** I pick the recommended preset, *"The movie was not bad,"* and walk through it phase-by-phase. Each token splits into the four canonical LSTM update steps — compute gates, forget, learn, output — and the demo lets a learner click through them one at a time. The narration follows the negation cue as it lands in a dedicated *negation* memory channel; when "bad" arrives, the LSTM remembers the negation and writes the new evidence into the *positive* channel instead of the negative one. The sentiment lands on positive. I then hover the gate and memory bars to surface the on-page tooltips, which name each gate and explain when it fires.

**Why this is effective.** The hardest thing to teach about LSTMs is *why memory matters*. Showing one moment where the cell state visibly rewrites the interpretation of the very next word — the negation flip — gives a clearer payoff than abstract gate equations. Walking phase-by-phase keeps the math concrete: every click corresponds to one term in the update rule, and the explanation card narrates that term in English.

**Intentional design choices.** I went rule-based rather than training a real LSTM. Learned weights give a dense cell state with no human-readable axes; here the four channels (positive, negative, negation, contrast) carry meanings the user can name. Every value on screen has a label, and the diagram annotates $c_{t-1} \to c_t$ along the top wire so viewers tie what they see to the equations card.

**How I iterated.** I traced every preset numerically and found one ("long-range contrast") whose result contradicted its stated lesson; I rephrased it so the contrast cue lands after the prior sentiment instead of before. I also collapsed a hidden 0.65 multiplier into the displayed forget value so on-screen numbers match the simulation exactly.
