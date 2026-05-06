/* ============================================================
   Watch an LSTM Remember
   Educational rule-based simulation of an LSTM doing sentiment.
   ============================================================ */

// ---------- Word dictionaries ----------

const positiveWords = new Set([
  "good", "great", "wonderful", "amazing", "excellent", "fun", "beautiful",
  "love", "loved", "best", "enjoyed", "fantastic", "brilliant", "delightful",
  "perfect", "happy", "joy", "joyful", "lovely", "nice", "thrilling"
]);

const negativeWords = new Set([
  "bad", "terrible", "awful", "boring", "slow", "worst", "hate", "hated",
  "poor", "confusing", "horrible", "dull", "ugly", "sad", "tedious",
  "disappointing", "weak", "messy", "annoying"
]);

const negationWords = new Set([
  "not", "never", "no", "hardly", "barely",
  "isn't", "wasn't", "don't", "didn't", "won't", "can't", "doesn't"
]);

const contrastWords = new Set([
  "but", "however", "although", "though", "yet"
]);

// ---------- Presets ----------

const PRESETS = [
  {
    text:    "The movie was not bad.",
    label:   "Negation flip",
    teaches: "Memory rewrites interpretation: 'not' + 'bad' → positive.",
    recommended: true
  },
  {
    text:    "The movie was wonderful.",
    label:   "Simple positive",
    teaches: "Positive evidence accumulates without surprises."
  },
  {
    text:    "The movie was terrible.",
    label:   "Simple negative",
    teaches: "Negative evidence accumulates without surprises."
  },
  {
    text:    "Although the beginning was slow, the ending was amazing.",
    label:   "Long-range contrast",
    teaches: "Contrast cue lets later evidence dominate the prediction."
  },
  {
    text:    "The acting was great, but the story was boring.",
    label:   "Mixed sentiment",
    teaches: "'But' downweights earlier sentiment; the second clause flips the verdict."
  }
];

const DEFAULT_PRESET = 0; // "not bad" — the central case (now first in the array)

// ---------- State ----------

const initialState = () => ({
  tokens: [],
  currentStep: -1,
  subPhase: -1,  // -1 = nothing done for currentStep; 0..3 = sub-phase just completed
  cellState:   { positive: 0, negative: 0, negation: 0, contrast: 0 },
  hiddenState: { positiveSignal: 0, negativeSignal: 0 },
  gates:       { forget: 0, input: 0, candidate: 0, output: 0 },
  sentiment:   { positiveProb: 0.5, negativeProb: 0.5, label: "Uncertain" }
});

const PHASE_NAMES = [
  "Compute gates",
  "Forget existing",
  "Learn new content",
  "Output & predict"
];

let state = initialState();
let history = [];
let autoTimer = null;
let speedMs = 600;

// ---------- DOM refs ----------

const $ = (id) => document.getElementById(id);
const presetGrid       = $("presetGrid");
const customInput      = $("customInput");
const loadBtn          = $("loadSentenceBtn");
const setupCard        = $("setupCard");
const setupExpandBtn   = $("setupExpandBtn");
const loadedSentenceEl = $("loadedSentenceText");
const tokenTimeline  = $("tokenTimeline");
const stepCounter    = $("stepCounter");
const lstmDiagram    = $("lstmDiagram");
const gateBars       = $("gateBars");
const memoryBars     = $("memoryBars");
const memorySummary  = $("memorySummary");
const sentimentBars  = $("sentimentBars");
const sentimentLabel = $("sentimentLabel");
const stepExplanation= $("stepExplanation");
const prevBtn        = $("prevBtn");
const phaseBtn       = $("phaseBtn");
const nextBtn        = $("nextBtn");
const autoBtn        = $("autoBtn");
const resetBtn       = $("resetBtn");
const speedSlider    = $("speedSlider");
const speedDisplay   = $("speedDisplay");
const tooltipEl      = $("tooltip");
const mathEquations  = $("mathEquations");

// ============================================================
// Tokenization & classification
// ============================================================

function tokenize(s) {
  return s.replace(/[.,!?;:"()]/g, "").split(/\s+/).filter(Boolean);
}

function classifyToken(token) {
  const w = token.toLowerCase();
  if (negationWords.has(w)) return "negation";
  if (contrastWords.has(w)) return "contrast";
  if (positiveWords.has(w)) return "positive";
  if (negativeWords.has(w)) return "negative";
  return "neutral";
}

// ============================================================
// Gate computations
// ============================================================

function computeForgetGate(type) {
  if (type === "contrast") return 0.45;
  return 0.85;
}

function computeInputGate(type) {
  if (type === "negation") return 0.9;
  if (type === "positive" || type === "negative") return 0.85;
  if (type === "contrast") return 0.6;
  return 0.2;
}

function computeCandidate(token, type) {
  if (type === "positive") return 1;
  if (type === "negative") return -1;
  if (type === "negation") return 0; // stored in negation channel
  if (type === "contrast") return 0;
  return 0;
}

function computeOutputGate(type) {
  if (type === "positive" || type === "negative") return 0.85;
  if (type === "negation") return 0.35;
  if (type === "contrast") return 0.5;
  return 0.5;
}

// ============================================================
// Memory updates
// ============================================================

function applyForgetGate(f) {
  state.cellState.positive *= f;
  state.cellState.negative *= f;
  state.cellState.negation *= f;
  state.cellState.contrast *= f;
}

function applyInputGate(i, candidate, type) {
  const negationActive = state.cellState.negation > 0.4;
  const contrastBoost  = 1 + state.cellState.contrast * 0.5;

  if (type === "positive") {
    if (negationActive) state.cellState.negative += i * 0.8 * contrastBoost;
    else                state.cellState.positive += i * 1.0 * contrastBoost;
  } else if (type === "negative") {
    if (negationActive) state.cellState.positive += i * 0.8 * contrastBoost;
    else                state.cellState.negative += i * 1.0 * contrastBoost;
  } else if (type === "negation") {
    state.cellState.negation += i * 1.0;
  } else if (type === "contrast") {
    state.cellState.contrast += i * 1.0;
    state.cellState.positive *= 0.65;
    state.cellState.negative *= 0.65;
  }
}

function decayNegation(type) {
  if (type !== "negation") state.cellState.negation *= 0.75;
}

function computeHiddenState(o) {
  state.hiddenState.positiveSignal = o * Math.tanh(state.cellState.positive);
  state.hiddenState.negativeSignal = o * Math.tanh(state.cellState.negative);
}

function computeSentiment() {
  const pos = state.hiddenState.positiveSignal;
  const neg = state.hiddenState.negativeSignal;
  const expPos = Math.exp(pos);
  const expNeg = Math.exp(neg);
  const positiveProb = expPos / (expPos + expNeg);
  const negativeProb = expNeg / (expPos + expNeg);
  state.sentiment.positiveProb = positiveProb;
  state.sentiment.negativeProb = negativeProb;
  if      (positiveProb > 0.6) state.sentiment.label = "Positive";
  else if (negativeProb > 0.6) state.sentiment.label = "Negative";
  else                          state.sentiment.label = "Uncertain";
}

// ============================================================
// Sub-phase wrappers — each applies one phase of the LSTM update
// ============================================================

function phaseCompute(token, type) {
  state._negationWasActive = state.cellState.negation > 0.4;
  state.gates = {
    forget:    computeForgetGate(type),
    input:     computeInputGate(type),
    candidate: computeCandidate(token, type),
    output:    computeOutputGate(type)
  };
}

function phaseForget() {
  applyForgetGate(state.gates.forget);
}

function phaseLearn(type) {
  applyInputGate(state.gates.input, state.gates.candidate, type);
  decayNegation(type);
}

function phaseOutput() {
  computeHiddenState(state.gates.output);
  computeSentiment();
}

// ============================================================
// Step explanations
// ============================================================

function generateExplanation(token, type) {
  const tok = `<code>${token}</code>`;
  const negationWasActive = state._negationWasActive;

  if (type === "neutral") {
    return `${tok} carries little sentiment. The <span class="cl-input">input gate</span> stays low (~0.2), so almost nothing new is written into memory. Existing memory persists thanks to a high <span class="cl-forget">forget gate</span> (~0.85).`;
  }

  if (type === "positive") {
    if (negationWasActive) {
      return `${tok} is normally <strong class="flip">positive</strong>, but the LSTM still remembers a recent negation in its cell state — so it writes this into the <em>negative</em> channel instead. <strong class="flip">"not good" → negative</strong>.`;
    }
    return `${tok} is positive. The <span class="cl-input">input gate</span> opens (~0.85) and writes to the positive channel. The <span class="cl-output">output gate</span> reveals confident sentiment.`;
  }

  if (type === "negative") {
    if (negationWasActive) {
      return `${tok} is normally negative, but the LSTM remembers a recent negation. The interpretation flips: it writes into the <em>positive</em> channel instead. <strong class="flip">"not bad" → positive</strong>.`;
    }
    return `${tok} is negative. The <span class="cl-input">input gate</span> opens (~0.85) and writes to the negative channel.`;
  }

  if (type === "negation") {
    return `${tok} is a <strong class="warn">negation cue</strong>. The LSTM stores it in a dedicated <em>negation</em> memory slot. The <span class="cl-output">output gate</span> dips (~0.35) — the model isn't ready to commit to sentiment until it sees what is being negated. <strong class="flip">Watch the next sentiment word: its interpretation will flip.</strong>`;
  }

  if (type === "contrast") {
    return `${tok} signals contrast. The <span class="cl-forget">forget gate</span> drops (~0.45), partially erasing prior sentiment so future evidence has more weight. A contrast signal is written to memory.`;
  }

  return "";
}

// Phase-by-phase explanation: what just happened in this sub-step.
function generatePhaseExplanation(phase, token, type) {
  const tok = `<code>${token}</code>`;
  const g = state.gates;
  const fNum = g.forget.toFixed(2);
  const iNum = g.input.toFixed(2);
  const cNum = g.candidate >= 0 ? `+${g.candidate.toFixed(2)}` : g.candidate.toFixed(2);
  const oNum = g.output.toFixed(2);

  if (phase === 0) {
    let typeNote = "";
    if (type === "negation")  typeNote = ` It's a <strong class="warn">negation cue</strong>, so the input gate is high but the candidate has no sentiment content (it'll be stored in the dedicated negation channel instead).`;
    else if (type === "contrast") typeNote = ` It's a <strong>contrast cue</strong>, so the forget gate drops sharply — old sentiment is about to get partially wiped.`;
    else if (type === "positive") typeNote = ` It's a <strong style="color:var(--positive);">positive</strong> sentiment word, so the candidate is +1 and the input gate is high.`;
    else if (type === "negative") typeNote = ` It's a <strong style="color:var(--negative);">negative</strong> sentiment word, so the candidate is −1 and the input gate is high.`;
    else typeNote = ` It carries little sentiment, so the input gate stays low — almost nothing new will be written.`;
    return `<strong>Phase 1 of 4: Compute gates.</strong><br>
The cell reads ${tok} together with the previous hidden state h<sub>t-1</sub> and computes all four gate values:
<span class="cl-forget">forget</span> = ${fNum}, <span class="cl-input">input</span> = ${iNum}, candidate = ${cNum}, <span class="cl-output">output</span> = ${oNum}.
${typeNote} <em>No memory has changed yet</em> — these values just describe what the cell is <em>about</em> to do.`;
  }

  if (phase === 1) {
    const pct = Math.round((1 - g.forget) * 100);
    const note = g.forget >= 0.7
      ? `Old memory is mostly preserved — only ${pct}% gets wiped this step.`
      : `A large slice (${pct}%) of old memory is wiped this step. This is what happens on contrast cues like "but" — the LSTM lets later evidence dominate.`;
    return `<strong>Phase 2 of 4: Forget existing memory.</strong><br>
Multiply every cell-state channel by the <span class="cl-forget">forget gate</span> (${fNum}). This is the <code>f<sub>t</sub> · c<sub>t-1</sub></code> term in <code>c<sub>t</sub> = f<sub>t</sub>·c<sub>t-1</sub> + i<sub>t</sub>·g<sub>t</sub></code>.
${note}`;
  }

  if (phase === 2) {
    const learn = generateExplanation(token, type);
    return `<strong>Phase 3 of 4: Learn new content.</strong><br>
The <span class="cl-input">input gate</span> (${iNum}) decides how much of the candidate (${cNum}) lands in memory — this is the <code>i<sub>t</sub>·g<sub>t</sub></code> term that gets <em>added</em> to the (already-forgotten) cell state.<br>
${learn}`;
  }

  if (phase === 3) {
    const pos = (state.sentiment.positiveProb * 100).toFixed(0);
    const neg = (state.sentiment.negativeProb * 100).toFixed(0);
    const labelColor = state.sentiment.label === "Positive" ? "var(--positive)"
                     : state.sentiment.label === "Negative" ? "var(--negative)"
                     : "var(--muted)";
    const outputNote = g.output < 0.4
      ? `The output gate is <em>low</em> (${oNum}) — the cell is holding back, signalling that it isn't ready to commit yet (typical right after a negation cue).`
      : `The output gate is open (${oNum}), so the cell reveals its current memory confidently.`;
    return `<strong>Phase 4 of 4: Output &amp; predict.</strong><br>
Compute the hidden state <code>h<sub>t</sub> = o<sub>t</sub> · tanh(c<sub>t</sub>)</code>. ${outputNote}
The classifier softmaxes the positive vs. negative hidden-state signals → <strong style="color:${labelColor};">${state.sentiment.label}</strong> (${pos}% / ${neg}%).`;
  }

  return "";
}

function renderPhaseExplanation(phase, token, type) {
  stepExplanation.innerHTML = generatePhaseExplanation(phase, token, type);
}

function generateMemorySummary() {
  const cs = state.cellState;
  const parts = [];
  if (cs.negation > 0.4) parts.push("a recent <strong>negation</strong> cue");
  if (cs.contrast > 0.4) parts.push("a <strong>contrast</strong> signal");
  if (cs.positive > 0.5) parts.push(`<strong>positive</strong> evidence (${cs.positive.toFixed(2)})`);
  if (cs.negative > 0.5) parts.push(`<strong>negative</strong> evidence (${cs.negative.toFixed(2)})`);
  if (parts.length === 0) return "Cell state is mostly empty.";
  return `Memory is currently holding: ${parts.join(", ")}.`;
}

function finalSummary() {
  const label = state.sentiment.label;
  const cs = state.cellState;
  let why = "";
  if (cs.negation > 0.3 && cs.positive > cs.negative) {
    why = " The model used remembered <em>negation</em> to flip a negative word into positive.";
  } else if (cs.contrast > 0.3) {
    why = " The model down-weighted earlier sentiment after a <em>contrast</em> word, letting later evidence dominate.";
  } else if (cs.positive > cs.negative) {
    why = " Positive evidence accumulated across the sentence.";
  } else if (cs.negative > cs.positive) {
    why = " Negative evidence accumulated across the sentence.";
  }
  return `<strong>Final prediction: ${label}.</strong>${why}`;
}

// ============================================================
// Step controls
// ============================================================

function deepClone(obj) { return JSON.parse(JSON.stringify(obj)); }

let animLock = false;

// ---------- Animation helpers ----------

function pulseAlongPath(cell, pathId, durMs, color) {
  return new Promise(resolve => {
    const path = cell.querySelector(`[data-path-id="${pathId}"]`);
    if (!path) { resolve(); return; }
    const dot = el("circle", {
      r: 5, cx: 0, cy: 0, fill: color, "class": "pulse-dot"
    }, cell);
    const motion = el("animateMotion", {
      dur: `${durMs}ms`, fill: "freeze",
      begin: "indefinite", repeatCount: "1",
      path: path.getAttribute("d")
    }, dot);
    let done = false;
    const finish = () => {
      if (done) return; done = true;
      dot.remove();
      resolve();
    };
    motion.addEventListener("endEvent", finish);
    try { motion.beginElement(); } catch (e) { /* noop */ }
    setTimeout(finish, durMs + 60); // safety fallback
  });
}

function flashElement(cell, flashId, durMs) {
  cell.querySelectorAll(`[data-flash-id="${flashId}"]`).forEach(t => {
    t.classList.add("flashing");
    setTimeout(() => t.classList.remove("flashing"), durMs);
  });
}

function flashAndCountUp(cell, gateKey, targetVal, durMs) {
  const lbl = cell.querySelector(`[data-val="${gateKey}"]`);
  const box = cell.querySelector(`[data-tooltip-key="${gateKey}"]`);
  return new Promise(resolve => {
    if (box) box.classList.add("flashing");
    if (!lbl) { resolve(); return; }
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / durMs);
      lbl.textContent = (targetVal * t).toFixed(2);
      if (t < 1) requestAnimationFrame(tick);
      else {
        if (box) box.classList.remove("flashing");
        lbl.textContent = targetVal.toFixed(2);
        resolve();
      }
    }
    requestAnimationFrame(tick);
  });
}

function clearPulses() {
  document.querySelectorAll(".pulse-dot").forEach(d => d.remove());
  document.querySelectorAll(".flashing").forEach(e => e.classList.remove("flashing"));
}

// ---------- Step animation orchestrator ----------

async function animateStep(idx) {
  const cell = cellGroups[idx];
  if (!cell) return;

  const sf = Math.max(0.33, speedMs / 900);
  const T = (ms) => Math.max(40, ms * sf);

  // Reset gate labels to 0.00 for the count-up
  ["forget","input","candidate","output"].forEach(k => {
    const lbl = cell.querySelector(`[data-val="${k}"]`);
    if (lbl) lbl.textContent = "0.00";
  });

  // Phase A: x_t pulse + h_{t-1} pulse arrive in parallel
  await Promise.all([
    pulseAlongPath(cell, "x-riser",  T(420), "#2563eb"),
    pulseAlongPath(cell, "h-bus-in", T(420), "#7c3aed")
  ]);

  // Phase B: merged signal travels along bus → splits into 4 risers
  await pulseAlongPath(cell, "bus-spine", T(320), "#1f2937");
  await Promise.all([
    pulseAlongPath(cell, "riser-forget",    T(220), "#ca8a04"),
    pulseAlongPath(cell, "riser-input",     T(220), "#ca8a04"),
    pulseAlongPath(cell, "riser-candidate", T(220), "#ca8a04"),
    pulseAlongPath(cell, "riser-output",    T(220), "#ca8a04")
  ]);

  // Phase C: each gate "computes" — flash + value count-up in parallel
  await Promise.all([
    flashAndCountUp(cell, "forget",    state.gates.forget,    T(500)),
    flashAndCountUp(cell, "input",     state.gates.input,     T(500)),
    flashAndCountUp(cell, "candidate", state.gates.candidate, T(500)),
    flashAndCountUp(cell, "output",    state.gates.output,    T(500))
  ]);

  // Phase D: gate outputs flow up
  await Promise.all([
    pulseAlongPath(cell, "forget-up",    T(280), "#ef4444"),
    pulseAlongPath(cell, "input-up",     T(320), "#22c55e"),
    pulseAlongPath(cell, "candidate-up", T(320), "#a855f7"),
    pulseAlongPath(cell, "output-up",    T(320), "#eab308")
  ]);
  flashElement(cell, "forget-times", T(300));
  flashElement(cell, "input-times",  T(300));

  // Phase E: input × → +
  await pulseAlongPath(cell, "input-to-plus", T(220), "#22c55e");
  flashElement(cell, "plus-node", T(280));

  // Phase F: cell-state pulses out (right) and down (toward tanh)
  await Promise.all([
    pulseAlongPath(cell, "cellstate-out",  T(380), "#0f172a"),
    pulseAlongPath(cell, "cellstate-down", T(280), "#0f172a")
  ]);
  flashElement(cell, "tanh-ellipse", T(300));

  // Phase G: tanh → output × → h_t (right + up)
  await pulseAlongPath(cell, "tanh-to-ox", T(220), "#dc2626");
  flashElement(cell, "output-times", T(300));
  await Promise.all([
    pulseAlongPath(cell, "ht-up",    T(320), "#7c3aed"),
    pulseAlongPath(cell, "ht-right", T(320), "#7c3aed")
  ]);
  flashElement(cell, "ht-bubble", T(400));
}

// ---------- Sub-phase animation (shorter, only the relevant segment) ----------

async function animateSubPhase(cell, phase) {
  if (!cell) return;
  const sf = Math.max(0.33, speedMs / 900);
  const T = (ms) => Math.max(40, ms * sf);

  if (phase === 0) {
    // Read input + compute gates
    ["forget","input","candidate","output"].forEach(k => {
      const lbl = cell.querySelector(`[data-val="${k}"]`);
      if (lbl) lbl.textContent = "0.00";
    });
    await Promise.all([
      pulseAlongPath(cell, "x-riser",  T(420), "#2563eb"),
      pulseAlongPath(cell, "h-bus-in", T(420), "#7c3aed")
    ]);
    await pulseAlongPath(cell, "bus-spine", T(320), "#1f2937");
    await Promise.all([
      pulseAlongPath(cell, "riser-forget",    T(180), "#ca8a04"),
      pulseAlongPath(cell, "riser-input",     T(180), "#ca8a04"),
      pulseAlongPath(cell, "riser-candidate", T(180), "#ca8a04"),
      pulseAlongPath(cell, "riser-output",    T(180), "#ca8a04")
    ]);
    await Promise.all([
      flashAndCountUp(cell, "forget",    state.gates.forget,    T(450)),
      flashAndCountUp(cell, "input",     state.gates.input,     T(450)),
      flashAndCountUp(cell, "candidate", state.gates.candidate, T(450)),
      flashAndCountUp(cell, "output",    state.gates.output,    T(450))
    ]);
  } else if (phase === 1) {
    // Forget: forget gate flows up to the × on the cell-state wire
    await pulseAlongPath(cell, "forget-up", T(320), "#ef4444");
    flashElement(cell, "forget-times", T(380));
  } else if (phase === 2) {
    // Learn: input + candidate combine at × → +
    await Promise.all([
      pulseAlongPath(cell, "input-up",     T(340), "#22c55e"),
      pulseAlongPath(cell, "candidate-up", T(340), "#a855f7")
    ]);
    flashElement(cell, "input-times", T(320));
    await pulseAlongPath(cell, "input-to-plus", T(260), "#22c55e");
    flashElement(cell, "plus-node", T(320));
    await pulseAlongPath(cell, "cellstate-out", T(380), "#0f172a");
  } else if (phase === 3) {
    // Output: cell-state → tanh → output × → h_t
    await pulseAlongPath(cell, "cellstate-down", T(280), "#0f172a");
    flashElement(cell, "tanh-ellipse", T(320));
    await pulseAlongPath(cell, "tanh-to-ox", T(260), "#dc2626");
    await pulseAlongPath(cell, "output-up", T(340), "#eab308");
    flashElement(cell, "output-times", T(320));
    await Promise.all([
      pulseAlongPath(cell, "ht-up",    T(340), "#7c3aed"),
      pulseAlongPath(cell, "ht-right", T(340), "#7c3aed")
    ]);
    flashElement(cell, "ht-bubble", T(420));
  }
}

// ---------- Step controls ----------

// Apply remaining sub-phases of the current token silently (no animation).
// Used when the user clicks "Next token" while mid-phase.
function completeRemainingPhasesSilently() {
  if (state.currentStep < 0 || state.subPhase === 3) return;
  const token = state.tokens[state.currentStep];
  const type  = classifyToken(token);
  if (state.subPhase < 0) phaseCompute(token, type);
  if (state.subPhase < 1) phaseForget();
  if (state.subPhase < 2) phaseLearn(type);
  if (state.subPhase < 3) phaseOutput();
  state.subPhase = 3;
}

async function stepForward() {
  if (animLock) return;
  if (state.currentStep >= state.tokens.length - 1 && state.subPhase === 3) return;
  animLock = true;

  // If user sub-phased into the current token, finish it silently before crossing.
  completeRemainingPhasesSilently();

  if (state.currentStep === -1) collapseSetup();

  history.push(deepClone(state));
  state.currentStep += 1;
  state.subPhase = -1;
  const token = state.tokens[state.currentStep];
  const type  = classifyToken(token);

  // Apply all four phases for the new token
  phaseCompute(token, type);
  phaseForget();
  phaseLearn(type);
  phaseOutput();
  state.subPhase = 3;

  // 1) Slide the diagram + update active/faded classes
  const idx = state.currentStep;
  const tx = (VIEW_W - CELL_W) / 2 - idx * CELL_W;
  const roll = document.getElementById("lstmRoll");
  if (roll) roll.style.transform = `translateX(${tx}px)`;
  cellGroups.forEach((cg, j) => {
    cg.classList.toggle("active", j === idx);
    cg.classList.toggle("faded",  j !== idx);
  });

  // 2) Update side panels + timeline immediately
  renderTimeline();
  renderGates();
  renderMemory();
  renderSentiment();
  renderExplanation();
  updateStepCounter();
  updateButtons();

  // 3) Wait for slide CSS transition (~0.7s) then animate
  await new Promise(r => setTimeout(r, 720));
  await animateStep(idx);

  if (state.currentStep === state.tokens.length - 1) {
    stepExplanation.innerHTML += `<br><br>${finalSummary()}`;
  }

  animLock = false;
  updateButtons();
}

// Advance one sub-phase. If currently at end-of-token (subPhase === 3) or
// before-start (currentStep === -1), this crosses the token boundary into
// the new token's phase 0.
async function stepPhase() {
  if (animLock) return;
  if (state.currentStep >= state.tokens.length - 1 && state.subPhase === 3) return;
  animLock = true;

  const startNewToken = state.currentStep === -1 || state.subPhase === 3;

  if (startNewToken) {
    if (state.currentStep === -1) collapseSetup();
    history.push(deepClone(state));
    state.currentStep += 1;
    state.subPhase = -1;

    // Slide diagram to new cell and update active/faded
    const idx = state.currentStep;
    const tx = (VIEW_W - CELL_W) / 2 - idx * CELL_W;
    const roll = document.getElementById("lstmRoll");
    if (roll) roll.style.transform = `translateX(${tx}px)`;
    cellGroups.forEach((cg, j) => {
      cg.classList.toggle("active", j === idx);
      cg.classList.toggle("faded",  j !== idx);
    });
    renderTimeline();
    updateStepCounter();
    updateButtons();
    await new Promise(r => setTimeout(r, 720));
  }

  const idx = state.currentStep;
  const token = state.tokens[idx];
  const type = classifyToken(token);
  const nextPhase = state.subPhase + 1; // 0..3

  // Apply this phase's math
  if (nextPhase === 0) phaseCompute(token, type);
  else if (nextPhase === 1) phaseForget();
  else if (nextPhase === 2) phaseLearn(type);
  else if (nextPhase === 3) phaseOutput();

  state.subPhase = nextPhase;

  // Render side panels immediately
  renderTimeline();
  renderGates();
  renderMemory();
  renderSentiment();
  renderPhaseExplanation(nextPhase, token, type);
  updateStepCounter();
  updateButtons();

  // Brief sub-phase animation
  await animateSubPhase(cellGroups[idx], nextPhase);

  if (state.currentStep === state.tokens.length - 1 && state.subPhase === 3) {
    stepExplanation.innerHTML += `<br><br>${finalSummary()}`;
  }

  animLock = false;
  updateButtons();
}

function stepBackward() {
  if (history.length === 0) return;
  if (animLock) {
    clearPulses();
    animLock = false;
  }
  state = history.pop();
  render();
}

function resetState() {
  stopAuto();
  clearPulses();
  animLock = false;
  history = [];
  const tokens = state.tokens.slice();
  state = initialState();
  state.tokens = tokens;
  render();
  expandSetup();
  stepExplanation.innerHTML = `Click <strong>Start ▶</strong> for a full token, or <strong>▶ 1. Compute gates</strong> to step through one phase at a time.`;
}

function loadSentence(s) {
  stopAuto();
  history = [];
  state = initialState();
  state.tokens = tokenize(s);
  buildDiagram();
  render();
  if (loadedSentenceEl) loadedSentenceEl.textContent = s;
  expandSetup(); // any new sentence resets the setup card to expanded
  stepExplanation.innerHTML = `Loaded a sentence with ${state.tokens.length} tokens.<br>
Click <strong>Start ▶</strong> to advance one full token (computes all four phases at once and plays the full animation).<br>
Or click <strong>▶ 1. Compute gates</strong> to step through one phase at a time and read what each phase does.`;
}

function collapseSetup() {
  if (setupCard) setupCard.classList.add("collapsed");
}
function expandSetup() {
  if (setupCard) setupCard.classList.remove("collapsed");
}

// ============================================================
// Auto-play
// ============================================================

async function startAuto() {
  if (autoTimer) return;
  autoTimer = true;
  autoBtn.textContent = "Pause";
  autoBtn.classList.add("active");
  while (autoTimer && state.currentStep < state.tokens.length - 1) {
    await stepForward();
    if (autoTimer) await new Promise(r => setTimeout(r, 250));
  }
  stopAuto();
}

function stopAuto() {
  autoTimer = null;
  autoBtn.textContent = "Auto-play";
  autoBtn.classList.remove("active");
}

// ============================================================
// Rendering
// ============================================================

function render() {
  renderTimeline();
  renderLSTMDiagram();
  renderGates();
  renderMemory();
  renderSentiment();
  renderExplanation();
  updateButtons();
  updateStepCounter();
}

function renderTimeline() {
  tokenTimeline.innerHTML = state.tokens.map((tok, i) => {
    const type = classifyToken(tok);
    let cls = "token-pill";
    if (i < state.currentStep) cls += " done";
    else if (i === state.currentStep) cls += " current";
    else cls += " future";
    return `<span class="${cls}" data-type="${type}">${tok}</span>`;
  }).join("");
}

function updateStepCounter() {
  if (state.tokens.length === 0) { stepCounter.textContent = "—"; return; }
  if (state.currentStep < 0) {
    stepCounter.textContent = `Step 0 of ${state.tokens.length}`;
  } else {
    const tok = state.tokens[state.currentStep];
    const phaseHint = (state.subPhase >= 0 && state.subPhase < 3)
      ? ` · phase ${state.subPhase + 1}/4: ${PHASE_NAMES[state.subPhase]}`
      : "";
    stepCounter.textContent = `Step ${state.currentStep + 1} of ${state.tokens.length} — "${tok}"${phaseHint}`;
  }
}

function updateButtons() {
  prevBtn.disabled = history.length === 0;

  const atVeryEnd = state.currentStep >= state.tokens.length - 1 && state.subPhase === 3;
  nextBtn.disabled  = atVeryEnd;
  phaseBtn.disabled = atVeryEnd;

  // Full-step button label
  nextBtn.textContent = state.currentStep < 0 ? "Start ▶" : "Next token ▶";

  // Phase button: label always names the *next* phase to be applied
  let nextPhaseIdx;
  if (state.currentStep < 0 || state.subPhase === 3) {
    nextPhaseIdx = 0; // crossing token boundary on next click
  } else {
    nextPhaseIdx = state.subPhase + 1;
  }
  phaseBtn.textContent = `▶ ${nextPhaseIdx + 1}. ${PHASE_NAMES[nextPhaseIdx]}`;
}

// ----- Gates -----

const GATE_META = [
  { key: "forget",    name: "Forget",    desc: "How much old memory to keep" },
  { key: "input",     name: "Input",     desc: "How much new info to write" },
  { key: "output",    name: "Output",    desc: "How much memory to reveal" }
];

const CANDIDATE_META = { key: "candidate", name: "Candidate (g<sub>t</sub>)" };

function signedDisplay(v) {
  if (v > 0)   return `+${v.toFixed(2)}`;
  if (v < 0)   return `−${Math.abs(v).toFixed(2)}`; // proper minus sign
  return ` 0.00`;
}

function renderGates() {
  const gateRows = GATE_META.map(g => {
    const v = state.gates[g.key];
    const pct = v * 100;
    return `
      <div class="bar-row" data-key="${g.key}" data-tooltip="gate-${g.key}">
        <span class="bar-label">${g.name}</span>
        <div class="bar-track">
          <div class="bar-fill unsigned" style="width:${pct}%;"></div>
        </div>
        <span class="bar-value">${v.toFixed(2)}</span>
      </div>`;
  }).join("");

  const cv = state.gates.candidate;
  const cPct = Math.abs(cv) * 50;
  const cOffset = cv >= 0 ? `left:50%;` : `right:50%;left:auto;`;
  const candidateRow = `
    <div class="bar-row candidate-row" data-key="candidate" data-tooltip="gate-candidate">
      <span class="bar-label">
        <span class="bar-mainlabel">${CANDIDATE_META.name}</span>
        <span class="bar-sublabel">new content, not a gate</span>
      </span>
      <div class="bar-track signed-track">
        <div class="bar-fill" style="width:${cPct}%; ${cOffset}"></div>
      </div>
      <span class="bar-value">${signedDisplay(cv)}</span>
    </div>`;

  gateBars.innerHTML = gateRows + candidateRow;

  // Wire gate tooltips
  GATE_META.forEach(g => {
    const row = gateBars.querySelector(`[data-key="${g.key}"]`);
    addTooltip(row, () => ({
      html: `<div class="tt-title">${g.name} gate</div><div class="tt-body">${GATE_TOOLTIPS[g.key]}</div>`
    }));
  });
  const candRow = gateBars.querySelector(`[data-key="candidate"]`);
  addTooltip(candRow, () => ({
    html: `<div class="tt-title">${CANDIDATE_META.name}</div><div class="tt-body">${GATE_TOOLTIPS.candidate}</div>`
  }));
}

const GATE_TOOLTIPS = {
  forget:    "Controls how much of the previous cell state survives. High = remember nearly everything; low = wipe most of it. Drops on contrast words like 'but'.",
  input:     "Controls how much new information gets written into the cell state. High on sentiment and negation words; low on neutral filler like 'the'.",
  candidate: "<strong>Not a gate</strong> — this is the <strong>new content</strong> being proposed (+1 for positive words, −1 for negative). The input gate decides how much of it actually lands in memory.",
  output:    "Controls how much of the cell state is revealed as the hidden state used by the classifier. Dips after 'not' because the model isn't yet sure what is being negated."
};

// ----- Memory -----

const MEM_META = [
  { key: "positive", name: "Positive" },
  { key: "negative", name: "Negative" },
  { key: "negation", name: "Negation" },
  { key: "contrast", name: "Contrast" }
];

const MEM_TOOLTIPS = {
  positive: "Accumulated positive sentiment evidence. Grows when positive words enter without active negation.",
  negative: "Accumulated negative sentiment evidence.",
  negation: "Tracks recent negation cues like 'not'. Decays over the next ~3 tokens. While active, sentiment-word writes get flipped.",
  contrast: "Tracks recent contrast cues like 'but', 'although'. While active, new evidence is amplified and old evidence was already partially forgotten."
};

function renderMemory() {
  memoryBars.innerHTML = MEM_META.map(m => {
    const v = state.cellState[m.key];
    const pct = Math.min(Math.abs(v) * 50, 100);
    return `
      <div class="bar-row" data-key="${m.key}" data-tooltip="mem-${m.key}">
        <span class="bar-label">${m.name}</span>
        <div class="bar-track">
          <div class="bar-fill unsigned" style="width:${pct}%"></div>
        </div>
        <span class="bar-value">${v.toFixed(2)}</span>
      </div>`;
  }).join("");

  MEM_META.forEach(m => {
    const row = memoryBars.querySelector(`[data-key="${m.key}"]`);
    addTooltip(row, () => ({
      html: `<div class="tt-title">${m.name} memory</div><div class="tt-body">${MEM_TOOLTIPS[m.key]}</div>`
    }));
  });

  memorySummary.innerHTML = generateMemorySummary();
}

// ----- Sentiment -----

function renderSentiment() {
  const pos = state.sentiment.positiveProb;
  const neg = state.sentiment.negativeProb;
  sentimentBars.innerHTML = `
    <div class="bar-row" data-key="positive">
      <span class="bar-label">Positive</span>
      <div class="bar-track"><div class="bar-fill unsigned" style="width:${pos*100}%"></div></div>
      <span class="bar-value">${(pos*100).toFixed(0)}%</span>
    </div>
    <div class="bar-row" data-key="negative">
      <span class="bar-label">Negative</span>
      <div class="bar-track"><div class="bar-fill unsigned" style="width:${neg*100}%"></div></div>
      <span class="bar-value">${(neg*100).toFixed(0)}%</span>
    </div>`;

  sentimentLabel.textContent = state.sentiment.label;
  sentimentLabel.className = "sentiment-label " + state.sentiment.label.toLowerCase();
}

// ----- Explanation -----

function renderExplanation() {
  if (state.currentStep < 0) return;
  const token = state.tokens[state.currentStep];
  const type = classifyToken(token);
  stepExplanation.innerHTML = generateExplanation(token, type);
}

// ============================================================
// LSTM SVG diagram (Olah-style, rolled-out time steps)
// ============================================================

const SVG_NS = "http://www.w3.org/2000/svg";

const CELL_W = 320;          // per-cell width in viewBox units
const CELL_H = 200;          // per-cell visible body height
const VIEW_W = 460;          // viewport shows active cell + ~70px sliver of each neighbor
const VIEW_H = 340;          // total SVG height (room for x_t below + h_t above)

// Cell-internal coordinates (relative to cell origin = top-left of cell-body rect)
// Top cell-state wire at y=50, bottom h_{t-1} input wire at y=150
const Y_TOP = 50, Y_MID = 150;

let cellGroups = [];    // populated by buildDiagram

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}

function buildDiagram() {
  lstmDiagram.innerHTML = "";
  cellGroups = [];
  const N = state.tokens.length;
  if (N === 0) return;

  const svg = el("svg", {
    viewBox: `0 0 ${VIEW_W} ${VIEW_H}`,
    preserveAspectRatio: "xMidYMid meet"
  });

  // Sliding row group
  const roll = el("g", { id: "lstmRoll" }, svg);

  // Cell body sits inside the SVG with vertical offset so x_t bubble fits below
  // and h_t bubble fits above. Body top-left corner = (xOffset+10, 60).
  for (let i = 0; i < N; i++) {
    const xOff = i * CELL_W;
    cellGroups.push(buildCell(i, xOff, roll));
  }

  lstmDiagram.appendChild(svg);
  // Initial position: center cell 0 (active step)
  document.getElementById("lstmRoll").style.transform = `translateX(${(VIEW_W - CELL_W) / 2}px)`;

  // Wire tooltips on activation boxes (per active cell — easier: bind to all and
  // the .faded ones get pointer-events:none from CSS).
  cellGroups.forEach(cell => {
    cell.querySelectorAll("[data-tooltip-key]").forEach(box => {
      const key = box.dataset.tooltipKey;
      addTooltip(box, () => ({
        html: `<div class="tt-title">${gateTooltipTitle(key)}</div><div class="tt-body">${GATE_TOOLTIPS[key]}</div>`
      }));
    });
  });
}

function gateTooltipTitle(k) {
  return ({ forget: "Forget gate (σ)", input: "Input gate (σ)", candidate: "Candidate memory (tanh)", output: "Output gate (σ)" })[k] || k;
}

function buildCell(index, xOffset, parent) {
  const token = state.tokens[index];
  const g = el("g", {
    "class": "cell-group",
    transform: `translate(${xOffset}, 0)`
  }, parent);

  // Cell body coords (relative to this group's origin)
  const BX = 10, BY = 60, BW = CELL_W - 20, BH = CELL_H;
  const TOP_Y = BY + 30;     // top cell-state wire (c_{t-1} → c_t)
  const ACT_Y = BY + 110;    // activation box top
  const ACT_H = 22;
  const BUS_Y = ACT_Y + ACT_H + 8;  // h_{t-1} + x_t bus, just below activation row

  // Activation box centers — well-spaced so labels don't collide
  const X_F = BX + 40;       // forget σ
  const X_I = BX + 90;       // input σ
  const X_C = BX + 150;      // candidate tanh (wider)
  const X_TANH = BX + 195;   // tanh ellipse + cell-state branch point (own column)
  const X_O = BX + 235;      // output σ AND output × column

  // ---- Inter-cell wire stubs (drawn outside cell body) ----
  el("path", { "class": "wire wire-cell", d: `M 0 ${TOP_Y} H ${BX}` }, g);
  el("path", { "class": "wire wire-cell", d: `M ${BX + BW} ${TOP_Y} H ${CELL_W}` }, g);
  el("path", { "class": "wire", d: `M 0 ${BUS_Y} H ${BX}` }, g);

  // ---- Cell body ----
  el("rect", { "class": "cell-body", x: BX, y: BY, width: BW, height: BH }, g);

  // ---- Faded "A" letter (shown when cell is inactive) ----
  el("text", { "class": "cell-letter", x: BX + BW / 2, y: BY + BH / 2 }, g).textContent = "A";

  // ---- Internals group (fades when cell is inactive) ----
  const ints = el("g", { "class": "cell-internals" }, g);

  // h_{t-1}/x_t bus across the bottom of the cell, reaching all gates
  el("path", { "class": "wire", d: `M ${BX} ${BUS_Y} H ${X_O}` }, ints);

  // Vertical risers from bus up into each activation box bottom
  [X_F, X_I, X_C, X_O].forEach(x => {
    el("path", { "class": "wire", d: `M ${x} ${BUS_Y} V ${ACT_Y + ACT_H}` }, ints);
  });

  // ---- Activation boxes ----
  const boxes = [
    { x: X_F, key: "forget",    glyph: "σ",    w: 28 },
    { x: X_I, key: "input",     glyph: "σ",    w: 28 },
    { x: X_C, key: "candidate", glyph: "tanh", w: 36 },
    { x: X_O, key: "output",    glyph: "σ",    w: 28 }
  ];
  boxes.forEach(b => {
    el("rect", {
      "class": "activation-box",
      x: b.x - b.w / 2, y: ACT_Y, width: b.w, height: ACT_H,
      "data-tooltip-key": b.key
    }, ints);
    el("text", {
      "class": "activation-text",
      x: b.x, y: ACT_Y + ACT_H / 2 + 1
    }, ints).textContent = b.glyph;
    // Numeric value label: above the box for forget/input/candidate;
    // to the LEFT of the box for output (because the × op sits above output σ)
    const isOutput = b.key === "output";
    el("text", {
      "class": "gate-num",
      x: isOutput ? b.x - b.w / 2 - 4 : b.x,
      y: isOutput ? ACT_Y + ACT_H / 2 + 1 : ACT_Y - 5,
      "text-anchor": isOutput ? "end" : "middle",
      "data-val": b.key
    }, ints).textContent = "0.00";
  });

  // ---- Forget × on top wire ----
  drawOp(ints, X_F, TOP_Y, "×", "forget-times");
  // Top cell-state wire entering the forget × from the left (c_{t-1} pathway)
  el("path", { "class": "wire wire-cell", d: `M ${BX} ${TOP_Y} H ${X_F - 10}` }, ints);
  el("path", { "class": "wire", d: `M ${X_F} ${ACT_Y} V ${TOP_Y + 10}` }, ints);

  // ---- + on top wire (right of forget ×) ----
  const X_PLUS = BX + 120;
  drawOp(ints, X_PLUS, TOP_Y, "+", "plus-node");
  el("path", { "class": "wire wire-cell", d: `M ${X_F + 10} ${TOP_Y} H ${X_PLUS - 10}` }, ints);

  // ---- Input × below + (combines input σ and candidate tanh outputs) ----
  const Y_IX = TOP_Y + 35;
  drawOp(ints, X_PLUS, Y_IX, "×", "input-times");
  el("path", { "class": "wire", d: `M ${X_I} ${ACT_Y} V ${Y_IX} H ${X_PLUS - 10}` }, ints);
  el("path", { "class": "wire", d: `M ${X_C} ${ACT_Y} V ${Y_IX} H ${X_PLUS + 10}` }, ints);
  el("path", { "class": "wire", d: `M ${X_PLUS} ${Y_IX - 10} V ${TOP_Y + 10}` }, ints);

  // ---- After +, top cell-state wire continues right ----
  el("path", { "class": "wire wire-cell", d: `M ${X_PLUS + 10} ${TOP_Y} H ${BX + BW}` }, ints);

  // ---- Cell-state branch DOWN at X_TANH, through tanh, then RIGHT to output × ----
  // Horizontal flow: tanh ellipse and output × are on the SAME y but DIFFERENT x columns.
  // This way the h_t up-wire (at X_O column further right) does NOT cross the tanh ellipse.
  const Y_TANH = TOP_Y + 30;
  const Y_OX   = Y_TANH;     // same row as tanh — they're in horizontal flow
  const ellipse = el("ellipse", { "class": "tanh-ellipse", cx: X_TANH, cy: Y_TANH, rx: 16, ry: 10 }, ints);
  ellipse.setAttribute("data-flash-id", "tanh-ellipse");
  el("text", { "class": "tanh-ellipse-text", x: X_TANH, y: Y_TANH + 1 }, ints).textContent = "tanh";
  // Top-wire branch DOWN to tanh top
  el("path", { "class": "wire wire-cell", d: `M ${X_TANH} ${TOP_Y} V ${Y_TANH - 10}` }, ints);

  // ---- Output × in its own column to the RIGHT of tanh ----
  drawOp(ints, X_O, Y_OX, "×", "output-times");
  // Horizontal wire from tanh ellipse RIGHT to output × left edge
  el("path", { "class": "wire wire-cell", d: `M ${X_TANH + 16} ${Y_TANH} H ${X_O - 10}` }, ints);
  // Vertical wire from output σ top UP to output × bottom
  el("path", { "class": "wire", d: `M ${X_O} ${ACT_Y} V ${Y_OX + 10}` }, ints);

  // ---- h_t output: T-junction to the RIGHT of output × so the up-wire avoids the tanh column ----
  const X_HT = X_O + 18;
  // Short stub from × right edge to T-junction
  el("path", { "class": "wire", d: `M ${X_O + 10} ${Y_OX} H ${X_HT}` }, ints);
  // Up to h_t bubble
  el("path", { "class": "wire", d: `M ${X_HT} ${Y_OX} V ${BY - 8}` }, ints);
  // Down then right to next cell's hidden bus
  el("path", { "class": "wire", d: `M ${X_HT} ${Y_OX} V ${BUS_Y} H ${CELL_W}` }, ints);

  // ---- x_t bubble (blue, below cell) — shows the actual token ----
  const xtCY = BY + BH + 32;
  const xtR = 24;
  // Font size scales with word length so it fits inside the bubble
  const tokenFs = token.length <= 3 ? 14
                : token.length <= 5 ? 12
                : token.length <= 7 ? 10
                : token.length <= 9 ? 8 : 7;
  el("circle", { "class": "io-bubble x", cx: BX + 22, cy: xtCY, r: xtR }, g);
  el("text", {
    "class": "io-text x",
    x: BX + 22, y: xtCY + 1,
    "font-size": tokenFs
  }, g).textContent = token;
  el("path", { "class": "wire", d: `M ${BX + 22} ${xtCY - xtR} V ${BUS_Y}` }, g);
  // Small x_i subscript label below the bubble
  el("text", {
    "class": "io-text x",
    x: BX + 22, y: xtCY + xtR + 12,
    "font-size": "10",
    "font-weight": "500"
  }, g).textContent = `x${sub(index)}`;

  // ---- h_t bubble (purple, above cell, on the T-junction column) ----
  const htCY = BY - 35;
  const htBubble = el("circle", { "class": "io-bubble h", cx: X_HT, cy: htCY, r: 18 }, g);
  htBubble.setAttribute("data-flash-id", "ht-bubble");
  el("text", { "class": "io-text h", x: X_HT, y: htCY + 1 }, g).textContent = `h${sub(index)}`;

  // ---- Animation guide paths (invisible — used by pulseAlongPath) ----
  const animPaths = {
    "x-riser":         `M ${BX + 22} ${xtCY - 24} V ${BUS_Y}`,
    "h-bus-in":        `M 0 ${BUS_Y} H ${BX + 22}`,
    "bus-spine":       `M ${BX + 22} ${BUS_Y} H ${X_O}`,
    "riser-forget":    `M ${X_F} ${BUS_Y} V ${ACT_Y + ACT_H}`,
    "riser-input":     `M ${X_I} ${BUS_Y} V ${ACT_Y + ACT_H}`,
    "riser-candidate": `M ${X_C} ${BUS_Y} V ${ACT_Y + ACT_H}`,
    "riser-output":    `M ${X_O} ${BUS_Y} V ${ACT_Y + ACT_H}`,
    "forget-up":       `M ${X_F} ${ACT_Y} V ${TOP_Y + 10}`,
    "input-up":        `M ${X_I} ${ACT_Y} V ${Y_IX} H ${X_PLUS - 10}`,
    "candidate-up":    `M ${X_C} ${ACT_Y} V ${Y_IX} H ${X_PLUS + 10}`,
    "output-up":       `M ${X_O} ${ACT_Y} V ${Y_OX + 10}`,
    "input-to-plus":   `M ${X_PLUS} ${Y_IX - 10} V ${TOP_Y + 10}`,
    "cellstate-out":   `M ${X_PLUS + 10} ${TOP_Y} H ${BX + BW}`,
    "cellstate-down":  `M ${X_TANH} ${TOP_Y} V ${Y_TANH - 10}`,
    "tanh-to-ox":      `M ${X_TANH + 16} ${Y_TANH} H ${X_O - 10}`,
    "ht-up":           `M ${X_O + 10} ${Y_OX} H ${X_HT} V ${BY - 8}`,
    "ht-right":        `M ${X_O + 10} ${Y_OX} H ${X_HT} V ${BUS_Y} H ${CELL_W}`
  };
  const guides = el("g", { "class": "anim-guides" }, g);
  for (const [id, d] of Object.entries(animPaths)) {
    el("path", { "data-path-id": id, d, fill: "none", stroke: "none" }, guides);
  }

  return g;
}

function sub(i) {
  // Returns subscript index, e.g., 0,1,2,... rendered as ₀, ₁, ₂
  const map = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"];
  return String(i).split("").map(d => map[+d] || d).join("");
}

function drawOp(parent, cx, cy, glyph, flashId) {
  const c = el("circle", { "class": "op-circle", cx, cy, r: 10 }, parent);
  if (flashId) c.setAttribute("data-flash-id", flashId);
  el("text", { "class": "op-text", x: cx, y: cy + 1 }, parent).textContent = glyph;
  return c;
}

function renderLSTMDiagram() {
  if (cellGroups.length === 0) return;
  const idx = Math.max(0, state.currentStep);
  const t = (VIEW_W - CELL_W) / 2 - idx * CELL_W;
  const roll = document.getElementById("lstmRoll");
  if (roll) roll.style.transform = `translateX(${t}px)`;

  cellGroups.forEach((cg, i) => {
    cg.classList.toggle("active", i === idx);
    cg.classList.toggle("faded",  i !== idx);
  });

  // Update numeric gate values inside the active cell
  const active = cellGroups[idx];
  if (!active) return;
  ["forget","input","candidate","output"].forEach(key => {
    const lbl = active.querySelector(`[data-val="${key}"]`);
    if (lbl) lbl.textContent = state.gates[key].toFixed(2);
  });
}

// ============================================================
// Tooltip system
// ============================================================

function showTooltip(anchor, html) {
  tooltipEl.innerHTML = html;
  tooltipEl.style.top = "-9999px";
  tooltipEl.style.left = "-9999px";
  tooltipEl.classList.add("visible");

  const rect = anchor.getBoundingClientRect();
  const tw = tooltipEl.offsetWidth;
  const th = tooltipEl.offsetHeight;
  const margin = 10;

  let top = rect.bottom + margin;
  if (top + th > window.innerHeight - margin) top = rect.top - th - margin;

  let left = rect.left;
  if (left + tw > window.innerWidth - margin) left = window.innerWidth - tw - margin;
  if (left < margin) left = margin;

  tooltipEl.style.top  = top  + "px";
  tooltipEl.style.left = left + "px";
}

function hideTooltip() { tooltipEl.classList.remove("visible"); }

function addTooltip(el, getContent) {
  if (!el) return;
  el.addEventListener("mouseenter", () => {
    const { html } = getContent();
    showTooltip(el, html);
  });
  el.addEventListener("mouseleave", hideTooltip);
}

// ============================================================
// Math equations (KaTeX)
// ============================================================

const EQUATIONS = [
  { tex: "f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)",  note: "forget gate" },
  { tex: "i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)",  note: "input gate" },
  { tex: "g_t = \\tanh(W_g \\cdot [h_{t-1}, x_t] + b_g)",   note: "candidate memory" },
  { tex: "o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)",  note: "output gate" },
  { tex: "c_t = f_t \\odot c_{t-1} + i_t \\odot g_t",       note: "cell state update" },
  { tex: "h_t = o_t \\odot \\tanh(c_t)",                    note: "hidden state output" }
];

function renderMath() {
  mathEquations.innerHTML = EQUATIONS.map(e => {
    const rendered = katex.renderToString(e.tex, { throwOnError: false, displayMode: true });
    return `<div>${rendered} <span style="color:var(--muted);font-size:0.78rem;">— ${e.note}</span></div>`;
  }).join("");
}

// ============================================================
// Wiring
// ============================================================

let activePresetIndex = -1; // -1 means custom sentence is loaded

function renderPresets() {
  presetGrid.innerHTML = PRESETS.map((p, i) => {
    const cls = [
      "preset-card",
      p.recommended ? "recommended" : "",
      i === activePresetIndex ? "active" : ""
    ].filter(Boolean).join(" ");
    const tag = p.recommended ? `<span class="preset-tag">★ Start here</span>` : "";
    return `
      <button class="${cls}" data-preset="${i}" type="button">
        <div class="preset-header">
          <span class="preset-label">${p.label}</span>
          ${tag}
        </div>
        <div class="preset-text">&ldquo;${p.text}&rdquo;</div>
        <div class="preset-teaches">${p.teaches}</div>
      </button>`;
  }).join("");

  presetGrid.querySelectorAll(".preset-card").forEach(card => {
    card.addEventListener("click", () => {
      const i = parseInt(card.dataset.preset);
      activePresetIndex = i;
      loadSentence(PRESETS[i].text);
      renderPresets(); // re-render so active highlight follows
    });
  });
}

function init() {
  activePresetIndex = DEFAULT_PRESET;
  renderPresets();
  renderMath();
  loadSentence(PRESETS[DEFAULT_PRESET].text);
}

loadBtn.addEventListener("click", () => {
  const v = customInput.value.trim();
  if (v) {
    activePresetIndex = -1;
    loadSentence(v);
    renderPresets();
  }
});

customInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadBtn.click();
});

nextBtn.addEventListener("click", stepForward);
phaseBtn.addEventListener("click", stepPhase);
prevBtn.addEventListener("click", stepBackward);
resetBtn.addEventListener("click", resetState);

if (setupExpandBtn) setupExpandBtn.addEventListener("click", expandSetup);

autoBtn.addEventListener("click", () => {
  if (autoTimer) stopAuto();
  else startAuto();
});

speedSlider.addEventListener("input", (e) => {
  speedMs = parseInt(e.target.value);
  const sf = Math.max(0.33, speedMs / 900);
  // approximate step animation total ≈ 3.4s × scale + 0.7s slide
  const total = (3400 * sf + 700) / 1000;
  speedDisplay.textContent = `${total.toFixed(1)}s/step`;
});

init();
