# LSTM Sentiment Demo Implementation Plan

## Project Title

**Watch an LSTM Remember: Step-by-Step Sentiment Classification**

## One-Sentence Pitch

This demo lets users step through a sentence token by token and see how an LSTM updates its memory using gates, then uses the final hidden state to classify sentiment.

## Main Learning Goal

The demo should teach one core idea:

> An LSTM improves on a vanilla RNN by maintaining a cell state and using gates to decide what to forget, what to store, and what to output.

The sentiment task gives the memory updates a concrete purpose. Instead of only showing abstract vectors, the demo shows how memory helps interpret phrases such as:

```text
"The movie was not bad."
```

---

# 1. MVP Feature Set

Build this first before adding polish.

## Required User Flow

1. User selects a preset sentence or enters a custom sentence.
2. Demo tokenizes the sentence into words.
3. User clicks **Next Step**.
4. The current token is highlighted.
5. The LSTM cell diagram updates.
6. Gate values update.
7. Cell state memory summary updates.
8. Hidden state updates.
9. Sentiment prediction updates.
10. At the final token, the demo shows the final classification.

## Required Controls

- **Preset sentence dropdown**
- **Custom sentence input**
- **Start / Reset button**
- **Previous Step button**
- **Next Step button**
- **Auto-play button**
- **Speed slider** for auto-play

## Required Visual Panels

The page should have these panels:

1. **Sentence Timeline**
2. **LSTM Cell Visualization**
3. **Gate Values**
4. **Memory State**
5. **Sentiment Prediction**
6. **Explanation Box**

---

# 2. Recommended Page Layout

Use a clean two-column layout.

```text
┌────────────────────────────────────────────────────────────┐
│ Title + short description                                  │
├────────────────────────────────────────────────────────────┤
│ Sentence input + preset selector                           │
├────────────────────────────────────────────────────────────┤
│ Token timeline                                             │
├───────────────────────────────┬────────────────────────────┤
│ LSTM cell diagram             │ Gate values + memory       │
│                               │ Sentiment prediction       │
├───────────────────────────────┴────────────────────────────┤
│ Step explanation box                                        │
├────────────────────────────────────────────────────────────┤
│ Controls: Previous / Next / Auto-play / Reset               │
└────────────────────────────────────────────────────────────┘
```

## Suggested Visual Hierarchy

- Top: What the user is doing.
- Middle: The LSTM cell and current computation.
- Right side: State values and prediction.
- Bottom: Plain-English explanation for the current step.

---

# 3. Suggested Preset Sentences

Use preset examples so the demo works reliably.

## Preset 1: Simple Positive

```text
The movie was wonderful.
```

Purpose:

- Easy positive sentiment.
- Shows simple accumulation of positive evidence.

## Preset 2: Simple Negative

```text
The movie was terrible.
```

Purpose:

- Easy negative sentiment.
- Shows simple accumulation of negative evidence.

## Preset 3: Negation

```text
The movie was not bad.
```

Purpose:

- Shows why memory matters.
- The model needs to remember "not" when reading "bad."

## Preset 4: Long-Range Contrast

```text
Although the beginning was slow, the ending was amazing.
```

Purpose:

- Shows the LSTM preserving useful context across a longer sentence.
- Demonstrates that recent positive evidence can dominate final sentiment.

## Preset 5: Mixed Sentiment

```text
The acting was great, but the story was boring.
```

Purpose:

- Shows that later negative evidence can shift the final prediction.

---

# 4. Core Conceptual Model

This does **not** need to be a real trained LSTM.

Use a simplified educational simulation. The goal is correctness of intuition, not ML training.

## Real LSTM Equations

You can show these in a collapsible "Math View" section:

```text
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)
g_t = tanh(W_g [h_{t-1}, x_t] + b_g)
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

Where:

- `f_t` is the forget gate.
- `i_t` is the input gate.
- `g_t` is the candidate memory.
- `o_t` is the output gate.
- `c_t` is the cell state.
- `h_t` is the hidden state.

## Simplified Demo Interpretation

In the demo, represent memory with a few interpretable dimensions:

```javascript
memory = {
  positive: 0,
  negative: 0,
  negation: 0,
  contrast: 0
};
```

This allows the user to understand what the LSTM is remembering.

---

# 5. Suggested State Representation

Use this JavaScript state object:

```javascript
const state = {
  tokens: [],
  currentStep: -1,

  cellState: {
    positive: 0,
    negative: 0,
    negation: 0,
    contrast: 0
  },

  hiddenState: {
    positiveSignal: 0,
    negativeSignal: 0
  },

  gates: {
    forget: 0,
    input: 0,
    candidate: 0,
    output: 0
  },

  sentiment: {
    positiveProb: 0.5,
    negativeProb: 0.5,
    label: "Neutral / uncertain"
  },

  history: []
};
```

Store `history` so the **Previous Step** button can restore earlier states.

---

# 6. Token Scoring Heuristics

Define word groups.

```javascript
const positiveWords = new Set([
  "good", "great", "wonderful", "amazing", "excellent",
  "fun", "beautiful", "love", "loved", "best", "enjoyed"
]);

const negativeWords = new Set([
  "bad", "terrible", "awful", "boring", "slow",
  "worst", "hate", "hated", "poor", "confusing"
]);

const negationWords = new Set([
  "not", "never", "no", "hardly", "barely", "isn't",
  "wasn't", "don't", "didn't"
]);

const contrastWords = new Set([
  "but", "however", "although", "though", "yet"
]);
```

## Token Effect Rules

### Positive word

If no active negation:

```text
increase positive memory
```

If active negation:

```text
increase negative memory less strongly or flip interpretation
```

Example:

```text
"not good" -> negative
```

### Negative word

If no active negation:

```text
increase negative memory
```

If active negation:

```text
increase positive memory
```

Example:

```text
"not bad" -> positive
```

### Negation word

```text
store negation memory
```

The negation memory should decay slowly over the next few tokens.

### Contrast word

```text
increase contrast memory
partially forget earlier sentiment
make future sentiment more influential
```

Example:

```text
"The acting was great, but the story was boring."
```

After "but", later negative evidence should matter more.

---

# 7. Gate Heuristic Design

The gate values should be visually meaningful.

## Forget Gate

Meaning:

> How much previous memory should be kept?

Recommended behavior:

- Normal word: high forget gate, around `0.85`
- Contrast word: lower forget gate, around `0.45`
- Strong sentiment word after contrast: medium, around `0.65`

Example:

```javascript
function computeForgetGate(tokenType) {
  if (tokenType === "contrast") return 0.45;
  return 0.85;
}
```

## Input Gate

Meaning:

> How much new information should be written into memory?

Recommended behavior:

- Sentiment word: high, around `0.85`
- Negation word: high, around `0.9`
- Neutral word: low, around `0.2`
- Contrast word: medium, around `0.6`

```javascript
function computeInputGate(tokenType) {
  if (tokenType === "positive" || tokenType === "negative") return 0.85;
  if (tokenType === "negation") return 0.9;
  if (tokenType === "contrast") return 0.6;
  return 0.2;
}
```

## Candidate Memory

Meaning:

> What new content could be added?

Use a signed value:

```text
positive word -> +1
negative word -> -1
negation word -> special negation memory
neutral word -> 0
```

## Output Gate

Meaning:

> How much memory should be revealed as the hidden state?

Recommended behavior:

- Sentiment word: high, around `0.85`
- Neutral word: medium, around `0.5`
- Negation word: lower, around `0.35`

Reason:

- After "not", the model stores negation, but may not make a confident sentiment prediction yet.
- After "bad", the model combines "not" and "bad" and reveals stronger sentiment.

---

# 8. Step Update Algorithm

For each token:

1. Normalize token.
2. Classify token type.
3. Compute gate values.
4. Apply forget gate to previous memory.
5. Add candidate memory through input gate.
6. Compute hidden state through output gate.
7. Compute sentiment probabilities.
8. Save state to history.
9. Render all panels.

## Pseudocode

```javascript
function stepForward() {
  saveHistory();

  state.currentStep += 1;
  const token = state.tokens[state.currentStep];
  const tokenType = classifyToken(token);

  const forget = computeForgetGate(tokenType);
  const input = computeInputGate(tokenType);
  const candidate = computeCandidate(token, tokenType);
  const output = computeOutputGate(tokenType);

  state.gates = { forget, input, candidate, output };

  applyForgetGate(forget);
  applyInputGate(input, candidate, tokenType);
  decayNegation();
  computeHiddenState(output);
  computeSentiment();

  render();
}
```

---

# 9. Memory Update Details

## Apply Forget Gate

```javascript
function applyForgetGate(forget) {
  state.cellState.positive *= forget;
  state.cellState.negative *= forget;
  state.cellState.negation *= forget;
  state.cellState.contrast *= forget;
}
```

## Apply Input Gate

```javascript
function applyInputGate(input, candidate, tokenType) {
  const negationActive = state.cellState.negation > 0.4;
  const contrastBoost = 1 + state.cellState.contrast * 0.5;

  if (tokenType === "positive") {
    if (negationActive) {
      state.cellState.negative += input * 0.8 * contrastBoost;
    } else {
      state.cellState.positive += input * 1.0 * contrastBoost;
    }
  }

  if (tokenType === "negative") {
    if (negationActive) {
      state.cellState.positive += input * 0.8 * contrastBoost;
    } else {
      state.cellState.negative += input * 1.0 * contrastBoost;
    }
  }

  if (tokenType === "negation") {
    state.cellState.negation += input * 1.0;
  }

  if (tokenType === "contrast") {
    state.cellState.contrast += input * 1.0;
    state.cellState.positive *= 0.65;
    state.cellState.negative *= 0.65;
  }
}
```

## Decay Negation

Negation should not last forever.

```javascript
function decayNegation() {
  state.cellState.negation *= 0.75;
}
```

## Compute Hidden State

```javascript
function computeHiddenState(output) {
  state.hiddenState.positiveSignal = output * Math.tanh(state.cellState.positive);
  state.hiddenState.negativeSignal = output * Math.tanh(state.cellState.negative);
}
```

## Compute Sentiment

Use a simple softmax over positive and negative signal.

```javascript
function computeSentiment() {
  const pos = state.hiddenState.positiveSignal;
  const neg = state.hiddenState.negativeSignal;

  const expPos = Math.exp(pos);
  const expNeg = Math.exp(neg);

  const positiveProb = expPos / (expPos + expNeg);
  const negativeProb = expNeg / (expPos + expNeg);

  state.sentiment.positiveProb = positiveProb;
  state.sentiment.negativeProb = negativeProb;

  if (positiveProb > 0.6) {
    state.sentiment.label = "Positive";
  } else if (negativeProb > 0.6) {
    state.sentiment.label = "Negative";
  } else {
    state.sentiment.label = "Neutral / uncertain";
  }
}
```

---

# 10. Visual Design Details

## Token Timeline

Show each token as a pill.

Example:

```text
[The] [movie] [was] [not] [bad]
```

Current token should be highlighted.

Past tokens should look completed.

Future tokens should be faded.

## LSTM Cell Diagram

The diagram should show:

```text
c_{t-1} -- forget gate --┐
                         + -- c_t
candidate -- input gate -┘

c_t -- tanh -- output gate -- h_t
```

Use animated arrows if possible.

Recommended colors:

- Cell state: blue
- Forget gate: red/orange
- Input gate: green
- Candidate memory: purple
- Output gate: yellow
- Hidden state: teal

## Gate Bars

Each gate should have:

- Name
- Value from 0 to 1
- Progress bar
- One-sentence explanation

Example:

```text
Forget gate: 0.85
Mostly keeps previous memory.
```

## Memory State Panel

Show memory dimensions as bars:

```text
Positive memory: ███████░░░ 0.72
Negative memory: ██░░░░░░░░ 0.21
Negation memory: ████░░░░░░ 0.43
Contrast memory: ██░░░░░░░░ 0.19
```

Also include a natural-language summary:

```text
The LSTM is currently remembering that a negation was recently seen.
```

## Sentiment Panel

Show:

- Positive probability bar
- Negative probability bar
- Final/current label

Example:

```text
Current prediction: Positive
Positive: 78%
Negative: 22%
```

## Explanation Box

This is very important for user-friendliness.

At each step, generate a short explanation.

Example for "not":

```text
The word "not" is a negation cue. The LSTM writes this into its cell state, but it does not yet make a strong sentiment prediction because it needs to see what word is being negated.
```

Example for "bad":

```text
The word "bad" is usually negative, but the LSTM still remembers the earlier word "not", so it flips the interpretation toward positive sentiment.
```

---

# 11. Suggested File Structure

Submit as a zip file.

```text
[netID]_lstm_demo.zip
├── index.html
├── style.css
├── script.js
├── assets/
│   └── optional-diagram.png
├── [netID]_demo_video.mp4
└── [netID]_demo_rationale.pdf
```

The rationale PDF is optional but highly recommended.

---

# 12. HTML Skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Watch an LSTM Remember</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <main class="app">
    <header class="hero">
      <h1>Watch an LSTM Remember</h1>
      <p>
        Step through a sentence and see how an LSTM updates memory
        to classify sentiment.
      </p>
    </header>

    <section class="setup-card">
      <label for="presetSelect">Choose a sentence:</label>
      <select id="presetSelect"></select>

      <label for="customInput">Or enter your own:</label>
      <input id="customInput" type="text" placeholder="Type a short sentence..." />

      <button id="loadSentenceBtn">Load Sentence</button>
    </section>

    <section class="timeline-card">
      <h2>Token Timeline</h2>
      <div id="tokenTimeline"></div>
    </section>

    <section class="main-grid">
      <section class="cell-card">
        <h2>LSTM Cell</h2>
        <div id="lstmDiagram"></div>
      </section>

      <aside class="side-panel">
        <section class="gates-card">
          <h2>Gates</h2>
          <div id="gateBars"></div>
        </section>

        <section class="memory-card">
          <h2>Cell State Memory</h2>
          <div id="memoryBars"></div>
          <p id="memorySummary"></p>
        </section>

        <section class="sentiment-card">
          <h2>Sentiment Prediction</h2>
          <div id="sentimentBars"></div>
          <p id="sentimentLabel"></p>
        </section>
      </aside>
    </section>

    <section class="explanation-card">
      <h2>What happened at this step?</h2>
      <p id="stepExplanation"></p>
    </section>

    <section class="controls-card">
      <button id="prevBtn">Previous Step</button>
      <button id="nextBtn">Next Step</button>
      <button id="autoBtn">Auto-play</button>
      <button id="resetBtn">Reset</button>

      <label for="speedSlider">Speed</label>
      <input id="speedSlider" type="range" min="300" max="2000" value="900" />
    </section>

    <section class="math-card">
      <details>
        <summary>Show LSTM equations</summary>
        <pre>
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)
g_t = tanh(W_g [h_{t-1}, x_t] + b_g)
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
        </pre>
      </details>
    </section>
  </main>

  <script src="script.js"></script>
</body>
</html>
```

---

# 13. CSS Design Guidelines

Use a modern card-based style.

## Recommended Look

- Light background.
- White cards.
- Rounded corners.
- Soft shadows.
- Large readable text.
- Clear buttons.
- Smooth transitions.

## Suggested CSS Variables

```css
:root {
  --bg: #f6f7fb;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --border: #e5e7eb;

  --cell: #3b82f6;
  --forget: #ef4444;
  --input: #22c55e;
  --candidate: #a855f7;
  --output: #eab308;
  --hidden: #14b8a6;

  --positive: #22c55e;
  --negative: #ef4444;
}
```

## Layout CSS

```css
.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px;
}

.main-grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr);
  gap: 24px;
}

.setup-card,
.timeline-card,
.cell-card,
.gates-card,
.memory-card,
.sentiment-card,
.explanation-card,
.controls-card,
.math-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 20px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
```

---

# 14. JavaScript Implementation Checklist

## Initialization

- Define preset sentences.
- Populate dropdown.
- Load default sentence.
- Render empty initial state.

## Tokenization

Simple tokenization is enough:

```javascript
function tokenize(sentence) {
  return sentence
    .replace(/[.,!?]/g, "")
    .split(/\s+/)
    .filter(Boolean);
}
```

## Main Functions

Implement these:

```javascript
function loadSentence(sentence) {}
function resetState() {}
function stepForward() {}
function stepBackward() {}
function classifyToken(token) {}
function computeForgetGate(tokenType) {}
function computeInputGate(tokenType) {}
function computeCandidate(token, tokenType) {}
function computeOutputGate(tokenType) {}
function applyForgetGate(forget) {}
function applyInputGate(input, candidate, tokenType) {}
function decayNegation() {}
function computeHiddenState(output) {}
function computeSentiment() {}
function generateExplanation(token, tokenType) {}
function generateMemorySummary() {}
function render() {}
```

## Render Functions

Break `render()` into smaller functions:

```javascript
function render() {
  renderTimeline();
  renderLSTMDiagram();
  renderGates();
  renderMemory();
  renderSentiment();
  renderExplanation();
  updateButtons();
}
```

---

# 15. LSTM Diagram Implementation Options

## Option A: HTML/CSS Diagram

This is easiest.

Use divs for gates and arrows. Animate with CSS classes.

Pros:

- Easy to style.
- Easy to update gate values.
- No SVG complexity.

Cons:

- Less precise layout.

## Option B: SVG Diagram

This is more polished.

Use SVG arrows and circles for operations:

- `×` for multiplication
- `+` for addition
- `tanh`
- `σ`

Pros:

- Looks like a real LSTM diagram.
- Good for presentation.

Cons:

- More implementation time.

## Recommendation

Use **SVG** if you have time. Since you want the demo polished, SVG will look much more professional.

---

# 16. Suggested SVG Diagram Elements

Include:

```text
c_{t-1} horizontal line
forget gate multiplier
input gate multiplier
candidate memory
addition node
c_t output
tanh node
output gate multiplier
h_t output
```

Use text labels:

```text
Forget gate f_t
Input gate i_t
Candidate g_t
Output gate o_t
Cell state c_t
Hidden state h_t
```

Color each component based on its role.

When a gate value changes, update:

- gate opacity
- gate bar
- arrow thickness
- numeric label

Example:

```text
Forget gate = 0.20 -> thin faded arrow
Forget gate = 0.90 -> thick bright arrow
```

This makes the concept intuitive.

---

# 17. User-Friendly Details

These are worth adding because you have time.

## Tooltips

Add hover tooltips:

- Forget gate: “Controls how much old memory is kept.”
- Input gate: “Controls how much new information is stored.”
- Output gate: “Controls how much memory is revealed as output.”
- Cell state: “Long-term memory pathway.”
- Hidden state: “Short-term output passed to the classifier.”

## Step Counter

Show:

```text
Step 4 of 5
Current token: "not"
```

## Final Summary

When the sentence is complete, show:

```text
Final prediction: Positive

Why?
The model remembered the negation "not" and used it to reinterpret "bad" as positive.
```

## Reset on Input Change

If the user changes the sentence, automatically reset the state.

## Mobile Friendliness

Make the grid collapse on small screens:

```css
@media (max-width: 900px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}
```

---

# 18. Optional Advanced Features

Add only after the core demo works.

## Feature 1: Vanilla RNN Comparison

Add a small side panel:

```text
Vanilla RNN memory: fades quickly
LSTM memory: controlled by gates
```

This reinforces why LSTMs exist.

## Feature 2: Math / Intuition Toggle

Allow users to switch between:

- **Intuition Mode**
- **Equation Mode**

Intuition Mode shows phrases like:

```text
"keep memory"
"write new info"
"reveal output"
```

Equation Mode shows:

```text
f_t, i_t, g_t, o_t, c_t, h_t
```

## Feature 3: Manual Gate Sliders

Let users manually override gates.

Controls:

- Forget gate slider
- Input gate slider
- Output gate slider

This can be very engaging, but it may distract from the main narrative. Keep it hidden under an “Advanced Controls” section.

## Feature 4: Export Demo State

Add a button to copy a short summary of the current step. This is not necessary, but useful for debugging.

---

# 19. Common Mistakes to Avoid

## Mistake 1: Too much math

The demo should teach intuition first. Put equations in a collapsible section.

## Mistake 2: Free-form input only

Free-form input may behave weirdly. Use presets as the main path.

## Mistake 3: Too many concepts

Do not also try to teach Transformers, GRUs, or attention in the main demo.

## Mistake 4: Calling it a real trained model

Be transparent in the rationale:

```text
The demo uses a simplified rule-based simulation of LSTM-like memory updates to make the internal mechanics interpretable.
```

This is acceptable because the goal is educational visualization.

## Mistake 5: Dense paragraphs

Use short explanations, labels, bars, and animations.

---

# 20. Demo Video Script

Keep the video under 2 minutes.

## Suggested Script

```text
This demo is called Watch an LSTM Remember. It teaches how an LSTM updates memory while reading a sentence for sentiment classification.

I will start with the sentence: "The movie was not bad."

The sentence is processed one token at a time. At each step, the current word is highlighted.

For neutral words like "the" and "movie", the input gate is low because there is not much sentiment information to store.

When the model sees "not", the input gate becomes high and the LSTM stores a negation cue in its cell state. The output is still uncertain because the model does not yet know what word is being negated.

When the model sees "bad", the word is usually negative. But the cell state still remembers "not", so the model interprets "not bad" as positive.

The sentiment panel updates after each step, showing how the hidden state is used for classification.

The main idea is that an LSTM uses gates to decide what to forget, what to store, and what to reveal as output. This helps it preserve important information over a sequence.
```

---

# 21. Optional Rationale PDF Draft

Use this as the basis for `[netID]_demo_rationale.pdf`.

```text
This demo visualizes how an LSTM updates memory during sentiment classification. I chose this topic because students often learn the LSTM equations without developing an intuition for what the gates actually do. The demo processes a sentence one token at a time and shows how the forget gate, input gate, candidate memory, output gate, cell state, and hidden state change at each step. I used sentiment classification because it gives the memory updates a concrete purpose: the model must remember cues such as negation and contrast to make a final prediction. For example, in "not bad," the LSTM stores the negation cue and later uses it to reinterpret the negative word. The demo uses a simplified rule-based simulation rather than a trained model so that the internal state is interpretable and suitable for teaching. I intentionally kept the interface focused, with preset sentences, step controls, gate bars, memory summaries, and a collapsible math section.
```

Word count is under 300 words.

---

# 22. Implementation Order

Follow this order.

## Phase 1: Basic Working Demo

1. Create `index.html`, `style.css`, and `script.js`.
2. Add preset sentence dropdown.
3. Implement tokenization.
4. Implement Next / Previous / Reset.
5. Render token timeline.
6. Implement state updates.
7. Render sentiment probabilities.

## Phase 2: LSTM Visualization

1. Add gate bars.
2. Add memory bars.
3. Add explanation box.
4. Add simple LSTM diagram.
5. Highlight current step.

## Phase 3: Polish

1. Improve card layout.
2. Add animations.
3. Add tooltips.
4. Add auto-play.
5. Add final summary.
6. Add collapsible equations.
7. Test all presets.

## Phase 4: Submission

1. Record demo video under 2 minutes.
2. Write rationale PDF.
3. Zip all files.
4. Confirm `index.html` opens locally.
5. Confirm filenames match instructions.

---

# 23. Minimum Acceptance Checklist

Before submitting, make sure:

- [ ] The demo opens by double-clicking `index.html`.
- [ ] It uses only HTML, CSS, and JavaScript.
- [ ] Preset sentences work.
- [ ] Custom input does not crash.
- [ ] Next / Previous / Reset work.
- [ ] Auto-play works or is removed.
- [ ] Gate values update each step.
- [ ] Memory state updates each step.
- [ ] Sentiment prediction updates each step.
- [ ] Explanations are short and clear.
- [ ] The final prediction makes sense for presets.
- [ ] The video is under 2 minutes.
- [ ] The zip includes `[netID]_demo_video.mp4`.
- [ ] The optional rationale is named `[netID]_demo_rationale.pdf`.

---

# 24. Final Recommendation

Build the demo around the preset sentence:

```text
The movie was not bad.
```

This is the clearest example because the user can see:

1. `not` gets stored in memory.
2. `bad` is interpreted using that memory.
3. The final prediction becomes positive.

Once that works, add the longer contrast examples.

A polished version of this project should feel like a classroom teaching tool, not just a code visualization.
