const INITIAL = {
  w: -4, target: 3, lr: 0.1,
  grad: 0, accumulatedGrad: 0,
  useZeroGrad: true, optimizer: "sgd",
  momentum: 0.9, velocity: 0,
  beta1: 0.9, beta2: 0.999,
  adamM: 0, adamV: 0, adamT: 0, epsilon: 1e-8,
  stepCount: 0
};

let state = { ...INITIAL };
let autoInterval = null;
let stepping = false;

const canvas = document.getElementById("lossCanvas");
const ctx = canvas.getContext("2d");

const X_MIN = -6, X_MAX = 8;
const PAD = { top: 30, right: 20, bottom: 40, left: 48 };

function lossVal(w) { return (w - state.target) ** 2; }
function gradVal(w) { return 2 * (w - state.target); }

function toCanvasX(x) {
  const w = canvas.width - PAD.left - PAD.right;
  return PAD.left + ((x - X_MIN) / (X_MAX - X_MIN)) * w;
}

function toCanvasY(y, yMax) {
  const h = canvas.height - PAD.top - PAD.bottom;
  return PAD.top + h - (y / yMax) * h;
}

function drawCanvas() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

  const yMax = lossVal(X_MIN) * 1.1;
  const plotH = H - PAD.top - PAD.bottom;
  const plotW = W - PAD.left - PAD.right;

  ctx.strokeStyle = "#21262d";
  ctx.lineWidth = 1;
  for (let gx = Math.ceil(X_MIN); gx <= Math.floor(X_MAX); gx++) {
    const cx = toCanvasX(gx);
    ctx.beginPath(); ctx.moveTo(cx, PAD.top); ctx.lineTo(cx, PAD.top + plotH);
    ctx.stroke();
  }

  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top + plotH);
  ctx.lineTo(PAD.left + plotW, PAD.top + plotH);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top);
  ctx.lineTo(PAD.left, PAD.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#8b949e";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  for (let gx = Math.ceil(X_MIN); gx <= Math.floor(X_MAX); gx += 2) {
    ctx.fillText(gx, toCanvasX(gx), PAD.top + plotH + 16);
  }
  ctx.textAlign = "right";
  const yTicks = [0, Math.round(yMax * 0.33), Math.round(yMax * 0.66), Math.round(yMax)];
  yTicks.forEach(yt => {
    const cy = toCanvasY(yt, yMax);
    if (cy >= PAD.top && cy <= PAD.top + plotH)
      ctx.fillText(yt, PAD.left - 6, cy + 4);
  });
  ctx.textAlign = "center";
  ctx.fillText("w", PAD.left + plotW + 10, PAD.top + plotH + 4);
  ctx.save();
  ctx.translate(14, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("loss", 0, 0);
  ctx.restore();

  const targetX = toCanvasX(state.target);
  ctx.strokeStyle = "#3fb95055";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(targetX, PAD.top); ctx.lineTo(targetX, PAD.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#3fb95088";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("min", targetX, PAD.top - 8);

  ctx.strokeStyle = "#58a6ff";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = X_MIN + (i / 200) * (X_MAX - X_MIN);
    const cy = toCanvasY(Math.min(lossVal(x), yMax), yMax);
    i === 0 ? ctx.moveTo(toCanvasX(x), cy) : ctx.lineTo(toCanvasX(x), cy);
  }
  ctx.stroke();

  const wClamped = Math.max(X_MIN, Math.min(X_MAX, state.w));
  const lossAtW = Math.min(lossVal(state.w), yMax);
  const dotX = toCanvasX(wClamped);
  const dotY = toCanvasY(lossAtW, yMax);
  const diverged = Math.abs(state.w) > 12;
  const dotColor = diverged ? "#f85149" : "#3fb950";

  if (state.grad !== 0 && !diverged) {
    const arrowLen = 28;
    const dir = state.grad > 0 ? -1 : 1;
    ctx.strokeStyle = "#d29922";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(dotX, dotY); ctx.lineTo(dotX + dir * arrowLen, dotY);
    ctx.stroke();
    ctx.fillStyle = "#d29922";
    ctx.beginPath();
    ctx.moveTo(dotX + dir * arrowLen, dotY);
    ctx.lineTo(dotX + dir * (arrowLen - 7), dotY - 4);
    ctx.lineTo(dotX + dir * (arrowLen - 7), dotY + 4);
    ctx.closePath(); ctx.fill();
  }

  ctx.beginPath(); ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
  ctx.fillStyle = dotColor + "33"; ctx.fill();
  ctx.beginPath(); ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
  ctx.fillStyle = dotColor; ctx.fill();

  ctx.font = "bold 12px system-ui";
  ctx.textAlign = "left";
  const labelX = Math.min(dotX + 10, W - 90);
  const labelY = Math.max(dotY - 10, PAD.top + 14);
  ctx.fillStyle = diverged ? "#f85149" : "#e6edf3";
  ctx.fillText(`w = ${state.w.toFixed(3)}`, labelX, labelY);
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px system-ui";
  ctx.fillText(`loss = ${lossVal(state.w).toFixed(3)}`, labelX, labelY + 14);
}

function updateStatePanel() {
  document.getElementById("val-w").textContent = state.w.toFixed(4);
  document.getElementById("val-loss").textContent = lossVal(state.w).toFixed(4);
  document.getElementById("val-grad").textContent = state.grad.toFixed(4);
  document.getElementById("val-accum").textContent = state.accumulatedGrad.toFixed(4);
  document.getElementById("val-lr").textContent = state.lr.toFixed(3);
  document.getElementById("val-steps").textContent = state.stepCount;
  if (state.optimizer === "momentum") {
    document.getElementById("val-velocity").textContent = state.velocity.toFixed(4);
  }
  if (state.optimizer === "adam") {
    const mHat = state.adamT > 0 ? state.adamM / (1 - state.beta1 ** state.adamT) : 0;
    const vHat = state.adamT > 0 ? state.adamV / (1 - state.beta2 ** state.adamT) : 0;
    document.getElementById("val-adam-m").textContent = mHat.toFixed(4);
    document.getElementById("val-adam-v").textContent = vHat.toFixed(6);
  }
}

function highlightLine(n) {
  for (let i = 0; i <= 5; i++)
    document.getElementById(`line-${i}`).classList.remove("active-line");
  if (n !== null)
    document.getElementById(`line-${n}`).classList.add("active-line");
}

function setHint(text) {
  document.getElementById("line-explanation").textContent = text || "";
}

function render() {
  drawCanvas();
  updateStatePanel();
}

// — Explainer content —

const ZEROG_CONTENT = {
  on: `<div class="explainer-title">✓ zero_grad() is ON</div>
       <div class="explainer-body">Before each <code>backward()</code>, PyTorch zeroes out the <code>.grad</code> attribute on every parameter. Each update uses only the <em>current</em> step's gradient — stable and correctly scaled.</div>`,
  off: `<div class="explainer-title">⚠ zero_grad() is OFF</div>
        <div class="explainer-body">PyTorch <em>adds</em> new gradients on top of existing <code>.grad</code> values — it never clears them unless you do. After N steps, the accumulated gradient is roughly N× larger than intended. Updates grow explosively. This is the most common training bug in PyTorch.</div>`
};

const OPT_CONTENT = {
  sgd: `<div class="explainer-title">SGD — Stochastic Gradient Descent</div>
        <div class="explainer-formula">w = w − lr × grad</div>
        <div class="explainer-body">Takes a direct step opposite the gradient. No memory of past steps — each update depends only on the current gradient. Simple and predictable, but sensitive to learning rate choice.</div>`,
  momentum: `<div class="explainer-title">SGD + Momentum</div>
             <div class="explainer-formula">v = β·v + grad &nbsp;&nbsp; w = w − lr·v</div>
             <div class="explainer-body">Maintains a velocity <em>v</em> that accumulates past gradients. With β=0.9, each step carries 90% of prior velocity — like a ball rolling downhill. Passes flat regions faster and dampens oscillation from large LR.</div>`,
  adam: `<div class="explainer-title">Adam — Adaptive Moment Estimation</div>
         <div class="explainer-formula">m̂ = m/(1−β₁ᵗ) &nbsp; v̂ = v/(1−β₂ᵗ) &nbsp; w = w − lr·m̂/(√v̂+ε)</div>
         <div class="explainer-body">Tracks gradient mean <em>m</em> and variance <em>v</em>. Divides by √v̂ so large or noisy gradients get automatically smaller steps — self-normalizing. Bias correction (1−βᵗ) prevents under-estimation at early steps. Usually needs less LR tuning than SGD.</div>`
};

function updateZerogExplainer() {
  const el = document.getElementById("zerog-explainer");
  el.innerHTML = state.useZeroGrad ? ZEROG_CONTENT.on : ZEROG_CONTENT.off;
  el.className = `explainer ${state.useZeroGrad ? "explainer-ok" : "explainer-warn"}`;
}

function updateOptExplainer() {
  document.getElementById("opt-explainer").innerHTML = OPT_CONTENT[state.optimizer];
}

function updateLrHint(lr) {
  let text;
  if (lr < 0.05)      text = "Very small — converges very slowly, many steps needed.";
  else if (lr < 0.2)  text = "Good — balanced speed and stability.";
  else if (lr < 0.5)  text = "Moderate — may oscillate slightly near the minimum.";
  else if (lr < 0.8)  text = "Large — likely to overshoot and bounce around.";
  else                text = "⚠ Very large — expect divergence, especially with zero_grad OFF.";
  document.getElementById("lr-hint").textContent = text;
}

// — Training step —

function trainingStep() {
  if (stepping) return;
  stepping = true;

  // Phase 1: zero_grad
  const oldAccum = state.accumulatedGrad;
  highlightLine(1);
  if (state.useZeroGrad) {
    state.accumulatedGrad = 0;
    setHint(`zero_grad(): cleared accumulated gradient (${oldAccum.toFixed(3)} → 0)`);
  } else {
    setHint(`zero_grad() skipped — accumulated gradient stays at ${oldAccum.toFixed(3)}`);
  }
  render();

  setTimeout(() => {
    // Phase 2: backward
    highlightLine(4);
    state.grad = gradVal(state.w);
    const prevAccum = state.accumulatedGrad;
    state.accumulatedGrad += state.grad;
    setHint(
      `backward(): grad = 2×(${state.w.toFixed(2)}−${state.target}) = ${state.grad.toFixed(3)}` +
      (state.useZeroGrad
        ? `, accumulated = ${state.accumulatedGrad.toFixed(3)}`
        : `, accumulated: ${prevAccum.toFixed(3)} + ${state.grad.toFixed(3)} = ${state.accumulatedGrad.toFixed(3)}`)
    );
    render();

    setTimeout(() => {
      // Phase 3: step
      highlightLine(5);
      const wBefore = state.w;

      if (state.optimizer === "sgd") {
        state.w -= state.lr * state.accumulatedGrad;
      } else if (state.optimizer === "momentum") {
        state.velocity = state.momentum * state.velocity + state.accumulatedGrad;
        state.w -= state.lr * state.velocity;
      } else if (state.optimizer === "adam") {
        state.adamT += 1;
        state.adamM = state.beta1 * state.adamM + (1 - state.beta1) * state.accumulatedGrad;
        state.adamV = state.beta2 * state.adamV + (1 - state.beta2) * state.accumulatedGrad ** 2;
        const mHat = state.adamM / (1 - state.beta1 ** state.adamT);
        const vHat = state.adamV / (1 - state.beta2 ** state.adamT);
        state.w -= state.lr * mHat / (Math.sqrt(vHat) + state.epsilon);
      }

      const delta = state.w - wBefore;
      setHint(`step(): w ${wBefore.toFixed(3)} → ${state.w.toFixed(3)} (Δw = ${delta > 0 ? "+" : ""}${delta.toFixed(3)})`);
      state.stepCount += 1;
      render();

      setTimeout(() => {
        highlightLine(null);
        setHint(null);
        stepping = false;
      }, 400);
    }, 450);
  }, 450);
}

function resetState() {
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    document.getElementById("btn-auto").textContent = "Auto Run";
  }
  state = { ...INITIAL, lr: state.lr, useZeroGrad: state.useZeroGrad, optimizer: state.optimizer };
  stepping = false;
  highlightLine(null);
  setHint(null);
  render();
}

function updateOptimizerRows() {
  document.getElementById("row-velocity").classList.toggle("hidden", state.optimizer !== "momentum");
  document.getElementById("row-adam-m").classList.toggle("hidden", state.optimizer !== "adam");
  document.getElementById("row-adam-v").classList.toggle("hidden", state.optimizer !== "adam");
}

// — Event listeners —

document.getElementById("btn-step").addEventListener("click", trainingStep);

document.getElementById("btn-auto").addEventListener("click", () => {
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    document.getElementById("btn-auto").textContent = "Auto Run";
  } else {
    autoInterval = setInterval(trainingStep, 1350);
    document.getElementById("btn-auto").textContent = "Pause";
  }
});

document.getElementById("btn-reset").addEventListener("click", resetState);

document.getElementById("toggle-zerog").addEventListener("click", () => {
  state.useZeroGrad = !state.useZeroGrad;
  const btn = document.getElementById("toggle-zerog");
  btn.textContent = state.useZeroGrad ? "zero_grad(): ON" : "zero_grad(): OFF";
  btn.className = state.useZeroGrad ? "toggle-on" : "toggle-off";
  updateZerogExplainer();
});

document.getElementById("slider-lr").addEventListener("input", (e) => {
  state.lr = parseFloat(e.target.value);
  document.getElementById("lr-display").textContent = state.lr.toFixed(2);
  document.getElementById("val-lr").textContent = state.lr.toFixed(3);
  updateLrHint(state.lr);
});

document.getElementById("select-optimizer").addEventListener("change", (e) => {
  state.optimizer = e.target.value;
  state.velocity = 0;
  state.adamM = 0; state.adamV = 0; state.adamT = 0;
  updateOptimizerRows();
  updateOptExplainer();
  render();
});

// — Init —
updateOptimizerRows();
updateZerogExplainer();
updateOptExplainer();
updateLrHint(state.lr);
render();
