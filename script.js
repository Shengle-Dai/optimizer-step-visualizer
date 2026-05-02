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

  // Background
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

  const yMax = lossVal(X_MIN) * 1.1;
  const plotH = H - PAD.top - PAD.bottom;
  const plotW = W - PAD.left - PAD.right;

  // Grid lines
  ctx.strokeStyle = "#21262d";
  ctx.lineWidth = 1;
  for (let gx = Math.ceil(X_MIN); gx <= Math.floor(X_MAX); gx++) {
    const cx = toCanvasX(gx);
    ctx.beginPath(); ctx.moveTo(cx, PAD.top); ctx.lineTo(cx, PAD.top + plotH);
    ctx.stroke();
  }

  // Axes
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

  // Axis labels
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  for (let gx = Math.ceil(X_MIN); gx <= Math.floor(X_MAX); gx += 2) {
    const cx = toCanvasX(gx);
    ctx.fillText(gx, cx, PAD.top + plotH + 16);
  }
  ctx.textAlign = "right";
  const yTicks = [0, Math.round(yMax * 0.33), Math.round(yMax * 0.66), Math.round(yMax)];
  yTicks.forEach(yt => {
    const cy = toCanvasY(yt, yMax);
    if (cy >= PAD.top && cy <= PAD.top + plotH)
      ctx.fillText(yt, PAD.left - 6, cy + 4);
  });

  ctx.textAlign = "center";
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px system-ui";
  ctx.fillText("w", PAD.left + plotW + 10, PAD.top + plotH + 4);
  ctx.save();
  ctx.translate(14, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("loss", 0, 0);
  ctx.restore();

  // Minimum dashed line
  const targetX = toCanvasX(state.target);
  ctx.strokeStyle = "#3fb95055";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(targetX, PAD.top);
  ctx.lineTo(targetX, PAD.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#3fb95088";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("min", targetX, PAD.top - 8);

  // Parabola
  ctx.strokeStyle = "#58a6ff";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  const steps = 200;
  for (let i = 0; i <= steps; i++) {
    const x = X_MIN + (i / steps) * (X_MAX - X_MIN);
    const y = lossVal(x);
    const cx = toCanvasX(x);
    const cy = toCanvasY(Math.min(y, yMax), yMax);
    i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // Parameter dot
  const wClamped = Math.max(X_MIN, Math.min(X_MAX, state.w));
  const lossAtW = Math.min(lossVal(state.w), yMax);
  const dotX = toCanvasX(wClamped);
  const dotY = toCanvasY(lossAtW, yMax);
  const diverged = Math.abs(state.w) > 12;
  const dotColor = diverged ? "#f85149" : "#3fb950";

  // Gradient arrow
  if (state.grad !== 0 && !diverged) {
    const arrowLen = 28;
    const dir = state.grad > 0 ? -1 : 1;
    ctx.strokeStyle = "#d29922";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(dotX, dotY);
    ctx.lineTo(dotX + dir * arrowLen, dotY);
    ctx.stroke();
    ctx.fillStyle = "#d29922";
    ctx.beginPath();
    ctx.moveTo(dotX + dir * arrowLen, dotY);
    ctx.lineTo(dotX + dir * (arrowLen - 7), dotY - 4);
    ctx.lineTo(dotX + dir * (arrowLen - 7), dotY + 4);
    ctx.closePath();
    ctx.fill();
  }

  ctx.beginPath();
  ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
  ctx.fillStyle = dotColor + "33";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
  ctx.fillStyle = dotColor;
  ctx.fill();

  // Labels
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
    const mHat = state.adamT > 0
      ? state.adamM / (1 - state.beta1 ** state.adamT)
      : 0;
    const vHat = state.adamT > 0
      ? state.adamV / (1 - state.beta2 ** state.adamT)
      : 0;
    document.getElementById("val-adam-m").textContent = mHat.toFixed(4);
    document.getElementById("val-adam-v").textContent = vHat.toFixed(6);
  }
}

function highlightLine(n) {
  for (let i = 0; i <= 5; i++) {
    document.getElementById(`line-${i}`).classList.remove("active-line");
  }
  if (n !== null) {
    document.getElementById(`line-${n}`).classList.add("active-line");
  }
}

const LINE_HINTS = {
  1: "Clearing accumulated gradients from previous iteration.",
  4: "Computing gradient: ∂loss/∂w = 2(w − target)",
  5: "Updating parameter using optimizer rule."
};

function setHint(lineIdx) {
  const el = document.getElementById("line-explanation");
  el.textContent = LINE_HINTS[lineIdx] || "";
}

function render() {
  drawCanvas();
  updateStatePanel();
}

function trainingStep() {
  if (stepping) return;
  stepping = true;

  // Phase 1: zero_grad
  highlightLine(1);
  setHint(1);
  if (state.useZeroGrad) {
    state.accumulatedGrad = 0;
  }
  render();

  setTimeout(() => {
    // Phase 2: backward
    highlightLine(4);
    setHint(4);
    state.grad = gradVal(state.w);
    state.accumulatedGrad += state.grad;
    render();

    setTimeout(() => {
      // Phase 3: step
      highlightLine(5);
      setHint(5);

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

      state.stepCount += 1;
      render();

      setTimeout(() => {
        highlightLine(null);
        setHint(null);
        stepping = false;
      }, 350);
    }, 420);
  }, 420);
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

const OPT_DESC = {
  sgd: "<strong>SGD:</strong> w = w − lr × grad",
  momentum: "<strong>SGD + Momentum:</strong> v = β·v + grad &nbsp;|&nbsp; w = w − lr·v",
  adam: "<strong>Adam:</strong> m̂ = m/(1−β₁ᵗ) &nbsp;|&nbsp; v̂ = v/(1−β₂ᵗ) &nbsp;|&nbsp; w = w − lr·m̂/(√v̂+ε)"
};

// — Event listeners —

document.getElementById("btn-step").addEventListener("click", trainingStep);

document.getElementById("btn-auto").addEventListener("click", () => {
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    document.getElementById("btn-auto").textContent = "Auto Run";
  } else {
    autoInterval = setInterval(trainingStep, 1300);
    document.getElementById("btn-auto").textContent = "Pause";
  }
});

document.getElementById("btn-reset").addEventListener("click", resetState);

document.getElementById("toggle-zerog").addEventListener("click", () => {
  state.useZeroGrad = !state.useZeroGrad;
  const btn = document.getElementById("toggle-zerog");
  const hint = document.getElementById("zerog-hint");
  if (state.useZeroGrad) {
    btn.textContent = "zero_grad(): ON";
    btn.className = "toggle-on";
    hint.textContent = "Gradients are cleared before each backward pass.";
  } else {
    btn.textContent = "zero_grad(): OFF";
    btn.className = "toggle-off";
    hint.textContent = "⚠ Gradients accumulate — updates will grow unstable!";
  }
});

document.getElementById("slider-lr").addEventListener("input", (e) => {
  state.lr = parseFloat(e.target.value);
  document.getElementById("lr-display").textContent = state.lr.toFixed(2);
  document.getElementById("val-lr").textContent = state.lr.toFixed(3);
});

document.getElementById("select-optimizer").addEventListener("change", (e) => {
  state.optimizer = e.target.value;
  state.velocity = 0;
  state.adamM = 0; state.adamV = 0; state.adamT = 0;
  updateOptimizerRows();
  document.getElementById("opt-desc").innerHTML = OPT_DESC[state.optimizer];
  render();
});

// Init
updateOptimizerRows();
render();
