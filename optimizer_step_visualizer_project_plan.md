# Extra Credit Demo Project Plan: Optimizer Step Visualizer

## Project Title

**Inside `optimizer.step()`: Why Training Loops Need `zero_grad`, `backward`, and `step`**

## Project Goal

This project is an interactive web demo that teaches what happens during a PyTorch-style training step. Instead of treating the optimizer as a black box, the demo shows how a model parameter moves across a simple loss landscape as the user toggles core training-loop operations.

The demo focuses on these lines:

```python
optim.zero_grad()
loss.backward()
optim.step()
```

The goal is to help students understand why each line matters, especially why removing `zero_grad()` causes gradients to accumulate across batches.

## Why This Topic Works

This topic is a strong fit for the extra credit assignment because it is:

- Directly connected to midterm-style material.
- Practical for students learning neural network training.
- Easy to visualize with a simple parameter and loss curve.
- Concise enough to implement cleanly in HTML, CSS, and JavaScript.
- Useful for explaining a common PyTorch bug.

Instead of building a full neural network, the demo uses a simple one-parameter optimization problem:

```text
loss(w) = (w - target)^2
```

This keeps the concept focused on the optimizer mechanics rather than model architecture.

## Main Concept

Training a model usually involves three key steps:

1. `zero_grad()` clears old gradients.
2. `backward()` computes the current gradient from the loss.
3. `step()` updates the parameter using the gradient.

If `zero_grad()` is skipped, gradients accumulate over multiple batches. This can make updates too large and unstable.

## Demo Layout

The webpage should have four main sections.

### 1. Loss Landscape Panel

Show a simple 1D loss curve:

```text
loss(w) = (w - 3)^2
```

Display a dot on the curve representing the current parameter value `w`.

As training runs, animate the dot moving toward the minimum.

Suggested labels:

- Current `w`
- Current loss
- Target/minimum
- Current gradient direction

### 2. Training Loop Code Panel

Display a short code snippet:

```python
for batch in data:
    optim.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optim.step()
```

Highlight the currently executing line during each animation step.

The user should clearly see which visual change corresponds to which code line.

### 3. Optimizer State Panel

Show numerical values:

- Current parameter `w`
- Current loss
- Current gradient
- Accumulated gradient
- Learning rate
- Momentum velocity, if momentum is enabled
- Adam first moment and second moment, if Adam is enabled

This panel helps connect the animation to the underlying math.

### 4. Controls Panel

Include the following controls:

- **Step button:** runs one training iteration.
- **Auto-run button:** repeatedly runs training steps.
- **Reset button:** resets `w`, gradients, and optimizer state.
- **Toggle `zero_grad()`:** on/off.
- **Learning rate slider:** for example, 0.01 to 1.0.
- **Optimizer selector:** SGD, SGD with Momentum, Adam.

Optional controls:

- Batch noise toggle.
- Starting position slider.
- Target value slider.

## Required Interactions

### Interaction 1: Toggle `zero_grad()`

When `zero_grad()` is ON:

- The accumulated gradient is reset before each backward pass.
- Updates are stable.
- The parameter moves smoothly toward the minimum.

When `zero_grad()` is OFF:

- Gradients accumulate across iterations.
- Updates grow too large.
- The parameter may overshoot or diverge.

This should be the central interaction of the demo.

### Interaction 2: Change Learning Rate

The learning rate slider should demonstrate:

- Very small learning rate: slow convergence.
- Reasonable learning rate: stable convergence.
- Very large learning rate: overshooting or divergence.

### Interaction 3: Compare Optimizers

The optimizer dropdown should include:

#### SGD

Update rule:

```text
w = w - lr * grad
```

Teaching point:

SGD moves directly against the current gradient.

#### SGD with Momentum

Update rule:

```text
v = beta * v + grad
w = w - lr * v
```

Teaching point:

Momentum adds velocity, allowing updates to carry information from previous gradients.

#### Adam

Simplified update rule:

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
w = w - lr * m / (sqrt(v) + epsilon)
```

Teaching point:

Adam adapts the update size using moving averages of the gradient and squared gradient.

## Suggested Visual Behavior

### When `zero_grad()` is ON

The accumulated gradient box should reset to zero before each backward pass.

Example visual sequence:

```text
zero_grad(): accumulated gradient = 0
backward(): accumulated gradient = current gradient
step(): w moves using accumulated gradient
```

### When `zero_grad()` is OFF

The accumulated gradient should keep growing.

Example visual sequence:

```text
backward(): accumulated gradient += current gradient
step(): w moves using accumulated gradient
```

The loss curve dot should start moving too aggressively, making the bug obvious.

## Implementation Plan

### Files

Submit the project as a zip file with this structure:

```text
[netID]_optimizer_demo.zip
├── index.html
├── style.css
├── script.js
├── [netID]_demo_video.mp4
└── [netID]_demo_rationale.pdf
```

The rationale PDF is optional but highly recommended.

## `index.html` Structure

Suggested structure:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Optimizer Step Visualizer</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <h1>Inside optimizer.step()</h1>
    <p>
      See how zero_grad(), backward(), and step() work together during training.
    </p>

    <section id="visualization">
      <canvas id="lossCanvas"></canvas>
    </section>

    <section id="codePanel">
      <pre><code id="codeBlock"></code></pre>
    </section>

    <section id="statePanel"></section>

    <section id="controls"></section>
  </main>

  <script src="script.js"></script>
</body>
</html>
```

## Core JavaScript State

Suggested state variables:

```javascript
let state = {
  w: -4,
  target: 3,
  lr: 0.1,
  grad: 0,
  accumulatedGrad: 0,
  useZeroGrad: true,
  optimizer: "sgd",
  momentum: 0.9,
  velocity: 0,
  beta1: 0.9,
  beta2: 0.999,
  adamM: 0,
  adamV: 0,
  adamT: 0,
  epsilon: 1e-8
};
```

## Core Math

Loss:

```javascript
function loss(w) {
  return (w - state.target) ** 2;
}
```

Gradient:

```javascript
function gradient(w) {
  return 2 * (w - state.target);
}
```

Training step:

```javascript
function trainingStep() {
  if (state.useZeroGrad) {
    state.accumulatedGrad = 0;
  }

  state.grad = gradient(state.w);
  state.accumulatedGrad += state.grad;

  if (state.optimizer === "sgd") {
    state.w -= state.lr * state.accumulatedGrad;
  }

  if (state.optimizer === "momentum") {
    state.velocity = state.momentum * state.velocity + state.accumulatedGrad;
    state.w -= state.lr * state.velocity;
  }

  if (state.optimizer === "adam") {
    state.adamT += 1;
    state.adamM = state.beta1 * state.adamM + (1 - state.beta1) * state.accumulatedGrad;
    state.adamV = state.beta2 * state.adamV + (1 - state.beta2) * state.accumulatedGrad ** 2;

    const mHat = state.adamM / (1 - state.beta1 ** state.adamT);
    const vHat = state.adamV / (1 - state.beta2 ** state.adamT);

    state.w -= state.lr * mHat / (Math.sqrt(vHat) + state.epsilon);
  }

  render();
}
```

## Design Tips

Keep the visual design simple and polished.

Recommended style:

- Large title at the top.
- One clear loss curve.
- Use cards for code, controls, and state.
- Use short explanations instead of dense paragraphs.
- Use animation to make each training step visible.
- Keep the demo usable within 30 seconds.

Avoid:

- Long theoretical explanations.
- Too many optimizer options.
- A complicated neural network.
- Dense mathematical derivations.
- Too many sliders.

## Demo Video Plan

The video should be under 2 minutes.

Suggested script:

1. Introduce the demo: “This visualizes a PyTorch-style training step.”
2. Show normal training with `zero_grad()` enabled.
3. Explain that the parameter moves toward the minimum.
4. Turn off `zero_grad()`.
5. Show accumulated gradients growing and updates becoming unstable.
6. Change the learning rate to show slow vs unstable convergence.
7. Switch between SGD, Momentum, and Adam.
8. Conclude with the main lesson.

## Optional Rationale Draft

This demo visualizes the mechanics of a PyTorch-style optimizer step. I chose this topic because students often memorize the training-loop lines without understanding what each line actually changes. The demo uses a one-parameter loss function so that the behavior of `zero_grad()`, `backward()`, and `step()` is visible without the complexity of a full neural network. The central interaction is toggling `zero_grad()`: when it is enabled, gradients are cleared before each update; when it is disabled, gradients accumulate and can cause unstable optimization. I also included a learning-rate slider and optimizer selector so students can compare SGD, momentum, and Adam on the same loss surface. The design is intentionally concise: the curve, code highlight, and optimizer-state panel work together to connect code, math, and behavior. This makes the demo useful for a student seeing model training for the first time.

## Minimum Viable Version

If time is limited, implement only:

- Loss curve.
- Parameter dot.
- Step/reset buttons.
- `zero_grad()` toggle.
- Learning rate slider.
- SGD only.

This version is already enough to make a strong demo.

## Stronger Version

If there is more time, add:

- Auto-run animation.
- Momentum optimizer.
- Adam optimizer.
- Highlighted code line animation.
- Accumulated gradient visualization.
- Short tooltip explanations.

## Final Recommendation

Build the minimum version first, then polish the UI. After that, add Momentum and Adam only if the core demo already works smoothly.

A polished small demo is better than an ambitious demo with bugs.
