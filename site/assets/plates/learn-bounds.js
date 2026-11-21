// Plate: Synaptic weight bounds (soft_bound, hard_bound, no_bound).
// Compares bounding factor curves and dynamic weight trajectories under repeated potentiation and depression.
// Mirrors conex/behaviors/synapses/learning.py lines 15-27.

import * as M from '../js/models/learning.js';
import * as A from '../js/anim.js';

export const meta = {
  id: 'learn-bounds',
  title: 'Synaptic weight bounds',
  conex: ['soft_bound', 'hard_bound', 'no_bound', 'SimpleSTDP'],
  source: 'conex/behaviors/synapses/learning.py',
};

export function mount(root, ui) {
  const { controls, figures, notes } = ui.layout(root);

  const hero = ui.hero(figures, { height: 300 });

  const state = {
    w_min: 0.0,
    w_max: 1.0,
    w0: 0.5,
    dw_plus: 0.05,
    dw_minus: 0.05,
    steps: 20,
  };

  let player = null;
  let cachedTrajectories = null;
  let currentStep = state.steps * 2;

  const curveLegend = document.createElement('div');
  curveLegend.style.cssText = 'font-size:12px;margin:0 0 6px 48px;display:flex;gap:1.5rem;align-items:center;';
  curveLegend.innerHTML =
    `<span><strong style="color:var(--c1,#0f7b7f);font-size:15px;line-height:1;">―</strong> hard_bound</span>` +
    `<span><strong style="color:var(--c2,#d95b43);font-size:15px;line-height:1;">―</strong> no_bound</span>` +
    `<span><strong style="color:var(--c3,#b98a1f);font-size:15px;line-height:1;">―</strong> soft_bound</span>`;
  figures.appendChild(curveLegend);

  const curvePlot = ui.plot(figures, {
    height: 190,
    xLabel: 'weight w',
    xUnit: 'a.u.',
    yLabel: 'bound factor B(w)',
    yUnit: 'factor',
  });

  const trajPlot = ui.plot(figures, {
    height: 220,
    xLabel: 'time step',
    xUnit: 'steps',
    yLabel: 'weight w(t)',
    yUnit: 'a.u.',
  });

  function computeTrajectories() {
    const trajectories = {};
    for (const bound of ['soft_bound', 'hard_bound', 'no_bound']) {
      trajectories[bound] = M.simulateBoundTrajectory({
        w0: state.w0,
        dw_plus: state.dw_plus,
        dw_minus: state.dw_minus,
        n_plus: state.steps,
        n_minus: state.steps,
        w_min: state.w_min,
        w_max: state.w_max,
        bound,
      });
    }
    return trajectories;
  }

  // Hero: three weights as beads on rails between w_min and w_max, pushed by
  // potentiation (first half) then depression (second half). The bound factor
  // B(w) is drawn as a spring stiffness; the hard_bound bead sticks at w_max.
  function drawHero(t) {
    const ctx = hero.ctx, W = hero.width, H = hero.height;
    const c1 = ui.color('c1'), c2 = ui.color('c2'), c3 = ui.color('c3');
    const ink = ui.color('ink'), muted = ui.color('muted');
    ctx.clearRect(0, 0, W, H);
    const trajs = cachedTrajectories;
    const bounds = [
      { name: 'hard_bound', color: c1 },
      { name: 'no_bound', color: c2 },
      { name: 'soft_bound', color: c3 },
    ];
    // rail range spans every trajectory (no_bound can leave [w_min, w_max])
    let lo = Infinity, hi = -Infinity;
    for (const b of bounds) for (const w of trajs[b.name]) { lo = Math.min(lo, w); hi = Math.max(hi, w); }
    const pad = 0.05 * (hi - lo || 1);
    lo -= pad; hi += pad;
    const railX0 = W * 0.12, railX1 = W * 0.88;
    const railY = (i) => H * (0.22 + i * 0.28);
    const sx = (w) => railX0 + A.boundsBeadPos(w, lo, hi) * (railX1 - railX0);

    // w_min / w_max markers
    ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    for (const wm of [state.w_min, state.w_max]) {
      ctx.beginPath(); ctx.moveTo(sx(wm), H * 0.1); ctx.lineTo(sx(wm), H * 0.9); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('w_min', sx(state.w_min), H * 0.09);
    ctx.fillText('w_max', sx(state.w_max), H * 0.09);

    bounds.forEach((b, i) => {
      const w = trajs[b.name][t];
      const factor = M.BOUNDS[b.name](w, state.w_min, state.w_max);
      const y = railY(i);
      // rail
      ctx.strokeStyle = muted; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(railX0, y); ctx.lineTo(railX1, y); ctx.stroke();
      // spring from the w_min end to the bead; stiffness = bound factor
      const bx = sx(w);
      const stiff = A.boundsSpring(factor);
      ctx.strokeStyle = b.color; ctx.lineWidth = 1 + 3 * stiff; ctx.globalAlpha = 0.3 + 0.7 * stiff;
      const coils = 8, span = bx - railX0;
      ctx.beginPath(); ctx.moveTo(railX0, y);
      for (let k = 1; k < coils; k++) {
        const x = railX0 + (span * k) / coils;
        ctx.lineTo(x, y + (k % 2 ? -6 : 6) * (0.4 + stiff));
      }
      ctx.lineTo(bx, y); ctx.stroke();
      ctx.globalAlpha = 1;
      // bead
      ctx.fillStyle = b.color;
      ctx.beginPath(); ctx.arc(bx, y, 8, 0, Math.PI * 2); ctx.fill();
      // label + value
      ctx.fillStyle = ink; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(`${b.name}  w = ${w.toFixed(3)}`, railX1 + 8, y);
    });
  }

  function renderAt(step) {
    if (!cachedTrajectories) cachedTrajectories = computeTrajectories();
    const totalSteps = state.steps * 2;
    const t = Math.max(0, Math.min(Math.floor(step), totalSteps));
    currentStep = t;

    // 1. Draw bound curves
    const nPts = 60;
    const pad = 0.2 * (state.w_max - state.w_min);
    const spanStart = state.w_min - pad;
    const spanEnd = state.w_max + pad;
    const wVals = [];
    const softVals = [];
    const hardVals = [];
    const noVals = [];

    for (let i = 0; i < nPts; i++) {
      const w = spanStart + (i * (spanEnd - spanStart)) / (nPts - 1);
      wVals.push(w);
      softVals.push(M.softBound(w, state.w_min, state.w_max));
      hardVals.push(M.hardBound(w, state.w_min, state.w_max));
      noVals.push(M.noBound(w, state.w_min, state.w_max));
    }

    curvePlot.draw(
      [
        { x: wVals, y: hardVals, color: 'c1' },
        { x: wVals, y: noVals, color: 'c2' },
        { x: wVals, y: softVals, color: 'c3' },
      ],
      {
        vlines: [
          { x: state.w_min, label: 'w_min' },
          { x: state.w_max, label: 'w_max' },
        ],
        hlines: [{ y: 0, label: '0' }],
      }
    );

    // 2. Draw trajectories with revealX
    const stepXs = [];
    for (let s = 0; s <= totalSteps; s++) stepXs.push(s);

    trajPlot.draw(
      [
        { x: stepXs, y: cachedTrajectories.hard_bound, label: 'hard_bound', color: 'c1' },
        { x: stepXs, y: cachedTrajectories.no_bound, label: 'no_bound', color: 'c2' },
        { x: stepXs, y: cachedTrajectories.soft_bound, label: 'soft_bound', color: 'c3' },
      ],
      {
        xRange: [0, totalSteps],
        revealX: t,
        vlines: [{ x: state.steps, label: 'potentiation → depression' }],
        hlines: [
          { y: state.w_min, label: 'w_min' },
          { y: state.w_max, label: 'w_max' },
        ],
      }
    );

    // 3. Hero: beads on rails
    drawHero(t);
    const phase = t <= state.steps ? 'plus' : 'minus';
    const hw = cachedTrajectories.hard_bound[t];
    hero.narration(A.boundsNarration('hard_bound', hw, state.w_min, state.w_max, phase));
  }

  function update() {
    cachedTrajectories = computeTrajectories();
    if (player && typeof player.setDuration === 'function') {
      player.setDuration(state.steps * 2);
    }
    renderAt(currentStep);
  }

  // --- Controls ---
  const secBounds = ui.section(controls, 'Weight bounds');
  ui.slider(secBounds, {
    key: 'w_min',
    label: 'w_min',
    unit: 'a.u.',
    min: -0.5,
    max: 0.2,
    step: 0.05,
    value: state.w_min,
    onInput: (v) => { state.w_min = v; update(); },
  });
  ui.slider(secBounds, {
    key: 'w_max',
    label: 'w_max',
    unit: 'a.u.',
    min: 0.8,
    max: 1.5,
    step: 0.05,
    value: state.w_max,
    onInput: (v) => { state.w_max = v; update(); },
  });
  ui.slider(secBounds, {
    key: 'w0',
    label: 'initial weight w0',
    unit: 'a.u.',
    min: 0.0,
    max: 1.0,
    step: 0.05,
    value: state.w0,
    onInput: (v) => { state.w0 = v; update(); },
  });

  const secLearning = ui.section(controls, 'Learning steps');
  ui.slider(secLearning, {
    key: 'dw_plus',
    label: 'potentiation step dw+',
    unit: 'a.u.',
    min: 0.01,
    max: 0.15,
    step: 0.01,
    value: state.dw_plus,
    onInput: (v) => { state.dw_plus = v; update(); },
  });
  ui.slider(secLearning, {
    key: 'dw_minus',
    label: 'depression step dw−',
    unit: 'a.u.',
    min: 0.01,
    max: 0.15,
    step: 0.01,
    value: state.dw_minus,
    onInput: (v) => { state.dw_minus = v; update(); },
  });
  ui.slider(secLearning, {
    key: 'steps',
    label: 'steps per phase',
    unit: 'steps',
    min: 10,
    max: 40,
    step: 5,
    value: state.steps,
    onInput: (v) => { state.steps = v; update(); },
  });

  // Animation player
  if (typeof ui.player === 'function') {
    player = ui.player(figures, {
      duration: state.steps * 2,
      unit: 'steps',
      playSeconds: 5,
      loop: true,
      onFrame: (t) => renderAt(t),
    });
  }

  hero.onResize(() => renderAt(currentStep));
  ui.onTheme(update);
  update();

  // Notes
  ui.note(
    notes,
    `Biological synapses are constrained by receptor packing limits, vesicle pools, ` +
    `and energetic costs. Without bounding mechanisms, associative synaptic plasticity can drive ` +
    `synapses to explode unboundedly or invert to negative weights. CoNeX implements three bounding ` +
    `regimes (learning.py lines 15-27): no_bound, hard_bound, and soft_bound, applicable independently ` +
    `to positive and negative learning.`
  );

  ui.equation(
    notes,
    `<var>soft_bound</var>(<var>w</var>, <var>w</var><sub>min</sub>, <var>w</var><sub>max</sub>) = ` +
    `(<var>w</var> − <var>w</var><sub>min</sub>)·(<var>w</var><sub>max</sub> − <var>w</var>)<br>` +
    `<var>hard_bound</var>(<var>w</var>, <var>w</var><sub>min</sub>, <var>w</var><sub>max</sub>) = ` +
    `[<var>w</var> &gt; <var>w</var><sub>min</sub>]·[<var>w</var> &lt; <var>w</var><sub>max</sub>]<br>` +
    `<var>no_bound</var>(<var>w</var>, <var>w</var><sub>min</sub>, <var>w</var><sub>max</sub>) = 1`
  );

  ui.note(
    notes,
    `Dynamics comparison: In no_bound, weights grow linearly and cross zero into unphysical negative territory ` +
    `during depression. In hard_bound, weights update with full constant increments until they reach the boundary, ` +
    `at which point plasticity ceases discontinuously. In soft_bound, updates scale symmetrically with the quadratic ` +
    `factor (w − w_min)(w_max − w), smoothly decelerating updates near both bounds with maximal plasticity in the center. ` +
    `ML analogy: hard_bound corresponds to gradient projection or hard weight clipping (as in WGAN-GP or ReLU clamps); ` +
    `soft_bound acts as an intrinsic smooth barrier regularizer preventing weight runaway without external clipping.`
  );

  ui.note(
    notes,
    `CoNeX quirk (mirrored as-is): In conex/behaviors/synapses/learning.py:19-20, hard_bound is evaluated ` +
    `with strict inequality: (w > w_min) * (w < w_max). In SimpleSTDP (learning.py:101-110), both potentiation ` +
    `(dw_plus) and depression (dw_minus) scale multiplicatively by their bound factor. Once repeated potentiation ` +
    `drives a weight to w_max exactly (or depression drives it to w_min), hard_bound evaluates to 0. Consequently, ` +
    `subsequent depression cannot move the weight either because dw_minus is multiplied by 0, permanently trapping ` +
    `the weight at the boundary. This real CoNeX behavior is mirrored as-is in our simulation and golden tests.`
  );

  // CoNeX code snippet
  ui.code(notes, () => {
    return [
      `from conex import SimpleSTDP, soft_bound, hard_bound, no_bound`,
      `# Bounding functions can be configured independently for positive and negative STDP:`,
      `SimpleSTDP(a_plus=${state.dw_plus}, a_minus=${state.dw_minus},`,
      `           w_min=${state.w_min}, w_max=${state.w_max},`,
      `           positive_bound="soft_bound", negative_bound="soft_bound")`,
    ].join('\n');
  });

  return {
    destroy() {
      if (player && typeof player.pause === 'function') {
        player.pause();
      }
    },
  };
}
