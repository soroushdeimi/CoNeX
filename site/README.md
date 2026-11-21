# CoNeX site

A static teaching site for the [CoNeX](https://github.com/cnrl/CoNeX) library.
Plain HTML + CSS + ES modules, no build step. It is served as-is and deployed
to GitHub Pages from this folder.

## Run locally

```bash
cd site && python3 -m http.server 8000
```

Then open <http://localhost:8000/>. The landing page is `index.html`; the
atlas is `atlas/index.html`; a single plate is `atlas/plate.html?id=<id>`.

## Check screenshot

To take a check screenshot of the example plate at phone width:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --window-size=390,900 --virtual-time-budget=3000 \
  --screenshot=/tmp/x.png "http://localhost:8000/atlas/plate.html?id=_example"
```

## Check the layout (no horizontal overflow)

With the local server running, this drives headless Chrome over the DevTools
protocol (no npm packages) and lists every element whose right edge exceeds
the viewport on the landing, atlas and example-plate pages. It exits non-zero
if any page overflows:

```bash
node site/tests/check_layout.mjs 390     # phone width
node site/tests/check_layout.mjs 1440    # desktop width
```

Set `CHROME_BIN` to point at a different Chrome if needed.

## Add a plate

A plate is one ES module in `assets/plates/<id>.js` plus one entry in the
matching `atlas/catalog/<area>.json`. Follow the rules in
`.worker/CONTRACT.md` exactly: the module exports `meta` and `mount(root, ui)`,
builds the page only with `ui.js` calls, gives every model parameter a control
with its CoNeX name, unit and default, includes at least one live figure, a
notes section, and a "CoNeX code" box. The pure math lives in
`assets/js/models/<area>.js` (no DOM) and is tested in `tests/<area>.test.mjs`
against golden numbers from `scripts/site/golden_<area>.py`.

Run the tests with:

```bash
node --test site/tests/
```

## Animating a plate

Every plate animates. For a time-based plate, compute the full run once when a
parameter changes, then let `ui.player` reveal it. `ui.player(parent, {duration,
unit, playSeconds, loop, onFrame})` renders a play/pause button, a restart
button, a speed select (0.25x–4x), a scrubber and a `t = …` readout, and calls
`onFrame(t)` every animation frame and after a seek; model time advances at
`speed * duration / playSeconds` per real second. Inside `onFrame`, redraw the
figure with `plot.draw(series, {revealX: t})` (or `raster.draw(spikes, nRows,
tMax, {revealT: t})`) so only the part of the run up to `t` is shown, with a
cursor line at the current time. On a parameter change, recompute the run and
continue from the current `t` (clamped to the duration). The player autoplays on
mount, pauses while the tab is hidden, and starts paused at the end for users
who prefer reduced motion.

Both `ui.player` and `ui.live` accept an optional `readoutFn(t)` that replaces
the default `t = …` readout, so a plate can name the process instead of a bare
time. For example `readoutFn: (t) => 'I = ' + t + ' pA'` shows the swept
current, and `readoutFn: (t) => 'step ' + t + '/16: C'` shows a step counter.
When omitted, the default `t = … <unit>` readout is unchanged.

## Live (oscilloscope) animation

For continuous-time plates, `ui.live` steps the real model each frame instead of
revealing a precomputed run. `ui.live(parent, {dt, unit, modelPerSecond,
windowSize, step, reset, onFrame})` advances model time at
`speed * modelPerSecond` per real second, calling `step()` once per `dt` of
model time (a fractional accumulator carries the remainder; at most 2000 steps
per frame, the rest dropped and the readout shows "(slowed)"). The plate keeps a
ring buffer of the last `windowSize` of model time and redraws it each frame with
a scrolling axis: `plot.draw(series, {xRange: [t - window, t]})` or
`raster.draw(spikes, nRows, tMax, {xRange: [t - window, t]})`. It renders
play/pause, reset, the shared speed slider (0.1x–10x) and a `t = …` readout,
with no scrubber. Slider changes apply on the next step without a reset; controls
that change sizes or seeds call `reset()`. It autoplays on mount, pauses while the
tab is hidden, and starts paused for users who prefer reduced motion.

## Try-this experiment line

`ui.tryThis(figures, text)` adds a short highlighted "Try this" line — a star
icon plus one sentence naming what to change and what to watch — at the top of
the figures column, above the hero. Call it once in `mount`, before creating
the hero, e.g. `ui.tryThis(figures, 'Raise I; watch threshold crossings become
more frequent.')`.

## Heatmap colour bar

`ui.heatmap(parent, {rows, cols, label, height, colorBar})` draws a grid of
cells. Pass `colorBar: true` to add a vertical colour bar on the right with the
range's min and max labelled (formatted from the range's own tick step, so
`0..1` shows "0"/"1" and `0..100` shows "0"/"100"). The cell ramp blends from a
brighter low end (`heatLow`) up to the series colour, so low values stay
visible in dark mode. The colour bar is off by default.

## Hero animation and narration

`ui.hero(parent, {height})` is the large animated diagram at the top of the
figures column: a HiDPI canvas (like `ui.canvas`) that also redraws on theme
change, plus a one-line narration caption under it. `hero.narration(text)` sets
that line; `ui.narration(parent)` returns a standalone `{set(text), get()}`
caption. `ui.ease(f)` is the shared ease-in-out (smoothstep) used to tween
discrete-step plates between step k and k+1.
