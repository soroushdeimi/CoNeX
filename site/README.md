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
