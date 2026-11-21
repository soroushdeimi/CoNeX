// Layout checker: drives headless Chrome over the DevTools protocol (no npm
// packages; node has fetch and WebSocket built in) and lists, for every page,
// (1) elements whose right edge exceeds the viewport and (2) a .plate-controls
// block whose box overlaps the .plate-notes block (the sticky-controls bug).
// Run:
//   node site/tests/check_layout.mjs [width ...]   (default: 390 1440)
//   THEME=dark node site/tests/check_layout.mjs    (light is the default)
// Exits 0 if no page overflows or overlaps, 1 otherwise.
import { spawn } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const CHROME =
  process.env.CHROME_BIN ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const widths = process.argv.slice(2).map(Number).filter((n) => Number.isFinite(n) && n > 0);
if (!widths.length) widths.push(390, 1440);
const HEIGHT = 900;
const BASE = process.env.BASE_URL || 'http://localhost:8000';
const THEME = process.env.THEME === 'dark' ? 'dark' : 'light';

const here = dirname(fileURLToPath(import.meta.url));
const platesDir = join(here, '..', 'assets', 'plates');
const PAGES = [
  '/index.html',
  '/atlas/index.html',
  ...readdirSync(platesDir)
    .filter((f) => f.endsWith('.js'))
    .map((f) => `/atlas/plate.html?id=${encodeURIComponent(f.slice(0, -3))}`),
];

const PORT = 9400 + Math.floor(Math.random() * 100);

const chrome = spawn(
  CHROME,
  [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    '--no-first-run',
    '--no-default-browser-check',
    `--user-data-dir=/tmp/check_layout_${process.pid}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
);

async function getJSON(path) {
  const r = await fetch(`http://127.0.0.1:${PORT}${path}`);
  return r.json();
}
async function waitTargets() {
  for (let i = 0; i < 60; i++) {
    try {
      const t = await getJSON('/json');
      if (Array.isArray(t) && t.length) return t;
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('Chrome did not open a debug target');
}

let idc = 0;
const pending = new Map();
function send(ws, method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++idc;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
}

const targets = await waitTargets();
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { resolve, reject } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? reject(new Error(m.error.message)) : resolve(m.result);
  }
};

await send(ws, 'Page.enable');

const EXPR = `(() => {
  const w = window.innerWidth;
  const overflow = [...document.querySelectorAll('*')]
    .filter((e) => {
      const r = e.getBoundingClientRect();
      if (r.right <= w + 1) return false;
      // Known false positive: a <code> inside a scrollable box may extend
      // past the viewport while the box itself scrolls.
      let p = e.parentElement;
      while (p) {
        const ox = getComputedStyle(p).overflowX;
        if (ox === 'auto' || ox === 'scroll') return false;
        p = p.parentElement;
      }
      return true;
    })
    .map((e) => e.tagName + '.' + String(e.className).split(' ').join('.') +
      ' right=' + Math.round(e.getBoundingClientRect().right * 10) / 10);
  let overlap = null;
  const controls = document.querySelector('.plate-controls');
  const notes = document.querySelector('.plate-notes');
  if (controls && notes) {
    const c = controls.getBoundingClientRect();
    const n = notes.getBoundingClientRect();
    const sameColumn = c.left < n.right && n.left < c.right;
    if (sameColumn && c.bottom > n.top + 1) {
      overlap = '.plate-controls bottom=' + Math.round(c.bottom * 10) / 10 +
        ' > .plate-notes top=' + Math.round(n.top * 10) / 10;
    }
  }
  return JSON.stringify({ innerW: w, scrollW: document.documentElement.scrollWidth, overflow, overlap });
})()`;

let failed = false;
for (const width of widths) {
  await send(ws, 'Emulation.setDeviceMetricsOverride', {
    width,
    height: HEIGHT,
    deviceScaleFactor: 1,
    mobile: false,
  });
  // Set the theme before any page script runs (ui.js reads it on load).
  const { identifier } = await send(ws, 'Page.addScriptToEvaluateOnNewDocument', {
    source:
      THEME === 'dark'
        ? "localStorage.setItem('conex-theme', 'dark');"
        : "localStorage.removeItem('conex-theme');",
  });
  for (const path of PAGES) {
    await send(ws, 'Page.navigate', { url: BASE + path });
    await new Promise((r) => setTimeout(r, 2500));
    let data;
    try {
      const { result } = await send(ws, 'Runtime.evaluate', {
        expression: EXPR,
        returnByValue: true,
      });
      data = JSON.parse(result.value);
    } catch (err) {
      failed = true;
      console.log(`FAIL ${path} [${width}px ${THEME}] (page did not load: ${err.message})`);
      continue;
    }
    if (data.overflow.length || data.overlap) {
      failed = true;
      console.log(`FAIL ${path} [${width}px ${THEME}] (innerW=${data.innerW}, scrollW=${data.scrollW})`);
      for (const o of data.overflow) console.log(`   overflow: ${o}`);
      if (data.overlap) console.log(`   overlap: ${data.overlap}`);
    } else {
      console.log(`OK   ${path} [${width}px ${THEME}] (innerW=${data.innerW}, scrollW=${data.scrollW})`);
    }
  }
  await send(ws, 'Page.removeScriptToEvaluateOnNewDocument', { identifier }).catch(() => {});
}

ws.close();
chrome.kill();
process.exit(failed ? 1 : 0);
