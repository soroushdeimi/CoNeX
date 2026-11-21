// Layout checker: drives headless Chrome over the DevTools protocol (no npm
// packages; node has fetch and WebSocket built in) and lists every element
// whose right edge exceeds the viewport. Run:
//   node site/tests/check_layout.mjs [width]
// Exits 0 if no page overflows, 1 otherwise.
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const CHROME =
  process.env.CHROME_BIN ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const WIDTH = Number(process.argv[2] || 390);
const HEIGHT = 900;
const BASE = process.env.BASE_URL || 'http://localhost:8000';
const PAGES = [
  '/index.html',
  '/atlas/index.html',
  '/atlas/plate.html?id=_example',
];

const here = dirname(fileURLToPath(import.meta.url));
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

await send(ws, 'Emulation.setDeviceMetricsOverride', {
  width: WIDTH,
  height: HEIGHT,
  deviceScaleFactor: 1,
  mobile: false,
});
await send(ws, 'Page.enable');

const EXPR = `(() => {
  const w = window.innerWidth;
  const out = [...document.querySelectorAll('*')]
    .filter((e) => e.getBoundingClientRect().right > w + 1)
    .map((e) => e.tagName + '.' + String(e.className).split(' ').join('.') +
      ' right=' + Math.round(e.getBoundingClientRect().right * 10) / 10);
  return JSON.stringify({ innerW: w, scrollW: document.documentElement.scrollWidth, overflow: out });
})()`;

let failed = false;
for (const path of PAGES) {
  await send(ws, 'Page.navigate', { url: BASE + path });
  await new Promise((r) => setTimeout(r, 2500));
  const { result } = await send(ws, 'Runtime.evaluate', {
    expression: EXPR,
    returnByValue: true,
  });
  const data = JSON.parse(result.value);
  if (data.overflow.length) {
    failed = true;
    console.log(`FAIL ${path} (innerW=${data.innerW}, scrollW=${data.scrollW})`);
    for (const o of data.overflow) console.log(`   ${o}`);
  } else {
    console.log(`OK   ${path} (innerW=${data.innerW}, scrollW=${data.scrollW})`);
  }
}

ws.close();
chrome.kill();
process.exit(failed ? 1 : 0);
