// Loads the area catalogs listed in catalog/index.json. A missing file is
// skipped (other workers are still building them). Several files may share one
// "area" (e.g. learning.json and learning-numenta.json); their plates are
// grouped under a single area, whose title and order come from the first file
// that defines them.

async function fetchCatalog(name) {
  try {
    // pages live in site/atlas/, so the catalog dir is relative to them
    const res = await fetch(`catalog/${name}.json`, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function loadCatalogs() {
  let names;
  try {
    const res = await fetch('catalog/index.json', { cache: 'no-store' });
    names = res.ok ? (await res.json()).catalogs : null;
  } catch {
    names = null;
  }
  if (!Array.isArray(names) || !names.length) {
    // fall back to the known areas if the index is missing
    names = ['neurons', 'plasticity', 'htm', 'tbt'];
  }

  const files = (await Promise.all(names.map(fetchCatalog))).filter(Boolean);

  // Group plates by the "area" field; keep first-seen order and title.
  const groups = new Map();
  for (const f of files) {
    const area = f.area;
    if (!groups.has(area)) {
      groups.set(area, { area, title: f.title, order: f.order, plates: [] });
    }
    groups.get(area).plates.push(...(f.plates || []));
  }

  return [...groups.values()].sort((a, b) => (a.order || 0) - (b.order || 0));
}
