// Loads the four area catalogs. A missing file is skipped (other workers
// are still building them); the rest are returned sorted by `order`.

const AREAS = ['neurons', 'plasticity', 'htm', 'tbt'];

export async function loadCatalogs() {
  const results = await Promise.all(
    AREAS.map(async (area) => {
      try {
        // pages live in site/atlas/, so the catalog dir is relative to them
        const res = await fetch(`catalog/${area}.json`, { cache: 'no-store' });
        if (!res.ok) return null;
        return await res.json();
      } catch {
        return null;
      }
    })
  );
  return results
    .filter(Boolean)
    .sort((a, b) => (a.order || 0) - (b.order || 0));
}
