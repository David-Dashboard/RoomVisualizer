"""Self-contained web viewer for a reconstructed scene.

Writes ``viewer.html`` plus a ``vendor/`` copy of three.js next to the exported
``scene.glb`` / ``scene.json``.  No network access is needed at view time.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "viewer_assets"

VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoomVisualizer</title>
<link rel="icon" href="data:,">
<style>
  :root {
    --bg: #10131a; --panel: #171b24; --line: #262c39;
    --text: #e6e9ef; --muted: #99a2b4; --accent: #6ea8fe;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden;
    background: var(--bg); color: var(--text);
    font: 13px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  #stage { position: absolute; top: 0; left: 0; right: 310px; bottom: 0; }
  #panel { position: absolute; top: 0; right: 0; bottom: 0; width: 310px;
    background: var(--panel); border-left: 1px solid var(--line);
    display: flex; flex-direction: column; z-index: 10; }
  #panel header { padding: 14px 16px; border-bottom: 1px solid var(--line); }
  #panel h1 { margin: 0; font-size: 14px; letter-spacing: .02em; }
  #stats { margin-top: 6px; color: var(--muted); font-size: 11px; }
  .group { border-bottom: 1px solid var(--line); padding: 10px 16px; }
  .group h2 { margin: 0 0 8px; font-size: 11px; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted); font-weight: 600; }
  label.row { display: flex; align-items: center; gap: 8px; padding: 3px 0;
    cursor: pointer; user-select: none; }
  label.row input { accent-color: var(--accent); }
  #objects { flex: 1 1 0; min-height: 0; overflow-y: auto; padding: 10px 16px 24px; }
  .obj { display: flex; align-items: center; gap: 8px; padding: 4px 6px;
    border-radius: 6px; cursor: pointer; }
  .obj:hover { background: #1e2431; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
  .obj .name { flex: 1; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; }
  .obj .size { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  #hint { position: absolute; left: 16px; bottom: 14px; color: var(--muted);
    font-size: 11px; z-index: 10; }
  #error { position: absolute; inset: 0; display: none; place-content: center;
    padding: 40px; text-align: center; color: var(--muted); z-index: 20; }
  #error code { color: var(--accent); }
  .visually-hidden { position: absolute; width: 1px; height: 1px; margin: -1px;
    padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .obj { font: inherit; color: inherit; background: none; border: 0;
    width: 100%; text-align: left; }
  .obj:focus-visible, canvas:focus-visible { outline: 2px solid var(--accent);
    outline-offset: 2px; }
  /* Swatches are the only key linking a row to a shape, so they must survive
     forced-colors mode, where every background is otherwise overridden. */
  @media (forced-colors: active) {
    .swatch { forced-color-adjust: none; border: 1px solid ButtonText; }
  }
  @media (max-width: 720px), (max-height: 560px) {
    /* Scroll the whole panel rather than only the object list: at phone
       heights the fixed groups leave the list a couple of rows tall, so its
       own scrollbar is not enough to reach the objects comfortably. */
    #panel { width: 100%; height: 55%; top: auto; overflow-y: auto; }
    /* And stop the stage from extending underneath it.  With `inset: 0` the
       canvas filled the viewport and the camera framed the room at the
       vertical centre -- which is exactly where the opaque panel sits, so on
       a phone the room was rendered entirely behind it. */
    #stage { right: 0; bottom: 55%; }
    #objects { flex: none; min-height: auto; overflow: visible; }
    #hint { display: none; }
  }
</style>
</head>
<body>
<main id="stage" aria-label="3D scene">
  <p id="scene-description" class="visually-hidden">Loading the scene&hellip;</p>
</main>
<div id="hint">drag or arrow keys to orbit &middot; scroll or +/- to zoom &middot; right-drag to pan &middot; click or Enter on an object to frame it &middot; Home resets</div>
<div id="error" role="alert">
  <div>
    <p id="reason">Could not load the scene.</p>
    <p>Serve this folder over HTTP:</p>
    <p><code>roomviz view .</code> &nbsp;or&nbsp; <code>python -m http.server</code></p>
  </div>
</div>
<aside id="panel" aria-label="Scene controls">
  <header>
    <h1>RoomVisualizer</h1>
    <div id="stats" role="status" aria-live="polite">loading&hellip;</div>
  </header>
  <div class="group">
    <h2>Layers</h2>
    <label class="row"><input type="checkbox" id="t-cloud" checked> Point cloud</label>
    <label class="row"><input type="checkbox" id="t-objects" checked> Object points</label>
    <label class="row"><input type="checkbox" id="t-boxes"> Bounding boxes</label>
    <label class="row"><input type="checkbox" id="t-walls" checked> Walls</label>
    <label class="row"><input type="checkbox" id="t-floor" checked> Floor &amp; ceiling</label>
    <label class="row"><input type="checkbox" id="t-bylabel"> Colour by object class</label>
  </div>
  <div class="group">
    <h2>Point size</h2>
    <input type="range" id="point-size" min="1" max="12" step="0.5" value="3"
           aria-label="Point size">
  </div>
  <div class="group" style="border-bottom:none; padding-bottom:4px">
    <h2>Objects</h2>
  </div>
  <div id="objects" role="list"></div>
</aside>

<script>
// Deliberately a classic script, and deliberately before the importmap: a
// module's *import statements* are resolved before its body runs, so on
// file:// the imports fail with a CORS error and nothing inside the module
// ever executes -- including any guard written at the top of it.  Only a
// non-module script can report this.
if (location.protocol === 'file:') {
  document.getElementById('reason').textContent =
    'Opened from the filesystem. Browsers block loading JavaScript modules ' +
    'over file://, so this page must be served over HTTP.';
  document.getElementById('error').style.display = 'grid';
  document.getElementById('stats').textContent = 'not served over HTTP';
} else {
  // If the module never gets as far as rendering, say so rather than sitting
  // on "loading..." forever (a missing vendor/ directory does exactly that).
  window.addEventListener('error', (e) => {
    if (e.target && e.target.tagName === 'SCRIPT') {
      document.getElementById('reason').textContent =
        'Could not load ' + (e.target.src || 'a script') + '.';
      document.getElementById('error').style.display = 'grid';
      document.getElementById('stats').textContent = 'load failed';
    }
  }, true);
  setTimeout(() => {
    if (!window.roomviz &&
        document.getElementById('stats').textContent.startsWith('loading')) {
      document.getElementById('reason').textContent =
        'The viewer scripts did not finish loading. Is vendor/ present?';
      document.getElementById('error').style.display = 'grid';
      document.getElementById('stats').textContent = 'load failed';
    }
  }, 10000);
}
</script>
<script type="importmap">
{ "imports": { "three": "./vendor/three.module.js",
               "three/addons/": "./vendor/jsm/" } }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from './vendor/jsm/controls/OrbitControls.js';
import { GLTFLoader } from './vendor/jsm/loaders/GLTFLoader.js';

const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10131a);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.0));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(3, 6, 4);
scene.add(key);

const camera = new THREE.PerspectiveCamera(55, 1, 0.02, 500);
const controls = new OrbitControls(camera, renderer.domElement);
// Inertial glide is motion the user did not ask for; honour their preference.
controls.enableDamping =
  !matchMedia('(prefers-reduced-motion: reduce)').matches;

// The canvas is the product, so it has to be reachable and drivable without a
// mouse.  OrbitControls already ships arrow-key panning; orbit and zoom are
// wired explicitly below.
renderer.domElement.tabIndex = 0;
renderer.domElement.setAttribute('role', 'application');
renderer.domElement.setAttribute('aria-label',
  '3D room view. Arrow keys orbit, plus and minus zoom, Home resets the view.');
renderer.domElement.setAttribute('aria-describedby', 'scene-description');
controls.listenToKeyEvents(renderer.domElement);

const ORBIT_STEP = Math.PI / 24;   // 7.5 degrees
function orbitBy(dTheta, dPhi) {
  const offset = camera.position.clone().sub(controls.target);
  const spherical = new THREE.Spherical().setFromVector3(offset);
  spherical.theta += dTheta;
  spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi + dPhi));
  camera.position.copy(controls.target).add(
    new THREE.Vector3().setFromSpherical(spherical));
  controls.update();
}
function zoomBy(factor) {
  const offset = camera.position.clone().sub(controls.target);
  camera.position.copy(controls.target).add(offset.multiplyScalar(factor));
  controls.update();
}
renderer.domElement.addEventListener('keydown', (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  const handlers = {
    ArrowLeft: () => orbitBy(-ORBIT_STEP, 0),
    ArrowRight: () => orbitBy(ORBIT_STEP, 0),
    ArrowUp: () => orbitBy(0, -ORBIT_STEP),
    ArrowDown: () => orbitBy(0, ORBIT_STEP),
    '+': () => zoomBy(0.85), '=': () => zoomBy(0.85), PageUp: () => zoomBy(0.85),
    '-': () => zoomBy(1.18), PageDown: () => zoomBy(1.18),
    Home: () => frameBox(new THREE.Box3().setFromObject(scene)),
  };
  const handler = handlers[event.key];
  if (handler) { event.preventDefault(); handler(); }
});

// Buckets of nodes, filled in as the glTF is walked.
const layers = { cloud: [], objects: [], boxes: [], walls: [], floor: [] };
const byInstance = new Map();
const objectBounds = new Map();   // node -> cached world-space Box3
let sceneMeta = null;

function bucketFor(name) {
  if (name === 'cloud') return 'cloud';
  if (name.startsWith('object__')) return 'objects';
  if (name.startsWith('box__')) return 'boxes';
  if (name.startsWith('surface__wall')) return 'walls';
  if (name.startsWith('surface__')) return 'floor';
  return null;
}

function applyPointSize(size) {
  scene.traverse((node) => {
    if (node.isPoints) {
      node.material.size = size / 500;
      node.material.sizeAttenuation = true;
      node.material.needsUpdate = true;
    }
  });
}

function frameBox(box, padding = 1.35) {
  framedBounds = box;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 0.25) * padding;
  const distance = radius / Math.sin((camera.fov * Math.PI) / 360);
  const direction = new THREE.Vector3(0.85, 0.5, 0.95).normalize();
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(distance / 500, 0.01);
  camera.far = distance * 40;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function buildObjectList(meta) {
  const host = document.getElementById('objects');
  host.innerHTML = '';
  for (const obj of meta.objects) {
    // A real button, not a div: the object list is the only way a keyboard
    // user can inspect anything, so it has to be focusable and operable.
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'obj';
    row.setAttribute('role', 'listitem');
    const [w, h, d] = obj.size;
    // Labels come from scene.json, which comes from a user-supplied labels.json
    // or a checkpoint's id2label -- untrusted text.  Build the row with
    // textContent so a label can never inject markup or script.
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    const [cr, cg, cb] = obj.color.map((v) => Math.max(0, Math.min(255, v | 0)));
    swatch.style.background = `rgb(${cr},${cg},${cb})`;
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = String(obj.label).split(';')[0];
    const size = document.createElement('span');
    size.className = 'size';
    size.textContent = `${w.toFixed(2)}x${h.toFixed(2)}x${d.toFixed(2)}m`;
    row.replaceChildren(swatch, name, size);
    const detail = `${obj.point_count} points, seen in ${obj.observations} frame(s)`;
    row.title = detail;
    // title= is a mouse tooltip; the accessible name has to carry it too.
    row.setAttribute('aria-label',
      `${String(obj.label).split(';')[0]}, ` +
      `${w.toFixed(2)} by ${h.toFixed(2)} by ${d.toFixed(2)} metres. ${detail}. ` +
      `Activate to frame this object.`);
    row.addEventListener('click', () => {
      const node = byInstance.get(obj.instance_id);
      if (!node) return;
      frameBox(new THREE.Box3().setFromObject(node));
      announce(`Framed ${String(obj.label).split(';')[0]}.`);
    });
    host.appendChild(row);
  }
}

function setVisible(bucket, visible) {
  for (const node of layers[bucket]) node.visible = visible;
}

function announce(message) {
  // #stats is the page's live region; reuse it for transient confirmations.
  const stats = document.getElementById('stats');
  stats.textContent = message;
  clearTimeout(announce._timer);
  announce._timer = setTimeout(() => { stats.textContent = summaryText; }, 4000);
}
let summaryText = '';

function describeScene(meta) {
  // Everything needed for a genuine text alternative is already in
  // scene.json; without this the visualisation is simply absent from the
  // accessibility tree.
  const room = meta.room || {};
  const extent = room.extent || [0, 0, 0];
  const parts = [
    `3D view of a room about ${extent[0].toFixed(1)} by ${extent[2].toFixed(1)} ` +
    `metres and ${extent[1].toFixed(1)} metres high` +
    (room.floor_area ? `, floor area ${room.floor_area.toFixed(1)} square metres` : '') +
    '.',
  ];
  const lo = room.bounds_min || [0, 0, 0];
  const hi = room.bounds_max || [1, 1, 1];
  const place = (c) => {
    const fx = (c[0] - lo[0]) / Math.max(hi[0] - lo[0], 1e-6);
    const fz = (c[2] - lo[2]) / Math.max(hi[2] - lo[2], 1e-6);
    const across = fx < 0.33 ? 'left' : fx > 0.67 ? 'right' : 'centre';
    const along = fz < 0.33 ? 'near' : fz > 0.67 ? 'far' : 'middle';
    return `${along} ${across}`;
  };
  if (meta.objects.length) {
    parts.push(`${meta.objects.length} objects: ` + meta.objects.map((o) =>
      `${String(o.label).split(';')[0]}, ` +
      `${o.size[0].toFixed(2)} by ${o.size[1].toFixed(2)} by ` +
      `${o.size[2].toFixed(2)} metres, ${place(o.centroid)}`).join('; ') + '.');
  } else {
    parts.push('No objects were reconstructed.');
  }
  const surfaces = meta.surfaces || [];
  if (surfaces.length) {
    parts.push(`${surfaces.length} surfaces: ` + surfaces.map(
      (s) => `${s.kind} ${s.area.toFixed(1)} square metres`).join(', ') + '.');
  }
  if (meta.gravity_aligned === false) {
    parts.push('Note: this scene is not gravity aligned, so heights and sizes ' +
               'are measured in the camera frame rather than the room frame.');
  }
  return parts.join(' ');
}

// --- click an object in the 3D view to frame it ---------------------------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pressAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => {
  pressAt = { x: e.clientX, y: e.clientY };
});
// Compare against a distance, not a boolean: a real mouse jitters a pixel or
// two between press and release, and treating that as a drag cancels the pick.
const DRAG_SLOP = 5;
function movedTooFar(e) {
  if (!pressAt) return true;
  return Math.hypot(e.clientX - pressAt.x, e.clientY - pressAt.y) > DRAG_SLOP;
}
renderer.domElement.addEventListener('pointerup', (event) => {
  if (event.button !== 0 || movedTooFar(event)) return;  // orbiting, not picking
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  // Pick against each object's bounding box rather than its points.  A ray
  // through a point cloud passes *between* the points unless the pick radius
  // is tuned to the sampling density, which makes clicking feel unreliable;
  // ray-versus-box always hits and costs nothing for a few dozen objects.
  const hit = new THREE.Vector3();
  let best = null;
  let bestDistance = Infinity;
  for (const node of layers.objects) {
    if (!node.visible) continue;
    const box = objectBounds.get(node);
    if (!box || !raycaster.ray.intersectBox(box, hit)) continue;
    const distance = raycaster.ray.origin.distanceTo(hit);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = box;
    }
  }
  if (best) frameBox(best);
});

function setColourByLabel(enabled) {
  if (!sceneMeta) return;
  for (const obj of sceneMeta.objects) {
    const node = byInstance.get(obj.instance_id);
    if (!node || !node.isPoints) continue;
    if (enabled) {
      node.material.vertexColors = false;
      node.material.color.setRGB(
        obj.color[0] / 255, obj.color[1] / 255, obj.color[2] / 255,
        THREE.SRGBColorSpace);
    } else {
      node.material.vertexColors = true;
      node.material.color.setRGB(1, 1, 1);
    }
    node.material.needsUpdate = true;
  }
}

for (const [id, bucket] of [['t-cloud', 'cloud'], ['t-objects', 'objects'],
                            ['t-boxes', 'boxes'], ['t-walls', 'walls'],
                            ['t-floor', 'floor']]) {
  const input = document.getElementById(id);
  input.addEventListener('change', () => setVisible(bucket, input.checked));
}
document.getElementById('t-bylabel').addEventListener('change', (e) => {
  const on = e.target.checked;
  setColourByLabel(on);
  // The full cloud sits on top of the object points, so leaving it on would
  // speckle the flat class colours with the original photo colours.  Toggle it
  // for the user rather than silently rendering a muddle - and move the
  // checkboxes too, so what is on screen matches what is drawn.
  const cloud = document.getElementById('t-cloud');
  cloud.checked = !on;
  setVisible('cloud', cloud.checked);
  if (on) {
    // Colouring object points is pointless if they are hidden, and turning the
    // cloud off while everything else is off would leave a blank stage.
    const objects = document.getElementById('t-objects');
    objects.checked = true;
    setVisible('objects', true);
  }
  // This control silently changes two others; say so, or a screen-reader user
  // is left with a layer state they were never told about.
  announce(on
    ? 'Colour by object class on. Point cloud hidden, object points shown.'
    : 'Colour by object class off. Point cloud shown.');
});
document.getElementById('point-size').addEventListener('input', (e) =>
  applyPointSize(parseFloat(e.target.value)));

let framedBounds = null;
function resize() {
  const width = stage.clientWidth, height = stage.clientHeight;
  // Note: no `false` third argument -- three.js must set the canvas CSS
  // size as well as its backing store, or on a HiDPI display the canvas
  // ends up devicePixelRatio times too large in CSS pixels and the scene
  // renders off-screen.
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
  // Also refresh the device pixel ratio: it changes when a window moves
  // between displays, and setPixelRatio is otherwise only read once at load.
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
}
addEventListener('resize', () => {
  resize();
  // Rotating a phone swaps which axis is constrained; without re-framing, the
  // scene can end up mostly off-screen.
  if (framedBounds) frameBox(framedBounds);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function fail(message) {
  const pane = document.getElementById('error');
  pane.querySelector('#reason').textContent = message;
  pane.style.display = 'grid';
  document.getElementById('stats').textContent = 'load failed';
}

async function main() {
  const meta = await fetch('scene.json').then((r) => {
    if (!r.ok) throw new Error('scene.json: HTTP ' + r.status);
    return r.json();
  });
  sceneMeta = meta;
  const gltf = await new GLTFLoader().loadAsync('scene.glb');

  gltf.scene.traverse((node) => {
    if (!node.isMesh && !node.isPoints) return;
    // userData.name preserves the original glTF name; node.name is sanitised.
    const name = node.userData.name || node.name || '';
    const bucket = bucketFor(name);
    if (bucket) layers[bucket].push(node);

    if (node.isMesh) {
      // GLTFLoader hands the same cached default material to every mesh that
      // declares none, so styling one surface would restyle every box too.
      node.material = node.material.clone();
      // Meshes carry vertex colours but no glTF material, so GLTFLoader
      // substitutes its default -- which is metalness 1.  A fully rough metal
      // with no environment map has no diffuse term, so every surface and box
      // renders near-black regardless of its colour.
      if ('metalness' in node.material) node.material.metalness = 0.0;
      if ('roughness' in node.material) node.material.roughness = 0.9;
      if (name.startsWith('surface__')) {
        node.material.side = THREE.DoubleSide;
        node.material.transparent = true;
        node.material.opacity = 0.75;
        node.material.depthWrite = false;
      }
    }
    const match = /^object__(\\d+)__/.exec(name);
    if (match && node.isPoints) byInstance.set(parseInt(match[1], 10), node);
  });

  scene.add(gltf.scene);
  applyPointSize(parseFloat(document.getElementById('point-size').value));
  setVisible('boxes', false);
  buildObjectList(meta);

  const room = meta.room || {};
  const extent = room.extent || [0, 0, 0];
  summaryText =
    `${meta.summary.object_count} objects | ${meta.summary.surface_count} surfaces | ` +
    `${meta.summary.point_count.toLocaleString()} points | ` +
    `${extent.map((v) => v.toFixed(1)).join(' x ')} m`;
  document.getElementById('stats').textContent = summaryText;
  document.getElementById('scene-description').textContent = describeScene(meta);

  // A handle for scripting and for automated checks: everything the page
  // builds, reachable from the console.
  // COLOR_0 is written as sRGB bytes but glTF declares it linear, so three.js
  // would render the whole cloud washed out.  Convert once, at load.
  const seenColorAttributes = new Set();
  gltf.scene.traverse((node) => {
    const attribute = node.geometry && node.geometry.attributes.color;
    if (!attribute || seenColorAttributes.has(attribute)) return;
    seenColorAttributes.add(attribute);
    const colour = new THREE.Color();
    for (let i = 0; i < attribute.count; i++) {
      colour.setRGB(
        attribute.getX(i), attribute.getY(i), attribute.getZ(i),
        THREE.SRGBColorSpace);
      attribute.setXYZ(i, colour.r, colour.g, colour.b);
    }
    attribute.needsUpdate = true;
  });

  // Bounding boxes are computed once: the scene is static, and recomputing
  // them per click would walk every point on every pick.
  for (const node of layers.objects) {
    objectBounds.set(node, new THREE.Box3().setFromObject(node));
  }

  window.roomviz = {
    THREE, scene, camera, controls, layers, byInstance, objectBounds, meta, frameBox,
  };

  resize();
  frameBox(new THREE.Box3().setFromObject(gltf.scene));
  animate();
}

main().catch((err) => {
  console.error(err);
  fail(String(err && err.message ? err.message : err));
});
</script>
</body>
</html>
"""


def write_viewer(directory: str | Path) -> Path:
    """Write ``viewer.html`` and the vendored three.js into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    vendor = directory / "vendor"
    if vendor.exists():
        shutil.rmtree(vendor)
    if not ASSETS_DIR.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(
            f"viewer assets missing at {ASSETS_DIR}; reinstall the package"
        )
    shutil.copytree(ASSETS_DIR, vendor)

    path = directory / "viewer.html"
    path.write_text(VIEWER_HTML, encoding="utf-8")
    log.info("wrote %s", path)
    return path
