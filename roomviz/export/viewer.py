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
  #stage { position: absolute; inset: 0; }
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
  #objects { flex: 1; overflow-y: auto; padding: 10px 16px 24px; }
  .obj { display: flex; align-items: center; gap: 8px; padding: 4px 6px;
    border-radius: 6px; cursor: pointer; }
  .obj:hover { background: #1e2431; }
  .obj.dim { opacity: .4; }
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
  @media (max-width: 720px) { #panel { width: 100%; height: 45%; top: auto; } }
</style>
</head>
<body>
<div id="stage"></div>
<div id="hint">drag to orbit &middot; scroll to zoom &middot; right-drag to pan &middot; click an object to frame it</div>
<div id="error">
  <div>
    <p>Could not load <code>scene.glb</code>.</p>
    <p>Browsers block local file access for security, so serve this folder over HTTP:</p>
    <p><code>roomviz view .</code> &nbsp;or&nbsp; <code>python -m http.server</code></p>
  </div>
</div>
<aside id="panel">
  <header>
    <h1>RoomVisualizer</h1>
    <div id="stats">loading&hellip;</div>
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
    <input type="range" id="point-size" min="1" max="12" step="0.5" value="3">
  </div>
  <div class="group" style="border-bottom:none; padding-bottom:4px">
    <h2>Objects</h2>
  </div>
  <div id="objects"></div>
</aside>

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
controls.enableDamping = true;

// Buckets of nodes, filled in as the glTF is walked.
const layers = { cloud: [], objects: [], boxes: [], walls: [], floor: [] };
const byInstance = new Map();
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
    const row = document.createElement('div');
    row.className = 'obj';
    const [w, h, d] = obj.size;
    row.innerHTML =
      `<span class="swatch" style="background: rgb(${obj.color.join(',')})"></span>` +
      `<span class="name">${obj.label.split(';')[0]}</span>` +
      `<span class="size">${w.toFixed(2)}x${h.toFixed(2)}x${d.toFixed(2)}m</span>`;
    row.title = `${obj.point_count} points, seen in ${obj.observations} frame(s)`;
    row.addEventListener('click', () => {
      const node = byInstance.get(obj.instance_id);
      if (!node) return;
      frameBox(new THREE.Box3().setFromObject(node));
    });
    host.appendChild(row);
  }
}

function setVisible(bucket, visible) {
  for (const node of layers[bucket]) node.visible = visible;
}

function setColourByLabel(enabled) {
  if (!sceneMeta) return;
  for (const obj of sceneMeta.objects) {
    const node = byInstance.get(obj.instance_id);
    if (!node || !node.isPoints) continue;
    if (enabled) {
      node.material.vertexColors = false;
      node.material.color.setRGB(
        obj.color[0] / 255, obj.color[1] / 255, obj.color[2] / 255);
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
  setColourByLabel(e.target.checked);
  // The full cloud sits on top of the object points, so leaving it on would
  // speckle the flat class colours with the original photo colours.  Toggle it
  // for the user rather than silently rendering a muddle - and move the
  // checkbox too, so the state on screen still matches what is drawn.
  const cloud = document.getElementById('t-cloud');
  cloud.checked = !e.target.checked;
  setVisible('cloud', cloud.checked);
});
document.getElementById('point-size').addEventListener('input', (e) =>
  applyPointSize(parseFloat(e.target.value)));

function resize() {
  const width = stage.clientWidth, height = stage.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function main() {
  const meta = await fetch('scene.json').then((r) => r.json());
  sceneMeta = meta;
  const gltf = await new GLTFLoader().loadAsync('scene.glb');

  gltf.scene.traverse((node) => {
    if (!node.isMesh && !node.isPoints) return;
    // userData.name preserves the original glTF name; node.name is sanitised.
    const name = node.userData.name || node.name || '';
    const bucket = bucketFor(name);
    if (bucket) layers[bucket].push(node);

    if (node.isMesh && name.startsWith('surface__')) {
      node.material.side = THREE.DoubleSide;
      node.material.transparent = true;
      node.material.opacity = 0.75;
      node.material.depthWrite = false;
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
  document.getElementById('stats').textContent =
    `${meta.summary.object_count} objects | ${meta.summary.surface_count} surfaces | ` +
    `${meta.summary.point_count.toLocaleString()} points | ` +
    `${extent.map((v) => v.toFixed(1)).join(' x ')} m`;

  resize();
  frameBox(new THREE.Box3().setFromObject(gltf.scene));
  animate();
}

main().catch((err) => {
  console.error(err);
  document.getElementById('error').style.display = 'grid';
  document.getElementById('stats').textContent = 'load failed';
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
