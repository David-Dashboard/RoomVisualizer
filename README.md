# RoomVisualizer

Turn photos or video of a room into an interactive 3D scene in which **every
object is segmented separately** and the **walls, floor and ceiling** are
recovered as explicit planes.

```bash
roomviz reconstruct living_room.mp4 -o output
roomviz view output
```

Input: a single image, a video, or a folder of images.
Output: a browsable 3D scene, per-object point clouds, and a JSON description
of the room with everything measured in metres.

---

## What it produces

```
output/
├── viewer.html          interactive 3D viewer (self-contained, no CDN)
├── vendor/              three.js, copied locally
├── scene.glb            the whole scene, one named node per object and surface
├── scene.ply            the full coloured point cloud
├── scene.json           objects, dimensions, wall planes, camera poses
└── objects/
    ├── 000_sofa.ply
    ├── 001_table.ply
    └── ...
```

`scene.json` is the structured deliverable:

```json
{
  "up_axis": "+Y", "units": "metres",
  "room": { "floor_area": 14.3, "room_height": 2.7, "extent": [5.0, 2.7, 4.0] },
  "objects": [
    { "instance_id": 0, "label": "sofa", "size": [1.40, 0.63, 1.16],
      "centroid": [3.98, 0.31, 3.09], "observations": 8,
      "node": "object__0__sofa", "color": [170, 90, 110] }
  ],
  "surfaces": [
    { "kind": "wall", "normal": [1.0, 0.0, 0.0], "offset": -3.2,
      "quad": [[...], [...], [...], [...]], "area": 12.8 }
  ]
}
```

Objects are toggled individually in the viewer, or loaded straight into
Blender / MeshLab / anything that reads glTF or PLY.

---

## Install

```bash
git clone https://github.com/David-Dashboard/RoomVisualizer
cd RoomVisualizer
pip install -e ".[models]"
```

`[models]` pulls in torch + transformers for the neural backends. Without it
the package still installs and runs, but you have to supply depth and masks
yourself (see [Bring your own depth](#bring-your-own-depth)).

A GPU is optional. On CPU expect roughly 5-20 s per frame for the default
models; `--device cuda` is a good deal faster.

---

## Try it without any model weights

A synthetic room ships with the repo, complete with ground-truth depth and
masks, so you can exercise the whole pipeline offline:

```bash
python examples/make_demo_room.py --out examples/demo

roomviz reconstruct examples/demo/room.mp4 -o output \
    --depth-backend file --depth-dir examples/demo/depth \
    --seg-backend  file --seg-dir   examples/demo/segmentation \
    --hfov 95

roomviz view output
```

The room is 5.00 x 2.70 x 4.00 m with four pieces of furniture, and
`examples/demo/ground_truth.json` records their exact dimensions so you can
check the output against them.

---

## How it works

```
image / video
      │
      ├─► keyframe sampling ─────────────► RGB frames
      │
      ├─► monocular depth ──────────────► metric depth (m)
      ├─► panoptic segmentation ────────► per-pixel instances + wall/floor/ceiling
      │
      ├─► ORB + PnP odometry ───────────► camera pose per frame
      │
      ├─► back-projection & fusion ─────► world-space point cloud
      │        ├─ 3D clustering ────────► per-frame object detections
      │        └─ cross-frame association ► global object instances
      │
      ├─► gravity + Manhattan alignment ► floor at y=0, walls axis-aligned
      └─► sequential RANSAC ────────────► wall / floor / ceiling planes
```

A few decisions worth calling out:

**Metric depth, not relative.** The default checkpoint is a *metric* indoor
model, so depth comes out in metres. That is what makes the camera odometry,
the fused geometry and the reported object dimensions all consistent with one
another. Relative checkpoints work too, but their absolute scale is a guess.

**Objects are split in 3D, not in 2D.** Panoptic "stuff" classes return one
mask per class, so three paintings arrive as a single segment. Splitting them
in 3D — where genuinely separate objects are separated in space — handles that
without also splitting an object that an occluder cut in two on screen.

**Occlusion edges are discarded.** Back-projecting across a depth
discontinuity smears "flying pixels" through empty space between foreground
and background. Pixels sitting on a large local depth jump are dropped.

**The pipeline under-reports rather than invents.** If the base of a bookcase
is hidden behind a table in every frame, its reported height is the part that
was actually seen.

---

## Usage

```bash
roomviz reconstruct INPUT [-o OUTPUT] [options]
roomviz view [DIRECTORY] [-p PORT]
roomviz backends
```

`roomviz photo.jpg` is shorthand for `roomviz reconstruct photo.jpg`.

### Camera

Scale accuracy depends on the camera's field of view. In order of preference:

```bash
--intrinsics FX FY CX CY   # exact, in original-resolution pixels
--hfov 68                  # horizontal field of view in degrees
```

For a single image, EXIF `FocalLengthIn35mmFilm` is used automatically when
present. Otherwise the default assumption is 60°, which is a typical phone
main camera; a wide-angle lens needs telling.

### Commonly useful options

| Option | What it does |
| --- | --- |
| `--max-frames N` | keyframes taken from a video (default 24) |
| `--max-side N` | working resolution (default 768) |
| `--min-sharpness F` | drop blurry frames; try `50`-`200` for handheld video |
| `--device cuda` | run the models on a GPU |
| `--voxel 0.02` | finer/coarser reconstruction |
| `--depth-trunc 8` | ignore depth past N metres (helps with windows) |
| `--no-align` | keep the original camera frame, skip gravity alignment |
| `--open` | serve and open the viewer when finished |

`roomviz reconstruct --help` lists everything.

### Python API

```python
from roomviz import PipelineConfig, run

result = run("living_room.mp4", "output", PipelineConfig(max_frames=16, hfov_deg=68))

for obj in result.scene.objects:
    lo, hi = obj.aabb
    print(f"{obj.label}: {hi - lo} m")

for surface in result.scene.surfaces:
    print(surface.kind, surface.area, "m^2")
```

---

## Bring your own depth

Monocular depth is the weakest link in any single-camera reconstruction. If
your capture already has real depth — an iPhone/iPad LiDAR scan, a RealSense
recording, a rendered dataset — feed it in directly and the geometry improves
dramatically:

```bash
roomviz reconstruct scan.mp4 -o output \
    --depth-backend file --depth-dir depth/ \
    --intrinsics 1450 1450 960 540
```

`depth/` holds one file per **original video frame**, named `000000.npy`,
`000001.npy`, … (`.npy` float metres, or 16-bit PNG/TIFF with
`--depth-scale 0.001` for millimetres). Masks work the same way via
`--seg-backend file --seg-dir masks/`, where `masks/` holds integer segment-id
maps plus a `labels.json` of `{"1": "wall", "2": "floor", "3": "chair"}`.

Depth and segmentation are independent, so you can mix a real depth sensor
with the neural segmenter, or vice versa.

---

## Getting good results

* **Move the camera.** Reconstruction quality comes from parallax. A slow walk
  across the room beats a pan from one spot; pure rotation gives odometry
  nothing to work with.
* **Cover the floor and ceiling.** They anchor gravity alignment and the room
  height. If the camera never looks down, the floor plane is guesswork.
* **Keep it sharp.** Motion blur destroys feature matching. Use
  `--min-sharpness` to drop the worst frames.
* **Tell it the field of view.** Everything is metric; a wrong FOV scales the
  entire room.
* **Resolution matters more than frame count.** A distant sofa spanning 25
  pixels cannot be measured accurately no matter how many frames it appears in.

### Limitations

* Frame-to-frame odometry has no loop closure or bundle adjustment, so drift
  accumulates over long trajectories. Keep clips short, or supply poses.
* Object labels come from the segmentation model's vocabulary (ADE20K by
  default). Anything outside it is labelled as the nearest class it knows.
* Walls are fitted as planes, so curved or heavily cluttered walls come out as
  several planar patches.
* Only surfaces the camera saw are reconstructed — there is no completion of
  unobserved geometry.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # 91 tests
ruff check .
```

The test suite ray-traces a synthetic room with known dimensions, then runs
the real pipeline over it with ground-truth depth and masks in place of the
neural backends. That covers back-projection, odometry, clustering,
cross-frame association, gravity alignment, plane fitting and export, and
every assertion is checked against the room's actual measurements — object
sizes land within a few centimetres.

```
roomviz/
├── media/        video and image loading, keyframe selection
├── perception/   depth and segmentation backends (pluggable)
├── geometry/     intrinsics, back-projection, planes, alignment
├── fusion/       camera odometry, multi-frame scene fusion
└── export/       PLY, glTF, scene JSON, web viewer
```

Adding a backend is a decorator:

```python
from roomviz.perception.base import register_depth

@register_depth("my-model")
def build(cfg):
    return MyDepthBackend(cfg)
```

## License

MIT — see [LICENSE](LICENSE).
