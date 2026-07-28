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
  "room": { "observed_floor_area": 14.3, "floor_coverage": 0.72,
            "room_height": 2.7, "extent": [5.0, 2.7, 4.0] },
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

**`observed_floor_area` is not the room's floor area.** It is the area of the
floor the camera actually saw — film half a room and it halves, with no other
symptom. In the example above the room is 5.0 x 4.0 m (20 m²) and only 14.3 m²
of floor was in shot. `floor_coverage` tells you how much is missing. If you
need gross internal area, this is a lower bound, not an answer.

`caveats` is a machine-readable list of everything that should make you
distrust a result — an assumed camera, a missing ceiling, partial floor
coverage, an implausible room height. Check it in any batch pipeline; the
equivalent warnings go to stderr, which nobody reads overnight.

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

**The segmenter's instance decision sets a threshold, not a veto.** Panoptic
"stuff" classes return one mask per class, so several objects can arrive as a
single segment — those get split apart in 3D. A "thing" mask is split too, but
only across a wide, unoccluded gap: the model calling it one object buys
reluctance, not immunity. Making that a veto meant three paintings sharing one
mask reconstructed as a single 3.6 m object, and whether they did depended on
whether the word "painting" appeared in a hand-typed list — so the list is now
a hint consulted only when the segmenter offers no flag of its own, and it can
only move the split distance, never switch splitting off.

Distance alone cannot carry that decision, because an occluder's shadow is
routinely *wider* than the space between two genuinely separate objects. So
every candidate split is checked back against the image: if the pixels between
two pieces belong to a nearer surface, they are one object seen past an
obstruction, and are rejoined. Where two clusters came from one mask, they may
also be rejoined later if another view connects them; where they came from
different masks in the same frame, they are never merged, however close
together they sit.

**Occlusion edges are discarded.** Back-projecting across a depth
discontinuity smears "flying pixels" through empty space between foreground
and background. Pixels sitting on a large local depth jump are dropped.

**The pipeline under-reports rather than invents.** If the base of a bookcase
is hidden behind a table in every frame, its reported height is the part that
was actually seen. The same goes for the room: the near wall behind the camera
is simply absent rather than guessed at.

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
present. An explicit `--hfov` overrides EXIF. Otherwise the default assumption
is 60°, and **the tool warns you loudly**, because everything scales with it.

This is the single largest source of wrong numbers. A wrong field of view does
not look wrong — it produces a self-consistent room of the wrong size. At 60°
on a 95° capture the demo room comes back 1.6 m tall with a 37 cm sofa. The
reconstruction is checked for plausibility and will warn, but a subtler error
(70° guessed for a 60° lens) passes silently at ~15% off.

`scene.json` records where the camera came from, so you can always tell a
measured reconstruction from a guessed one:

```json
"intrinsics": { "fx": 554.3, "provenance": "assumed_default" }
```

`provenance` is one of `explicit_intrinsics`, `hfov_flag`, `exif` or
`assumed_default`. Treat `assumed_default` as "shape is right, scale is a
guess".

Video carries no EXIF, so for video you should always pass `--hfov` or
`--intrinsics`.

### Commonly useful options

| Option | What it does |
| --- | --- |
| `--max-frames N` | keyframes taken from a video (default 24) |
| `--frame-stride N` | sample every Nth video frame (0 = automatic) |
| `--max-side N` | working resolution (default 768) |
| `--min-sharpness F` | drop blurry frames; try `50`-`200` for handheld video |
| `--device cuda` | run the models on a GPU |
| `--voxel 0.02` | finer/coarser reconstruction |
| `--depth-trunc 8` | ignore depth past N metres (helps with windows) |
| `--no-align` | keep the original camera frame, skip gravity alignment |
| `--depth-min M` | ignore depth closer than this |
| `--open` | serve and open the viewer when finished |

`roomviz view` takes `--host 0.0.0.0` to serve the viewer to a phone on the
same network, and walks up a few ports if the one you asked for is busy.

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

**The structural names matter.** Only these are recognised as room shell, after
splitting on `;` `,` `/`: `wall`/`walls`; `floor`/`flooring`/`rug`/`carpet`;
`ceiling`; `windowpane`/`window`; `door`; `column`/`pillar`; `stairs`. Anything
else — including `wall_1` or `left_wall` — becomes an *object*, and you will
get no wall planes at all.

Other things worth knowing about sidecars:

* Files are matched by **source frame index, 0-based** (`000000.npy` is the
  first video frame). A 1-indexed export is detected and refused, because it
  would otherwise pair every frame with the previous frame's data.
* Depth may be `float32`/`float64` (metres) or an integer type (scaled by
  `--depth-scale`, default mm). `--depth-scale` is ignored for float input.
* Zero, negative, NaN and infinite depths are all treated as invalid.
* In a mask, `-1` means "unlabelled"; `0` is a normal segment id.

If one mask covers several objects — the panoptic "stuff" case — say so, and
they will be separated in 3D rather than fused into one:

```json
{"1": "wall", "2": {"label": "books", "thing": false}}
```

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
* Two objects of the same class that are never visible in the same frame and
  end up within `--object-split-gap` of each other can be merged into one.
  Co-visibility in any single frame prevents this.
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
pytest          # 123 tests
ruff check .
```

The test suite ray-traces a synthetic room with known dimensions, then runs
the real pipeline over it with ground-truth depth and masks in place of the
neural backends. That covers back-projection, odometry, clustering,
cross-frame association, gravity alignment, plane fitting and export, and
every assertion is checked against the room's actual measurements.

**What the numbers mean.** On that scene, horizontal object extents come back
within 1–3 cm and the room height within about 1 cm. Those are measurements of
one favourable configuration — perfect depth, perfect masks, a well-lit
textured room and a smooth camera path — not a guarantee for your footage. The
suite is built to constrain them rather than merely display them: it includes
a 1%-depth-noise run, a narrow-field-of-view run, a run on the shipped
defaults, adversarial cases (objects sharing one mask, a row of near-identical
objects, an object occluded mid-span), and two-sided assertions so that
shrinking every object would fail just as loudly as inflating one.

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
