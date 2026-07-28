# Handoff

State of `roomviz` as of `a689c21`, branch `claude/3d-viz-segmentation-rql6bs` (PR #1).
Written to be read by whoever picks this up next, including a future me.

Read this before trusting any number in the README or the PR description.

---

## What it is

Images, a video, or a folder of photos in; an interactive 3D scene out, with
every object segmented individually and walls/floor/ceiling recovered as
explicit planes. Metric throughout. ~5,700 lines of package, ~6,500 of tests.

```
keyframe sampling -> monocular metric depth + panoptic segmentation
                  -> ORB/PnP camera odometry
                  -> back-projection & multi-frame fusion
                  -> gravity + Manhattan alignment
                  -> sequential RANSAC for the room shell
                  -> PLY / GLB / scene.json / self-contained three.js viewer
```

---

## The single most important thing

**The whole package was built in an environment where `huggingface.co` was
blocked.** The two neural backends — `perception/depth.py` and
`perception/segment.py` — were therefore *never executed once* during
development. Everything downstream of them is verified against ray-traced
ground truth; those two files were reviewed code, not working code.

That is not a footnote. **Every defect found once a real user ran it was in
that unreachable region or in the paths feeding it**, and there were four in
the first two runs:

| Defect | Consequence |
| --- | --- |
| Default `seg_model` id did not exist on the hub | Hard failure; HF answers a missing repo with 401, so it read as an auth/proxy problem |
| COCO-panoptic compound labels (`wall-brick`, `floor-wood`) mapped to "object" | A room with **no walls, floor or ceiling** — silently |
| EXIF gated on `len(frames) == 1` | A *folder* of photos ignored its focal length and guessed the field of view |
| HEIC unsupported | iPhone's default format failed with "no images found in directory" |

All four are fixed. **Assume more remain in the same region.** The lesson is
not "those were the bugs" but "code that has never run is not code that works,
however carefully reviewed."

---

## Verified vs not

**Verified against ground truth** (synthetic ray-traced room, exact depth and
masks substituted for the networks): back-projection, odometry, fusion,
alignment, plane fitting, all four export formats, the viewer, the CLI.
Measured on a real MP4 through the CLI: table 1.22 vs 1.20 m true, sofa 1.41
vs 1.40, chair 0.62 vs 0.60, room height 2.698 vs 2.70, floor coverage 90.5%,
zero caveats.

**Not verified:** the two neural backends (25–31% covered), the CUDA and MPS
device branches, and — critically — **the whole pipeline on real-world
imagery**. As of this writing no real capture has produced a good result. See
"Open investigation".

---

## Test suite

`437 tests, 436 passing, 1 strict xfail`, `ruff` clean. Full run ~118 s;
`-m "not slow"` covers the bulk in ~72 s.

**Mutation score 91.9% (147/160)** — 48.1% -> 80.0% in a dedicated pass, then
the 32 survivors triaged individually for a further 19 kills. The remaining 13
are *equivalent mutants* that no test can kill; they are enumerated with
evidence in the PR description. Do not "fix" them: each would require
asserting an implementation detail, and this suite has spent three rounds
removing exactly that.

Two disciplines are load-bearing here and should survive contact with future
contributors:

1. **Every test must be mutation-checked.** Break the behaviour it claims to
   constrain; confirm it goes red; restore. A test that passes both ways is
   worse than no test, because it reads as coverage.
2. **Run the checker with bytecode caching off.** A stale `.pyc` from a
   mutation run makes clean code execute mutant bytecode and reports a *false
   kill*. This bit me: `PYTHONDONTWRITEBYTECODE=1`.

Harnesses live in the session scratchpad, not the repo: `surv32.py` (the 32
mutants respecified by source text so they survive line drift), `run32.py`
(parallel worktrees), `killcheck.py` (single mutant). Worth re-creating in
`tools/` if this continues.

### Three tests that were deleted or rewritten, and why

- A narrow-FOV test **omitted the assertion that would fail**. At the package
  default 60° HFOV, odometry drift is 0.1188 m against the 0.08 m bound
  asserted elsewhere. It is now a **strict xfail** carrying the measured
  50–95° sweep. This is a real unfixed limitation, not a test artifact.
- An 8 cm tolerance was justified by a comment **invented after the fact** —
  it cited "~5 cm measured" for a 3.55 cm result, and of three regressions it
  claimed to catch, two slip straight through (5% scale error: 0.0356 m;
  10%-long intrinsics: 0.0429 m).
- A RANSAC test asserted a property **the production code had been changed to
  satisfy**. Replaced with accuracy, residual-vs-noise-floor,
  seed-independence and outlier-fraction stability.

If you find yourself tuning a test until it passes, stop — that is how all
three of those got written.

---

## Architecture, and the one non-obvious decision

Layout is conventional: `media/` loads, `perception/` runs the networks (or
reads precomputed sidecars), `geometry/` holds the primitives, `fusion/`
builds the scene, `export/` writes it, `pipeline.py` sequences it, `cli.py`
wraps it.

**The non-obvious decision is how object splitting works**, and it is worth
understanding before touching `fusion/scene_fusion.py`:

A panoptic "stuff" mask can hold several objects (three paintings in one
`painting` mask), so masks are split in 3D. Originally a hand-curated word
list decided *whether* a mask could be split at all — and because `painting`
was *in* that list, three paintings reconstructed as one 3.59 m slab. Classes
nobody had typed were split with no restraint: a pleated curtain returned
eight objects.

Splitting is now **ungated**. The segmenter's own thing/stuff flag sets *how
wide a gap must be* before two blobs are separate objects; the word list is a
hint consulted only when the model offers no flag. Being wrong about a class
now moves a distance, never switches off a behaviour.

Distance alone cannot carry that decision, because **an occluder's shadow is
routinely wider than the space between two genuinely separate objects**. So
every candidate split is checked back against the image: if the pixels between
two pieces belong to a *nearer* surface, it is one object seen past an
obstruction, and the pieces rejoin.

Merging requires **positive provenance** — some frame that saw both pieces
under one segment id. Keying on the mere *absence* of a contradiction (which
holds for any pair never co-visible) fused two chairs 6 cm apart into one
object, and five stools into two. The merge relation is a symmetric predicate
over the original instances, unioned in content-derived order, so the
partition is a function of the instance *set*: a 400-scene × 6-permutation
sweep goes from 25 order-dependent to 0.

`README.md` documents the user-facing contract; the reasoning above lives in
the module docstrings.

---

## Open investigation — read this before debugging a bad result

A real iPhone 15 Pro Max capture (0.5× ultra-wide) has **not** yet produced a
usable reconstruction. Two runs so far:

**Run 1, video, `--hfov 111`:** 189 objects, extent 33.0 × 12.44 × 26.28 m, no
floor, objects smeared 9–12 m along the viewing direction.

The diagnostic number is **189 objects**. From 24 frames at ~8 detections
each, one instance per detection means **nothing merged across frames** —
instances only merge when they occupy the same world space, so the camera
poses were wrong. Everything else follows: no coplanar floor, hence no floor
plane, hence no gravity alignment, hence the nonsense height.

Leading hypothesis: **keyframe sampling**. `max_frames` defaults to 24 and
spreads over the whole clip, so a 60 s walkthrough sampled one keyframe every
2.5 s — consecutive keyframes share almost no features and ORB/PnP has nothing
to match. This now warns up front (`087636b`), but **the hypothesis is
unconfirmed**.

**What is needed to confirm it:** run with `-vv` and read the per-link
`frame N pose: X/Y inliers, translation Z m` lines. Low inlier counts or
erratic translations confirm it. Healthy ones refute it, and the search moves
to depth or distortion.

**Run 2 is pending** — a photo capture of a single room, walking around a
central occluder. Success looks like 10–30 objects, a floor, and a 2.3–2.9 m
height.

### If the geometry is still wrong once sampling is ruled out

Ranked by suspicion, and none of these are yet eliminated:

1. **Ultra-wide is out of distribution for the depth model.** Depth Anything
   V2 metric-indoor infers absolute scale partly from perspective cues and was
   trained overwhelmingly on normal-FOV imagery. A 111° frame may produce
   systematically wrong *metric* depth, which no `--hfov` value can correct.
   **Test by capturing the same room at 1× (~69°) and comparing.**
2. **No distortion model at all.** `CameraIntrinsics` is pure pinhole — there
   are no distortion coefficients anywhere. iOS corrects most ultra-wide
   barrel distortion when Lens Correction is on, but residual bowing hits
   plane fitting hardest, which is the most fragile output.
3. **Odometry drift.** Sequential frame-to-frame only. See below.

---

## Known limitations, ranked by how much they will hurt

1. **Odometry misses its own drift bound at the default 60° field of view**
   (0.1188 m against 0.08 m; 1.688° against 1.5°). Drift roughly triples
   between 70° and 65°. Recorded as a strict xfail rather than hidden.
2. **No loop closure, no bundle adjustment, no relocalisation.** Verified by
   grep — there is nothing to correct accumulated error. This is the hard
   ceiling on scene size, and the reason a whole apartment is not reachable by
   tuning. See "Next steps".
3. **One input per run.** `reconstruct` takes a single path. Multiple captures
   cannot be merged or refined; more footage of the same room currently makes
   drift *worse*, not accuracy better.
4. **Ceiling and room height survive only a 1.12× margin on field of view**
   (present at 85°, gone by 75°) — the tightest margin measured anywhere in
   the project. Object sizes are far more robust than room dimensions.
5. Curved walls come out as several planar patches. Labels are limited to the
   segmentation model's vocabulary. `OCCLUDER_MARGIN` and
   `MIN_FRAGMENT_RATIO` are chosen constants, not derived —
   `MIN_FRAGMENT_RATIO` changed no outcome in any end-to-end run and guards a
   constructed case rather than an observed one.

### Harness cliffs

The synthetic harness sits close to several of its own failure points, and
those positions are now recorded as tests so a regression that moves one is
caught: texture amplitude 2.25×, camera travel 2.3×, depth noise ~2×, HFOV
1.6× for objects but **1.12× for ceiling and room height**. Two claimed cliffs
did not reproduce: there is no 3-frame cliff (2 frames still reconstructs
everything), and the HFOV cliff depends entirely on which assertion is meant.

---

## Next steps, in the order I would do them

1. **Confirm or refute the sampling hypothesis** with `-vv` inlier counts.
   Everything else is guesswork until the trajectory is known-good on one
   room. Nothing below is worth starting first.
2. **Capture the same room at 1× and compare against 0.5×.** One run
   discriminates between "geometry is wrong" and "the depth model is out of
   distribution on ultra-wide" — a question no amount of reading the code can
   settle.
3. **Replace pose estimation with a real SfM/SLAM backend (COLMAP).** This is
   the biggest available quality win and the only route to whole-apartment
   capture. Depth, segmentation, fusion and export stay as they are; only the
   pose source changes. Cost: a heavyweight external dependency and much
   slower runs. The alternative — per-room reconstructions registered through
   doorway overlap — is lighter but does not fix drift *within* a large open
   space, and registering across a doorway with little shared geometry is
   genuinely hard.
4. **CI.** There is none. A suite this size with no automation will rot.
5. Smaller, real, and cheap: read focal length from QuickTime metadata so
   video stops falling back to an assumed FOV (currently only stills get EXIF);
   fix the ~2% bias where the EXIF path divides by 36 mm (frame width) when the
   35mm-equivalent convention is defined on the diagonal.

---

## Conventions worth keeping

- **Comments explain *why*, and every number is measured or derived.** If a
  tolerance is empirical, it says so and names the figure. The one comment
  that invented its justification is described above; it did real damage,
  because it read as authoritative.
- **Warnings name the remedy.** A test asserts that taking the advice in the
  keyframe warning actually silences it — advice that does not work is worse
  than none.
- **The artefact carries its own caveats.** `scene.json` has a machine-readable
  `caveats` list, because an overnight batch writes stderr to a log nobody
  reads. Anything that should hold a result back for review must be in there.
- **Never claim an artefact this run did not write**, and never leave a stale
  one beside a fresh `scene.json` — the viewer takes geometry from `scene.glb`
  and labels from `scene.json`, so a mismatched pair renders the previous
  reconstruction annotated with the current one's numbers.
