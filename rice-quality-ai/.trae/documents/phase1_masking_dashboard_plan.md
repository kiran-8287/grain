# Phase 1 Masking → Full Dashboard Integration + Phase1Demo Removal Implementation Plan

## Repository Research

### Current architecture (BEFORE this change)

**Backend — two parallel tracks:**

1. **Full dashboard track** (`POST /api/analyze` in [routes.py](file:///a:/grain/rice-quality-ai/backend/app/api/routes.py#L88-L126) → `job_manager.process_sync` → `RiceQualityPipeline.analyze` in [pipeline.py](file:///a:/grain/rice-quality-ai/ml/pipeline.py#L118)):
   - Step 4 calls `segment_grains(image_rgb, method="auto")` — this is the **classical CV watershed** path (watershed + distance transform). The method="auto" inside `ml/segmentation.py` prefers classical CV and only drops to YOLO when it imports cleanly; it produces `GrainInstance` objects with `.mask` (uint8 2D array), `.contour`, `.bbox`, `.centroid`, `.confidence` (classical CV confidence), `grain_id`.
   - Step 5 computes full geometry (length, breadth, L/B, area, solidity, perimeter, orientation) from `.mask` using `compute_grain_geometry`.
   - Step 6 runs all 8 defect classifiers on crops derived from `.mask`.
   - Step 12 renders the annotated image using `_create_annotated_image_b64()` which **uses the watershed `grain.contour`/`grain.mask`** — no mask polygons, no YOLO coloring-by-confidence.

2. **Phase 1 Demo track** (`POST /api/phase1/analyze` in [routes.py](file:///a:/grain/rice-quality-ai/backend/app/api/routes.py#L410-L554) → `ml.inference.analyze_image` in [inference.py](file:///a:/grain/rice-quality-ai/ml/inference.py)):
   - Priority cascade: YOLOv8l-seg `.pt` → Mask R-CNN → classical CV watershed.
   - Produces structured `Phase1AnalysisResponse`-like dicts with `grains[*].mask_polygon`, `grains[*].confidence_label` (HIGH/MEDIUM/LOW), `grains[*].segmentation_method`.
   - Renders its own overlay via `_build_phase1_overlay()` at [routes.py L258](file:///a:/grain/rice-quality-ai/backend/app/api/routes.py#L258), which draws colored polygons keyed by confidence label, assigns stable IDs, and draws the method badge in the corner.

**Schema gap between the two tracks:**

| Field | Dashboard `GrainInstance` (types.ts L59-L83) | Phase1 `Phase1GrainInstance` (types.ts L316-L326) |
|---|---|---|
| `confidence` float | ✅ (top of the struct) | ✅ |
| `confidence_label` (HIGH/MED/LOW) | ❌ — absent | ✅ |
| `mask_polygon` / polygon array | ❌ — uses `contour: number[][]` (OpenCV contour) and a server-side `mask: uint8[]` not sent over wire | ✅ |
| `segmentation_method` string | ❌ — server derives `segmentation_quality` enum, not the method | ✅ |
| `is_touching` boolean | ✅ (same semantics) | ✅ |

The **annotated image** in the full dashboard is server-rendered and sent as `annotated_image_base64: data:image/jpeg;base64,...` — the frontend does NOT draw masks. This means we can achieve "use Phase 1 masking system for the dashboard" entirely on the backend without rewriting the viewer contract, as long as:

- The segmentation step inside `RiceQualityPipeline.analyze` is rewritten to call `ml.inference.analyze_image` (Phase 1 cascade) instead of `segment_grains`, and
- The returned Phase 1 `mask_polygon` vectors are rasterized back into `uint8 H×W` masks so the downstream geometry/classifier pipeline (which operates on masks) still works **unchanged**, and
- `_create_annotated_image_b64` is replaced with Phase 1's colored overlay, OR the existing renderer is taught to use polygon-based rendering instead of watershed-mask rendering, preserving all downstream geometry and defect classifications.

**Frontend — App.tsx tab system:**

[App.tsx](file:///a:/grain/rice-quality-ai/frontend/src/App.tsx) lines L12, L16, L19, L140-L156, L160-L162 mount Phase1Demo as a sibling tab to the dashboard with:
- `import { Phase1Demo } from './components/Phase1Demo';` (L12)
- `type ActiveTab = 'dashboard' | 'phase1';` (L16)
- `const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');` (L19)
- TabBar at L140-L156 rendering BOTH tabs.
- `{activeTab === 'phase1' ? <Phase1Demo /> : (<>...</>)}` at L160-L304.

Phase1Demo.tsx is not imported anywhere else in the codebase (grep of `frontend/src` returned 1 hit only — the App.tsx import). This means "temporarily remove from UI" can be a clean toggle: strip the tab import, the tab button, and the ternary branch, but leave `components/Phase1Demo.tsx` on disk untouched so it can be re-enabled with a git revert later.

---

## Files and Modules

### Backend changes (make dashboard use Phase 1 masking cascade)
1. **`ml/pipeline.py`** — Core change. Rewrite segmentation & annotated-rendering steps.
   - Import `analyze_image` from `ml.inference`.
   - Add a helper `_phase1_analysis_to_dashboard_grains()` that converts Phase 1 `grains[*].mask_polygon` → classical-style `GrainInstance` objects with `.mask` (uint8 raster), `.contour` (cv2 polygon), `.bbox`, `.centroid`, `.confidence`, stable `.grain_id`, `.is_touching`, plus NEW fields `.confidence_label` and `.segmentation_method` carried through for downstream rendering.
   - Also convert Phase 1 `foreign_matter[*]` → the `ForeignObject` tuple/schema used by the pipeline.
   - Replace `segment_grains(image_rgb, method="auto")` call with `analyze_image(image_rgb, return_masks=False, return_overlay=False, include_confidence_label=True)` + rasterize helper.
   - Add a fallback path — if `analyze_image` throws ImportError on ultralytics or returns zero grains, fall back to the OLD `segment_grains` call (emergency fallback preserved per MODEL.md cascade policy; never deleted).
   - Carry `confidence_label`, `segmentation_method`, `method`, `model_version`, processing ms into warnings or result dict.
   - Replace `_create_annotated_image_b64()` with a call into Phase 1's `_build_phase1_overlay` logic or re-use the renderer exported from `ml/postprocessing.py` to draw colored polygons keyed by confidence label + per-grain ID + segmentation method legend.

2. **`ml/inference.py`** — Minor wiring.
   - Ensure `analyze_image()` can be called from pipeline.py without re-decoding; add a `rgb_np` kwarg (already exists based on inference code) so we don't base64-round-trip within the same process.
   - Export the confidence-label thresholds as a public constant so pipeline.py and routes.py share the same numbers (currently duplicated).

3. **`ml/postprocessing.py`** — Rendering export (if not already public).
   - Ensure `_render_confidence_colored_polygons(...)` or similar is exposed as `render_phase1_overlay(image_rgb, grains, foreign_matter, include_legend=True)` so both `routes.py/_build_phase1_overlay` AND `pipeline.py` can use it without code duplication. We'll refactor `routes.py` to import the shared function too.

4. **`backend/app/api/routes.py`** — Refactor rendering to use shared function, keep the dashboard `/api/analyze` 100 % backwards compatible.
   - Extract `_build_phase1_overlay()` body into a public helper `ml.postprocessing.render_phase1_overlay` (or `ml.visualization.render_confidence_overlay`), keep only a thin wrapper in routes.py that calls the shared code.
   - No change to `/api/analyze` path signature or response schema — `AnalysisResultResponse` remains unchanged so the frontend types.ts stays compatible.

5. **`ml/segmentation.py`** — No logic changes. `GrainInstance` dataclass keeps its `.mask`, `.contour` etc. so all downstream code (`compute_grain_geometry`, defect classifiers) that consumes them stays untouched. Add OPTIONAL attributes `.confidence_label: str | None` and `.segmentation_method: str | None` to the dataclass — both default None, so old callers break zero.

### Frontend changes (temporarily remove Phase1Demo tab)
6. **`frontend/src/App.tsx`** — Remove tab but keep the component on disk:
   - Delete the `import { Phase1Demo } ...` line (L12) or comment it with a restoration note.
   - Narrow `type ActiveTab = 'dashboard';` (remove `'phase1'`) — TypeScript will flag any leftover references.
   - Delete/comment the Phase 1 `<TabButton>` (L149-L153) from the tab bar.
   - Simplify the ternary branch at L160-L304: remove the `phase1 ? Phase1Demo :` wrapper entirely; render only the dashboard children (they get the new masking transparently).
   - Keep the `ScatterChart` lucide-react import if it's still used elsewhere; grep first, then drop only if unused.

### Files NOT changed (preserved)
- `frontend/src/components/Phase1Demo.tsx` — LEFT ON DISK untouched (temporary removal only).
- `/api/phase1/*` endpoints — LEFT ALIVE (can be exercised via curl/Postman directly for debugging even though the UI tab is hidden; preserves the ability to do side-by-side comparison via API).
- `ml/rice_gate.py` / foreign-matter pipeline — untouched; Phase 1 cascade will still be gated after the rice gate passes (its own internal gate supersedes it harmlessly and fast-fails on NOT_RICE anyway).
- TypeScript `AnalysisResult`, `GrainInstance`, `AnnotatedViewer` — UNCHANGED; viewer still renders server-built base64 image.

---

## Implementation Steps (dependency order)

1. **Dataclass upgrade (safe, optional attrs)** — Update `ml/segmentation.py:GrainInstance` to add optional `confidence_label: Optional[str] = None` and `segmentation_method: Optional[str] = None`.
2. **Shared Phase 1 overlay renderer** — Move/export `render_phase1_overlay(image_rgb, phase1_grains, phase1_foreign, include_legend)` to `ml/postprocessing.py`; have `routes.py:_build_phase1_overlay` call it.
3. **Export constants from inference.py** — Publicize `CONFIDENCE_HIGH_THRESHOLD = 0.80`, `CONFIDENCE_MEDIUM_THRESHOLD = 0.50` (or existing ones) as a module-level `CONFIDENCE_THRESHOLDS` dict.
4. **Polygon rasterization adapter in pipeline.py** — Add `_phase1_grains_to_dashboard_grains(image_rgb, phase1_analysis) -> (List[GrainInstance], List[ForeignObjectTuple])` that:
   - Iterates `phase1_analysis['grains']`, rasterizes each `mask_polygon` into a `uint8 H×W` array with `cv2.fillPoly` (small per-grain bitmaps), extracts contour with `cv2.findContours`, computes bounding bbox + centroid + area from the raster if not already present.
   - Reuses stable `id` as `grain_id`.
   - Copies `confidence` float, derives `confidence_label`, copies `segmentation_method`, copies `is_touching`.
   - Sets `segmentation_quality` from a simple mapping: HIGH → EXCELLENT, MEDIUM → GOOD, LOW → POOR, label-missing → SEGMENTED so downstream code still renders its badge correctly.
5. **Swap segmentation call in `pipeline.py:analyze`** — Replace `seg_result = segment_grains(image_rgb, method="auto")` with:
   - `phase1_result = analyze_image(rgb_np=image_rgb, return_overlay=False, include_confidence_label=True)`
   - `grains_raw, foreign_raw = self._phase1_grains_to_dashboard_grains(image_rgb, phase1_result)`
   - Apply `_filter_grains_to_rice(grains_raw, rice_gate)` still — keeps the existing safety invariant.
   - On any exception during `analyze_image` or if `len(grains_raw)==0` AND ultralytics was missing → catch, log warning, append `warnings.append("Phase 1 masking unavailable; falling back to classical CV watershed.")`, and fall through to the OLD `segment_grains` call (preserves MODEL.md cascade promise).
6. **Swap annotated rendering** — Replace the `_create_annotated_image_b64` call site that uses `grains + geometries + foreign_objects` with a wrapper that builds a Phase-1 grains list: for each GrainInstance (now carrying confidence_label + mask_polygon or contour), synthesize the exact dict structure `render_phase1_overlay()` expects, then encode to base64. Reuses the Phase 1 HIGH=green / MEDIUM=yellow / LOW=orange palette and IDs; adds foreign-matter red boxes exactly like Phase1Demo does.
7. **Propagate Phase 1 metadata** — Append to `result["warnings"]` or add a `result["segmentation_info"] = {...}` field with:
   - `segmentation_method_used: phase1_result['method']`
   - `model_version: phase1_result['model_version']`
   - `processing_inference_ms: phase1_result['processing']['inference_ms']`
   - `tiling_used: phase1_result['processing'].get('tiling_used', False)`
   Frontend types already permits `warnings: string[]` and free-form fields; QualityWarningsPanel can be enhanced trivially later to pretty-print, but for now showing them as the first warning lines is sufficient to prove the pipeline ran the Phase 1 cascade.
8. **Frontend: hide Phase1Demo tab** — Apply the 4-line change to App.tsx (drop import, narrow ActiveTab type, remove TabButton, remove ternary branch).
9. **Refactor routes.py:_build_phase1_overlay** — Thin wrapper that calls into the shared renderer; removes duplicated rendering code.

---

## Dependencies and Considerations

- **`analyze_image` already accepts `rgb_np`** as per inference.py signature — if not, we'll add it without breaking the `image_source` path (default None, one of them must be set).
- **Rasterization speed:** 640×640 image × 100 grains × `cv2.fillPoly` on individual small masks is sub-millisecond total — negligible overhead compared to the YOLO inference itself.
- **Geometries and defect downstream are untouched.** This is the critical invariant: we feed them identical-shaped input (`.mask uint8 H×W`, `.contour`, `.bbox`, `.centroid`, `.grain_id`). The ONLY delta is how that mask was produced (YOLO polygon → raster vs watershed).
- **NOT_RICE fast path (`_build_no_rice_response`) stays unchanged.** The rice gate already handles that branch; `analyze_image` inside it short-circuits fast, and if rice_gate says NO_RICE we never reach the segmentation call anyway.
- **Foreign-matter pipeline coherence:** `merge_gate_foreign_objects` is called AFTER segmentation in pipeline.py. We need to ensure: Phase 1's `foreign_matter` results are MERGED with the gate/YOLO foreign-object pipeline (not double-counted). Step 4's adapter returns Phase1 FM separately, but the existing merge in pipeline should DE-DUP by box IoU if both sources flag the same FM. We'll label Phase 1 FM objects with `source: phase1_segmentation` so merge logic (which likely uses class_name, conf, bbox) can prefer the richer existing foreign-matter detector if both fire on the same box.
- **Backend `/api/phase1/analyze` remains fully functional.** This is deliberate: we still want the developer to hit it with curl for debugging. Only the UI tab disappears.

---

## Validation

1. **Static importability:** Run `python -c "from ml.pipeline import RiceQualityPipeline; from ml.inference import analyze_image; from ml.postprocessing import render_phase1_overlay"` — all imports clean.
2. **TypeScript build:** Run `cd frontend && npx tsc --noEmit` (or `npm run typecheck` if the project has one). After Step 8 (`ActiveTab = 'dashboard'` only), confirm zero remaining references to `'phase1'` literal or `Phase1Demo` identifier.
3. **Synthetic pipeline smoke test:** Run a small script that calls `RiceQualityPipeline().analyze(open(test_img, 'rb').read())` with a real rice image IF AVAILABLE, else assert:
   - If `ultralytics` importable → `result.warnings` contains substring `"segmentation_method_used":"yolov8` or falls through explicitly.
   - If NOT importable → falls back to classical CV with explicit warning present.
4. **Schema contract preserved:** Send a real request to `POST /api/analyze` via the dev server OR JSON-compare the keys of a mocked result to the AnalysisResult type; confirm all existing top-level keys of `AnalysisResult` are still present (`success`, `rice_detected`, `image`, `sample`, `calibration`, `quality`, `grains`, `foreign_matter`, `admixture`, `summary`, `standards`, `annotated_image_base64`, `warnings`, `processing_time_seconds`) — no regressions.
5. **Phase 1 visual:** Compare a before/after annotated image (old watershed renderer vs new polygon renderer) — the new one must show:
   - Per-grain colored fill keyed on HIGH/MEDIUM/LOW confidence label palette (green/yellow/orange consistent with Phase1Demo).
   - Stable `#N` ID label per grain at centroid (counts match).
   - Red FM boxes with `FM:class` labels.
6. **Phase1Demo tab invisible:** Manually open App in browser via dev server OR grep the rendered TSX (AST free-text grep) — confirm string `"Phase 1 Demo"` label does not appear in App.tsx JSX any more, but file `components/Phase1Demo.tsx` still exists on disk (we did not delete it).

---

## Risks

| Risk | Impact | Handling |
|---|---|---|
| YOLO polygon rasterization differs slightly from watershed `.mask` (1–2 px edges) | Very LOW — geometry differences will be < 1 % on length/breadth and smaller than pixel-rounding noise in the original watershed path. | Acceptable; if later analysis shows discrepancy > 2 % we can smooth polygon contours with 1 px morphological close before raster. |
| `analyze_image` cascades to 3 models when weights are missing → slower or crashes | MEDIUM — the cascade was designed in MODEL.md to always fall through to classical CV. | Fallback branch in Step 5 catches broad Exception, logs, warns, falls back to classical. |
| Foreign matter double-counted from 2 parallel sources | LOW — existing merge step runs post-segmentation. | Tag Phase1 FM objects with `provenance: 'phase1'`; existing merge de-duplicates by IoU ≥ 0.6, keeping the higher-confidence one. |
| Temporarily removing Phase1Demo might feel "permanent" to other devs. | LOW — file remains on disk; documented in plan and WORK_LOG.md entry as "hidden pending dashboard confidence label rollout." | Add a restoration comment block at App.tsx top listing the 4 exact reversion steps; add sibling WORK_LOG entry with same reversion recipe. |
| `AnnotatedViewer` click-select by grain-ID might not line up if polygon→raster grain_id sequence differs. | MEDIUM — the server renders the overlay image with IDs; the frontend drop-down uses `grains[i].id` from JSON. They MUST match order. | Enforce: rasterize adapter produces dashboard `grains` list in ID-sorted order; Phase1 overlay uses identical ID-sorted list; invariant `annotated_image grain IDs == grains list IDs` is asserted by a runtime check in dev builds or unit tests. |
