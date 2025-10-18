class ForgeCouple {

    /** The fc_mapping \<div\> */
    static container = { "t2i": undefined, "i2i": undefined };
    /** The actual \<tbody\> */
    static mappingTable = { "t2i": undefined, "i2i": undefined };

    /** The floating \<button\>s for row controls */
    static rowButtons = { "t2i": undefined, "i2i": undefined };

    /** The \<input\> for preview resolution */
    static previewResolution = { "t2i": undefined, "i2i": undefined };
    /** The \<button\> to trigger preview */
    static previewButton = { "t2i": undefined, "i2i": undefined };

    /** The ForgeCoupleDataframe class */
    static dataframe = { "t2i": undefined, "i2i": undefined };
    /** The ForgeCoupleBox class */
    static bbox = { "t2i": undefined, "i2i": undefined };

    /** The \<input\> for SendTo buttons */
    static pasteField = { "t2i": undefined, "i2i": undefined };
    /** The \<input\> for internal updates */
    static entryField = { "t2i": undefined, "i2i": undefined };

    /** The ForgeCoupleMaskHandler class */
    static maskHandler = { "t2i": undefined, "i2i": undefined };
    /** Cached ShapeToolManager references */
    static shapeTools = { "t2i": null, "i2i": null };
    /** Active shape tool per mode */
    static shapeToolSelection = { "t2i": "RECTANGLE", "i2i": "RECTANGLE" };
    /** Canvas feature toggles per mode */
    static canvasFeatures = {
        "t2i": { snapToGrid: false, gridSize: 0.1, smartGuides: false, aspectLock: false, aspectRatio: 1 },
        "i2i": { snapToGrid: false, gridSize: 0.1, smartGuides: false, aspectLock: false, aspectRatio: 1 },
    };
    /** Coordinate input references */
    static shapeCoordInputs = { "t2i": [], "i2i": [] };
    /** Hidden coordinate synchronisation fields */
    static shapeCoordField = { "t2i": undefined, "i2i": undefined };
    /** Shape metadata hidden fields */
    static shapeMetadataField = { "t2i": undefined, "i2i": undefined };
    /** Global shape property controls */
    static shapePropertyControls = { "t2i": {}, "i2i": {} };
    /** Available blend mode options */
    static availableBlendModes = [];
    /** Pending shapes waiting for mask assignment */
    static pendingShapes = { "t2i": null, "i2i": null };

    /**
     * After updating the mappings, trigger a preview
     * @param {string} mode "t2i" | "i2i"
     */
    static preview(mode) {
        let res = null;
        let w = -1;
        let h = -1;

        setTimeout(() => {
            if (mode === "t2i") {
                w = parseInt(document.getElementById("txt2img_width").querySelector("input").value);
                h = parseInt(document.getElementById("txt2img_height").querySelector("input").value);
            } else {
                const i2i_size = document.getElementById("img2img_column_size").querySelector(".tab-nav");

                if (i2i_size.children[0].classList.contains("selected")) {
                    // Resize to
                    w = parseInt(document.getElementById("img2img_width").querySelector("input").value);
                    h = parseInt(document.getElementById("img2img_height").querySelector("input").value);
                } else {
                    // Resize by
                    res = document.getElementById("img2img_scale_resolution_preview")?.querySelector(".resolution")?.textContent;
                }
            }

            if (w > 100 && h > 100)
                res = `${w}x${h}`;

            if (!res)
                return;

            this.previewResolution[mode].value = res;
            updateInput(this.previewResolution[mode]);

            this.previewButton[mode].click();
        }, 100);
    }

    static setShapeTool(mode, toolType) {
        const normalisedMode = (mode === "i2i") ? "i2i" : "t2i";
        const normalisedTool = (toolType || "RECTANGLE").toUpperCase();
        this.shapeToolSelection[normalisedMode] = normalisedTool;
        this.bbox[normalisedMode]?.setShapeTool?.(normalisedTool);
    }

    static toggleSnapToGrid(mode, enabled, gridSize) {
        const state = this.canvasFeatures[mode];
        if (!state)
            return;

        state.snapToGrid = !!enabled;
        const parsedSize = Number(gridSize);
        if (Number.isFinite(parsedSize))
            state.gridSize = parsedSize || state.gridSize;

        this.bbox[mode]?.setSnapToGrid?.(state.snapToGrid, state.gridSize);
    }

    static toggleSmartGuides(mode, enabled) {
        const state = this.canvasFeatures[mode];
        if (!state)
            return;

        state.smartGuides = !!enabled;
        this.bbox[mode]?.setSmartGuides?.(state.smartGuides);
    }

    static toggleAspectLock(mode, enabled, ratio) {
        const state = this.canvasFeatures[mode];
        if (!state)
            return;

        state.aspectLock = !!enabled;

        const coerceRatio = (value) => {
            if (typeof value === "string") {
                const trimmed = value.trim();
                if (!trimmed)
                    return null;
                if (trimmed.includes(":")) {
                    const [wRaw, hRaw] = trimmed.split(":");
                    const w = Number(wRaw);
                    const h = Number(hRaw);
                    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0)
                        return w / h;
                    return null;
                }
                const numeric = Number(trimmed);
                return Number.isFinite(numeric) && numeric > 0 ? numeric : null;
            }
            if (Number.isFinite(value) && value > 0)
                return Number(value);
            return null;
        };

        const parsedRatio = coerceRatio(ratio);
        if (parsedRatio !== null)
            state.aspectRatio = parsedRatio;

        this.bbox[mode]?.setAspectLock?.(state.aspectLock, state.aspectRatio);
    }

    static setBlendModes(modes) {
        if (!Array.isArray(modes))
            return;

        const normalized = Array.from(new Set(
            modes
                .map((mode) => (typeof mode === "string" ? mode.trim() : String(mode)))
                .filter((mode) => mode.length > 0)
                .map((mode) => mode.toUpperCase())
        ));

        if (normalized.length)
            this.availableBlendModes = normalized;
    }

    static getBlendModes() {
        if (Array.isArray(this.availableBlendModes) && this.availableBlendModes.length)
            return this.availableBlendModes;
        return ["NORMAL", "MULTIPLY", "OVERLAY"];
    }

    static updateShapeCoords(mode, x, y, w, h) {
        this.bbox[mode]?.updateShapeCoords?.(x, y, w, h);
    }

    static updateShapeMetadata(mode, metadata) {
        this.bbox[mode]?.updateShapeMetadata?.(metadata);
    }

    static syncShapeMetadata(mode, shapeData) {
        const handler = this.maskHandler[mode];
        const hasSelection = handler?.hasSelection?.()
            ?? (typeof handler?.getSelectedIndex === "function" ? handler.getSelectedIndex() >= 0 : false);

        let effectiveShape = shapeData;

        if (shapeData && !hasSelection) {
            this.pendingShapes[mode] = shapeData;
            handler?.setPendingIndicator?.(true);
        } else if (hasSelection) {
            if (!shapeData && this.pendingShapes[mode]) {
                effectiveShape = this.pendingShapes[mode];
                this.pendingShapes[mode] = null;
            } else if (shapeData) {
                this.pendingShapes[mode] = null;
            }

            handler?.setPendingIndicator?.(Boolean(this.pendingShapes[mode]));
            if (handler?.updateSelectedShape)
                handler.updateSelectedShape(effectiveShape ?? null);
        } else {
            handler?.setPendingIndicator?.(Boolean(this.pendingShapes[mode]));
        }

        const displayShape = effectiveShape
            ?? (hasSelection
                ? handler?.getSelectedShape?.() ?? null
                : this.pendingShapes[mode]);

        this._updateGlobalShapeControls(mode, displayShape);
        const coords = this._shapeToCoords(displayShape);
        this._setCoordinateInputsEnabled(mode, Boolean(coords));
        this._updateCoordinateInputs(mode, coords);
    }

    static _shapeToCoords(shape) {
        if (!shape || typeof shape !== "object")
            return null;

        const clamp01 = (value) => Math.min(Math.max(value, 0), 1);

        if (shape.shape_type === "RECTANGLE") {
            const { x1, y1, x2, y2 } = shape.parameters || {};
            if ([x1, y1, x2, y2].some((v) => typeof v !== "number"))
                return null;
            return {
                x: clamp01(x1),
                y: clamp01(y1),
                w: clamp01(x2 - x1),
                h: clamp01(y2 - y1),
            };
        }

        if (shape.shape_type === "ELLIPSE") {
            const { cx, cy, rx, ry } = shape.parameters || {};
            if ([cx, cy, rx, ry].some((v) => typeof v !== "number"))
                return null;
            return {
                x: clamp01(cx),
                y: clamp01(cy),
                w: clamp01(rx * 2),
                h: clamp01(ry * 2),
            };
        }

        const bounds = this.#shapeBounds(shape);
        if (!bounds)
            return null;

        return {
            x: clamp01(bounds.x1),
            y: clamp01(bounds.y1),
            w: clamp01(bounds.x2 - bounds.x1),
            h: clamp01(bounds.y2 - bounds.y1),
        };
    }

    static _updateCoordinateInputs(mode, coords) {
        const inputs = this.shapeCoordInputs[mode];
        if (!inputs || inputs.length < 4)
            return;

        if (!coords) {
            inputs.forEach((input) => {
                if (!input)
                    return;
                input.value = "";
                input.disabled = true;
            });
            const field = this.shapeCoordField[mode];
            if (field) {
                field.value = "";
                if (typeof updateInput === "function")
                    updateInput(field);
            }
            return;
        }

        inputs.forEach((input) => {
            if (input)
                input.disabled = false;
        });

        const values = [coords.x, coords.y, coords.w, coords.h];
        inputs.forEach((input, idx) => {
            if (!input)
                return;
            const value = values[idx];
            input.value = Number.isFinite(value) ? value.toFixed(2) : "";
        });

        const field = this.shapeCoordField[mode];
        if (field) {
            field.value = JSON.stringify(coords);
            if (typeof updateInput === "function")
                updateInput(field);
        }
    }

    static _setCoordinateInputsEnabled(mode, enabled) {
        const inputs = this.shapeCoordInputs[mode];
        if (!inputs || !inputs.length)
            return;
        inputs.forEach((input) => {
            if (input)
                input.disabled = !enabled;
        });
    }

    static _collectShapeDefaults(mode) {
        const controls = this.shapePropertyControls[mode] || {};
        const edges = controls.featherEdges || {};
        const edgeValues = {};
        ["top", "right", "bottom", "left"].forEach((key) => {
            const input = edges[key];
            if (!input)
                return;
            const value = this.#clamp01(input.value);
            if (Number.isFinite(value))
                edgeValues[key] = value;
        });
        const hasEdges = Object.keys(edgeValues).length > 0;
        return {
            blend_mode: controls.blend?.value || "NORMAL",
            feather: parseFloat(controls.feather?.value ?? "0") || 0,
            hardness: parseFloat(controls.hardness?.value ?? "1") || 1,
            z_order: parseInt(controls.z?.value ?? "0", 10) || 0,
            ...(hasEdges ? { feather_edges: edgeValues } : {}),
        };
    }

    static _updateGlobalShapeControls(mode, shape) {
        const controls = this.shapePropertyControls[mode] || {};
        if (!shape || typeof shape !== "object") {
            if (controls.featherEdges) {
                Object.values(controls.featherEdges).forEach((input) => {
                    if (!input)
                        return;
                    input.value = "";
                    if (input.dataset)
                        input.dataset.value = "";
                });
            }
            return;
        }

        if (controls.blend && shape.blend_mode)
            controls.blend.value = shape.blend_mode;

        if (controls.feather && Number.isFinite(shape.feather)) {
            controls.feather.value = shape.feather;
            controls.feather.dataset.value = shape.feather.toFixed(2);
        }

        if (controls.featherEdges) {
            const shapeEdges = shape.feather_edges || {};
            ["top", "right", "bottom", "left"].forEach((key) => {
                const input = controls.featherEdges[key];
                if (!input)
                    return;
                const value = Number.isFinite(shapeEdges[key])
                    ? this.#clamp01(shapeEdges[key])
                    : Number.isFinite(shape.feather)
                        ? this.#clamp01(shape.feather)
                        : 0;
                input.value = value;
                if (input.dataset)
                    input.dataset.value = value.toFixed(2);
            });
        }

        if (controls.hardness && Number.isFinite(shape.hardness)) {
            controls.hardness.value = shape.hardness;
            controls.hardness.dataset.value = shape.hardness.toFixed(2);
        }

        if (controls.z && Number.isFinite(shape.z_order))
            controls.z.value = shape.z_order;
    }

    static ensureMaskSelection(mode) {
        const handler = this.maskHandler[mode];
        if (!handler)
            return false;

        if (typeof handler.hasSelection === "function" && handler.hasSelection())
            return true;

        if (typeof handler.ensureSelection === "function") {
            const index = handler.ensureSelection();
            return Number.isInteger(index) && index >= 0;
        }

        return false;
    }

    static applyPendingShape(mode) {
        const pending = this.pendingShapes[mode];
        if (!pending)
            return;

        const hasSelection = this.ensureMaskSelection(mode);
        if (!hasSelection) {
            this.pendingShapes[mode] = pending;
            this.maskHandler[mode]?.setPendingIndicator?.(true);
            return;
        }

        this.pendingShapes[mode] = null;
        this.syncShapeMetadata(mode, pending);
    }

    /**
     * Update the color of the rows based on the order and selection
     * @param {string} mode "t2i" | "i2i"
     */
    static updateColors(mode) {
        const [color, row] = this.dataframe[mode].updateColors();

        if (color) {
            this.bbox[mode].showBox(color, row);
            return row;
        }
        else {
            this.bbox[mode].hideBox();
            this.rowButtons[mode].style.display = "none";
            return null;
        }
    }

    /**
     * When using SendTo buttons, refresh the table
     * @param {string} mode "t2i" | "i2i"
     */
    static onPaste(mode) {
        const infotext = this.pasteField[mode].value;
        if (!infotext.trim())
            return;

        let entries = [];
        try {
            const parsed = JSON.parse(infotext);
            if (Array.isArray(parsed))
                entries = parsed;
        } catch {
            console.warn("ForgeCouple: Failed to parse mapping paste payload");
            return;
        }

        const mappings = [];
        const shapes = [];
        entries.forEach((entry) => {
            if (Array.isArray(entry)) {
                mappings.push(entry);
                shapes.push(null);
                return;
            }
            if (entry && typeof entry === "object") {
                const mapping = Array.isArray(entry.mapping)
                    ? entry.mapping
                    : (Array.isArray(entry.values) ? entry.values : null);
                if (Array.isArray(mapping)) {
                    mappings.push(mapping);
                    shapes.push(entry.shape ?? null);
                }
                return;
            }
        });

        this.dataframe[mode].onPaste(mappings);

        const handler = this.maskHandler[mode];
        if (handler && shapes.some((shape) => !!shape)) {
            shapes.forEach((shape, index) => {
                if (shape)
                    handler.setShapeData(index, shape);
            });
        }

        this.preview(mode);

        this.pasteField[mode].value = "";
        updateInput(this.pasteField[mode]);
    }

    /**
     * When clicking on a row, update the index
     * @param {string} mode "t2i" | "i2i"
     */
    static onSelect(mode) {
        const cell = this.updateColors(mode);

        if (cell) {
            const bounding = cell.querySelector("td").getBoundingClientRect();
            const bounding_container = this.container[mode].getBoundingClientRect();
            this.rowButtons[mode].style.top = `calc(${bounding.top - bounding_container.top}px - 1.5em)`;
            this.rowButtons[mode].style.display = "block";
        } else
            this.rowButtons[mode].style.display = "none";
    }

    /**
     * When editing the mapping, update the internal JSON
     * @param {string} mode "t2i" | "i2i"
     */
    static onEntry(mode) {
        const rows = this.mappingTable[mode].querySelectorAll("tr");

        const vals = Array.from(rows, (row) => {
            const cells = Array.from(row.querySelectorAll("td"));
            if (!cells.length)
                return null;

            const firstCell = cells[0];
            const shape = this.#readShapeFromCell(firstCell);

            const fallbackWeight = Number.isFinite(shape?.weight)
                ? Number(shape.weight)
                : 1;
            const mapping = shape
                ? this.#boundsToMapping(this.#shapeBounds(shape), this.#readWeightFromCells(cells, fallbackWeight))
                : this.#extractNumericMapping(cells, fallbackWeight);

            if (!Array.isArray(mapping) || mapping.length !== 5)
                return null;

            const entry = {
                type: shape ? "shape" : "mapping",
                mapping,
            };

            if (shape)
                entry.shape = shape;

            return entry;
        }).filter((entry) => entry !== null);

        const json = JSON.stringify(vals);
        this.entryField[mode].value = json;
        updateInput(this.entryField[mode]);
    }

    static #readWeightFromCells(cells, fallback) {
        if (!Array.isArray(cells) || cells.length < 5)
            return fallback;
        const raw = parseFloat(cells[4].textContent);
        return Number.isFinite(raw) ? raw : fallback;
    }

    static #readShapeFromCell(cell) {
        if (!cell)
            return null;
        const raw = cell.dataset?.shape ?? "";
        if (!raw || !raw.trim().startsWith("{"))
            return null;
        try {
            return JSON.parse(raw);
        } catch {
            return null;
        }
    }

    static #extractNumericMapping(cells, fallbackWeight = 1) {
        const mapping = [];
        for (let i = 0; i < 5; i++) {
            const cell = cells[i];
            if (!cell) {
                mapping.push(i === 4 ? fallbackWeight : 0);
                continue;
            }
            const value = parseFloat(cell.textContent);
            if (Number.isFinite(value))
                mapping.push(value);
            else
                mapping.push(i === 4 ? fallbackWeight : 0);
        }

        if (mapping.length < 5)
            mapping.push(fallbackWeight);

        mapping[0] = this.#clamp01(mapping[0]);
        mapping[1] = this.#clamp01(mapping[1]);
        mapping[2] = this.#clamp01(mapping[2]);
        mapping[3] = this.#clamp01(mapping[3]);
        mapping[4] = Number.isFinite(mapping[4]) ? mapping[4] : fallbackWeight;

        if (mapping[0] > mapping[1])
            [mapping[0], mapping[1]] = [mapping[1], mapping[0]];
        if (mapping[2] > mapping[3])
            [mapping[2], mapping[3]] = [mapping[3], mapping[2]];

        return mapping.slice(0, 5);
    }

    static #boundsToMapping(bounds, weight) {
        if (!bounds) {
            return [
                0,
                0,
                0,
                0,
                Number.isFinite(weight) ? weight : 1,
            ];
        }

        const x1 = this.#clamp01(bounds.x1);
        const x2 = this.#clamp01(bounds.x2);
        const y1 = this.#clamp01(bounds.y1);
        const y2 = this.#clamp01(bounds.y2);

        return [
            Math.min(x1, x2),
            Math.max(x1, x2),
            Math.min(y1, y2),
            Math.max(y1, y2),
            Number.isFinite(weight) ? weight : 1,
        ];
    }

    static #shapeBounds(shape) {
        if (!shape || typeof shape !== "object")
            return null;

        const type = (shape.shape_type || "").toUpperCase();
        const params = shape.parameters || {};

        switch (type) {
            case "RECTANGLE": {
                const x1 = this.#clamp01(Number(params.x1));
                const y1 = this.#clamp01(Number(params.y1));
                const x2 = this.#clamp01(Number(params.x2));
                const y2 = this.#clamp01(Number(params.y2));
                return { x1: Math.min(x1, x2), x2: Math.max(x1, x2), y1: Math.min(y1, y2), y2: Math.max(y1, y2) };
            }
            case "ELLIPSE": {
                const cx = this.#clamp01(Number(params.cx));
                const cy = this.#clamp01(Number(params.cy));
                const rx = Math.max(0, Number(params.rx));
                const ry = Math.max(0, Number(params.ry));
                return {
                    x1: this.#clamp01(cx - rx),
                    x2: this.#clamp01(cx + rx),
                    y1: this.#clamp01(cy - ry),
                    y2: this.#clamp01(cy + ry),
                };
            }
            case "POLYGON": {
                const points = Array.isArray(params.points) ? params.points : [];
                if (points.length < 3)
                    return null;
                let minX = Infinity;
                let maxX = -Infinity;
                let minY = Infinity;
                let maxY = -Infinity;
                points.forEach(([px, py]) => {
                    const x = this.#clamp01(Number(px));
                    const y = this.#clamp01(Number(py));
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                });
                if (!Number.isFinite(minX) || !Number.isFinite(minY))
                    return null;
                return { x1: minX, x2: maxX, y1: minY, y2: maxY };
            }
            case "BEZIER": {
                const controls = Array.isArray(params.control_points) ? params.control_points : [];
                const bezierBounds = this.#bezierBounds(controls);
                return bezierBounds;
            }
            default:
                return null;
        }
    }

    static #bezierBounds(controlPoints) {
        if (!Array.isArray(controlPoints) || controlPoints.length < 4)
            return null;

        const segmentCount = Math.floor((controlPoints.length - 1) / 3);
        if (segmentCount <= 0)
            return null;

        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        const sampleCount = 32;

        for (let segment = 0; segment < segmentCount; segment += 1) {
            const base = segment * 3;
            const p0 = controlPoints[base];
            const p1 = controlPoints[base + 1];
            const p2 = controlPoints[base + 2];
            const p3 = controlPoints[base + 3];

            for (let i = 0; i <= sampleCount; i += 1) {
                if (segment > 0 && i === 0)
                    continue;
                const t = i / sampleCount;
                const omt = 1 - t;
                const omt2 = omt * omt;
                const t2 = t * t;

                const x = (
                    omt2 * omt * Number(p0?.[0] ?? 0)
                    + 3 * omt2 * t * Number(p1?.[0] ?? 0)
                    + 3 * omt * t2 * Number(p2?.[0] ?? 0)
                    + t2 * t * Number(p3?.[0] ?? 0)
                );
                const y = (
                    omt2 * omt * Number(p0?.[1] ?? 0)
                    + 3 * omt2 * t * Number(p1?.[1] ?? 0)
                    + 3 * omt * t2 * Number(p2?.[1] ?? 0)
                    + t2 * t * Number(p3?.[1] ?? 0)
                );

                const clampedX = this.#clamp01(x);
                const clampedY = this.#clamp01(y);
                minX = Math.min(minX, clampedX);
                maxX = Math.max(maxX, clampedX);
                minY = Math.min(minY, clampedY);
                maxY = Math.max(maxY, clampedY);
            }
        }

        if (!Number.isFinite(minX) || !Number.isFinite(minY))
            return null;

        return { x1: minX, x2: maxX, y1: minY, y2: maxY };
    }

    static #clamp01(value) {
        const num = Number(value);
        if (!Number.isFinite(num))
            return 0;
        return Math.min(Math.max(num, 0), 1);
    }

    /**
     * Link the buttons related to the mapping
     * @param {Element} ex
     * @param {string} mode "t2i" | "i2i"
     */
    static #registerButtons(ex, mode) {
        ex.querySelector(".fc_reset_btn").onclick = () => { this.dataframe[mode].reset(); };
        ex.querySelector("#fc_up_btn").onclick = (e) => { this.dataframe[mode].newRowAbove(e.shiftKey); };
        ex.querySelector("#fc_dn_btn").onclick = (e) => { this.dataframe[mode].newRowBelow(e.shiftKey); };
        ex.querySelector("#fc_del_btn").onclick = (e) => { this.dataframe[mode].deleteRow(e.shiftKey); };
    }

    /** Hook some elements to automatically refresh the resolution */
    static #registerResolutionHandles() {

        [["txt2img", "t2i"], ["img2img", "i2i"]].forEach(([tab, mode]) => {
            const btns = document.getElementById(`${tab}_dimensions_row`)?.querySelectorAll("button");
            if (btns != null)
                btns.forEach((btn) => { btn.onclick = () => { this.preview(mode); } });
        });

        const i2i_size_btns = document.getElementById("img2img_column_size").querySelector(".tab-nav");
        i2i_size_btns.addEventListener("click", () => { this.preview("i2i"); });

        const tabs = document.getElementById('tabs').querySelector('.tab-nav');
        tabs.addEventListener("click", () => {
            if (tabs.children[0].classList.contains("selected"))
                this.preview("t2i");
            if (tabs.children[1].classList.contains("selected"))
                this.preview("i2i");
        });

    }

    /**
     * Remove Gradio Image Junks...
     * @param {string} mode "t2i" | "i2i"
     */
    static hideButtons(mode) {
        this.maskHandler[mode].hideButtons();
    }

    /**
     * After editing masks, refresh the preview rows
     * @param {string} mode "t2i" | "i2i"
     */
    static populateMasks(mode) {
        this.maskHandler[mode].generatePreview();
    }

    /**
     * After changing Global Effect, re-sync the prompts
     * @param {string} mode "t2i" | "i2i"
     */
    static onBackgroundChange(mode) {
        this.maskHandler[mode].syncPrompts();
    }

    static setup() {
        ["t2i", "i2i"].forEach((mode) => {
            const ex = document.getElementById(`forge_couple_${mode}`);
            const mapping_btns = ex.querySelector(".fc_mapping_btns");

            this.container[mode] = ex.querySelector(".fc_mapping");
            this.container[mode].appendChild(mapping_btns);

            const separator = ex.querySelector(".fc_separator").querySelector("input");
            const promptField = document.getElementById(`${mode === "t2i" ? "txt" : "img"}2img_prompt`).querySelector("textarea");

            this.dataframe[mode] = new ForgeCoupleDataframe(this.container[mode], mode, separator);

            this.mappingTable[mode] = this.container[mode].querySelector("tbody");

            this.rowButtons[mode] = ex.querySelector(".fc_row_btns");
            this.rowButtons[mode].style.display = "none";

            this.rowButtons[mode].querySelectorAll("button").forEach((btn) => {
                btn.setAttribute('style', 'margin: auto !important');
            });

            this.container[mode].appendChild(this.rowButtons[mode]);

            this.previewResolution[mode] = ex.querySelector(".fc_preview_res").querySelector("input");
            this.previewButton[mode] = ex.querySelector(".fc_preview");

            const previewHost = ex.querySelector(".fc_preview_img");
            const preview_img = previewHost?.querySelector("img") ?? ex.querySelector("img");
            if (!preview_img) {
                console.warn("ForgeCouple: mapping preview image element not found for mode", mode);
                return;
            }
            preview_img.ondragstart = (e) => { e.preventDefault(); return false; };
            const previewContainer = preview_img.closest(".image-container") ?? preview_img.parentElement;
            if (previewContainer)
                previewContainer.style.overflow = "visible";

            // Dispose of existing instance before creating a new one
            if (this.bbox[mode]?.dispose) {
                this.bbox[mode].dispose();
            }

            this.bbox[mode] = new ForgeCoupleBox(preview_img, mode);
            this.shapeTools[mode] = this.bbox[mode]?.getShapeToolManager?.() ?? null;

            const bg_btns = ex.querySelector(".fc_bg_btns");
            if (bg_btns && previewContainer)
                previewContainer.appendChild(bg_btns);

            ForgeCoupleImageLoader.setup(preview_img, bg_btns ? bg_btns.querySelectorAll("button") : [])

            this.pasteField[mode] = ex.querySelector(".fc_paste_field").querySelector("textarea");
            this.entryField[mode] = ex.querySelector(".fc_entry_field").querySelector("textarea");

            const weightsField = ex.querySelector(".fc_msk_weights").querySelector("textarea");
            const shapeFieldContainer = ex.querySelector(".fc_shape_metadata");
            const shapeField = shapeFieldContainer ? shapeFieldContainer.querySelector("textarea") : null;
            const operationField = ex.querySelector(".fc_msk_op").querySelector("textarea");
            const operationButton = ex.querySelector(".fc_msk_op_btn");
            const loadButton = ex.querySelector(".fc_msk_io").querySelectorAll("button")[1];
            const propContainer = ex.querySelector(".fc_shape_properties");
            if (propContainer?.dataset?.blendModes) {
                try {
                    const blendModes = JSON.parse(propContainer.dataset.blendModes);
                    this.setBlendModes(blendModes);
                } catch (error) {
                    console.warn("ForgeCouple: failed to parse blend mode dataset", error);
                }
            }

            this.shapeMetadataField[mode] = shapeField;

            this.maskHandler[mode] = new ForgeCoupleMaskHandler(
                ex.querySelector(".fc_msk"),
                ex.querySelector(".fc_msk_gal"),
                ex.querySelector(".fc_masks"),
                separator,
                ex.querySelector(".fc_global_effect"),
                promptField,
                weightsField,
                shapeField,
                operationField,
                operationButton,
                loadButton,
                mode
            );

            const coordRow = ex.querySelector(".fc_coord_inputs");
            const coordInputs = coordRow ? Array.from(coordRow.querySelectorAll("input")) : [];
            this.shapeCoordInputs[mode] = coordInputs;
            this._setCoordinateInputsEnabled(mode, false);

            this.shapeCoordField[mode] = ex.querySelector(".fc_active_shape_coords")?.querySelector("textarea") ?? null;

            if (coordInputs.length) {
                coordInputs.forEach((_input, idx, arr) => {
                    _input.addEventListener("input", () => {
                        const rawValues = arr.map((inp) => Number(inp.value));
                        const clamp01 = (value) => {
                            const num = Number.isFinite(value) ? value : 0;
                            return Math.min(Math.max(num, 0), 1);
                        };

                        let nx = clamp01(rawValues[0]);
                        let ny = clamp01(rawValues[1]);
                        const widthInput = clamp01(rawValues[2]);
                        const heightInput = clamp01(rawValues[3]);

                        const maxX = clamp01(nx + Math.max(widthInput, 0));
                        const maxY = clamp01(ny + Math.max(heightInput, 0));

                        const width = Math.max(maxX - nx, 0);
                        const height = Math.max(maxY - ny, 0);

                        const clamped = [nx, ny, width, height];

                        this.updateShapeCoords(mode, nx, ny, width, height);

                        arr.forEach((input, index) => {
                            if (input)
                                input.value = clamped[index].toFixed(2);
                        });

                        const field = this.shapeCoordField[mode];
                        if (field) {
                            field.value = JSON.stringify({
                                x: clamped[0],
                                y: clamped[1],
                                w: clamped[2],
                                h: clamped[3],
                            });
                            if (typeof updateInput === "function")
                                updateInput(field);
                        }
                    });
                });
            }

            const shapeToolButtons = ex.querySelectorAll(".fc_shape_tools button[data-tool]");
            shapeToolButtons.forEach((btn) => {
                btn.addEventListener("click", () => {
                    shapeToolButtons.forEach((b) => b.classList.remove("active"));
                    btn.classList.add("active");
                    this.setShapeTool(mode, btn.dataset.tool || "RECTANGLE");
                });
            });
            if (shapeToolButtons.length) {
                const activeTool = this.shapeToolSelection[mode];
                shapeToolButtons.forEach((btn) => {
                    if ((btn.dataset.tool || "").toUpperCase() === activeTool)
                        btn.classList.add("active");
                });
                this.setShapeTool(mode, activeTool);
            } else
                this.setShapeTool(mode, this.shapeToolSelection[mode]);

            const featureRow = ex.querySelector(".fc_canvas_features");
            if (featureRow) {
                const gridToggle = featureRow.querySelector("input.fc_snap_grid");
                const gridSize = featureRow.querySelector("input.fc_grid_size");
                const guidesToggle = featureRow.querySelector("input.fc_smart_guides");
                const aspectToggle = featureRow.querySelector("input.fc_aspect_lock");
                const aspectRatio = featureRow.querySelector("input.fc_aspect_ratio");

                if (gridToggle) {
                    gridToggle.addEventListener("change", () => {
                        const size = gridSize ? parseFloat(gridSize.value) : this.canvasFeatures[mode].gridSize;
                        this.toggleSnapToGrid(mode, gridToggle.checked, size);
                    });
                }
                if (gridSize) {
                    gridSize.addEventListener("change", () => {
                        const enabled = gridToggle ? gridToggle.checked : true;
                        this.toggleSnapToGrid(mode, enabled, parseFloat(gridSize.value));
                    });
                }
                if (guidesToggle) {
                    guidesToggle.addEventListener("change", () => {
                        this.toggleSmartGuides(mode, guidesToggle.checked);
                    });
                }
                if (aspectToggle) {
                    aspectToggle.addEventListener("change", () => {
                        const ratioValue = aspectRatio ? aspectRatio.value : this.canvasFeatures[mode].aspectRatio;
                        this.toggleAspectLock(mode, aspectToggle.checked, ratioValue);
                    });
                }
                if (aspectRatio) {
                    aspectRatio.addEventListener("change", () => {
                        const enabled = aspectToggle ? aspectToggle.checked : true;
                        this.toggleAspectLock(mode, enabled, aspectRatio.value);
                    });
                }

                const state = this.canvasFeatures[mode];
                if (gridToggle) gridToggle.checked = state.snapToGrid;
                if (gridSize) gridSize.value = state.gridSize;
                if (guidesToggle) guidesToggle.checked = state.smartGuides;
                if (aspectToggle) aspectToggle.checked = state.aspectLock;
                if (aspectRatio) aspectRatio.value = Number(state.aspectRatio).toFixed(2);

                this.toggleSnapToGrid(mode, state.snapToGrid, state.gridSize);
                this.toggleSmartGuides(mode, state.smartGuides);
                this.toggleAspectLock(mode, state.aspectLock, state.aspectRatio);
            }

            if (propContainer) {
                const propBlend = propContainer.querySelector("select.fc_prop_blend");
                const propFeather = propContainer.querySelector("input.fc_prop_feather");
                const propHardness = propContainer.querySelector("input.fc_prop_hardness");
                const propZ = propContainer.querySelector("input.fc_prop_z");
                const propFeatherEdges = {
                    top: propContainer.querySelector("input.fc_prop_feather_edge_top"),
                    right: propContainer.querySelector("input.fc_prop_feather_edge_right"),
                    bottom: propContainer.querySelector("input.fc_prop_feather_edge_bottom"),
                    left: propContainer.querySelector("input.fc_prop_feather_edge_left"),
                };

                this.shapePropertyControls[mode] = {
                    blend: propBlend,
                    feather: propFeather,
                    hardness: propHardness,
                    z: propZ,
                    featherEdges: propFeatherEdges,
                };

                if (propFeather)
                    propFeather.dataset.value = Number.parseFloat(propFeather.value || "0").toFixed(2);
                if (propHardness)
                    propHardness.dataset.value = Number.parseFloat(propHardness.value || "1").toFixed(2);
                Object.values(propFeatherEdges).forEach((input) => {
                    if (!input)
                        return;
                    input.dataset.value = this.#clamp01(input.value).toFixed(2);
                });

                const applyDefaults = () => {
                    this.bbox[mode]?.setShapeDefaults?.(this._collectShapeDefaults(mode));
                };

                if (propBlend) {
                    propBlend.addEventListener("change", () => {
                        this.updateShapeMetadata(mode, { blend_mode: propBlend.value });
                        applyDefaults();
                    });
                }

                if (propFeather) {
                    propFeather.addEventListener("input", () => {
                        const value = this.#clamp01(propFeather.value);
                        propFeather.value = value;
                        propFeather.dataset.value = value.toFixed(2);
                        this.updateShapeMetadata(mode, { feather: value });
                        applyDefaults();
                    });
                }

                if (propHardness) {
                    propHardness.addEventListener("input", () => {
                        const value = this.#clamp01(propHardness.value);
                        propHardness.value = value;
                        propHardness.dataset.value = value.toFixed(2);
                        this.updateShapeMetadata(mode, { hardness: value });
                        applyDefaults();
                    });
                }

                if (propZ) {
                    propZ.addEventListener("change", () => {
                        this.updateShapeMetadata(mode, { z_order: parseInt(propZ.value, 10) || 0 });
                        applyDefaults();
                    });
                }

                Object.entries(propFeatherEdges).forEach(([edge, input]) => {
                    if (!input)
                        return;
                    input.addEventListener("input", () => {
                        const raw = parseFloat(input.value);
                        const value = this.#clamp01(Number.isFinite(raw) ? raw : 0);
                        input.value = value;
                        input.dataset.value = value.toFixed(2);
                        this.updateShapeMetadata(mode, { feather_edges: { [edge]: value } });
                        applyDefaults();
                    });
                });

                applyDefaults();
            } else {
                this.shapePropertyControls[mode] = { featherEdges: {} };
                this.bbox[mode]?.setShapeDefaults?.(this._collectShapeDefaults(mode));
            }

            this.#registerButtons(ex, mode);
            ForgeCoupleObserver.observe(
                mode,
                promptField,
                () => {
                    this.dataframe[mode].syncPrompts();
                    this.maskHandler[mode].syncPrompts();
                }
            );
        });

        this.#registerResolutionHandles();
    }

}

onUiLoaded(() => { ForgeCouple.setup(); })
