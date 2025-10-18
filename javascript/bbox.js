class ForgeCoupleBox {

    /** "t2i" | "i2i" */
    #mode = undefined;

    /** The background image */
    #img = undefined;

    /** The bounding of the image */
    get #imgBound() { return this.#img.getBoundingClientRect(); }

    /** The bounding box currently selected */
    #box = undefined;

    /** Canvas overlay for advanced shape previews */
    #overlay = undefined;

    /** 2D context for overlay canvas */
    #overlayCtx = undefined;

    /** Keyboard handler reference */
    #keyHandler = null;

    /** Resize handler reference */
    #resizeHandler = null;

    /** Click handler reference (mousedown on image) */
    #clickHandler = null;

    /** Hover handler reference (mousemove on tab) */
    #hoverHandler = null;

    /** Up handlers references (mouseup and mouseleave on tab) */
    #upHandlers = { mouseup: null, mouseleave: null };

    /** Tab element reference */
    #tab = null;

    /** Shape tool manager */
    #shapeToolManager = undefined;

    /** Canvas guides helper */
    #canvasGuides = undefined;

    /** Shape currently being edited */
    #currentShape = null;

    /** Active shape tool identifier */
    #activeShapeTool = "LEGACY";

    /** Snap to grid toggle */
    #snapToGrid = false;

    /** Grid size (fraction of canvas width / height) */
    #gridSize = 0.1;

    /** Smart guides toggle */
    #smartGuides = false;

    /** Aspect lock toggle */
    #aspectLock = false;

    /** Aspect ratio (width / height) */
    #aspectRatio = 1;

    /** Default metadata applied to new shapes */
    #shapeDefaults = {
        z_order: 0,
        blend_mode: "NORMAL",
        feather: 0,
        hardness: 1,
        weight: 1,
        feather_edges: null,
    };

    /** Display color for current shape overlay */
    #currentColor = "rgba(0,255,255,0.8)";

    /** The bounding of the container */
    get #boxBound() { return this.#box.getBoundingClientRect(); }

    /** Booleans representing whether each edge is used for resizing */
    #resize = {};
    /** Delta between the image and the container, when the image is not a square */
    #padding = {};
    /** The pixel distance to the window edge */
    #margin = {};
    /** The step size (1%) for moving and resizing */
    #step = {};

    /** Currently selected row */
    #cachedRow = null;

    /** @param {Element} image @param {string} mode */
    constructor(image, mode) {
        const box = document.createElement("div");
        box.classList.add(`fc_bbox`);
        box.style.display = "none";

        this.#mode = mode;
        this.#img = image;
        this.#box = box;

        this.#setupShapeTools(image.parentElement);

        this.#tab = document.getElementById((mode === "t2i") ? "tab_txt2img" : "tab_img2img");
        this.#registerClick(this.#tab);
        this.#registerHover(this.#tab);
        this.#registerUp(this.#tab);

        image.parentElement.appendChild(box);
    }

    #registerClick(tab) {
        this.#clickHandler = (e) => {
            if (e.button !== 0)
                return;

            if (this.#isShapeToolActive()) {
                this.#handleShapeMouseDown(e);
                return;
            }

            this.isValid = (this.#img.style.cursor != "default");
            this.isResize = (this.#resize.L || this.#resize.R || this.#resize.T || this.#resize.B);

            if (this.isValid) {
                this.#initCoord();

                this.init = {
                    X: e.clientX,
                    Y: e.clientY,
                    left: this.#style2value(this.#box.style.left),
                    top: this.#style2value(this.#box.style.top)
                };

                tab.style.cursor = this.#img.style.cursor;
            }
        };
        this.#img.addEventListener("mousedown", this.#clickHandler);
    }

    #registerHover(tab) {
        this.#hoverHandler = (e) => {

            if (this.#isShapeToolActive()) {
                this.#handleShapeMouseMove(e);
                return;
            }

            if (!this.isValid) {
                this.#checkMouse(e.clientX, e.clientY);
                return;
            }

            if (this.isResize)
                this.#resizeLogic(e.clientX, e.clientY)
            else
                this.#offsetLogic(e.clientX, e.clientY)

        };
        tab.addEventListener("mousemove", this.#hoverHandler);
    }

    #registerUp(tab) {
        const createUpHandler = (ev) => (e) => {
            if (this.#isShapeToolActive()) {
                if (ev === "mouseup" && e.button === 0) {
                    this.#handleShapeMouseUp(e);
                } else if (ev === "mouseleave") {
                    this.#handleShapeMouseUp();
                }
                return;
            }

            if (!this.isValid || (ev === "mouseup" && e.button !== 0))
                return;

            const vals = this.#styleToMapping();
            const cells = this.#cachedRow.querySelectorAll("td");

            if (vals && !Array.isArray(vals)) {
                const json = JSON.stringify(vals);
                if (cells.length) {
                    cells[0].textContent = json;
                    cells[0].dataset.shape = json;
                    for (let i = 1; i < cells.length; i++) {
                        cells[i].textContent = "";
                    }
                }
            } else if (Array.isArray(vals)) {
                for (let i = 0; i < vals.length && i < cells.length; i++)
                    cells[i].textContent = Number(Math.max(0.0, vals[i])).toFixed(2);
            }

            this.isValid = false;
            tab.style.cursor = "unset";

            ForgeCouple.onEntry(this.#mode);
        };

        this.#upHandlers.mouseup = createUpHandler("mouseup");
        this.#upHandlers.mouseleave = createUpHandler("mouseleave");

        tab.addEventListener("mouseup", this.#upHandlers.mouseup);
        tab.addEventListener("mouseleave", this.#upHandlers.mouseleave);
    }


    /** @param {number} mouseX @param {number} mouseY */
    #resizeLogic(mouseX, mouseY) {
        const FC_minimumSize = 32;

        if (this.#resize.R) {
            const W = this.#clampMinMax(mouseX - this.#boxBound.left, FC_minimumSize,
                this.#imgBound.right + this.#padding.left - this.#margin.left - this.init.left
            );

            this.#box.style.width = `${this.#step.w * Math.round(W / this.#step.w)}px`;
        } else if (this.#resize.L) {
            const rightEdge = this.#style2value(this.#box.style.left) + this.#style2value(this.#box.style.width);
            const W = this.#clampMinMax(this.#boxBound.right - mouseX, FC_minimumSize, rightEdge - this.#padding.left)

            this.#box.style.left = `${rightEdge - this.#step.w * Math.round(W / this.#step.w)}px`;
            this.#box.style.width = `${this.#step.w * Math.round(W / this.#step.w)}px`;
        }

        if (this.#resize.B) {
            const H = this.#clampMinMax(mouseY - this.#boxBound.top, FC_minimumSize,
                this.#imgBound.bottom + this.#padding.top - this.#margin.top - this.init.top
            );

            this.#box.style.height = `${this.#step.h * Math.round(H / this.#step.h)}px`;
        } else if (this.#resize.T) {
            const bottomEdge = this.#style2value(this.#box.style.top) + this.#style2value(this.#box.style.height);
            const H = this.#clampMinMax(this.#boxBound.bottom - mouseY, FC_minimumSize, bottomEdge - this.#padding.top);

            this.#box.style.top = `${bottomEdge - this.#step.h * Math.round(H / this.#step.h)}px`;
            this.#box.style.height = `${this.#step.h * Math.round(H / this.#step.h)}px`;
        }
    }

    /** @param {number} mouseX @param {number} mouseY */
    #offsetLogic(mouseX, mouseY) {
        const deltaX = mouseX - this.init.X;
        const deltaY = mouseY - this.init.Y;

        const newLeft = this.#clampMinMax(this.init.left + deltaX,
            this.#padding.left, this.#imgBound.width - this.#boxBound.width + this.#padding.left);

        const newTop = this.#clampMinMax(this.init.top + deltaY,
            this.#padding.top, this.#imgBound.height - this.#boxBound.height + this.#padding.top);

        this.#box.style.left = `${this.#step.w * Math.round(newLeft / this.#step.w)}px`;
        this.#box.style.top = `${this.#step.h * Math.round(newTop / this.#step.h)}px`;
    }

    /**
     * When a row is selected, display its corresponding bounding box, as well as initialize the coordinates
     * @param {string} color
     * @param {Element} row
     */
    showBox(color, row) {
        this.#cachedRow = row;

        if (color)
            this.#currentColor = color;

        const shape = this.#extractShapeFromRow(row);
        if (shape) {
            this.#currentShape = shape;
            this.#renderShapeOverlay(shape, color);
            if (this.#overlay)
                this.#overlay.style.display = "block";
            this.#box.style.display = "none";
            return;
        }

        setTimeout(() => {
            this.#initCoord();
            this.#box.style.background = color;
            this.#box.style.display = "block";
        }, 25);
    }

    hideBox() {
        this.#cachedRow = null;
        this.#box.style.display = "none";
        if (this.#overlay) {
            this.#overlayCtx?.clearRect(0, 0, this.#overlay.width, this.#overlay.height);
            this.#overlay.style.display = "none";
        }
        this.#currentShape = null;
    }

    #setupShapeTools(container) {
        if (!container || typeof window === "undefined")
            return;

        const ShapeToolManagerRef = window.ShapeToolManager;
        const CanvasGuidesRef = window.CanvasGuides;

        if (!ShapeToolManagerRef || !CanvasGuidesRef)
            return;

        if (!["relative", "absolute", "fixed"].includes(getComputedStyle(container).position))
            container.style.position = "relative";

        const overlay = document.createElement("canvas");
        overlay.classList.add("fc_canvas_overlay");
        overlay.style.position = "absolute";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.zIndex = "200";
        overlay.style.pointerEvents = "none";
        overlay.style.display = "none";
        container.appendChild(overlay);

        this.#overlay = overlay;
        this.#overlayCtx = overlay.getContext("2d");

        this.#canvasGuides = new CanvasGuidesRef({ overlayCanvas: overlay });
        this.#canvasGuides.setGrid(this.#snapToGrid, this.#gridSize);
        this.#canvasGuides.setGuides(this.#smartGuides);

        this.#shapeToolManager = new ShapeToolManagerRef({
            canvas: overlay,
            guides: this.#canvasGuides
        });
        this.#shapeToolManager.setMetadataDefaults(this.#shapeDefaults);
        this.#updateToolOptions();

        if (!this.#keyHandler) {
            this.#keyHandler = (event) => {
                if (event.key !== "Enter")
                    return;
                if (!this.#isShapeToolActive())
                    return;
                const rect = this.#imgBound;
                this.#shapeToolManager.updateCanvasMetrics(rect.width, rect.height, this.#collectExistingShapes());
                const shape = this.#shapeToolManager?.handleKeyDown?.(event);
                if (!shape)
                    return;
                event.preventDefault();
                this.#commitShape(shape);
                this.#shapeToolManager.currentTool?.cancelDrawing?.();
            };
            window.addEventListener("keydown", this.#keyHandler, { passive: false });
        }

        if (!this.#resizeHandler) {
            this.#resizeHandler = () => this.#resizeOverlay();
            window.addEventListener("resize", this.#resizeHandler);
            if (this.#img instanceof HTMLImageElement)
                this.#img.addEventListener("load", this.#resizeHandler);
        }

        this.#resizeOverlay();
    }

    #resizeOverlay() {
        if (!this.#overlay)
            return;

        const rect = this.#imgBound;
        this.#overlay.width = rect.width;
        this.#overlay.height = rect.height;
        this.#overlay.style.width = `${rect.width}px`;
        this.#overlay.style.height = `${rect.height}px`;
    }

    #isShapeToolActive() {
        return Boolean(this.#shapeToolManager && this.#activeShapeTool && this.#activeShapeTool !== "LEGACY");
    }

    #updateToolOptions() {
        if (!this.#shapeToolManager || !this.#shapeToolManager.setToolOptions)
            return;

        this.#shapeToolManager.setToolOptions({
            aspectLock: this.#aspectLock,
            aspectRatio: this.#aspectRatio,
            snapToGrid: this.#snapToGrid,
            gridSize: this.#gridSize,
            guides: this.#canvasGuides
        });
    }

    setShapeTool(toolType) {
        if (!this.#shapeToolManager) {
            this.#activeShapeTool = "LEGACY";
            return;
        }

        const normalised = (toolType || "").toUpperCase();
        const supported = ["RECTANGLE", "ELLIPSE", "POLYGON", "BEZIER"];
        if (!supported.includes(normalised)) {
            this.#activeShapeTool = "LEGACY";
            return;
        }

        this.#activeShapeTool = normalised;
        this.#shapeToolManager.setTool(normalised);
        this.#img.style.cursor = "crosshair";
    }

    setAspectLock(enabled, ratio) {
        this.#aspectLock = Boolean(enabled);

        if (typeof ratio === "string" && ratio.includes(":")) {
            const [w, h] = ratio.split(":").map(Number);
            if (!Number.isNaN(w) && !Number.isNaN(h) && w > 0 && h > 0)
                this.#aspectRatio = w / h;
        } else if (typeof ratio === "number" && ratio > 0) {
            this.#aspectRatio = ratio;
        }

        this.#updateToolOptions();
    }

    setSnapToGrid(enabled, gridSize) {
        this.#snapToGrid = Boolean(enabled);
        if (typeof gridSize === "number" && gridSize > 0)
            this.#gridSize = gridSize;

        this.#canvasGuides?.setGrid(this.#snapToGrid, this.#gridSize);
        this.#updateToolOptions();
        this.#renderShapeOverlay(this.#currentShape, this.#currentColor);
    }

    setSmartGuides(enabled) {
        this.#smartGuides = Boolean(enabled);
        this.#canvasGuides?.setGuides(this.#smartGuides);
        this.#renderShapeOverlay(this.#currentShape, this.#currentColor);
    }

    getShapeToolManager() {
        return this.#shapeToolManager;
    }

    setShapeDefaults(defaults) {
        if (!defaults)
            return;

        this.#shapeDefaults = {
            ...this.#shapeDefaults,
            ...defaults,
        };
        this.#shapeToolManager?.setMetadataDefaults(this.#shapeDefaults);
    }

    updateShapeMetadata(metadata) {
        if (!this.#currentShape || !metadata)
            return;

        const next = {
            ...this.#currentShape,
            ...metadata,
        };

        if (metadata.feather_edges) {
            const mergedEdges = {
                ...(this.#currentShape.feather_edges || {}),
                ...metadata.feather_edges,
            };
            const normalised = this.#normaliseFeatherEdges(mergedEdges);
            if (normalised)
                next.feather_edges = normalised;
            else
                delete next.feather_edges;
        }

        this.#currentShape = next;

        if (this.#cachedRow) {
            const json = JSON.stringify(this.#currentShape);
            const firstCell = this.#cachedRow.querySelector("td");
            if (firstCell) {
                firstCell.dataset.shape = json;
                firstCell.textContent = json;
            }
        }

        this.#renderShapeOverlay(this.#currentShape, this.#currentColor);
        if (typeof window.ForgeCouple?.syncShapeMetadata === "function")
            window.ForgeCouple.syncShapeMetadata(this.#mode, this.#currentShape);
    }

    updateShapeCoords(x, y, w, h) {
        if (!this.#currentShape)
            return;

        const clamp01 = (value) => this.#clampMinMax(Number(value), 0, 1);
        const nx = clamp01(x);
        const ny = clamp01(y);
        const width = clamp01(w);
        const height = clamp01(h);
        const maxX = this.#clampMinMax(nx + Math.max(width, 0), 0, 1);
        const maxY = this.#clampMinMax(ny + Math.max(height, 0), 0, 1);

        const originalBounds = this.#shapeBounds(this.#currentShape);
        const newBounds = { x1: nx, y1: ny, x2: maxX, y2: maxY };

        const updated = this.#applyShapeDefaults(this.#currentShape);

        if (updated.shape_type === "RECTANGLE") {
            updated.parameters = {
                x1: newBounds.x1,
                y1: newBounds.y1,
                x2: newBounds.x2,
                y2: newBounds.y2,
            };
        } else if (updated.shape_type === "ELLIPSE") {
            updated.parameters = {
                cx: nx,
                cy: ny,
                rx: Math.max(width / 2, 0.001),
                ry: Math.max(height / 2, 0.001),
            };
        } else if (updated.shape_type === "POLYGON") {
            const points = Array.isArray(updated.parameters?.points) ? updated.parameters.points : null;
            if (!points || points.length < 3 || !originalBounds)
                return;
            updated.parameters.points = this.#transformPointList(points, originalBounds, newBounds);
        } else if (updated.shape_type === "BEZIER") {
            const controls = Array.isArray(updated.parameters?.control_points) ? updated.parameters.control_points : null;
            if (!controls || controls.length < 4 || !originalBounds)
                return;
            updated.parameters.control_points = this.#transformPointList(controls, originalBounds, newBounds);
        } else {
            return;
        }

        this.#currentShape = updated;
        if (this.#cachedRow) {
            const json = JSON.stringify(updated);
            const firstCell = this.#cachedRow.querySelector("td");
            if (firstCell) {
                firstCell.dataset.shape = json;
                firstCell.textContent = json;
            }
        }

        this.#renderShapeOverlay(updated, this.#currentColor);
        if (typeof window.ForgeCouple?.syncShapeMetadata === "function")
            window.ForgeCouple.syncShapeMetadata(this.#mode, updated);
    }

    #collectExistingShapes() {
        const table = window.ForgeCouple?.mappingTable?.[this.#mode];
        if (!table)
            return this.#currentShape ? [this.#currentShape] : [];

        const rows = Array.from(table.querySelectorAll("tr"));
        const shapes = rows
            .map((row) => this.#extractShapeFromRow(row))
            .filter((shape) => !!shape);

        if (this.#currentShape && !shapes.some((shape) => JSON.stringify(shape) === JSON.stringify(this.#currentShape)))
            shapes.push(this.#currentShape);

        return shapes;
    }

    #extractShapeFromRow(row) {
        if (!row)
            return null;

        const cells = row.querySelectorAll("td");
        if (!cells.length)
            return null;

        const raw = cells[0].dataset?.shape ?? cells[0].textContent ?? "";
        if (!raw || !raw.trim().startsWith("{"))
            return null;

        try {
            return JSON.parse(raw);
        } catch {
            return null;
        }
    }

    #handleShapeMouseDown(event) {
        if (!this.#shapeToolManager)
            return;

        window.ForgeCouple?.ensureMaskSelection?.(this.#mode);

        event.preventDefault();
        this.#resizeOverlay();

        const rect = this.#imgBound;
        const point = this.#eventToNormalized(event, rect);
        this.#shapeToolManager.updateCanvasMetrics(rect.width, rect.height, this.#collectExistingShapes());

        const preview = this.#shapeToolManager.handleMouseDown(event, point);
        if (preview) {
            const enriched = this.#applyShapeDefaults(preview);
            this.#renderShapeOverlay(enriched, this.#currentColor, true);
        } else {
            this.#renderShapeOverlay(this.#currentShape, this.#currentColor);
        }
    }

    #handleShapeMouseMove(event) {
        if (!this.#shapeToolManager)
            return;

        const rect = this.#imgBound;
        const point = this.#eventToNormalized(event, rect);
        this.#shapeToolManager.updateCanvasMetrics(rect.width, rect.height, this.#collectExistingShapes());

        const preview = this.#shapeToolManager.handleMouseMove(event, point);
        if (preview) {
            const enriched = this.#applyShapeDefaults(preview);
            this.#renderShapeOverlay(enriched, this.#currentColor, true);
        } else if (!this.#shapeToolManager.currentTool?.isDrawing) {
            this.#renderShapeOverlay(this.#currentShape, this.#currentColor);
        }

        this.#img.style.cursor = "crosshair";
    }

    #handleShapeMouseUp(event) {
        if (!this.#shapeToolManager)
            return;

        let shape = null;
        if (event) {
            const rect = this.#imgBound;
            const point = this.#eventToNormalized(event, rect);
            shape = this.#shapeToolManager.handleMouseUp(event, point);
        } else {
            this.#shapeToolManager.currentTool?.cancelDrawing?.();
        }

        if (shape)
            this.#commitShape(shape);
        else
            this.#renderShapeOverlay(this.#currentShape, this.#currentColor);
    }

    #commitShape(shape) {
        if (!shape)
            return;

        const enriched = this.#applyShapeDefaults(shape);
        if (!enriched)
            return;

        const clamp01 = (value) => this.#clampMinMax(Number(value), 0, 1);
        const params = enriched.parameters ?? {};

        switch (enriched.shape_type) {
            case "RECTANGLE": {
                const x1 = clamp01(params.x1);
                const x2 = clamp01(params.x2);
                const y1 = clamp01(params.y1);
                const y2 = clamp01(params.y2);
                enriched.parameters = {
                    ...params,
                    x1: Math.min(x1, x2),
                    y1: Math.min(y1, y2),
                    x2: Math.max(x1, x2),
                    y2: Math.max(y1, y2),
                };
                break;
            }
            case "ELLIPSE": {
                enriched.parameters = {
                    ...params,
                    cx: clamp01(params.cx),
                    cy: clamp01(params.cy),
                    rx: clamp01(params.rx),
                    ry: clamp01(params.ry),
                };
                break;
            }
            case "POLYGON": {
                const points = Array.isArray(params.points) ? params.points : [];
                enriched.parameters = {
                    ...params,
                    points: points.map((point) => {
                        if (!Array.isArray(point) || point.length < 2)
                            return [0, 0];
                        return [clamp01(point[0]), clamp01(point[1])];
                    }),
                };
                break;
            }
            case "BEZIER": {
                const controlPoints = Array.isArray(params.control_points) ? params.control_points : [];
                enriched.parameters = {
                    ...params,
                    control_points: controlPoints.map((point) => {
                        if (!Array.isArray(point) || point.length < 2)
                            return [0, 0];
                        return [clamp01(point[0]), clamp01(point[1])];
                    }),
                };
                break;
            }
            default:
                enriched.parameters = { ...params };
        }

        this.#currentShape = enriched;
        this.#renderShapeOverlay(enriched, this.#currentColor);

        if (this.#cachedRow) {
            const json = JSON.stringify(enriched);
            const firstCell = this.#cachedRow.querySelector("td");
            if (firstCell) {
                firstCell.dataset.shape = json;
                firstCell.textContent = json;
            }
        }

        window.ForgeCouple?.ensureMaskSelection?.(this.#mode);
        if (typeof window.ForgeCouple?.syncShapeMetadata === "function")
            window.ForgeCouple.syncShapeMetadata(this.#mode, enriched);
    }

    #eventToNormalized(event, rect) {
        const x = this.#clampMinMax(event.clientX - rect.left, 0, rect.width);
        const y = this.#clampMinMax(event.clientY - rect.top, 0, rect.height);
        return [x / rect.width, y / rect.height];
    }

    #renderShapeOverlay(shape, color, preview = false) {
        if (!this.#overlayCtx || !this.#overlay)
            return;

        this.#resizeOverlay();
        const width = this.#overlay.width;
        const height = this.#overlay.height;

        this.#overlayCtx.clearRect(0, 0, width, height);

        if (this.#snapToGrid)
            this.#canvasGuides?.drawGrid(width, height);

        if (this.#smartGuides && shape)
            this.#canvasGuides?.drawSmartGuides(shape, this.#collectExistingShapes(), width, height);

        if (!shape) {
            if (this.#snapToGrid || this.#smartGuides)
                this.#overlay.style.display = "block";
            else
                this.#overlay.style.display = "none";
            return;
        }

        this.#overlay.style.display = "block";
        this.#overlayCtx.save();
        this.#overlayCtx.strokeStyle = color || this.#currentColor;
        this.#overlayCtx.lineWidth = 2;
        this.#overlayCtx.setLineDash(preview ? [4, 4] : [6, 4]);
        window.ShapeTool?.drawShape(this.#overlayCtx, shape, { width, height });
        this.#overlayCtx.restore();
    }

    #applyShapeDefaults(shape) {
        if (!shape)
            return null;

        const parameters = shape.parameters ? JSON.parse(JSON.stringify(shape.parameters)) : {};
        const baseEdges = (shape.feather_edges !== undefined && shape.feather_edges !== null)
            ? shape.feather_edges
            : this.#shapeDefaults.feather_edges;
        const normalisedEdges = this.#normaliseFeatherEdges(baseEdges);
        const enriched = {
            shape_type: shape.shape_type,
            parameters,
            z_order: shape.z_order ?? this.#shapeDefaults.z_order,
            blend_mode: shape.blend_mode ?? this.#shapeDefaults.blend_mode,
            feather: shape.feather ?? this.#shapeDefaults.feather,
            hardness: shape.hardness ?? this.#shapeDefaults.hardness,
            weight: shape.weight ?? this.#shapeDefaults.weight,
        };
        if (normalisedEdges)
            enriched.feather_edges = normalisedEdges;
        return enriched;
    }

    #normaliseFeatherEdges(edges) {
        if (!edges || typeof edges !== "object")
            return null;

        const allowed = ["top", "right", "bottom", "left"];
        const result = {};
        let hasValue = false;
        allowed.forEach((key) => {
            if (!(key in edges))
                return;
            const value = this.#clampMinMax(Number(edges[key]), 0, 1);
            if (Number.isFinite(value)) {
                result[key] = value;
                hasValue = true;
            }
        });

        return hasValue ? result : null;
    }

    #transformPointList(points, originalBounds, newBounds) {
        if (!Array.isArray(points))
            return [];

        const oldWidth = Math.max(originalBounds.x2 - originalBounds.x1, 0);
        const oldHeight = Math.max(originalBounds.y2 - originalBounds.y1, 0);
        const newWidth = Math.max(newBounds.x2 - newBounds.x1, 0);
        const newHeight = Math.max(newBounds.y2 - newBounds.y1, 0);

        return points.map(([px, py]) => {
            const baseX = this.#clampMinMax(Number(px), 0, 1);
            const baseY = this.#clampMinMax(Number(py), 0, 1);
            const relX = oldWidth > 1e-6 ? this.#clampMinMax((baseX - originalBounds.x1) / oldWidth, 0, 1) : 0.5;
            const relY = oldHeight > 1e-6 ? this.#clampMinMax((baseY - originalBounds.y1) / oldHeight, 0, 1) : 0.5;
            const nextX = this.#clampMinMax(newBounds.x1 + relX * newWidth, 0, 1);
            const nextY = this.#clampMinMax(newBounds.y1 + relY * newHeight, 0, 1);
            return [nextX, nextY];
        });
    }

    #shapeBounds(shape) {
        if (!shape || typeof shape !== "object")
            return null;

        const type = (shape.shape_type || "").toUpperCase();
        const params = shape.parameters || {};

        switch (type) {
            case "RECTANGLE": {
                const x1 = this.#clampMinMax(Number(params.x1), 0, 1);
                const y1 = this.#clampMinMax(Number(params.y1), 0, 1);
                const x2 = this.#clampMinMax(Number(params.x2), 0, 1);
                const y2 = this.#clampMinMax(Number(params.y2), 0, 1);
                return { x1: Math.min(x1, x2), x2: Math.max(x1, x2), y1: Math.min(y1, y2), y2: Math.max(y1, y2) };
            }
            case "ELLIPSE": {
                const cx = this.#clampMinMax(Number(params.cx), 0, 1);
                const cy = this.#clampMinMax(Number(params.cy), 0, 1);
                const rx = Math.max(Number(params.rx) || 0, 0);
                const ry = Math.max(Number(params.ry) || 0, 0);
                return {
                    x1: this.#clampMinMax(cx - rx, 0, 1),
                    x2: this.#clampMinMax(cx + rx, 0, 1),
                    y1: this.#clampMinMax(cy - ry, 0, 1),
                    y2: this.#clampMinMax(cy + ry, 0, 1),
                };
            }
            case "POLYGON": {
                const points = Array.isArray(params.points) ? params.points : [];
                if (points.length < 3)
                    return null;
                let minX = 1;
                let maxX = 0;
                let minY = 1;
                let maxY = 0;
                points.forEach(([px, py]) => {
                    const x = this.#clampMinMax(Number(px), 0, 1);
                    const y = this.#clampMinMax(Number(py), 0, 1);
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                });
                return { x1: minX, x2: maxX, y1: minY, y2: maxY };
            }
            case "BEZIER": {
                const controls = Array.isArray(params.control_points) ? params.control_points : [];
                return this.#bezierBounds(controls);
            }
            default:
                return null;
        }
    }

    #bezierBounds(controlPoints) {
        if (!Array.isArray(controlPoints) || controlPoints.length < 4)
            return null;

        const segmentCount = Math.floor((controlPoints.length - 1) / 3);
        if (segmentCount <= 0)
            return null;

        let minX = 1;
        let maxX = 0;
        let minY = 1;
        let maxY = 0;

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

                const clampedX = this.#clampMinMax(x, 0, 1);
                const clampedY = this.#clampMinMax(y, 0, 1);
                minX = Math.min(minX, clampedX);
                maxX = Math.max(maxX, clampedX);
                minY = Math.min(minY, clampedY);
                maxY = Math.max(maxY, clampedY);
            }
        }

        return { x1: minX, x2: maxX, y1: minY, y2: maxY };
    }

    #initCoord() {
        if (this.#cachedRow == null)
            return;

        const mapping = this.#mappingToStyle(this.#cachedRow);
        if (mapping && !Array.isArray(mapping)) {
            this.#currentShape = mapping;
            this.#renderShapeOverlay(mapping, this.#currentColor);
            if (this.#overlay)
                this.#overlay.style.display = "block";
            this.#box.style.display = "none";
            return;
        }

        if (!Array.isArray(mapping) || mapping.length < 4)
            return;

        const [from_x, delta_x, from_y, delta_y] = mapping;
        const { width, height } = this.#imgBound;

        if (width === height) {
            this.#padding.left = 0.0;
            this.#padding.top = 0.0;
        } else if (width > height) {
            const ratio = height / width;
            this.#padding.left = 0.0;
            this.#padding.top = 256.0 * (1.0 - ratio);
        } else {
            const ratio = width / height;
            this.#padding.left = 256.0 * (1.0 - ratio);
            this.#padding.top = 0.0;
        }

        this.#step.w = width / 100.0;
        this.#step.h = height / 100.0;

        this.#margin.left = this.#imgBound.left;
        this.#margin.top = this.#imgBound.top;

        this.#box.style.width = `${width * delta_x}px`;
        this.#box.style.height = `${height * delta_y}px`;

        this.#box.style.left = `${this.#padding.left + width * from_x}px`;
        this.#box.style.top = `${this.#padding.top + height * from_y}px`;
    }


    /** @param {number} mouseX @param {number} mouseY */
    #checkMouse(mouseX, mouseY) {
        const FC_resizeBorder = 8;

        if (this.#box.style.display == "none") {
            this.#img.style.cursor = "default";
            return;
        }

        const { left, right, top, bottom } = this.#boxBound;

        if (mouseX < left - FC_resizeBorder || mouseX > right + FC_resizeBorder || mouseY < top - FC_resizeBorder || mouseY > bottom + FC_resizeBorder) {
            this.#img.style.cursor = "default";
            return;
        }

        this.#resize.L = mouseX < left + FC_resizeBorder;
        this.#resize.R = mouseX > right - FC_resizeBorder;
        this.#resize.T = mouseY < top + FC_resizeBorder;
        this.#resize.B = mouseY > bottom - FC_resizeBorder;

        if (!(this.#resize.L || this.#resize.T || this.#resize.R || this.#resize.B)) {
            this.#img.style.cursor = "move";
            return;
        }

        if (this.#resize.R && this.#resize.B)
            this.#img.style.cursor = "nwse-resize";
        else if (this.#resize.R && this.#resize.T)
            this.#img.style.cursor = "nesw-resize";
        else if (this.#resize.L && this.#resize.B)
            this.#img.style.cursor = "nesw-resize";
        else if (this.#resize.L && this.#resize.T)
            this.#img.style.cursor = "nwse-resize";
        else if (this.#resize.R || this.#resize.L)
            this.#img.style.cursor = "ew-resize";
        else if (this.#resize.B || this.#resize.T)
            this.#img.style.cursor = "ns-resize";
    }


    /**
     * Convert the table row into coordinate ranges
     * @param {Element} row
     * @returns {number[]}
     */
    #mappingToStyle(row) {
        const cells = row.querySelectorAll("td");

        if (!cells.length)
            return [0, 0, 0, 0];

        const shape = this.#extractShapeFromRow(row);
        if (shape)
            return shape;

        const from_x = parseFloat(cells[0].textContent);
        const to_x = parseFloat(cells[1].textContent);
        const from_y = parseFloat(cells[2].textContent);
        const to_y = parseFloat(cells[3].textContent);

        return [from_x, to_x - from_x, from_y, to_y - from_y]
    }

    /**
     * Convert the coordinates of bounding box back into string
     * @returns {number[]}
     */
    #styleToMapping() {
        if (this.#currentShape) {
            try {
                return JSON.parse(JSON.stringify(this.#currentShape));
            } catch {
                return this.#currentShape;
            }
        }

        const { width, height } = this.#imgBound;
        const { left, right, top, bottom } = this.#boxBound;
        const { left: leftMargin, top: topMargin } = this.#margin;

        const from_x = (left - leftMargin) / width;
        const to_x = (right - leftMargin) / width;
        const from_y = (top - topMargin) / height;
        const to_y = (bottom - topMargin) / height;

        return [from_x, to_x, from_y, to_y];
    }

    /** @param {number} v @param {number} min @param {number} max @returns {number} */
    #clampMinMax(v, min, max) {
        return Math.min(Math.max(v, min), max);
    }

    /** @param {string} style @returns {number} */
    #style2value(style) {
        try {
            const re = /calc\((-?\d+(?:\.\d+)?)px\)/;
            return parseFloat(style.match(re)[1]);
        } catch {
            return parseFloat(style);
        }
    }

    /**
     * Clean up event listeners and resources when the instance is destroyed.
     * Call this method when removing the ForgeCoupleBox instance or when it's no longer needed.
     */
    dispose() {
        // Remove event listeners from image
        if (this.#clickHandler && this.#img) {
            this.#img.removeEventListener("mousedown", this.#clickHandler);
            this.#clickHandler = null;
        }

        // Remove event listeners from tab
        if (this.#tab) {
            if (this.#hoverHandler) {
                this.#tab.removeEventListener("mousemove", this.#hoverHandler);
                this.#hoverHandler = null;
            }

            if (this.#upHandlers.mouseup) {
                this.#tab.removeEventListener("mouseup", this.#upHandlers.mouseup);
                this.#upHandlers.mouseup = null;
            }

            if (this.#upHandlers.mouseleave) {
                this.#tab.removeEventListener("mouseleave", this.#upHandlers.mouseleave);
                this.#upHandlers.mouseleave = null;
            }

            this.#tab = null;
        }

        // Remove window event listeners
        if (this.#keyHandler) {
            window.removeEventListener("keydown", this.#keyHandler);
            this.#keyHandler = null;
        }

        if (this.#resizeHandler) {
            window.removeEventListener("resize", this.#resizeHandler);
            if (this.#img instanceof HTMLImageElement) {
                this.#img.removeEventListener("load", this.#resizeHandler);
            }
            this.#resizeHandler = null;
        }

        // Clean up canvas overlay if it exists
        if (this.#overlay && this.#overlay.parentElement) {
            this.#overlay.parentElement.removeChild(this.#overlay);
            this.#overlay = null;
            this.#overlayCtx = null;
        }

        // Clean up bounding box if it exists
        if (this.#box && this.#box.parentElement) {
            this.#box.parentElement.removeChild(this.#box);
            this.#box = null;
        }

        // Clear image reference
        this.#img = null;

        // Clear references to managers
        this.#shapeToolManager = null;
        this.#canvasGuides = null;
    }
}
