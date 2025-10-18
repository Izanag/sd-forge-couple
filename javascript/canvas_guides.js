class CanvasGuides {
    constructor(options = {}) {
        this.gridEnabled = options.gridEnabled ?? false;
        this.gridSize = options.gridSize ?? 0.1;
        this.guidesEnabled = options.guidesEnabled ?? false;
        this.snapThreshold = options.snapThreshold ?? 5;

        this._overlayCanvas = options.overlayCanvas ?? null;
        this._ctx = this._overlayCanvas?.getContext?.("2d") ?? null;
        this._activeGuides = [];
        this._gridCacheCanvas = null;
        this._gridCacheWidth = 0;
        this._gridCacheHeight = 0;
        this._gridCacheSize = this.gridSize;
        this._pendingGridDraw = null;
        this._gridDrawScheduled = false;
        this._gridDrawHandle = null;
        this._gridDrawUseRaf = false;
    }

    setOverlayCanvas(canvas) {
        this._overlayCanvas = canvas;
        this._ctx = canvas?.getContext?.("2d") ?? null;
        this._cancelScheduledGridDraw();
        this._invalidateGridCache();
    }

    setGrid(enabled, size) {
        const previousSize = this.gridSize;
        const previousEnabled = this.gridEnabled;

        this.gridEnabled = !!enabled;
        if (size !== undefined && size > 0) this.gridSize = size;

        if (!this.gridEnabled || previousSize !== this.gridSize || previousEnabled !== this.gridEnabled)
            this._invalidateGridCache();
    }

    setGuides(enabled) {
        this.guidesEnabled = !!enabled;
    }

    setSnapThreshold(pixels) {
        if (pixels > 0) this.snapThreshold = pixels;
    }

    clearOverlay() {
        this._cancelScheduledGridDraw();
        if (!this._ctx || !this._overlayCanvas) return;
        this._ctx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
        this._activeGuides = [];
    }

    drawGrid(width, height) {
        if (!this.gridEnabled || !this._ctx) return;
        if (!(width > 0) || !(height > 0)) return;

        this._pendingGridDraw = { width, height };

        if (this._gridDrawScheduled) return;
        this._gridDrawScheduled = true;

        const root =
            typeof window !== "undefined" ? window :
            (typeof globalThis !== "undefined" ? globalThis : null);
        const raf = root?.requestAnimationFrame;
        const schedule = (typeof raf === "function")
            ? raf.bind(root)
            : ((callback) => {
                const timeoutFn = root?.setTimeout ?? setTimeout;
                return timeoutFn.call ? timeoutFn.call(root, callback, 16) : timeoutFn(callback, 16);
            });

        this._gridDrawUseRaf = typeof raf === "function";
        this._gridDrawHandle = schedule(() => {
            this._gridDrawHandle = null;
            this._gridDrawScheduled = false;
            const args = this._pendingGridDraw;
            this._pendingGridDraw = null;
            if (!args) return;
            this._renderGrid(args.width, args.height);
        });
    }

    _renderGrid(width, height) {
        if (!this.gridEnabled || !this._ctx) return;
        if (!(width > 0) || !(height > 0)) return;

        const needsRedraw =
            !this._gridCacheCanvas ||
            this._gridCacheWidth !== width ||
            this._gridCacheHeight !== height ||
            this._gridCacheSize !== this.gridSize;

        if (needsRedraw) {
            const canvas = this._gridCacheCanvas ?? document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;

            const cacheCtx = canvas.getContext("2d");
            if (!cacheCtx) return;

            cacheCtx.clearRect(0, 0, width, height);
            cacheCtx.lineWidth = 1;
            cacheCtx.strokeStyle = "rgba(255,255,255,0.1)";

            const stepX = width * this.gridSize;
            const stepY = height * this.gridSize;

            if (stepX > 0) {
                for (let x = 0; x <= width; x += stepX) {
                    const pos = Math.round(x) + 0.5;
                    cacheCtx.beginPath();
                    cacheCtx.moveTo(pos, 0);
                    cacheCtx.lineTo(pos, height);
                    cacheCtx.stroke();
                }
            }

            if (stepY > 0) {
                for (let y = 0; y <= height; y += stepY) {
                    const pos = Math.round(y) + 0.5;
                    cacheCtx.beginPath();
                    cacheCtx.moveTo(0, pos);
                    cacheCtx.lineTo(width, pos);
                    cacheCtx.stroke();
                }
            }

            this._gridCacheCanvas = canvas;
            this._gridCacheWidth = width;
            this._gridCacheHeight = height;
            this._gridCacheSize = this.gridSize;
        }

        if (!this._gridCacheCanvas) return;

        this._ctx.save();
        this._ctx.drawImage(this._gridCacheCanvas, 0, 0);
        this._ctx.restore();
    }

    drawSmartGuides(activeShape, allShapes, width, height) {
        if (!this.guidesEnabled || !this._ctx) return;
        const ctx = this._ctx;
        ctx.save();
        ctx.strokeStyle = "rgba(0,255,255,0.8)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 2]);

        const guides = this._collectGuideLines(activeShape, allShapes, width, height);
        guides.forEach((line) => {
            ctx.beginPath();
            if (line.type === "vertical") {
                ctx.moveTo(line.position, 0);
                ctx.lineTo(line.position, height);
            } else {
                ctx.moveTo(0, line.position);
                ctx.lineTo(width, line.position);
            }
            ctx.stroke();
        });

        ctx.restore();
        this._activeGuides = guides;
    }

    snapToGrid(x, y, width, height) {
        if (!this.gridEnabled) return { x, y };
        const stepX = width * this.gridSize;
        const stepY = height * this.gridSize;

        const snappedX = Math.round(x / stepX) * stepX;
        const snappedY = Math.round(y / stepY) * stepY;
        return { x: snappedX, y: snappedY };
    }

    snapToGuides(x, y, allShapes, width, height) {
        if (!this.guidesEnabled) return { x, y };
        const threshold = this.snapThreshold;
        const lines = this._collectGuideLines(null, allShapes, width, height);

        let snappedX = x;
        let snappedY = y;
        let minDistX = Infinity;
        let minDistY = Infinity;

        lines.forEach((line) => {
            const distX = Math.abs(line.position - x);
            if (line.type === "vertical" && distX <= threshold && distX < minDistX) {
                minDistX = distX;
                snappedX = line.position;
            }
            const distY = Math.abs(line.position - y);
            if (line.type === "horizontal" && distY <= threshold && distY < minDistY) {
                minDistY = distY;
                snappedY = line.position;
            }
        });

        return { x: snappedX, y: snappedY };
    }

    getSnapPosition(x, y, width, height, allShapes) {
        let point = { x, y };
        if (this.gridEnabled) {
            point = this.snapToGrid(point.x, point.y, width, height);
        }
        if (this.guidesEnabled) {
            point = this.snapToGuides(point.x, point.y, allShapes, width, height);
        }
        return point;
    }

    _collectGuideLines(activeShape, allShapes, width, height) {
        const shapes = Array.isArray(allShapes) ? allShapes : [];
        const lines = [];
        const added = new Set();
        const addLine = (type, position) => {
            if (!Number.isFinite(position))
                return;
            const key = `${type}:${position.toFixed(4)}`;
            if (added.has(key))
                return;
            added.add(key);
            lines.push({ type, position });
        };

        const canvasWidth = Number.isFinite(width) ? width : 0;
        const canvasHeight = Number.isFinite(height) ? height : 0;

        if (canvasWidth > 0) {
            addLine("vertical", 0);
            addLine("vertical", canvasWidth / 2);
            addLine("vertical", canvasWidth);
        }
        if (canvasHeight > 0) {
            addLine("horizontal", 0);
            addLine("horizontal", canvasHeight / 2);
            addLine("horizontal", canvasHeight);
        }

        const processShape = (shape) => {
            if (!shape) return;
            const { shape_type: type, parameters } = shape;
            switch (type) {
                case "RECTANGLE": {
                    const { x1, x2, y1, y2 } = parameters;
                    const left = Number(x1) * canvasWidth;
                    const right = Number(x2) * canvasWidth;
                    const top = Number(y1) * canvasHeight;
                    const bottom = Number(y2) * canvasHeight;
                    const cx = (left + right) / 2;
                    const cy = (top + bottom) / 2;
                    addLine("vertical", left);
                    addLine("vertical", right);
                    addLine("horizontal", top);
                    addLine("horizontal", bottom);
                    addLine("vertical", cx);
                    addLine("horizontal", cy);
                    break;
                }
                case "ELLIPSE": {
                    const { cx, cy, rx, ry } = parameters;
                    const centerX = Number(cx) * canvasWidth;
                    const centerY = Number(cy) * canvasHeight;
                    const rxPx = Number(rx) * canvasWidth;
                    const ryPx = Number(ry) * canvasHeight;
                    const left = centerX - rxPx;
                    const right = centerX + rxPx;
                    const top = centerY - ryPx;
                    const bottom = centerY + ryPx;
                    addLine("vertical", centerX);
                    addLine("horizontal", centerY);
                    addLine("vertical", left);
                    addLine("vertical", right);
                    addLine("horizontal", top);
                    addLine("horizontal", bottom);
                    break;
                }
                case "POLYGON": {
                    const pts = parameters.points ?? [];
                    pts.forEach(([px, py]) => {
                        addLine("vertical", Number(px) * canvasWidth);
                        addLine("horizontal", Number(py) * canvasHeight);
                    });
                    break;
                }
                case "BEZIER": {
                    const cps = parameters.control_points ?? [];
                    cps.forEach(([px, py]) => {
                        addLine("vertical", Number(px) * canvasWidth);
                        addLine("horizontal", Number(py) * canvasHeight);
                    });
                    break;
                }
                default:
                    break;
            }
        };

        shapes.forEach(processShape);
        processShape(activeShape);

        return lines;
    }

    _invalidateGridCache() {
        this._cancelScheduledGridDraw();
        this._gridCacheCanvas = null;
        this._gridCacheWidth = 0;
        this._gridCacheHeight = 0;
        this._gridCacheSize = this.gridSize;
    }

    _cancelScheduledGridDraw() {
        this._pendingGridDraw = null;
        if (!this._gridDrawScheduled) return;

        const root =
            typeof window !== "undefined" ? window :
            (typeof globalThis !== "undefined" ? globalThis : null);

        if (this._gridDrawHandle != null) {
            if (this._gridDrawUseRaf && root?.cancelAnimationFrame) {
                root.cancelAnimationFrame(this._gridDrawHandle);
            } else if (!this._gridDrawUseRaf) {
                const clearFn = root?.clearTimeout ?? clearTimeout;
                if (typeof clearFn === "function") {
                    if (clearFn.call)
                        clearFn.call(root, this._gridDrawHandle);
                    else
                        clearFn(this._gridDrawHandle);
                }
            }
        }

        this._gridDrawHandle = null;
        this._gridDrawScheduled = false;
    }
}

if (typeof window !== "undefined") {
    window.CanvasGuides = CanvasGuides;
}
