class ShapeTool {
    constructor(options = {}) {
        this.options = {
            gridSize: 0.1,
            snapToGrid: false,
            aspectLock: false,
            aspectRatio: 1,
            metadata: {},
            guides: null,
            canvas: null,
            ...options,
        };

        this._isDrawing = false;
        this._previewShape = null;
        this._start = null;
        this._points = [];
    }

    setOptions(options) {
        this.options = {
            ...this.options,
            ...options,
        };
    }

    get canvas() {
        return this.options.canvas;
    }

    get context() {
        return this.canvas?.getContext?.("2d") ?? null;
    }

    get guides() {
        return this.options.guides;
    }

    get previewShape() {
        return this._previewShape;
    }

    get isDrawing() {
        return this._isDrawing;
    }

    startDrawing(origin) {
        this._isDrawing = true;
        this._start = origin;
        this._points = [origin];
    }

    updateDrawing(current) {
        if (!this._isDrawing) return;
        this._points[this._points.length - 1] = current;
    }

    finishDrawing() {
        this._isDrawing = false;
        this._start = null;
        const shape = this._previewShape;
        this._previewShape = null;
        this._points = [];
        return shape;
    }

    cancelDrawing() {
        this._isDrawing = false;
        this._start = null;
        this._previewShape = null;
        this._points = [];
    }

    onMouseDown(_event, normPoint, canvasInfo) {
        this.startDrawing(normPoint);
        return this.toShapeData(canvasInfo);
    }

    onMouseMove(_event, normPoint, canvasInfo) {
        if (!this.isDrawing) return null;
        this.updateDrawing(normPoint);
        this._previewShape = this.toShapeData(canvasInfo);
        return this.previewShape;
    }

    onMouseUp(_event, normPoint, canvasInfo) {
        if (!this.isDrawing) return null;
        this.updateDrawing(normPoint);
        const shape = this.toShapeData(canvasInfo);
        this.finishDrawing();
        return shape;
    }

    onKeyDown(_event, _canvasInfo) {
        return null;
    }

    render(ctx, canvasInfo) {
        if (!ctx || !this.previewShape) return;
        ShapeTool.drawShape(ctx, this.previewShape, canvasInfo);
    }

    toShapeData(_canvasInfo) {
        return null;
    }

    static drawShape(ctx, shapeData, canvasInfo) {
        if (!shapeData) return;
        const { width, height } = canvasInfo;
        const { shape_type: shapeType, parameters } = shapeData;
        ctx.save();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#00ffff";
        ctx.setLineDash([6, 4]);

        switch (shapeType) {
            case "RECTANGLE": {
                const { x1, y1, x2, y2 } = parameters;
                const x = x1 * width;
                const y = y1 * height;
                const w = (x2 - x1) * width;
                const h = (y2 - y1) * height;
                ctx.strokeRect(x, y, w, h);
                break;
            }
            case "ELLIPSE": {
                const { cx, cy, rx, ry } = parameters;
                ctx.beginPath();
                ctx.ellipse(
                    cx * width,
                    cy * height,
                    rx * width,
                    ry * height,
                    0,
                    0,
                    Math.PI * 2
                );
                ctx.stroke();
                break;
            }
            case "POLYGON": {
                const points = parameters.points ?? [];
                if (points.length <= 1) break;
                ctx.beginPath();
                points.forEach(([px, py], index) => {
                    const dx = px * width;
                    const dy = py * height;
                    if (index === 0) ctx.moveTo(dx, dy);
                    else ctx.lineTo(dx, dy);
                });
                ctx.closePath();
                ctx.stroke();
                break;
            }
            case "BEZIER": {
                const cps = parameters.control_points ?? [];
                if (cps.length < 4) break;
                ctx.beginPath();
                const start = cps[0];
                ctx.moveTo(start[0] * width, start[1] * height);
                for (let i = 1; i + 2 < cps.length; i += 3) {
                    const cp1 = cps[i];
                    const cp2 = cps[i + 1];
                    const end = cps[i + 2];
                    ctx.bezierCurveTo(
                        cp1[0] * width,
                        cp1[1] * height,
                        cp2[0] * width,
                        cp2[1] * height,
                        end[0] * width,
                        end[1] * height
                    );
                }
                ctx.stroke();
                break;
            }
            default:
                break;
        }

        ctx.restore();
    }

    static normalizeCoords(x, y, width, height) {
        return [x / width, y / height];
    }

    static denormalizeCoords(nx, ny, width, height) {
        return [nx * width, ny * height];
    }

    static snapToGrid(x, y, width, height, gridSize) {
        if (!gridSize || gridSize <= 0) return [x, y];
        const nx = Math.round((x / width) / gridSize) * gridSize;
        const ny = Math.round((y / height) / gridSize) * gridSize;
        return [nx * width, ny * height];
    }

    static checkAspectLock(width, height, aspectRatio) {
        if (!aspectRatio || aspectRatio <= 0) return [width, height];
        const ratio = aspectRatio;
        if (width / height > ratio) {
            width = height * ratio;
        } else {
            height = width / ratio;
        }
        return [width, height];
    }
}

class RectangleTool extends ShapeTool {
    constructor(options = {}) {
        super(options);
    }

    onMouseDown(event, normPoint, canvasInfo) {
        const { guides } = this;
        let [nx, ny] = normPoint;
        if (guides) {
            const snap = guides.getSnapPosition(
                nx * canvasInfo.width,
                ny * canvasInfo.height,
                canvasInfo.width,
                canvasInfo.height,
                canvasInfo.existingShapes
            );
            [nx, ny] = ShapeTool.normalizeCoords(
                snap.x,
                snap.y,
                canvasInfo.width,
                canvasInfo.height
            );
        }
        return super.onMouseDown(event, [nx, ny], canvasInfo);
    }

    onMouseMove(event, normPoint, canvasInfo) {
        if (!this.isDrawing) return null;
        let [nx, ny] = normPoint;
        if (this.guides) {
            const snap = this.guides.getSnapPosition(
                nx * canvasInfo.width,
                ny * canvasInfo.height,
                canvasInfo.width,
                canvasInfo.height,
                canvasInfo.existingShapes
            );
            [nx, ny] = ShapeTool.normalizeCoords(
                snap.x,
                snap.y,
                canvasInfo.width,
                canvasInfo.height
            );
        }
        this.updateDrawing([nx, ny]);
        this._previewShape = this.toShapeData(canvasInfo);
        return this.previewShape;
    }

    toShapeData(canvasInfo) {
        if (!this._start || this._points.length === 0) return null;

        const [startX, startY] = this._start;
        const [endX, endY] = this._points[this._points.length - 1];
        const x1 = Math.min(startX, endX);
        const x2 = Math.max(startX, endX);
        const y1 = Math.min(startY, endY);
        const y2 = Math.max(startY, endY);

        return this._composeShape("RECTANGLE", {
            x1, y1, x2, y2,
        }, canvasInfo);
    }

    _composeShape(type, parameters, canvasInfo) {
        const defaults = {
            z_order: canvasInfo?.zOrder ?? 0,
            blend_mode: canvasInfo?.blendMode ?? "NORMAL",
            feather: canvasInfo?.feather ?? 0,
            hardness: canvasInfo?.hardness ?? 1,
            weight: canvasInfo?.weight ?? 1,
        };

        if (this.options.aspectLock && this.options.aspectRatio > 0 && this._start && this._points.length) {
            const width = parameters.x2 - parameters.x1;
            const height = parameters.y2 - parameters.y1;
            const [w, h] = ShapeTool.checkAspectLock(width, height, this.options.aspectRatio);
            if (width !== 0 && height !== 0) {
                const cx = (parameters.x1 + parameters.x2) / 2;
                const cy = (parameters.y1 + parameters.y2) / 2;
                parameters.x1 = cx - w / 2;
                parameters.x2 = cx + w / 2;
                parameters.y1 = cy - h / 2;
                parameters.y2 = cy + h / 2;
            }
        }

        return {
            shape_type: type,
            parameters,
            ...defaults,
        };
    }
}

class EllipseTool extends ShapeTool {
    toShapeData(canvasInfo) {
        if (!this._start || this._points.length === 0) return null;

        const [startX, startY] = this._start;
        const [endX, endY] = this._points[this._points.length - 1];
        const cx = (startX + endX) / 2;
        const cy = (startY + endY) / 2;
        let rx = Math.abs(endX - startX) / 2;
        let ry = Math.abs(endY - startY) / 2;

        if (this.options.aspectLock && this.options.aspectRatio > 0) {
            const [w, h] = ShapeTool.checkAspectLock(rx * 2, ry * 2, this.options.aspectRatio);
            rx = Math.abs(w / 2);
            ry = Math.abs(h / 2);
        }

        const MIN_RADIUS = 0.001;
        rx = Math.max(rx, MIN_RADIUS);
        ry = Math.max(ry, MIN_RADIUS);

        return {
            shape_type: "ELLIPSE",
            parameters: { cx, cy, rx, ry },
            z_order: canvasInfo?.zOrder ?? 0,
            blend_mode: canvasInfo?.blendMode ?? "NORMAL",
            feather: canvasInfo?.feather ?? 0,
            hardness: canvasInfo?.hardness ?? 1,
            weight: canvasInfo?.weight ?? 1,
        };
    }
}

class PolygonTool extends ShapeTool {
    constructor(options = {}) {
        super(options);
        this._isClosed = false;
    }

    onMouseDown(event, normPoint, canvasInfo) {
        const { guides } = this;
        let [nx, ny] = normPoint;
        if (guides) {
            const snap = guides.getSnapPosition(
                nx * canvasInfo.width,
                ny * canvasInfo.height,
                canvasInfo.width,
                canvasInfo.height,
                canvasInfo.existingShapes
            );
            [nx, ny] = ShapeTool.normalizeCoords(
                snap.x,
                snap.y,
                canvasInfo.width,
                canvasInfo.height
            );
        }

        if (!this.isDrawing) {
            this.startDrawing([nx, ny]);
            // duplicate starting point for preview updates
            this._points = [[nx, ny], [nx, ny]];
            this._isClosed = false;
            return null;
        }

        const firstPoint = this._points[0];
        const [fx, fy] = firstPoint;
        const proximityThreshold = 10 / Math.max(canvasInfo.width, canvasInfo.height);
        const isCloseToStart = Math.hypot(fx - nx, fy - ny) < proximityThreshold;
        const isDoubleClick = Boolean(event?.detail >= 2);

        if (isCloseToStart || isDoubleClick) {
            this._isClosed = true;
            // lock current preview to starting point to close the loop
            this._points[this._points.length - 1] = [fx, fy];
            return this.onMouseUp(event, firstPoint, canvasInfo);
        }

        // commit current preview and append new preview anchor
        this._points[this._points.length - 1] = [nx, ny];
        this._points.push([nx, ny]);
        this._previewShape = this.toShapeData(canvasInfo);
        return this.previewShape;
    }

    onMouseMove(event, normPoint, canvasInfo) {
        if (!this.isDrawing || this._isClosed) return null;
        let [nx, ny] = normPoint;
        if (this.guides) {
            const snap = this.guides.getSnapPosition(
                nx * canvasInfo.width,
                ny * canvasInfo.height,
                canvasInfo.width,
                canvasInfo.height,
                canvasInfo.existingShapes
            );
            [nx, ny] = ShapeTool.normalizeCoords(
                snap.x,
                snap.y,
                canvasInfo.width,
                canvasInfo.height
            );
        }
        if (this._points.length)
            this._points[this._points.length - 1] = [nx, ny];
        this._previewShape = this.toShapeData(canvasInfo);
        return this.previewShape;
    }

    onMouseUp(event, normPoint, canvasInfo) {
        if (!this.isDrawing) return null;
        if (!this._isClosed) {
            this._previewShape = this.toShapeData(canvasInfo);
            return null;
        }

        const points = this._points.slice(0, -1); // drop trailing preview
        if (points.length < 3) {
            this.cancelDrawing();
            this._isClosed = false;
            return null;
        }

        const shape = {
            shape_type: "POLYGON",
            parameters: { points },
            z_order: canvasInfo?.zOrder ?? 0,
            blend_mode: canvasInfo?.blendMode ?? "NORMAL",
            feather: canvasInfo?.feather ?? 0,
            hardness: canvasInfo?.hardness ?? 1,
            weight: canvasInfo?.weight ?? 1,
        };

        this.cancelDrawing();
        this._isClosed = false;
        return shape;
    }

    toShapeData(canvasInfo) {
        const points = this._points.slice();
        if (points.length < 3) return null;
        if (!this._isClosed && points.length > 1)
            points[points.length - 1] = points[points.length - 1].slice();
        return {
            shape_type: "POLYGON",
            parameters: { points },
            z_order: canvasInfo?.zOrder ?? 0,
            blend_mode: canvasInfo?.blendMode ?? "NORMAL",
            feather: canvasInfo?.feather ?? 0,
            hardness: canvasInfo?.hardness ?? 1,
            weight: canvasInfo?.weight ?? 1,
        };
    }
}

class BezierTool extends ShapeTool {
    constructor(options = {}) {
        super(options);
        this._segmentIndex = 0;
    }

    onMouseDown(event, normPoint, canvasInfo) {
        const { guides } = this;
        let [nx, ny] = normPoint;
        if (guides) {
            const snap = guides.getSnapPosition(
                nx * canvasInfo.width,
                ny * canvasInfo.height,
                canvasInfo.width,
                canvasInfo.height,
                canvasInfo.existingShapes
            );
            [nx, ny] = ShapeTool.normalizeCoords(
                snap.x,
                snap.y,
                canvasInfo.width,
                canvasInfo.height
            );
        }

        if (!this.isDrawing) {
            this.startDrawing([nx, ny]);
            this._points = [[nx, ny], [nx, ny]];
            this._segmentIndex = 0;
            return null;
        }

        const isDoubleClick = Boolean(event?.detail >= 2);

        if (isDoubleClick) {
            this._commitPoint([nx, ny], false);
            return this._finalize(canvasInfo);
        }

        this._commitPoint([nx, ny], true);
        this._previewShape = this.toShapeData(canvasInfo);
        return this.previewShape;
    }

    onMouseMove(event, normPoint, canvasInfo) {
        if (!this.isDrawing) return null;
        let [nx, ny] = normPoint;
        if (this.guides) {
            const snap = this.guides.getSnapPosition(
                nx * canvasInfo.width,
                ny * canvasInfo.height,
                canvasInfo.width,
                canvasInfo.height,
                canvasInfo.existingShapes
            );
            [nx, ny] = ShapeTool.normalizeCoords(
                snap.x,
                snap.y,
                canvasInfo.width,
                canvasInfo.height
            );
        }

        if (this._points.length)
            this._points[this._points.length - 1] = [nx, ny];
        this._previewShape = this.toShapeData(canvasInfo);
        return this.previewShape;
    }

    onMouseUp(event, normPoint, canvasInfo) {
        if (!this.isDrawing) return null;
        this.updateDrawing(normPoint);
        this._previewShape = this.toShapeData(canvasInfo);
        return null;
    }

    onKeyDown(event, canvasInfo) {
        if (!this.isDrawing)
            return null;
        if (event.key?.toLowerCase() !== "enter")
            return null;
        event.preventDefault();
        return this._finalize(canvasInfo);
    }

    toShapeData(canvasInfo) {
        const points = this._points.slice();
        if (points.length < 4) return null;

        const effective = points.slice();
        if ((effective.length - 1) % 3 !== 0)
            effective.pop();
        if (effective.length < 4)
            return null;

        return {
            shape_type: "BEZIER",
            parameters: { control_points: effective },
            z_order: canvasInfo?.zOrder ?? 0,
            blend_mode: canvasInfo?.blendMode ?? "NORMAL",
            feather: canvasInfo?.feather ?? 0,
            hardness: canvasInfo?.hardness ?? 1,
            weight: canvasInfo?.weight ?? 1,
        };
    }

    _commitPoint(point, addPreview = true) {
        if (!Array.isArray(point) || point.length !== 2)
            return;
        if (this._points.length)
            this._points[this._points.length - 1] = [point[0], point[1]];
        else
            this._points.push([point[0], point[1]]);

        if (addPreview)
            this._points.push([point[0], point[1]]);

        this._segmentIndex += 1;
        if (this._segmentIndex >= 3)
            this._segmentIndex = 0;
    }

    _finalize(canvasInfo) {
        const points = this._points.slice();
        if (points.length > 1 && (points.length - 1) % 3 !== 0)
            points.pop();

        if (points.length < 4 || (points.length - 1) % 3 !== 0)
            return null;

        const shape = {
            shape_type: "BEZIER",
            parameters: { control_points: points },
            z_order: canvasInfo?.zOrder ?? 0,
            blend_mode: canvasInfo?.blendMode ?? "NORMAL",
            feather: canvasInfo?.feather ?? 0,
            hardness: canvasInfo?.hardness ?? 1,
            weight: canvasInfo?.weight ?? 1,
        };

        this.cancelDrawing();
        this._segmentIndex = 0;
        return shape;
    }
}

class ShapeToolManager {
    constructor(options = {}) {
        this.canvasInfo = {
            width: 1,
            height: 1,
            existingShapes: [],
            zOrder: 0,
            blendMode: "NORMAL",
            feather: 0,
            hardness: 1,
            weight: 1,
        };

        const baseOpts = {
            canvas: options.canvas ?? null,
            guides: options.guides ?? null,
        };

        this._tools = {
            RECTANGLE: new RectangleTool(baseOpts),
            ELLIPSE: new EllipseTool(baseOpts),
            POLYGON: new PolygonTool(baseOpts),
            BEZIER: new BezierTool(baseOpts),
        };

        this._current = this._tools.RECTANGLE;
    }

    setCanvas(canvas) {
        Object.values(this._tools).forEach((tool) => tool.setOptions({ canvas }));
    }

    setGuides(guides) {
        Object.values(this._tools).forEach((tool) => tool.setOptions({ guides }));
    }

    setToolOptions(options) {
        Object.values(this._tools).forEach((tool) => tool.setOptions(options));
    }

    setMetadataDefaults(defaults) {
        this.canvasInfo = {
            ...this.canvasInfo,
            ...defaults,
        };
    }

    updateCanvasMetrics(width, height, existingShapes = []) {
        this.canvasInfo.width = width;
        this.canvasInfo.height = height;
        this.canvasInfo.existingShapes = existingShapes;
    }

    setTool(toolName) {
        if (toolName in this._tools) {
            this._current?.cancelDrawing?.();
            this._current = this._tools[toolName];
        }
    }

    get currentTool() {
        return this._current;
    }

    handleMouseDown(event, point) {
        if (!this._current) return null;
        return this._current.onMouseDown(event, point, this.canvasInfo);
    }

    handleMouseMove(event, point) {
        if (!this._current) return null;
        return this._current.onMouseMove(event, point, this.canvasInfo);
    }

    handleMouseUp(event, point) {
        if (!this._current) return null;
        return this._current.onMouseUp(event, point, this.canvasInfo);
    }

    handleKeyDown(event) {
        if (!this._current) return null;
        return this._current.onKeyDown(event, this.canvasInfo);
    }
}

window.ShapeTool = ShapeTool;
window.RectangleTool = RectangleTool;
window.EllipseTool = EllipseTool;
window.PolygonTool = PolygonTool;
window.BezierTool = BezierTool;
window.ShapeToolManager = ShapeToolManager;
