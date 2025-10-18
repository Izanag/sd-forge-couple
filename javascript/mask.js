class ForgeCoupleMaskHandler {

    #group = undefined;
    #gallery = undefined;
    #preview = undefined;
    #separatorField = undefined;
    #background = undefined;
    #promptField = undefined;
    #weightField = undefined;
    #shapeField = undefined;
    #operationField = undefined;
    #operationButton = undefined;
    #loadButton = undefined;
    #shapeMetadata = [];
    #selectedIndex = -1;
    #mode = "t2i";

    /** @returns {string} */
    get #sep() {
        let sep = this.#separatorField.value.trim();
        sep = (!sep) ? "\n" : sep.replace(/\\n/g, "\n").split("\n").map(c => c.trim()).join("\n");
        return sep;
    }

    /** @returns {boolean} */
    get #selectionAvailable() { return !(this.#loadButton.disabled); }

    /**
     * @param {HTMLDivElement} group
     * @param {HTMLDivElement} gallery
     * @param {HTMLDivElement} preview
     * @param {HTMLInputElement} sep
     * @param {HTMLInputElement} background
     * @param {HTMLTextAreaElement} promptField
     * @param {HTMLTextAreaElement} shapeField
     * @param {HTMLTextAreaElement} op
     * @param {HTMLButtonElement} opButton
     * @param {HTMLButtonElement} loadButton
     * @param {"t2i" | "i2i"} mode
     */
    constructor(group, gallery, preview, sep, background, promptField, weightField, shapeField, op, opButton, loadButton, mode = "t2i") {
        this.#group = group;
        this.#gallery = gallery;
        this.#preview = preview;
        this.#separatorField = sep;
        this.#background = background;
        this.#promptField = promptField;
        this.#weightField = weightField;
        this.#shapeField = shapeField;
        this.#operationField = op;
        this.#operationButton = opButton;
        this.#loadButton = loadButton;
        this.#mode = mode;

        this.#separatorField.addEventListener("blur", () => { this.syncPrompts(); });
        if (this.#shapeField)
            this.#shapeField.addEventListener("change", () => { this.#parseShapeField(); });

        this.#parseShapeField();
    }

    /** @returns {HTMLDivElement[]} */
    get #allRows() {
        return this.#preview.querySelectorAll(".fc_mask_row");
    }

    hideButtons() {
        const undo = this.#group.querySelector("button[aria-label='Undo']");
        if (undo == null || undo.style.display === "none")
            return;

        undo.style.display = "none";

        const clear = this.#group.querySelector("button[aria-label='Clear']");
        clear.style.display = "none";

        const remove = this.#group.querySelector("button[aria-label='Remove Image']");
        remove.style.display = "none";

        const brush = this.#group.querySelector("button[aria-label='Use brush']");
        brush.firstElementChild.style.width = "20px";
        brush.firstElementChild.style.height = "20px";

        const color = this.#group.querySelector("button[aria-label='Select brush color']");
        color.firstElementChild.style.width = "20px";
        color.firstElementChild.style.height = "20px";

        brush.parentElement.parentElement.style.top = "var(--size-2)";
        brush.parentElement.parentElement.style.right = "var(--size-10)";
    }

    generatePreview() {
        const imgs = this.#gallery.querySelectorAll("img");
        const maskCount = imgs.length;

        // Clear Excess Rows
        while (this.#preview.children.length > maskCount)
            this.#preview.lastElementChild.remove();

        // Append Insufficient Rows
        while (this.#preview.children.length < maskCount) {
            const row = document.createElement("div");
            row.classList.add("fc_mask_row");
            this.#preview.appendChild(row);
        }

        this.#populateRows(this.#allRows, imgs);
        this.syncPrompts();
        this.parseWeights();
        this.#applyShapeMetadataToRows();

        if (!this.#selectionAvailable) {
            const lastSelected = this.#preview.querySelector(".selected");
            if (lastSelected) lastSelected.classList.remove("selected");
            this.#selectedIndex = -1;
            if (typeof window.ForgeCouple?.syncShapeMetadata === "function")
                window.ForgeCouple.syncShapeMetadata(this.#mode, null);
        } else if (this.#selectedIndex >= maskCount) {
            this.#selectedIndex = -1;
            if (typeof window.ForgeCouple?.syncShapeMetadata === "function")
                window.ForgeCouple.syncShapeMetadata(this.#mode, null);
        }
    }

    /** @param {HTMLDivElement} row */
    #constructRow(row) {
        if (row.hasOwnProperty("setup"))
            return;

        const img = document.createElement("img");
        img.setAttribute('style', 'width: 96px !important; height: 96px !important; object-fit: contain;');
        img.title = "Select this Mask";
        row.appendChild(img);
        row.img = img;

        img.addEventListener("click", () => { this.#onSelectRow(row); });

        const txt = document.createElement("input");
        txt.setAttribute('style', 'width: 80%;');
        txt.setAttribute("type", "text");
        row.appendChild(txt);
        row.txt = txt;

        txt.value = "";
        txt.addEventListener("blur", () => { this.#onSubmitPrompt(txt); });

        const weight = document.createElement("input");
        weight.setAttribute('style', 'width: 10%;');
        weight.setAttribute("type", "number");
        weight.title = "Weight";
        row.appendChild(weight);
        row.weight = weight;

        weight.value = Number(1.0).toFixed(2);
        weight.addEventListener("blur", () => { this.#onSubmitWeight(weight); });

        const meta = document.createElement("div");
        meta.classList.add("fc_shape_meta");
        row.appendChild(meta);

        const typeIndicator = document.createElement("span");
        typeIndicator.classList.add("fc_shape_type");
        typeIndicator.textContent = "RECT";
        meta.appendChild(typeIndicator);
        row.shapeType = typeIndicator;

        const blend = document.createElement("select");
        blend.classList.add("fc_shape_blend");
        const blendModes = (() => {
            const modes = window.ForgeCouple?.getBlendModes?.();
            return Array.isArray(modes) && modes.length ? modes : ["NORMAL", "MULTIPLY", "OVERLAY"];
        })();
        blendModes.forEach((mode) => {
            const option = document.createElement("option");
            option.value = mode;
            option.textContent = mode;
            blend.appendChild(option);
        });
        meta.appendChild(blend);
        row.blendMode = blend;

        const feather = document.createElement("input");
        feather.setAttribute("type", "range");
        feather.setAttribute("min", "0");
        feather.setAttribute("max", "1");
        feather.setAttribute("step", "0.01");
        feather.setAttribute("value", "0");
        feather.title = "Edge Feather";
        feather.classList.add("fc_shape_feather");
        meta.appendChild(feather);
        row.feather = feather;

        const hardness = document.createElement("input");
        hardness.setAttribute("type", "range");
        hardness.setAttribute("min", "0");
        hardness.setAttribute("max", "1");
        hardness.setAttribute("step", "0.01");
        hardness.setAttribute("value", "1");
        hardness.title = "Edge Hardness";
        hardness.classList.add("fc_shape_hardness");
        meta.appendChild(hardness);
        row.hardness = hardness;

        const featherEdgesContainer = document.createElement("div");
        featherEdgesContainer.classList.add("fc_shape_feather_edges");
        meta.appendChild(featherEdgesContainer);
        const featherEdgeInputs = {};
        ["top", "right", "bottom", "left"].forEach((edge) => {
            const edgeInput = document.createElement("input");
            edgeInput.setAttribute("type", "range");
            edgeInput.setAttribute("min", "0");
            edgeInput.setAttribute("max", "1");
            edgeInput.setAttribute("step", "0.01");
            edgeInput.setAttribute("value", "0");
            edgeInput.title = `Feather (${edge})`;
            edgeInput.classList.add("fc_shape_feather_edge");
            edgeInput.dataset.edge = edge;
            edgeInput.dataset.value = "0.00";
            featherEdgesContainer.appendChild(edgeInput);
            featherEdgeInputs[edge] = edgeInput;
            edgeInput.addEventListener("input", () => {
                const value = this.#clamp01(edgeInput.value);
                edgeInput.value = value;
                edgeInput.dataset.value = value.toFixed(2);
                edgeInput.title = `Feather (${edge}): ${edgeInput.dataset.value}`;
                this.#onShapeControlChange(row, "feather_edges", { [edge]: value });
            });
        });
        row.featherEdges = featherEdgeInputs;

        const zOrder = document.createElement("input");
        zOrder.setAttribute("type", "number");
        zOrder.setAttribute("step", "1");
        zOrder.value = "0";
        zOrder.title = "Layer Order";
        zOrder.classList.add("fc_shape_z");
        meta.appendChild(zOrder);
        row.zOrder = zOrder;

        blend.addEventListener("change", () => { this.#onShapeControlChange(row, "blend_mode", blend.value); });
        feather.addEventListener("input", () => {
            feather.dataset.value = Number.parseFloat(feather.value).toFixed(2);
            feather.title = `Edge Feather: ${feather.dataset.value}`;
            this.#onShapeControlChange(row, "feather", parseFloat(feather.value));
        });
        hardness.addEventListener("input", () => {
            hardness.dataset.value = Number.parseFloat(hardness.value).toFixed(2);
            hardness.title = `Edge Hardness: ${hardness.dataset.value}`;
            this.#onShapeControlChange(row, "hardness", parseFloat(hardness.value));
        });
        zOrder.addEventListener("change", () => {
            this.#onShapeControlChange(row, "z_order", parseInt(zOrder.value, 10) || 0);
        });

        const del = document.createElement("button");
        del.classList.add("del");
        del.textContent = "❌";
        del.title = "Delete this Mask";
        row.appendChild(del);

        del.addEventListener("click", () => { this.#onDeleteRow(row); });

        const up = document.createElement("button");
        up.classList.add("up");
        up.textContent = "^";
        up.title = "Move this Layer Up";
        row.appendChild(up);

        up.addEventListener("click", () => { this.#onShiftRow(row, true); });

        const down = document.createElement("button");
        down.classList.add("down");
        down.textContent = "^";
        down.title = "Move this Layer Down";
        row.appendChild(down);

        down.addEventListener("click", () => { this.#onShiftRow(row, false); });

        row.setup = true;
    }

    /** @param {HTMLDivElement[]} rows @param {HTMLImageElement[]} imgs */
    #populateRows(rows, imgs) {
        const len = rows.length;

        for (let i = 0; i < len; i++) {
            this.#constructRow(rows[i]);
            rows[i].img.src = imgs[i].src;
        }
    }

    #rowIndex(row) {
        return Array.from(this.#allRows).indexOf(row);
    }

    #onShapeControlChange(row, key, value) {
        const index = this.#rowIndex(row);
        if (index < 0)
            return;

        const existing = (typeof this.#shapeMetadata[index] === "object" && this.#shapeMetadata[index] !== null)
            ? this.#shapeMetadata[index]
            : { shape_type: "", parameters: {} };
        if (key === "feather_edges" && typeof value === "object" && value !== null) {
            const merged = { ...(existing.feather_edges || {}) };
            Object.entries(value).forEach(([edge, val]) => {
                const norm = this.#clamp01(val);
                if (Number.isFinite(norm))
                    merged[edge] = norm;
            });
            const cleaned = {};
            ["top", "right", "bottom", "left"].forEach((edge) => {
                if (Number.isFinite(merged[edge]))
                    cleaned[edge] = this.#clamp01(merged[edge]);
            });
            const edges = Object.keys(cleaned).length ? cleaned : null;
            this.#shapeMetadata[index] = {
                ...existing,
                feather_edges: edges,
            };
        } else {
            this.#shapeMetadata[index] = {
                ...existing,
                [key]: value,
            };
        }

        this.#serializeShapeField();
    }

    #parseShapeField() {
        if (!this.#shapeField) {
            this.#shapeMetadata = [];
            return;
        }

        const raw = this.#shapeField.value?.trim();
        if (!raw) {
            this.#shapeMetadata = [];
            return;
        }

        try {
            const parsed = JSON.parse(raw);
            this.#shapeMetadata = Array.isArray(parsed) ? parsed : [];
        } catch {
            this.#shapeMetadata = [];
        }
    }

    #serializeShapeField() {
        if (!this.#shapeField)
            return;

        const cleaned = this.#shapeMetadata.map((shape) => shape ?? null);
        this.#shapeField.value = JSON.stringify(cleaned);
        if (typeof updateInput === "function")
            updateInput(this.#shapeField);
    }

    #applyShapeMetadataToRows() {
        const rows = Array.from(this.#allRows);
        const count = rows.length;

        while (this.#shapeMetadata.length < count)
            this.#shapeMetadata.push(null);

        rows.forEach((row, index) => {
            const metadata = this.#shapeMetadata[index];
            this.#updateRowShapeControls(row, metadata);
        });

        this.#serializeShapeField();
    }

    #updateRowShapeControls(row, metadata) {
        if (!row.shapeType || !row.blendMode || !row.feather || !row.hardness || !row.zOrder)
            return;

        const shapeType = metadata?.shape_type ?? "";
        row.shapeType.textContent = shapeType ? shapeType.slice(0, 4).toUpperCase() : "NONE";

        const blend = metadata?.blend_mode ?? "NORMAL";
        row.blendMode.value = blend;

        const feather = Number.isFinite(metadata?.feather) ? metadata.feather : 0;
        row.feather.value = feather;
        row.feather.dataset.value = feather.toFixed(2);
        row.feather.title = `Edge Feather: ${row.feather.dataset.value}`;

        const hardness = Number.isFinite(metadata?.hardness) ? metadata.hardness : 1;
        row.hardness.value = hardness;
        row.hardness.dataset.value = hardness.toFixed(2);
        row.hardness.title = `Edge Hardness: ${row.hardness.dataset.value}`;

        const zOrder = Number.isFinite(metadata?.z_order) ? metadata.z_order : 0;
        row.zOrder.value = zOrder;

        if (row.weight && Number.isFinite(metadata?.weight))
            row.weight.value = Number(metadata.weight).toFixed(2);

        if (row.featherEdges) {
            const edges = metadata?.feather_edges || {};
            ["top", "right", "bottom", "left"].forEach((edge) => {
                const input = row.featherEdges[edge];
                if (!input)
                    return;
                const value = Number.isFinite(edges[edge]) ? this.#clamp01(edges[edge]) : 0;
                input.value = value;
                input.dataset.value = value.toFixed(2);
                input.title = `Feather (${edge}): ${input.dataset.value}`;
            });
        }
    }

    getShapeData(index) {
        if (index < 0 || index >= this.#shapeMetadata.length)
            return null;
        return this.#shapeMetadata[index];
    }

    setShapeData(index, shapeData) {
        if (index < 0)
            return;

        while (this.#shapeMetadata.length <= index)
            this.#shapeMetadata.push(null);

        const clonedShape = shapeData ? JSON.parse(JSON.stringify(shapeData)) : shapeData;
        this.#shapeMetadata[index] = clonedShape;

        const row = this.#allRows[index];
        if (row)
            this.#updateRowShapeControls(row, clonedShape);

        this.#serializeShapeField();
        this.setPendingIndicator(false);
    }

    updateSelectedShape(shapeData) {
        if (this.#selectedIndex < 0)
            return;
        this.setShapeData(this.#selectedIndex, shapeData);
    }

    hasSelection() {
        return this.#selectedIndex >= 0 && this.#selectedIndex < this.#allRows.length;
    }

    ensureSelection() {
        if (this.hasSelection())
            return this.#selectedIndex;
        const rows = Array.from(this.#allRows);
        if (!rows.length)
            return -1;
        this.#onSelectRow(rows[0]);
        return this.#selectedIndex;
    }

    getSelectedIndex() {
        return this.#selectedIndex;
    }

    getSelectedShape() {
        if (!this.hasSelection())
            return null;
        return this.getShapeData(this.#selectedIndex);
    }

    setPendingIndicator(isPending) {
        if (!this.#preview)
            return;
        if (isPending)
            this.#preview.classList.add("fc_pending_shape");
        else
            this.#preview.classList.remove("fc_pending_shape");
    }

    getAllShapeData() {
        return this.#shapeMetadata.slice();
    }

    /** @param {HTMLInputElement} field */
    #onSubmitPrompt(field) {
        const prompts = [];
        this.#allRows.forEach((row) => {
            prompts.push(row.txt.value);
        });

        const radio = this.#background.querySelector('div.wrap>label.selected>span');
        const background = radio.textContent;

        const existingPrompt = this.#promptField.value
            .split(this.#sep).map(line => line.trim());

        if (existingPrompt.length > 0) {
            if (background == "First Line")
                prompts.unshift(existingPrompt.shift());
            else if (background == "Last Line")
                prompts.push(existingPrompt.pop());
        }

        const oldLen = existingPrompt.length;
        const newLen = prompts.length;

        if ((newLen >= oldLen) || (oldLen === 0)) {
            this.#promptField.value = prompts.join(this.#sep);
            updateInput(this.#promptField);
        }
        else {
            const newPrompts = [...prompts, ...(existingPrompt.slice(newLen))];
            this.#promptField.value = newPrompts.join(this.#sep);
            updateInput(this.#promptField);
        }
    }

    /** @param {HTMLInputElement} field */
    #onSubmitWeight(field) {
        const w = this.#clamp05(field.value);
        field.value = Number(w).toFixed(2);
        this.parseWeights();
    }

    /** @param {HTMLDivElement} row */
    #onSelectRow(row) {
        const rows = Array.from(this.#allRows);
        const index = rows.indexOf(row);

        const lastSelected = this.#preview.querySelector(".selected");
        if (lastSelected) lastSelected.classList.remove("selected");

        row.classList.add("selected");
        this.#selectedIndex = index;

        if (typeof window.ForgeCouple?.syncShapeMetadata === "function") {
            const shape = this.getShapeData(index);
            window.ForgeCouple.syncShapeMetadata(this.#mode, shape || null);
        }
        if (typeof window.ForgeCouple?.applyPendingShape === "function")
            window.ForgeCouple.applyPendingShape(this.#mode);

        this.#operationField.value = `${index}`;
        updateInput(this.#operationField);
        this.#operationButton.click();
    }

    /** @param {HTMLDivElement} row */
    #onDeleteRow(row) {
        const rows = Array.from(this.#allRows);
        const index = rows.indexOf(row);

        if (index >= 0) {
            this.#shapeMetadata.splice(index, 1);
            this.#serializeShapeField();
        }

        this.#operationField.value = `-${index}`;
        updateInput(this.#operationField);
        this.#operationButton.click();
    }

    /** @param {HTMLDivElement} row @param {boolean} isUp */
    #onShiftRow(row, isUp) {
        const rows = Array.from(this.#allRows);
        const index = rows.indexOf(row);
        const target = isUp ? index - 1 : index + 1;

        if (target < 0 || target >= rows.length)
            return;

        const temp = this.#shapeMetadata[index];
        this.#shapeMetadata[index] = this.#shapeMetadata[target];
        this.#shapeMetadata[target] = temp;
        this.#serializeShapeField();

        this.#operationField.value = `${index}=${target}`;
        updateInput(this.#operationField);
        this.#operationButton.click();
    }

    syncPrompts() {
        const prompt = this.#promptField.value;
        let prompts = prompt.split(this.#sep).map(line => line.trim());

        const radio = this.#background.querySelector('div.wrap>label.selected>span');
        const background = radio.textContent;

        if (background == "First Line")
            prompts = prompts.slice(1);
        else if (background == "Last Line")
            prompts = prompts.slice(0, -1);

        const active = document.activeElement;
        this.#allRows.forEach((row, i) => {
            const promptCell = row.txt;

            // Skip editing Cell
            if (promptCell === active)
                return;

            if (i < prompts.length)
                promptCell.value = prompts[i].replace(/\n+/g, ", ").replace(/,+/g, ",");
            else
                promptCell.value = "";
        });
    }

    parseWeights() {
        const weights = [];
        this.#allRows.forEach((row) => {
            weights.push(row.weight.value);
        });

        this.#weightField.value = weights.join(",");
        updateInput(this.#weightField);

        weights.forEach((weight, index) => {
            if (!this.#shapeMetadata[index])
                return;
            this.#shapeMetadata[index] = {
                ...this.#shapeMetadata[index],
                weight: parseFloat(weight) || 0,
            };
        });
        this.#serializeShapeField();
    }

    /** @param {number} v @returns {number} */
    #clamp01(v) {
        let val = parseFloat(v);
        if (Number.isNaN(val))
            val = 0.0;
        return Math.min(Math.max(val, 0.0), 1.0);
    }

    /** @param {number} v @returns {number} */
    #clamp05(v) {
        let val = parseFloat(v);
        if (Number.isNaN(val))
            val = 0.0;

        return Math.min(Math.max(val, 0.0), 5.0);
    }
}
