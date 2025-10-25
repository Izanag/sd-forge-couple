class ForgeCoupleDataframe {

    static #default_mapping = [
        [0.0, 0.5, 0.0, 1.0, 1.0],
        [0.5, 1.0, 0.0, 1.0, 1.0]
    ];

    static get #columns() { return this.#tableHeader.length; }

    static #tableHeader = ["x1", "x2", "y1", "y2", "w", "prompt", "neg_prompt"];
    static #tableWidth = ["6%", "6%", "6%", "6%", "6%", "35%", "35%"];

    static #colors = [0, 30, 60, 120, 240, 280, 320];
    static #color(i) { return `hsl(${ForgeCoupleDataframe.#colors[i % 7]}, 36%, 36%)` }

    /** "t2i" | "i2i" */
    #mode = undefined;
    #promptField = undefined;
    #negPromptField = undefined;
    #separatorField = undefined;

    static #columnType(index) {
        if (index === this.#columns - 2)
            return "prompt";
        if (index === this.#columns - 1)
            return "neg_prompt";
        return "number";
    }

    get #sep() {
        let sep = this.#separatorField.value.trim();
        sep = (!sep) ? "\n" : sep.replace(/\\n/g, "\n").split("\n").map(c => c.trim()).join("\n");
        return sep;
    }

    #body = undefined;
    #selection = -1;

    /** @param {Element} div @param {string} mode @param {Element} separator */
    constructor(div, mode, separator) {
        this.#mode = mode;
        const promptWrapper = document.getElementById(`${mode === "t2i" ? "txt" : "img"}2img_prompt`);
        this.#promptField = promptWrapper ? promptWrapper.querySelector("textarea") : undefined;
        const negPromptWrapper = document.getElementById(`${mode === "t2i" ? "txt" : "img"}2img_neg_prompt`);
        this.#negPromptField = negPromptWrapper ? negPromptWrapper.querySelector("textarea") : undefined;
        this.#separatorField = separator;

        this.#separatorField.addEventListener("blur", () => {
            this.syncPrompts();
            this.syncNegativePrompts();
        });
        const table = document.createElement('table');


        const colgroup = document.createElement('colgroup');
        for (let c = 0; c < ForgeCoupleDataframe.#columns; c++) {
            const col = document.createElement('col');
            col.style.width = ForgeCoupleDataframe.#tableWidth[c];
            colgroup.appendChild(col);
        }
        table.appendChild(colgroup);


        const thead = document.createElement('thead');
        const thr = thead.insertRow();
        for (let c = 0; c < ForgeCoupleDataframe.#columns; c++) {
            const th = document.createElement('th');
            th.textContent = ForgeCoupleDataframe.#tableHeader[c];
            thr.appendChild(th);
        }
        table.appendChild(thead);


        const tbody = document.createElement('tbody');
        for (let r = 0; r < ForgeCoupleDataframe.#default_mapping.length; r++) {
            const tr = tbody.insertRow();

            for (let c = 0; c < ForgeCoupleDataframe.#columns; c++) {
                const td = tr.insertCell();
                const columnType = ForgeCoupleDataframe.#columnType(c);

                td.contentEditable = true;
                td.textContent = (columnType === "number") ? Number(ForgeCoupleDataframe.#default_mapping[r][c]).toFixed(2) : "";

                td.addEventListener("keydown", (e) => {
                    if (e.key == 'Enter') {
                        e.preventDefault();
                        td.blur();
                    }
                });

                td.addEventListener("blur", () => { this.#onSubmit(td, columnType); })
                td.onclick = () => { this.#onSelect(r); }
            }
        }
        table.appendChild(tbody);


        div.appendChild(table);
        this.#body = tbody;
    }

    /** @param {number} row */
    #onSelect(row) {
        this.#selection = (row === this.#selection) ? -1 : row;
        ForgeCouple.onSelect(this.#mode);
    }

    /** @param {Element} cell @param {"number"|"prompt"|"neg_prompt"} columnType */
    #onSubmit(cell, columnType) {
        if (columnType === "prompt" || columnType === "neg_prompt") {
            const columnIndex = (columnType === "prompt")
                ? ForgeCoupleDataframe.#columns - 2
                : ForgeCoupleDataframe.#columns - 1;
            const targetField = (columnType === "prompt") ? this.#promptField : this.#negPromptField;
            if (!targetField)
                return;

            const prompts = [];
            const rows = this.#body.querySelectorAll("tr");
            rows.forEach((row) => {
                const prompt = row.querySelectorAll("td")[columnIndex].textContent.trim();
                prompts.push(prompt);
            });

            const oldPrompts = targetField.value.split(this.#sep).map(line => line.trim());
            const modified = prompts.length;

            if (modified >= oldPrompts.length)
                targetField.value = prompts.join(this.#sep);
            else {
                const newPrompts = [...prompts, ...(oldPrompts.slice(modified))]
                targetField.value = newPrompts.join(this.#sep);
            }

            updateInput(targetField);
            return;
        } else {
            let val = this.#clamp01(cell.textContent,
                Array.from(cell.parentElement.children).indexOf(cell) === 4
            );
            val = Math.round(val / 0.01) * 0.01;
            cell.textContent = Number(val).toFixed(2);
            ForgeCouple.onSelect(this.#mode);
            ForgeCouple.onEntry(this.#mode);
        }
    }

    /** @param {number[][]} vals */
    onPaste(vals) {
        while (this.#body.querySelector("tr"))
            this.#body.deleteRow(0);

        const count = vals.length;

        for (let r = 0; r < count; r++) {
            const tr = this.#body.insertRow();

            for (let c = 0; c < ForgeCoupleDataframe.#columns; c++) {
                const td = tr.insertCell();
                const columnType = ForgeCoupleDataframe.#columnType(c);

                td.contentEditable = true;
                td.textContent = (columnType === "number") ? Number(vals[r][c]).toFixed(2) : "";

                td.addEventListener("keydown", (e) => {
                    if (e.key == 'Enter') {
                        e.preventDefault();
                        td.blur();
                    }
                });

                td.addEventListener("blur", () => { this.#onSubmit(td, columnType); })
                td.onclick = () => { this.#onSelect(r); }
            }
        }

        this.#selection = -1;
        ForgeCouple.onSelect(this.#mode);
        ForgeCouple.onEntry(this.#mode);
        this.syncPrompts();
        this.syncNegativePrompts();
    }

    reset() {
        while (this.#body.querySelector("tr"))
            this.#body.deleteRow(0);

        for (let r = 0; r < ForgeCoupleDataframe.#default_mapping.length; r++) {
            const tr = this.#body.insertRow();

            for (let c = 0; c < ForgeCoupleDataframe.#columns; c++) {
                const td = tr.insertCell();
                const columnType = ForgeCoupleDataframe.#columnType(c);

                td.contentEditable = true;
                td.textContent = (columnType === "number") ? Number(ForgeCoupleDataframe.#default_mapping[r][c]).toFixed(2) : "";

                td.addEventListener("keydown", (e) => {
                    if (e.key == 'Enter') {
                        e.preventDefault();
                        td.blur();
                    }
                });

                td.addEventListener("blur", () => { this.#onSubmit(td, columnType); })
                td.onclick = () => { this.#onSelect(r); }
            }
        }

        this.#selection = -1;
        ForgeCouple.onSelect(this.#mode);
        ForgeCouple.onEntry(this.#mode);
        this.syncPrompts();
        this.syncNegativePrompts();
    }

    /** @returns {number[][]} */
    #newRow() {
        const rows = this.#body.querySelectorAll("tr");
        const count = rows.length;

        const vals = Array.from(rows, row => {
            return Array.from(row.querySelectorAll("td"))
                .slice(0, -2).map(cell => parseFloat(cell.textContent));
        });

        const tr = this.#body.insertRow();

        for (let c = 0; c < ForgeCoupleDataframe.#columns; c++) {
            const td = tr.insertCell();
            const columnType = ForgeCoupleDataframe.#columnType(c);

            td.contentEditable = true;
            td.textContent = "";

            td.addEventListener("keydown", (e) => {
                if (e.key == 'Enter') {
                    e.preventDefault();
                    td.blur();
                }
            });

            td.addEventListener("blur", () => { this.#onSubmit(td, columnType); })
            td.onclick = () => { this.#onSelect(count); }
        }

        return vals;
    }

    /** @param {boolean} newline */
    newRowAbove(newline) {
        const vals = this.#newRow();

        const newVals = [
            ...vals.slice(0, this.#selection),
            [0.0, 1.0, 0.0, 1.0, 1.0],
            ...vals.slice(this.#selection)
        ];

        const count = newVals.length;
        const rows = this.#body.querySelectorAll("tr");

        for (let r = 0; r < count; r++) {
            const cells = rows[r].querySelectorAll("td");
            for (let c = 0; c < ForgeCoupleDataframe.#columns - 2; c++)
                cells[c].textContent = Number(newVals[r][c]).toFixed(2);
        }

        if (newline) {
            this.#insertPromptLine(this.#promptField, this.#selection);
            this.#insertPromptLine(this.#negPromptField, this.#selection);
        }

        this.#selection += 1;
        ForgeCouple.onSelect(this.#mode);
        ForgeCouple.onEntry(this.#mode);
        this.syncPrompts();
        this.syncNegativePrompts();
    }

    /** @param {boolean} newline */
    newRowBelow(newline) {
        const vals = this.#newRow();

        const newVals = [
            ...vals.slice(0, this.#selection + 1),
            [0.25, 0.75, 0.25, 0.75, 1.0],
            ...vals.slice(this.#selection + 1)
        ];

        const count = newVals.length;
        const rows = this.#body.querySelectorAll("tr");

        for (let r = 0; r < count; r++) {
            const cells = rows[r].querySelectorAll("td");
            for (let c = 0; c < ForgeCoupleDataframe.#columns - 2; c++)
                cells[c].textContent = Number(newVals[r][c]).toFixed(2);
        }

        if (newline) {
            this.#insertPromptLine(this.#promptField, this.#selection + 1);
            this.#insertPromptLine(this.#negPromptField, this.#selection + 1);
        }

        ForgeCouple.onSelect(this.#mode);
        ForgeCouple.onEntry(this.#mode);
        this.syncPrompts();
        this.syncNegativePrompts();
    }

    /** @param {boolean} removeText */
    deleteRow(removeText) {
        const rows = this.#body.querySelectorAll("tr");
        const count = rows.length;

        const vals = Array.from(rows, row => {
            return Array.from(row.querySelectorAll("td"))
                .slice(0, -2).map(cell => parseFloat(cell.textContent));
        });

        vals.splice(this.#selection, 1);
        this.#body.deleteRow(count - 1);

        for (let r = 0; r < count - 1; r++) {
            const cells = rows[r].querySelectorAll("td");
            for (let c = 0; c < ForgeCoupleDataframe.#columns - 2; c++)
                cells[c].textContent = Number(vals[r][c]).toFixed(2);
        }

        if (removeText) {
            this.#removePromptLine(this.#promptField, this.#selection);
            this.#removePromptLine(this.#negPromptField, this.#selection);
        }

        if (this.#selection == count - 1)
            this.#selection -= 1;

        ForgeCouple.onSelect(this.#mode);
        ForgeCouple.onEntry(this.#mode);
        this.syncPrompts();
        this.syncNegativePrompts();
    }

    /** @returns {[string, Element]} */
    updateColors() {
        const rows = this.#body.querySelectorAll("tr");

        rows.forEach((row, i) => {
            const color = ForgeCoupleDataframe.#color(i);

            if (this.#selection === i)
                row.style.background = `linear-gradient(to right, var(--table-row-focus) 80%, ${color})`;
            else {
                const stripe = `var(--table-${(i % 2 == 0) ? "odd" : "even"}-background-fill)`;
                row.style.background = `linear-gradient(to right, ${stripe} 80%, ${color})`;
            }
        });

        if (this.#selection < 0 || this.#selection > rows.length)
            return [null, null];
        else
            return [ForgeCoupleDataframe.#color(this.#selection), rows[this.#selection]];
    }

    syncPrompts() {
        if (!this.#promptField)
            return;

        const prompt = this.#promptField.value;
        const prompts = prompt.split(this.#sep).map(line => line.trim());
        const rows = this.#body.querySelectorAll("tr");
        const columnIndex = ForgeCoupleDataframe.#columns - 2;

        const active = document.activeElement;
        rows.forEach((row, i) => {
            const promptCell = row.querySelectorAll("td")[columnIndex];

            if (promptCell === active)
                return;

            if (i < prompts.length)
                promptCell.textContent = prompts[i].replace(/\n+/g, ", ").replace(/,+/g, ",");
            else
                promptCell.textContent = "";
        });
    }

    syncNegativePrompts() {
        if (!this.#negPromptField)
            return;

        const negPrompt = this.#negPromptField.value;
        const prompts = negPrompt.split(this.#sep).map(line => line.trim());
        const rows = this.#body.querySelectorAll("tr");
        const columnIndex = ForgeCoupleDataframe.#columns - 1;

        const active = document.activeElement;
        rows.forEach((row, i) => {
            const promptCell = row.querySelectorAll("td")[columnIndex];

            if (promptCell === active)
                return;

            if (i < prompts.length)
                promptCell.textContent = prompts[i].replace(/\n+/g, ", ").replace(/,+/g, ",");
            else
                promptCell.textContent = "";
        });
    }

    #insertPromptLine(field, index) {
        if (!field || index < 0)
            return;

        const prompts = field.value ? field.value.split(this.#sep).map(line => line.trim()) : [];
        const newPrompts = [
            ...prompts.slice(0, index),
            "",
            ...prompts.slice(index)
        ];
        field.value = newPrompts.join(this.#sep);
        updateInput(field);
    }

    #removePromptLine(field, index) {
        if (!field || index < 0)
            return;

        const prompts = field.value ? field.value.split(this.#sep).map(line => line.trim()) : [];
        if (!prompts.length || index >= prompts.length)
            return;

        prompts.splice(index, 1);
        field.value = prompts.join(this.#sep);
        updateInput(field);
    }

    /** @param {number} @param {boolean} w @returns {number} */
    #clamp01(v, w) {
        let val = parseFloat(v);
        if (Number.isNaN(val))
            val = 0.0;

        return Math.min(Math.max(val, 0.0), w ? 5.0 : 1.0);
    }

}
