import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Lists for LoRA Grid Plan: a row per LoRA with a tick, a dropdown of the loras folder and a remove button, and a row per
// weight with a tick, a number and a remove button. The hidden "loras" and "weights" widgets hold the rows as JSON
// [{name, enabled}] and [{weight, enabled}].

const STYLE = `
.loragrid { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--input-text); }
.loragrid-row { display: flex; align-items: center; gap: 6px; }
.loragrid-row select { flex: 1; min-width: 0; }
.loragrid-row input[type=number] { flex: 1; min-width: 0; }
.loragrid-row.off select, .loragrid-row.off input[type=number] { opacity: .5; }
.loragrid-row.missing select { color: #e66; }
.loragrid button { cursor: pointer; }
`;

function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}

async function loraFiles() {
    const response = await api.fetchApi("/models/loras");
    if (response.status !== 200) throw new Error(`${response.status} ${await response.text()}`);
    return response.json();
}

class RowList {
    constructor(node, name, addLabel, newRow, control) {
        this.node = node;
        this.newRow = newRow;
        this.control = control;
        this.widget = node.widgets.find((w) => w.name === name);
        this.widget.hidden = true;
        this.widget.computeSize = () => [0, -4];
        this.addLabel = addLabel;
        this.root = el("div", { className: "loragrid" });
        this.load();
    }

    load() {
        try {
            this.rows = JSON.parse(this.widget.value || "[]");
        } catch {
            this.rows = [];
        }
        this.draw();
    }

    save() {
        this.widget.value = JSON.stringify(this.rows);
        this.node.setDirtyCanvas?.(true, true);
    }

    async add() {
        try {
            this.rows.push({ ...(await this.newRow(this.rows)), enabled: true });
        } catch (e) {
            alert(`Couldn't read the loras folder: ${e.message}`);
            return;
        }
        this.save();
        this.draw();
    }

    async draw() {
        const context = await this.control.context?.();
        const rows = this.rows.map((row, i) => {
            const tick = el("input", { type: "checkbox", checked: row.enabled, onchange: () => { row.enabled = tick.checked; line.classList.toggle("off", !row.enabled); this.save(); } });
            const line = el("div", { className: `loragrid-row${row.enabled ? "" : " off"}${this.control.missing?.(row, context) ? " missing" : ""}` },
                tick,
                this.control.input(row, context, () => this.save()),
                el("button", { textContent: "✕", title: "Remove", onclick: () => { this.rows.splice(i, 1); this.save(); this.draw(); } }));
            return line;
        });
        this.root.replaceChildren(...rows, el("button", { textContent: this.addLabel, onclick: () => this.add() }));
    }
}

const LORAS = {
    async context() {
        try {
            return await loraFiles();
        } catch {
            return [];
        }
    },
    missing: (row, files) => !files.includes(row.name),
    input(row, files, save) {
        const names = files.includes(row.name) ? files : [row.name, ...files];
        const select = el("select", { onchange: () => { row.name = select.value; save(); } },
            ...names.map((name) => el("option", { value: name, textContent: name, selected: name === row.name })));
        return select;
    },
};

const WEIGHTS = {
    input(row, context, save) {
        const number = el("input", { type: "number", step: 0.05, value: row.weight, onchange: () => { row.weight = Number(number.value); save(); } });
        return number;
    },
};

app.registerExtension({
    name: "PersonalUtils.LoraGridPlan",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PersonalLoraGridPlan") return;
        document.head.append(el("style", { textContent: STYLE }));

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const loras = new RowList(this, "loras", "Add LoRA", async (rows) => {
                const files = await loraFiles();
                return { name: files.find((f) => !rows.some((r) => r.name === f)) ?? files[0] ?? "" };
            }, LORAS);
            const weights = new RowList(this, "weights", "Add weight", (rows) => ({ weight: rows.length ? Math.round((rows.at(-1).weight + 0.1) * 100) / 100 : 1 }), WEIGHTS);
            this.gridLists = [loras, weights];
            for (const [name, list] of [["loras_list", loras], ["weights_list", weights]]) {
                const widget = this.addDOMWidget(name, "lora_grid_list", list.root, { getMinHeight: () => 80 });
                widget.serialize = false;
                widget.options.serialize = false;
            }
            if (this.size[0] < 360) this.setSize([360, Math.max(this.size[1], 360)]);
            return result;
        };

        // Loading a workflow sets the widget values after the node is created.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.gridLists?.forEach((list) => list.load());
            return result;
        };
    },
});
