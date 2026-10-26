import { app } from "../../scripts/app.js";

// Shows the Prompt Picker's prompts as a checklist. The text box stays the source of truth: ticking a prompt here
// rewrites its "[x]" / "[ ]" header line, and editing the text box redraws the list.

const HEADER = /^(\s*)\[([ xX]?)\](.*)$/;

function parse(text) {
    const prompts = [];
    text.split("\n").forEach((line, i) => {
        const match = line.match(HEADER);
        if (match) prompts.push({ line: i, ticked: match[2].toLowerCase() === "x", name: match[3].trim() || `Prompt ${prompts.length + 1}` });
    });
    return prompts;
}

function setTicked(text, lineNumbers, ticked) {
    const lines = text.split("\n");
    for (const i of lineNumbers) lines[i] = lines[i].replace(HEADER, (_, indent, __, rest) => `${indent}[${ticked ? "x" : " "}]${rest}`);
    return lines.join("\n");
}

const STYLE = `
.prompt-picker { display: flex; flex-direction: column; gap: 4px; font: 12px sans-serif; color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #222); border-radius: 6px; padding: 6px; box-sizing: border-box; height: 100%; }
.prompt-picker-bar { display: flex; gap: 6px; align-items: center; }
.prompt-picker-bar span { flex: 1; opacity: 0.75; }
.prompt-picker-bar button { font: inherit; padding: 2px 8px; cursor: pointer; border-radius: 4px; border: 1px solid #555;
    background: var(--comfy-menu-bg, #333); color: inherit; }
.prompt-picker-list { overflow-y: auto; flex: 1; min-height: 0; }
.prompt-picker-list label { display: flex; gap: 6px; align-items: center; padding: 2px 0; cursor: pointer; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
.prompt-picker-empty { opacity: 0.6; padding: 4px 0; }
`;

app.registerExtension({
    name: "PromptPicker",
    setup() {
        const style = document.createElement("style");
        style.textContent = STYLE;
        document.head.append(style);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PromptPicker") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const prompts = this.widgets.find((w) => w.name === "prompts");

            const root = document.createElement("div");
            root.className = "prompt-picker";
            const bar = document.createElement("div");
            bar.className = "prompt-picker-bar";
            const count = document.createElement("span");
            const list = document.createElement("div");
            list.className = "prompt-picker-list";
            root.append(bar, list);

            const update = (text) => {
                prompts.value = text;
                prompts.callback?.(text);
                this.setDirtyCanvas?.(true, true);
                draw();
            };
            for (const [label, ticked] of [["All", true], ["None", false]]) {
                const button = document.createElement("button");
                button.textContent = label;
                button.onclick = () => update(setTicked(prompts.value, parse(prompts.value).map((p) => p.line), ticked));
                bar.append(button);
            }
            bar.prepend(count);

            let drawn = null;
            const draw = () => {
                const text = prompts.value ?? "";
                if (text === drawn) return;
                drawn = text;
                const items = parse(text);
                count.textContent = `${items.filter((p) => p.ticked).length} of ${items.length} ticked`;
                list.replaceChildren();
                if (!items.length) {
                    const empty = document.createElement("div");
                    empty.className = "prompt-picker-empty";
                    empty.textContent = "Add a prompt below: a \"[x] Name\" line, then the prompt text.";
                    list.append(empty);
                }
                for (const item of items) {
                    const label = document.createElement("label");
                    const box = document.createElement("input");
                    box.type = "checkbox";
                    box.checked = item.ticked;
                    box.onchange = () => update(setTicked(prompts.value, [item.line], box.checked));
                    label.append(box, item.name);
                    label.title = item.name;
                    list.append(label);
                }
            };

            const widget = this.addDOMWidget("checklist", "prompt_picker_checklist", root, {
                getMinHeight: () => 120,
                getMaxHeight: () => 320,
            });
            widget.serialize = false;
            widget.options.serialize = false;

            // Put the checklist above the text box.
            this.widgets.splice(this.widgets.indexOf(widget), 1);
            this.widgets.splice(this.widgets.indexOf(prompts), 0, widget);

            // The text box can change by typing, pasting or loading a workflow, polling catches all of them.
            const timer = setInterval(draw, 300);
            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                clearInterval(timer);
                return onRemoved?.apply(this, arguments);
            };
            draw();
            if (this.size[0] < 420 || this.size[1] < 520) this.setSize([Math.max(this.size[0], 420), Math.max(this.size[1], 520)]);
            return result;
        };
    },
});
