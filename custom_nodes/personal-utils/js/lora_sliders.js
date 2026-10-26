import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// List for LoRA Sliders and LoRA Styles: a row per LoRA with a tick, a strength slider limited to the range the catalog
// gives and a remove button, and a browse dialog with the catalog's descriptions to add them from. The hidden "loras"
// widget holds the rows as JSON [{name, strength, enabled}]. A LoRA with previews shows its first photo in the row and
// all of them in the browse dialog, a click on a photo opens them large.

const ALL = "All";
const ENDPOINTS = { PersonalLoraSliders: "/personal-utils/loras", PersonalLoraStyles: "/personal-utils/styles" };

const STYLE = `
.lorasl { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--input-text); }
.lorasl-row { display: flex; align-items: center; gap: 6px; }
.lorasl-row.missing .lorasl-name { color: #e66; }
.lorasl-name { flex: 0 0 38%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lorasl-row input[type=range] { flex: 1; min-width: 40px; }
.lorasl-row input[type=number] { width: 56px; }
.lorasl-triggers { display: flex; align-items: center; gap: 2px; white-space: nowrap; }
.lorasl button, .lorasl-dialog button { cursor: pointer; }
.lorasl-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0, 0, 0, .5); display: flex; align-items: center; justify-content: center; }
.lorasl-dialog { width: 520px; max-width: 90vw; max-height: 80vh; display: flex; flex-direction: column; gap: 8px; padding: 12px;
    background: var(--comfy-menu-bg); color: var(--input-text); border-radius: 6px; font-size: 13px; }
.lorasl-list { overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
.lorasl-item { padding: 6px 8px; border-radius: 4px; background: var(--comfy-input-bg); cursor: pointer; }
.lorasl-item:hover { outline: 1px solid var(--input-text); }
.lorasl-item small { display: block; opacity: .7; white-space: pre-line; }
.lorasl-thumb { flex: none; width: 36px; height: 36px; object-fit: cover; border-radius: 3px; cursor: zoom-in; }
.lorasl-thumbs { display: flex; gap: 4px; overflow-x: auto; margin-top: 4px; }
.lorasl-viewer { max-width: 90vw; max-height: 90vh; object-fit: contain; cursor: pointer; }
.lorasl-item em { color: #e96; font-style: normal; }
`;

function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}

// Descriptions are HTML in the catalog: only these tags are kept, a link keeps its http(s) href, a tag keeps the style
// properties below with plain values, anything else is unwrapped to its text. The parsed document is inert, nothing in
// it loads or runs.
const TAGS = new Set(["A", "B", "STRONG", "I", "EM", "U", "S", "CODE", "SPAN", "BR", "P", "UL", "OL", "LI", "H1", "H2", "H3", "H4", "HR"]);
const COLOR = /^(#[0-9a-f]{3,8}|[a-z]+|(rgb|hsl)a?\([\d\s,.%/]+\))$/i;
const STYLES = {
    "color": COLOR,
    "background-color": COLOR,
    "font-weight": /^[a-z0-9]+$/i,
    "font-style": /^[a-z]+$/i,
    "text-decoration": /^[a-z ]+$/i,
};

function cleanNodes(parent) {
    return [...parent.childNodes].flatMap((child) => {
        if (child.nodeType === Node.TEXT_NODE) return [child];
        if (child.nodeType !== Node.ELEMENT_NODE || ["SCRIPT", "STYLE"].includes(child.tagName)) return [];
        const children = cleanNodes(child);
        if (!TAGS.has(child.tagName)) return children;
        const clean = document.createElement(child.tagName);
        if (child.tagName === "A" && /^https?:\/\//i.test(child.getAttribute("href") ?? "")) {
            Object.assign(clean, { href: child.getAttribute("href"), target: "_blank", rel: "noopener", onclick: (e) => e.stopPropagation() });
        }
        for (const [property, allowed] of Object.entries(STYLES)) {
            const value = child.style.getPropertyValue(property).trim();
            if (allowed.test(value)) clean.style.setProperty(property, value);
        }
        clean.append(...children);
        return [clean];
    });
}

function descriptionNodes(html) {
    return cleanNodes(new DOMParser().parseFromString(html, "text/html").body);
}

function previewImage(url, className, onclick) {
    return el("img", { src: url, className, loading: "lazy", referrerPolicy: "no-referrer", onclick });
}

// Click steps to the next photo, a click outside closes it.
function viewPreviews(urls, index = 0) {
    const image = previewImage(urls[index], "lorasl-viewer", () => {
        index = (index + 1) % urls.length;
        image.src = urls[index];
    });
    const overlay = el("div", { className: "lorasl-overlay", onclick: (e) => { if (e.target === overlay) overlay.remove(); } }, image);
    document.body.append(overlay);
}

function previewUrls(info) {
    return (info?.previews ?? []).filter((url) => /^https?:\/\//i.test(url));
}

async function fetchCatalog(endpoint) {
    const response = await api.fetchApi(endpoint);
    if (response.status !== 200) throw new Error(`${response.status} ${await response.text()}`);
    return response.json();
}

class LoraList {
    constructor(node, endpoint) {
        this.node = node;
        this.endpoint = endpoint;
        this.widget = node.widgets.find((w) => w.name === "loras");
        this.widget.hidden = true;
        this.widget.computeSize = () => [0, -4];
        this.type = node.widgets.find((w) => w.name === "model_type");
        const onType = this.type.callback;
        this.type.callback = (...args) => {
            onType?.apply(this.type, args);
            this.draw();
        };
        this.catalog = new Map();
        this.root = el("div", { className: "lorasl" });
        this.load();
        this.refreshCatalog();
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

    async refreshCatalog() {
        try {
            this.catalog = new Map((await fetchCatalog(this.endpoint)).map((l) => [l.name, l]));
        } catch (e) {
            this.root.title = `Couldn't read the LoRA catalog: ${e.message}`;
        }
        this.draw();
    }

    add(lora) {
        if (!this.rows.some((r) => r.name === lora.name)) this.rows.push({ name: lora.name, strength: lora.default, enabled: true, triggers: true });
        this.save();
        this.draw();
    }

    // Rows of other model types stay in the list, hidden, and aren't applied.
    inType(info) {
        return this.type.value === ALL || info?.model === this.type.value;
    }

    draw() {
        const rows = this.rows.map((row, i) => {
            const info = this.catalog.get(row.name);
            if (info && !this.inType(info)) return null;
            const min = info?.min ?? Math.min(-2, row.strength);
            const max = info?.max ?? Math.max(2, row.strength);
            // Redrawing while the slider is dragged would replace the element under the mouse.
            const change = (value) => {
                row.strength = Number(value);
                number.value = slider.value = row.strength;
                this.save();
            };
            const slider = el("input", { type: "range", min, max, step: 0.05, value: row.strength, oninput: () => change(slider.value) });
            const number = el("input", { type: "number", step: 0.05, value: row.strength, onchange: () => change(number.value) });
            const check = el("input", { type: "checkbox", checked: !!row.triggers, onchange: () => { row.triggers = check.checked; this.save(); } });
            const triggers = info?.trigger_words
                ? el("label", { className: "lorasl-triggers", title: `Add the trigger words to the prompt: ${info.trigger_words}` }, check, "trigger")
                : "";
            const tick = el("input", { type: "checkbox", checked: row.enabled, onchange: () => { row.enabled = tick.checked; this.save(); } });
            const previews = previewUrls(info);
            return el("div", { className: `lorasl-row${info ? "" : " missing"}`, title: info ? new DOMParser().parseFromString(info.description, "text/html").body.textContent : "Not a LoRA of this node in the catalog" },
                tick,
                previews.length ? previewImage(previews[0], "lorasl-thumb", () => viewPreviews(previews)) : "",
                el("span", { className: "lorasl-name", textContent: row.name }),
                slider,
                number,
                triggers,
                el("button", { textContent: "✕", title: "Remove", onclick: () => { this.rows.splice(i, 1); this.save(); this.draw(); } }));
        });
        this.root.replaceChildren(...rows.filter(Boolean), el("button", { textContent: "Add LoRA…", onclick: () => this.browse() }));
    }

    async browse() {
        let loras;
        try {
            loras = await fetchCatalog(this.endpoint);
        } catch (e) {
            alert(`Couldn't read the LoRA catalog: ${e.message}`);
            return;
        }
        this.catalog = new Map(loras.map((l) => [l.name, l]));
        const search = el("input", { type: "search", placeholder: "Search" });
        const list = el("div", { className: "lorasl-list" });
        const overlay = el("div", { className: "lorasl-overlay", onclick: (e) => { if (e.target === overlay) overlay.remove(); } },
            el("div", { className: "lorasl-dialog" }, search, list));
        const fill = () => {
            const words = search.value.toLowerCase().split(/\s+/).filter(Boolean);
            list.replaceChildren(...loras
                .filter((l) => this.inType(l) && words.every((w) => `${l.name} ${l.description}`.toLowerCase().includes(w)))
                .map((l) => el("div", { className: "lorasl-item", onclick: () => { this.add(l); overlay.remove(); } },
                    l.name, l.model ? el("small", { textContent: l.model }) : "", l.installed ? "" : el("em", { textContent: "  not installed" }),
                    el("small", { textContent: `${l.min} to ${l.max}, default ${l.default}` }),
                    l.trigger_words ? el("small", { textContent: `Trigger words: ${l.trigger_words}` }) : "",
                    el("small", {}, ...descriptionNodes(l.description)),
                    previewUrls(l).length ? el("div", { className: "lorasl-thumbs" },
                        ...previewUrls(l).map((url, i, urls) => previewImage(url, "lorasl-thumb", (e) => { e.stopPropagation(); viewPreviews(urls, i); }))) : "",
                    l.author || l.page ? el("small", {}, l.author && `by ${l.author} `,
                        l.page && el("a", { href: l.page, target: "_blank", rel: "noopener", textContent: "page", onclick: (e) => e.stopPropagation() })) : "")));
        };
        search.oninput = fill;
        fill();
        document.body.append(overlay);
        search.focus();
    }
}

app.registerExtension({
    name: "PersonalUtils.LoraList",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!(nodeData.name in ENDPOINTS)) return;
        document.head.append(el("style", { textContent: STYLE }));

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.loraList = new LoraList(this, ENDPOINTS[nodeData.name]);
            const widget = this.addDOMWidget("list", "lora_sliders_list", this.loraList.root, { getMinHeight: () => 120 });
            widget.serialize = false;
            widget.options.serialize = false;
            if (this.size[0] < 420) this.setSize([420, Math.max(this.size[1], 220)]);
            return result;
        };

        // Loading a workflow sets the widget values after the node is created.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.loraList?.load();
            return result;
        };
    },
});
