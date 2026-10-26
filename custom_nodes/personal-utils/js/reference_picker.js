import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Grid for the Reference Picker. The hidden "selection" widget holds everything as JSON: the Reference Analyzer's last
// result, the weights, the picked photos per prompt id and the approved results. Anchors are written to the Reference
// Analyzer's "anchors" widget and unticked prompts to Pending Prompts' "skip" widget, since those nodes read them.
// The analyzer stays muted so Run doesn't analyze again, Analyze unmutes it for its own run.

const ANALYZER = "PersonalReferenceAnalyzer";
const PICKER = "PersonalReferencePicker";
const PENDING = "PersonalPendingPrompts";
const MUTED = 2; // LGraphEventMode.NEVER
const MAX_PICKS = 8;
const DEFAULT_WEIGHTS = { angle: 0.2, crop: 0.2, pose: 0.05, camera: 0.05, gaze: 0.05, light: 0.1, environment: 0.05, quality: 0.15, identity: 0.15 };

const STYLE = `
.refpick { display: flex; flex-direction: column; gap: 6px; height: 100%; box-sizing: border-box; padding: 6px; font: 12px sans-serif;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #222); border-radius: 6px; }
.refpick button, .refpick input[type=number] { font: inherit; padding: 2px 8px; border-radius: 4px; border: 1px solid #555;
    background: var(--comfy-menu-bg, #333); color: inherit; }
.refpick button { cursor: pointer; }
.refpick button.on { background: #3b5b8a; border-color: #5b84c4; }
.refpick input[type=number] { width: 44px; }
.refpick-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.refpick-status { flex: 1; opacity: 0.75; min-width: 120px; }
.refpick-status.warn { color: #e6b450; opacity: 1; }
.refpick-weights { display: grid; grid-template-columns: repeat(3, auto 1fr 30px); gap: 2px 8px; align-items: center; }
.refpick-weights input { width: 100%; }
.refpick-body { display: flex; gap: 6px; flex: 1; min-height: 0; }
.refpick-list { width: 240px; overflow-y: auto; flex-shrink: 0; }
.refpick-list p { margin: 0 0 4px; display: flex; gap: 4px; align-items: center; }
.refpick-list div { padding: 3px 4px; border-radius: 4px; cursor: pointer; }
.refpick-list div:hover { background: #ffffff10; }
.refpick-list div.sel { background: #3b5b8a; }
.refpick-list div.off { opacity: 0.5; }
.refpick-list input { margin: 0 4px 0 0; vertical-align: middle; }
.refpick-list b { margin-left: 4px; font-weight: normal; }
.refpick-list small, .refpick-head small, .refpick-tile small { display: block; opacity: 0.65; overflow: hidden; text-overflow: ellipsis; }
.refpick-list small, .refpick-head small { white-space: nowrap; }
.refpick-main { flex: 1; overflow-y: auto; min-width: 0; }
.refpick-head { padding: 2px 2px 6px; }
.refpick-head b { font-size: 13px; }
.refpick-head details { margin-top: 4px; opacity: 0.8; }
.refpick-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.refpick-tile { width: 140px; cursor: pointer; border: 2px solid transparent; border-radius: 6px; padding: 2px; box-sizing: border-box; }
.refpick-tile.picked { border-color: #5b9bf0; background: #5b9bf020; }
.refpick-tile.anchor { border-color: #e6b450; background: #e6b45020; }
.refpick-tile.excluded { opacity: 0.4; }
.refpick-thumb { position: relative; height: 130px; display: flex; justify-content: center; background: #0003; border-radius: 4px; overflow: hidden; }
.refpick-thumb span { position: relative; height: 100%; }
.refpick-thumb img { height: 100%; display: block; }
.refpick-thumb i { position: absolute; border: 2px solid #e6b450; box-sizing: border-box; }
.refpick-thumb em { position: absolute; top: 3px; left: 3px; background: #000a; padding: 0 5px; border-radius: 3px; font-style: normal; font-weight: bold; }
.refpick-tile small { white-space: normal; max-height: 30px; }
.refpick-empty { opacity: 0.6; padding: 12px; }
.refpick-result { width: 240px; border: 2px solid transparent; border-radius: 6px; padding: 2px; box-sizing: border-box; }
.refpick-result.approved { border-color: #5bbf6a; background: #5bbf6a20; }
.refpick-result img { width: 100%; height: 240px; object-fit: contain; display: block; background: #0003; border-radius: 4px; }
.refpick-result div { display: flex; gap: 4px; align-items: center; margin-top: 3px; }
.refpick-result span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
`;

function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}

function photoURL(key, type = "input") {
    const i = key.lastIndexOf("/");
    const query = new URLSearchParams({ filename: key.slice(i + 1), subfolder: i < 0 ? "" : key.slice(0, i), type });
    return api.apiURL(`/view?${query}`);
}

function sameSet(a, b) {
    return a.length === b.length && a.every((x) => b.includes(x));
}

class Picker {
    constructor(node) {
        this.node = node;
        this.widget = node.widgets.find((w) => w.name === "selection");
        this.widget.hidden = true;
        this.widget.computeSize = () => [0, -4];
        this.view = "prompts";
        this.current = null;
        this.status = "";
        this.results = [];
        this.load();
        this.build();
    }

    load() {
        let saved = {};
        try {
            saved = this.widget.value ? JSON.parse(this.widget.value) : {};
        } catch {
            saved = {};
        }
        this.state = { analysis: saved.analysis ?? null, picks: saved.picks ?? {}, n: saved.n ?? 4, weights: { ...DEFAULT_WEIGHTS, ...saved.weights },
            approved: saved.approved ?? [] };
        this.index();
        this.current ??= this.state.analysis?.prompts[0]?.id ?? null;
    }

    index() {
        this.photoIndex = new Map((this.state.analysis?.photos ?? []).map((p, i) => [p.key, i]));
    }

    save() {
        this.widget.value = JSON.stringify(this.state);
        this.widget.callback?.(this.widget.value);
        this.node.setDirtyCanvas?.(true, true);
    }

    analyzer() {
        return app.graph.findNodesByType(ANALYZER)[0] ?? null;
    }

    widgetOf(node, name) {
        return node?.widgets?.find((w) => w.name === name);
    }

    anchors() {
        const widget = this.widgetOf(this.analyzer(), "anchors");
        return (widget?.value ?? "").split("\n").map((s) => s.trim()).filter(Boolean);
    }

    setAnchors(keys) {
        const widget = this.widgetOf(this.analyzer(), "anchors");
        if (!widget) return;
        widget.value = keys.join("\n");
        widget.callback?.(widget.value);
    }

    pending() {
        return app.graph.findNodesByType(PENDING)[0] ?? null;
    }

    resultsFolder() {
        return this.widgetOf(this.pending(), "folder")?.value?.trim().replace(/^[\\/]+|[\\/]+$/g, "");
    }

    skipped() {
        const widget = this.widgetOf(this.pending(), "skip");
        return (widget?.value ?? "").split("\n").map((s) => s.trim()).filter(Boolean);
    }

    setSkipped(ids) {
        const widget = this.widgetOf(this.pending(), "skip");
        if (!widget) return this.setStatus("Add a Pending Prompts node between Load Prompt List and ▶ For Each to choose the prompts to run.");
        widget.value = ids.join("\n");
        widget.callback?.(widget.value);
        this.drawList();
    }

    async refreshResults() {
        const folder = this.resultsFolder();
        if (folder === undefined) return;
        const response = await api.fetchApi(`/personal-utils/results?${new URLSearchParams({ folder })}`);
        if (response.status !== 200) return this.setStatus(`Listing output/${folder} failed: ${response.status} ${response.statusText}`);
        this.results = await response.json();
        this.drawList();
        if (this.view === "results") this.drawMain();
    }

    toggleApproved(file) {
        const approved = this.state.approved;
        this.state.approved = approved.includes(file) ? approved.filter((f) => f !== file) : [...approved, file];
        this.save();
        this.drawList();
        this.drawMain();
    }

    async reject(file) {
        const folder = this.resultsFolder();
        const response = await api.fetchApi("/personal-utils/reject", { method: "POST", body: JSON.stringify({ folder, file }) });
        if (response.status !== 200) return this.setStatus(`Rejecting ${file} failed: ${await response.text()}`);
        this.state.approved = this.state.approved.filter((f) => f !== file);
        this.save();
        this.setStatus(`Moved ${file} to output/${folder}/rejected, its prompt runs again on the next Run.`);
        await this.refreshResults();
    }

    // Photos for a prompt, best first: the weighted mean of the components (0-100) minus the penalty, photos where the
    // subject wasn't found last.
    ranked(prompt) {
        const { components, photos } = this.state.analysis;
        const weights = components.map((c) => this.state.weights[c]);
        const total = weights.reduce((a, b) => a + b, 0) || 1;
        return prompt.scores
            .map((s, i) => ({
                key: photos[i].key, index: i, s,
                excluded: photos[i].found === false,
                score: Math.round(weights.reduce((a, w, j) => a + w * s[j], 0) / total - s[components.length]),
            }))
            .sort((a, b) => a.excluded - b.excluded || b.score - a.score);
    }

    suggest(prompt) {
        const { photos } = this.state.analysis;
        const picks = [];
        for (const { key, index, excluded } of this.ranked(prompt)) {
            if (picks.length >= this.state.n || excluded) break;
            if (picks.some((p) => photos[this.photoIndex.get(p)].dupes.includes(index))) continue;
            picks.push(key);
        }
        this.state.picks[prompt.id] = picks;
    }

    setAnalysis(analysis) {
        this.state.analysis = analysis;
        this.index();
        for (const [id, keys] of Object.entries(this.state.picks)) this.state.picks[id] = keys.filter((k) => this.photoIndex.has(k));
        for (const prompt of analysis.prompts) if (!this.state.picks[prompt.id]?.length) this.suggest(prompt);
        if (!analysis.prompts.some((p) => p.id === this.current)) this.current = analysis.prompts[0]?.id ?? null;
        this.status = `Analyzed ${analysis.photos.length} photos for ${analysis.prompts.length} prompts.`;
        this.save();
        this.draw();
    }

    async upload(files) {
        const folder = this.widgetOf(this.analyzer(), "folder")?.value?.trim().replace(/^[\\/]+|[\\/]+$/g, "");
        if (folder === undefined) return this.setStatus("Add a Reference Analyzer node first, the photos go to its folder.");
        let done = 0;
        for (const file of files) {
            this.setStatus(`Uploading ${done + 1}/${files.length}...`);
            const body = new FormData();
            body.append("image", file);
            body.append("type", "input");
            body.append("subfolder", folder);
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (response.status !== 200) return this.setStatus(`Upload of ${file.name} failed: ${response.status} ${response.statusText}`);
            done++;
        }
        this.setStatus(`Uploaded ${done} photos to input/${folder}, click Analyze.`);
    }

    async analyze() {
        const analyzer = this.analyzer();
        if (!analyzer) return this.setStatus("Add a Reference Analyzer node and connect it to Load Prompt List.");
        app.canvas.selectItems([analyzer]);
        analyzer.mode = 0;
        try {
            await app.extensionManager.command.execute("Comfy.QueueSelectedOutputNodes");
        } finally {
            analyzer.mode = MUTED;
            app.graph.setDirtyCanvas(true, true);
        }
        this.setStatus("Analyzing, this takes a while the first time...");
    }

    setStatus(text) {
        this.status = text;
        this.drawStatus();
    }

    build() {
        const picker = el("input", { type: "file", multiple: true, accept: "image/*" });
        picker.onchange = () => {
            const files = [...picker.files];
            picker.value = "";
            if (files.length) this.upload(files);
        };
        this.n = el("input", { type: "number", min: 1, max: MAX_PICKS, value: this.state.n, title: "Photos to suggest per prompt" });
        this.n.onchange = () => {
            this.state.n = Math.min(MAX_PICKS, Math.max(1, parseInt(this.n.value) || 4));
            this.n.value = this.state.n;
            this.save();
        };
        const resuggest = el("button", { textContent: "Re-suggest all", title: "Replace the picks of every prompt with the top suggestions" });
        resuggest.onclick = () => {
            if (!this.state.analysis || !confirm("Replace the picks of every prompt with the suggestions?")) return;
            for (const prompt of this.state.analysis.prompts) this.suggest(prompt);
            this.save();
            this.draw();
        };
        this.tabs = {
            prompts: el("button", { textContent: "Prompts", title: "Pick photos per prompt" }),
            anchors: el("button", { textContent: "Anchors", title: "Star the photos that clearly show the subject" }),
            results: el("button", { textContent: "Results", title: "Approve or reject the generated images" }),
        };
        for (const [view, button] of Object.entries(this.tabs)) button.onclick = () => {
            this.view = view;
            this.draw();
            if (view === "results") this.refreshResults();
        };
        const weightsButton = el("button", { textContent: "Weights" });
        this.weights = el("div", { className: "refpick-weights" });
        this.weights.style.display = "none";
        weightsButton.onclick = () => {
            this.weights.style.display = this.weights.style.display === "none" ? "" : "none";
            weightsButton.classList.toggle("on");
        };
        for (const name of Object.keys(DEFAULT_WEIGHTS)) {
            const value = el("span");
            const slider = el("input", { type: "range", min: 0, max: 1, step: 0.05, value: this.state.weights[name] });
            const show = () => (value.textContent = Number(slider.value).toFixed(2));
            slider.oninput = () => {
                this.state.weights[name] = Number(slider.value);
                show();
                this.drawMain();
            };
            slider.onchange = () => this.save();
            show();
            this.weights.append(el("span", { textContent: name }), slider, value);
        }
        this.statusEl = el("span", { className: "refpick-status" });
        const bar = el("div", { className: "refpick-bar" },
            el("button", { textContent: "Upload photos", onclick: () => picker.click() }),
            el("button", { textContent: "Analyze", onclick: () => this.analyze(), title: "Runs only the Reference Analyzer" }),
            this.tabs.prompts, this.tabs.anchors, this.tabs.results,
            el("span", { textContent: "suggest" }), this.n, resuggest, weightsButton, this.statusEl);
        this.list = el("div", { className: "refpick-list" });
        this.main = el("div", { className: "refpick-main" });
        this.root = el("div", { className: "refpick" }, bar, this.weights, el("div", { className: "refpick-body" }, this.list, this.main));
        this.draw();
    }

    draw() {
        if (this.n) this.n.value = this.state.n;
        for (const [view, button] of Object.entries(this.tabs)) button.classList.toggle("on", view === this.view);
        this.list.style.display = this.view === "prompts" ? "" : "none";
        this.drawList();
        this.drawMain();
        this.drawStatus();
    }

    drawStatus() {
        const analysis = this.state.analysis;
        const stale = analysis && !sameSet(this.anchors(), analysis.anchors);
        this.statusEl.textContent = stale ? "Anchors changed, click Analyze to use them." : this.status ||
            (analysis ? (analysis.anchors.length ? `${analysis.anchors.length} anchors.` : "No anchors yet: star photos of the subject under Anchors, then Analyze.") : "");
        this.statusEl.classList.toggle("warn", Boolean(stale));
    }

    drawList() {
        this.list.replaceChildren();
        const prompts = this.state.analysis?.prompts ?? [];
        if (!prompts.length) return;
        const skipped = this.skipped();
        this.list.append(el("p", {}, "Run",
            el("button", { textContent: "all", onclick: () => this.setSkipped([]) }),
            el("button", { textContent: "none", onclick: () => this.setSkipped(prompts.map((p) => p.id)) })));
        for (const prompt of prompts) {
            const count = this.state.picks[prompt.id]?.length ?? 0;
            const files = this.results.filter((r) => r.id === prompt.id).map((r) => r.file);
            const check = el("input", { type: "checkbox", checked: !skipped.includes(prompt.id), title: "Run this prompt when it has no image yet" });
            check.onclick = (e) => e.stopPropagation();
            check.onchange = () => this.setSkipped(check.checked ? this.skipped().filter((id) => id !== prompt.id) : [...this.skipped(), prompt.id]);
            const mark = !files.length ? "" : files.every((f) => this.state.approved.includes(f)) ? "✓" : "●";
            const row = el("div", { className: prompt.id === this.current ? "sel" : "", title: prompt.prompt },
                check, `${prompt.id} · ${prompt.title} (${count})`, el("b", { textContent: mark, title: mark === "✓" ? "Approved" : "Generated, not approved yet" }),
                el("small", { textContent: prompt.tags }));
            row.classList.toggle("off", skipped.includes(prompt.id));
            row.onclick = () => {
                this.current = prompt.id;
                this.drawList();
                this.drawMain();
            };
            this.list.append(row);
        }
    }

    tile(key, { score, s, excluded, picked, anchor }) {
        const { photos, components } = this.state.analysis;
        const photo = photos[this.photoIndex.get(key)];
        const name = key.slice(key.lastIndexOf("/") + 1);
        const image = el("img", { src: photoURL(key), loading: "lazy", draggable: false });
        const frame = el("span", {}, image);
        if (this.view === "anchors" && photo.box) {
            const [x0, y0, x1, y1] = photo.box;
            const box = el("i");
            Object.assign(box.style, { left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` });
            frame.append(box);
        }
        const thumb = el("div", { className: "refpick-thumb" }, frame);
        if (score !== undefined) thumb.append(el("em", { textContent: picked ? `${score} ✓${picked}` : score }));
        else if (anchor) thumb.append(el("em", { textContent: "★" }));
        const lines = [name];
        if (s) {
            lines.push(...components.map((c, i) => `${c}: ${s[i]}`));
            if (s[components.length]) lines.push(`penalty: -${s[components.length]}`);
        }
        lines.push(...photo.notes);
        const tile = el("div", { className: "refpick-tile", title: lines.join("\n") }, thumb, el("small", { textContent: photo.notes.join(", ") }));
        tile.classList.toggle("picked", Boolean(picked));
        tile.classList.toggle("anchor", Boolean(anchor));
        tile.classList.toggle("excluded", Boolean(excluded));
        return tile;
    }

    drawResults() {
        const folder = this.resultsFolder();
        if (folder === undefined) {
            this.main.append(el("div", { className: "refpick-empty", textContent: "Add a Pending Prompts node between Load Prompt List and ▶ For Each." }));
            return;
        }
        this.main.append(el("div", { className: "refpick-head" }, el("b", { textContent: "Results" }),
            el("small", { textContent: `Images in output/${folder}. Rejected ones move to output/${folder}/rejected and their prompt runs again on the next Run.` })));
        const titles = new Map(this.state.analysis.prompts.map((p) => [p.id, p.title]));
        const grid = el("div", { className: "refpick-grid" });
        for (const { file, id, mtime } of this.results) {
            const url = `${photoURL(`${folder}/${file}`, "output")}&t=${mtime}`;
            const approved = this.state.approved.includes(file);
            grid.append(el("div", { className: `refpick-result${approved ? " approved" : ""}`, title: file },
                el("a", { href: url, target: "_blank" }, el("img", { src: url, loading: "lazy", draggable: false })),
                el("div", {}, el("span", { textContent: `${id} · ${titles.get(id) ?? file}` }),
                    el("button", { textContent: approved ? "✓ Approved" : "Approve", className: approved ? "on" : "", onclick: () => this.toggleApproved(file) }),
                    el("button", { textContent: "Reject", onclick: () => this.reject(file) }))));
        }
        if (!this.results.length) grid.append(el("div", { className: "refpick-empty", textContent: "Nothing generated yet." }));
        this.main.append(grid);
    }

    drawMain() {
        const analysis = this.state.analysis;
        this.main.replaceChildren();
        if (!analysis) {
            this.main.append(el("div", { className: "refpick-empty", textContent: "Upload photos, then click Analyze. Star a few photos that clearly show the subject under Anchors and Analyze again." }));
            return;
        }
        const grid = el("div", { className: "refpick-grid" });
        if (this.view === "anchors") {
            const anchors = this.anchors();
            this.main.append(el("div", { className: "refpick-head" }, el("b", { textContent: "Anchors" }),
                el("small", { textContent: "Click the photos that clearly show the subject. The yellow box is the face picked as the subject." })));
            for (const { key } of analysis.photos) {
                const tile = this.tile(key, { anchor: anchors.includes(key) });
                tile.onclick = () => {
                    const current = this.anchors();
                    this.setAnchors(current.includes(key) ? current.filter((k) => k !== key) : [...current, key]);
                    this.drawMain();
                    this.drawStatus();
                };
                grid.append(tile);
            }
            this.main.append(grid);
            return;
        }
        if (this.view === "results") return this.drawResults();
        const prompt = analysis.prompts.find((p) => p.id === this.current);
        if (!prompt) return;
        const picks = this.state.picks[prompt.id] ?? [];
        const clear = el("button", { textContent: "Clear" });
        clear.onclick = () => { this.state.picks[prompt.id] = []; this.save(); this.drawList(); this.drawMain(); };
        const again = el("button", { textContent: "Suggest" });
        again.onclick = () => { this.suggest(prompt); this.save(); this.drawList(); this.drawMain(); };
        this.main.append(el("div", { className: "refpick-head" },
            el("b", { textContent: `${prompt.id} · ${prompt.title}` }), " ", again, " ", clear,
            el("small", { textContent: `${picks.length}/${MAX_PICKS} picked · ${prompt.tags}` }),
            el("details", {}, el("summary", { textContent: "prompt" }), prompt.prompt)));
        for (const entry of this.ranked(prompt)) {
            const index = picks.indexOf(entry.key);
            const tile = this.tile(entry.key, { ...entry, picked: index + 1 });
            tile.onclick = () => {
                const current = this.state.picks[prompt.id] ?? [];
                if (current.includes(entry.key)) this.state.picks[prompt.id] = current.filter((k) => k !== entry.key);
                else if (current.length < MAX_PICKS) this.state.picks[prompt.id] = [...current, entry.key];
                else return this.setStatus(`At most ${MAX_PICKS} photos per prompt.`);
                this.save();
                this.drawList();
                this.drawMain();
            };
            grid.append(tile);
        }
        this.main.append(grid);
    }
}

app.registerExtension({
    name: "PersonalUtils.ReferencePicker",
    setup() {
        document.head.append(el("style", { textContent: STYLE }));
        api.addEventListener("executed", ({ detail }) => {
            const analysis = detail?.output?.analysis?.[0];
            if (!analysis) return;
            for (const node of app.graph.findNodesByType(PICKER)) node.referencePicker?.setAnalysis(analysis);
        });
        api.addEventListener("execution_success", () => {
            for (const node of app.graph.findNodesByType(PICKER)) node.referencePicker?.refreshResults();
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PICKER) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.referencePicker = new Picker(this);
            const widget = this.addDOMWidget("grid", "reference_picker_grid", this.referencePicker.root, { getMinHeight: () => 420 });
            widget.serialize = false;
            widget.options.serialize = false;
            if (this.size[0] < 1000 || this.size[1] < 720) this.setSize([Math.max(this.size[0], 1000), Math.max(this.size[1], 720)]);
            return result;
        };

        // Loading a workflow sets the widget values after the node is created.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.referencePicker?.load();
            this.referencePicker?.draw();
            // The other nodes are configured after this one.
            setTimeout(() => this.referencePicker?.refreshResults());
            return result;
        };
    },
});
