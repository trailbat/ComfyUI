import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Upload button for Load Prompt List (YAML): saves the picked file to the input folder, replacing a file with the
// same name, and selects it.

app.registerExtension({
    name: "PersonalUtils.Upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PersonalLoadPromptList") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const file = this.widgets.find((w) => w.name === "file");

            const picker = document.createElement("input");
            picker.type = "file";
            picker.accept = ".yaml,.yml";
            picker.onchange = async () => {
                if (!picker.files.length) return;
                const body = new FormData();
                body.append("image", picker.files[0]);
                body.append("type", "input");
                body.append("overwrite", "true");
                picker.value = "";
                const response = await api.fetchApi("/upload/image", { method: "POST", body });
                if (response.status !== 200) {
                    alert(`Upload failed: ${response.status} ${response.statusText}`);
                    return;
                }
                const { name } = await response.json();
                if (!file.options.values.includes(name)) file.options.values.push(name);
                file.value = name;
                file.callback?.(name);
                this.setDirtyCanvas?.(true, true);
            };

            this.addWidget("button", "upload yaml", null, () => picker.click(), { serialize: false });
            return result;
        };
    },
});
