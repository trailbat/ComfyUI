# For Each loop built on node expansion, the same approach as the Inspire Pack's Foreach List nodes and BadCafeCode's
# execution-inversion-demo: when For Each ◀ gets a value and items are left, it returns a copy of every node between
# the two loop nodes with ▶ For Each fed the remaining items, and its result becomes the copy's result.

import torch

import comfy.utils
from comfy_execution.graph_utils import ExecutionBlocker, GraphBuilder, is_link
from server import PromptServer

ANY = "*"


class LoopState:
    """Passed from ▶ For Each to For Each ◀: the items left to run, starting with the current one, and the values
    collected so far. A copied ▶ For Each gets it back as its items to carry on from there."""

    def __init__(self, begin_id, remaining, results):
        self.begin_id = begin_id
        self.remaining = remaining
        self.results = results


def batch(values):
    """Values that are all images come out as one batch, sized to the first like Batch Images does, anything else as a list."""
    if not all(isinstance(v, torch.Tensor) for v in values):
        return values
    height, width = values[0].shape[1:3]
    return torch.cat([v if v.shape[1:3] == (height, width) else
                      comfy.utils.common_upscale(v.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                      for v in values])


class ForEachBegin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": (ANY, {"tooltip": "A list such as Load Prompt List's prompts, or a list output such as Prompt Picker's."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("FOREACH_FLOW", ANY, "INT", "INT")
    RETURN_NAMES = ("flow", "item", "index", "count")
    OUTPUT_TOOLTIPS = ("Connect to For Each ◀.", "The current item.", "The current item's position, starting at 0.", "The number of items.")
    FUNCTION = "begin"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Start of a loop: the nodes between this and For Each ◀ run once per item."

    def begin(self, items, unique_id):
        if len(items) == 1 and isinstance(items[0], (list, tuple, LoopState)):
            items = items[0]
        if isinstance(items, LoopState):
            remaining, results = items.remaining, items.results
        else:
            remaining, results = list(items), []
        if not remaining:  # e.g. Pending Prompts with nothing left to generate
            return (ExecutionBlocker(None),) * 4
        return (LoopState(unique_id[0], remaining, results), remaining[0], len(results), len(results) + len(remaining))


class ForEachEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FOREACH_FLOW", {"tooltip": "Connect ▶ For Each's flow output."}),
                "value": (ANY, {"tooltip": "The result of one run, e.g. the decoded image."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("batch",)
    # Runs even when nothing uses the batch, e.g. when every run is saved inside the loop, since it drives the loop.
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("Images as one batch, resized to the first image's size when they differ. Other values as a list.",)
    FUNCTION = "end"
    CATEGORY = "utils/personal"
    DESCRIPTION = "End of a loop: collects the value of every run."

    def end(self, flow, value, dynprompt, unique_id):
        results = flow.results + [value]
        remaining = flow.remaining[1:]
        PromptServer.instance.send_progress_text(f"{len(results)}/{len(results) + len(remaining)}", dynprompt.get_display_node_id(unique_id))
        if not remaining:
            return (batch(results),)

        children = {}  # node id -> ids of the nodes using its outputs, for everything upstream of this node
        stack = [unique_id]
        while stack:
            node_id = stack.pop()
            for v in dynprompt.get_node(node_id)["inputs"].values():
                if is_link(v):
                    if v[0] not in children:
                        children[v[0]] = []
                        stack.append(v[0])
                    children[v[0]].append(node_id)

        loop = {flow.begin_id}
        stack = [flow.begin_id]
        while stack:
            for child in children.get(stack.pop(), []):
                if child not in loop:
                    loop.add(child)
                    stack.append(child)

        graph = GraphBuilder()
        # The copy of this node gets a fixed id so ids don't grow with every run.
        copy_id = lambda node_id: "Recurse" if node_id == unique_id else node_id
        for node_id in loop:
            graph.node(dynprompt.get_node(node_id)["class_type"], copy_id(node_id)).set_override_display_id(node_id)
        for node_id in loop:
            node = graph.lookup_node(copy_id(node_id))
            for k, v in dynprompt.get_node(node_id)["inputs"].items():
                node.set_input(k, graph.lookup_node(copy_id(v[0])).out(v[1]) if is_link(v) and v[0] in loop else v)
        graph.lookup_node(flow.begin_id).set_input("items", LoopState(None, remaining, results))

        end = graph.lookup_node("Recurse")
        return {"result": (end.out(0),), "expand": graph.finalize()}
