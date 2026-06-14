from __future__ import annotations


class VlmServiceAdapter:
    def __init__(self, context: object) -> None:
        self.context = context

    def generate_report(self, tool_input: object) -> dict[str, object]:
        raise RuntimeError(
            "VLM service adapter is a deployment contract placeholder; "
            "replace generate_report with trained model inference before enabling checkpoint mode"
        )


def load_adapter(context: object) -> VlmServiceAdapter:
    return VlmServiceAdapter(context)
