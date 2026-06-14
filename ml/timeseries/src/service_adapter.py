from __future__ import annotations


class TimeSeriesServiceAdapter:
    def __init__(self, context: object) -> None:
        self.context = context

    def predict_csv(self, tool_input: object) -> dict[str, object]:
        raise RuntimeError(
            "timeseries service adapter is a deployment contract placeholder; "
            "replace predict_csv with trained model inference before enabling checkpoint mode"
        )


def load_adapter(context: object) -> TimeSeriesServiceAdapter:
    return TimeSeriesServiceAdapter(context)
