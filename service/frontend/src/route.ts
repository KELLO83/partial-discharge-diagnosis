export type InputRoute = "insufficient_input" | "timeseries_only" | "vlm_only" | "hybrid";

export type InputPresence = {
  readonly hasImage: boolean;
  readonly hasMetadata: boolean;
  readonly hasTimeseries: boolean;
};

export function selectInputRoute(input: InputPresence): InputRoute {
  if (input.hasImage && input.hasMetadata && input.hasTimeseries) {
    return "hybrid";
  }
  if (input.hasImage && input.hasMetadata) {
    return "vlm_only";
  }
  if (input.hasTimeseries) {
    return "timeseries_only";
  }
  return "insufficient_input";
}
