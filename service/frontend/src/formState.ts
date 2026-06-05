import type { FormPresenceInput } from "./types";
import type { InputPresence } from "./route";

export function buildInputPresence(input: FormPresenceInput): InputPresence {
  return {
    hasImage: input.hasImage,
    hasTimeseries: input.hasTimeseries,
    hasMetadata: metadataComplete(input.metadata),
  };
}

function metadataComplete(metadata: FormPresenceInput["metadata"]): boolean {
  const fields = [
    metadata.equipmentName,
    metadata.ratedVoltage,
    metadata.ratedCurrent,
    metadata.sensorType,
    metadata.temperature,
    metadata.humidity,
  ];
  return fields.every((value) => value.trim().length > 0);
}
