import { describe, expect, it } from "vitest";

import { buildInputPresence } from "./formState";

describe("buildInputPresence", () => {
  it("marks metadata present only when required fields are filled", () => {
    const state = buildInputPresence({
      hasImage: true,
      hasTimeseries: false,
      metadata: {
        equipmentName: "ACSR-OC",
        ratedVoltage: "22900V",
        ratedCurrent: "268A",
        sensorType: "HFCT",
        temperature: "19",
        humidity: "66",
      },
    });

    expect(state.hasMetadata).toBe(true);
  });

  it("marks metadata missing when one required field is empty", () => {
    const state = buildInputPresence({
      hasImage: true,
      hasTimeseries: false,
      metadata: {
        equipmentName: "ACSR-OC",
        ratedVoltage: "22900V",
        ratedCurrent: "268A",
        sensorType: "",
        temperature: "19",
        humidity: "66",
      },
    });

    expect(state.hasMetadata).toBe(false);
  });
});
