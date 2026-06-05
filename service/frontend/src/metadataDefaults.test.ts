import { describe, expect, it } from "vitest";

import { DEFAULT_METADATA } from "./metadataDefaults";

describe("DEFAULT_METADATA", () => {
  it("pre-fills every metadata field with a realistic example", () => {
    expect(DEFAULT_METADATA.equipmentName).toBe("ACSR-OC");
    expect(DEFAULT_METADATA.ratedVoltage).toBe("22900V");
    expect(DEFAULT_METADATA.ratedCurrent).toBe("268A");
    expect(DEFAULT_METADATA.sensorType).toBe("HFCT");
    expect(DEFAULT_METADATA.temperature).toBe("19");
    expect(DEFAULT_METADATA.humidity).toBe("66");
  });
});
