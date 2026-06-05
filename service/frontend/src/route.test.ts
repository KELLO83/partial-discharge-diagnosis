import { describe, expect, it } from "vitest";

import { selectInputRoute } from "./route";

describe("selectInputRoute", () => {
  it("routes to hybrid when image metadata and csv are provided", () => {
    const route = selectInputRoute({ hasImage: true, hasMetadata: true, hasTimeseries: true });

    expect(route).toBe("hybrid");
  });

  it("routes to vlm_only when image and metadata are provided", () => {
    const route = selectInputRoute({ hasImage: true, hasMetadata: true, hasTimeseries: false });

    expect(route).toBe("vlm_only");
  });

  it("routes to timeseries_only when only csv is provided", () => {
    const route = selectInputRoute({ hasImage: false, hasMetadata: false, hasTimeseries: true });

    expect(route).toBe("timeseries_only");
  });

  it("rejects metadata only input", () => {
    const route = selectInputRoute({ hasImage: false, hasMetadata: true, hasTimeseries: false });

    expect(route).toBe("insufficient_input");
  });
});
