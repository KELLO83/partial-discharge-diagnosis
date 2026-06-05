import { describe, expect, it } from "vitest";

import { fileDisplayText } from "./fileDisplay";

describe("fileDisplayText", () => {
  it("shows waiting copy when a file is missing", () => {
    const text = fileDisplayText(null, "PRPD PNG");

    expect(text).toBe("PRPD PNG 업로드 대기");
  });

  it("shows the selected file name", () => {
    const file = new File([""], "signal.csv", { type: "text/csv" });

    const text = fileDisplayText(file, "시계열 CSV");

    expect(text).toBe("signal.csv");
  });
});
