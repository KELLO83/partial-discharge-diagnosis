import { createRoot } from "react-dom/client";

import { App } from "./App";

const rootElement = document.getElementById("root");

if (rootElement !== null) {
  createRoot(rootElement).render(<App />);
}
