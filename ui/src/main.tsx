import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Story from "./Story";
import { SNAP } from "./snapshot";
import "./styles.css";

// Default is the step-by-step demo; the full dashboard lives at ?full=1.
const full = !SNAP && new URLSearchParams(location.search).has("full");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {full ? <App /> : <Story />}
  </StrictMode>,
);
