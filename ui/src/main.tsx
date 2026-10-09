import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Story from "./Story";
import "./styles.css";

// Default is the step-by-step demo; the full dashboard lives at ?full=1.
const full = new URLSearchParams(location.search).has("full");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {full ? <App /> : <Story />}
  </StrictMode>,
);
