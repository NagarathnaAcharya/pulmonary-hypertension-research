const pages = ["overview", "ph-model", "mortality", "external"];
const pageLabels = {
  overview: "Overview",
  "ph-model": "PH model",
  mortality: "Mortality analysis",
  external: "External validation",
};

const sidebar = document.querySelector("#sidebar");
const backdrop = document.querySelector("#sidebar-backdrop");
const toast = document.querySelector("#toast");
let toastTimer;

function showPage(name) {
  if (!pages.includes(name)) return;
  for (const page of pages) {
    const section = document.querySelector(`#page-${page}`);
    const active = page === name;
    section.hidden = !active;
    section.classList.toggle("active", active);
  }
  document.querySelectorAll("[data-page]").forEach((control) => {
    if (control.classList.contains("nav-item")) {
      control.classList.toggle("active", control.dataset.page === name);
      if (control.dataset.page === name) control.setAttribute("aria-current", "page");
      else control.removeAttribute("aria-current");
    }
  });
  document.querySelector("#breadcrumb-current").textContent = pageLabels[name];
  sidebar.classList.remove("open");
  backdrop.classList.remove("visible");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll("[data-page]").forEach((control) => {
  control.addEventListener("click", () => showPage(control.dataset.page));
});

document.querySelector("#menu-toggle").addEventListener("click", () => {
  sidebar.classList.toggle("open");
  backdrop.classList.toggle("visible", sidebar.classList.contains("open"));
});

backdrop.addEventListener("click", () => {
  sidebar.classList.remove("open");
  backdrop.classList.remove("visible");
});

function announce(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

document.querySelector("#export-button").addEventListener("click", () => {
  const summary = [
    "PulmoScope — research summary",
    "Dataset: MIMIC-IV v3.1; retrospective ICU cohort",
    "PH cohort: 7,815 patients; 72.3% classified as PH-positive by the study definition (24-hour median mPAP > 20 mmHg), not clinically diagnosed PAH",
    "Elevated mPAP classification: Model A logistic regression test AUROC 0.749 (95% CI 0.721–0.775)",
    "Adding cardiac output resulted in a negligible improvement in logistic regression test AUROC (0.749 to 0.750)",
    "Mortality landmark: XGBoost test AUROC 0.935 (95% CI 0.922–0.948); AUPRC 0.386",
    "External validation: 5-feature eICU model AUROC 0.590; calibration was poor",
    "Research only. Not a diagnostic or treatment tool.",
  ].join("\n");
  const file = new Blob([summary], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = "pulmoscope-research-summary.txt";
  link.click();
  URL.revokeObjectURL(url);
  announce("Research summary exported");
});
