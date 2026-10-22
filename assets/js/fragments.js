document.querySelectorAll("[data-expand-list]").forEach((button) => {
  const list = document.getElementById(button.dataset.expandList);
  if (!list) return;
  const extraRows = [...list.querySelectorAll("[data-extra]")];
  button.addEventListener("click", () => {
    const expanded = button.getAttribute("aria-expanded") !== "true";
    const listTop = list.getBoundingClientRect().top + window.scrollY;
    extraRows.forEach((row) => {
      row.hidden = !expanded;
    });
    button.setAttribute("aria-expanded", String(expanded));
    button.querySelector("[data-toggle-label]").textContent = expanded
      ? button.dataset.less
      : button.dataset.more;
    button.querySelector("[data-toggle-icon]").textContent = expanded
      ? "−"
      : "+";
    // Keep the beginning of the list in view when collapsing a long archive.
    if (!expanded && button.getBoundingClientRect().bottom < 0) {
      window.scrollTo({ top: Math.max(0, listTop - 100), behavior: "instant" });
    }
  });
});
