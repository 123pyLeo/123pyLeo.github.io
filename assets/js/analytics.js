(() => {
  const config = window.siteAnalytics;
  if (!config) return;
  // A blank configuration deliberately makes no analytics requests.
  if (config.websiteId && config.domains.includes(window.location.hostname)) {
    const script = document.createElement("script");
    script.defer = true;
    script.src = config.scriptUrl;
    script.dataset.websiteId = config.websiteId;
    script.dataset.domains = config.domains.join(",");
    document.head.appendChild(script);
  }
  if (config.shareUrl) {
    let url;
    try {
      url = new URL(config.shareUrl);
    } catch {
      return;
    }
    if (url.protocol !== "https:") return;
    const footer = document.querySelector("footer.footer");
    if (!footer) return;
    const link = document.createElement("a");
    link.className = "traffic-link";
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Traffic ↗";
    link.setAttribute("aria-label", "View public website traffic statistics");
    footer.appendChild(link);
  }
})();
