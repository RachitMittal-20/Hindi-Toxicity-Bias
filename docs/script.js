/* ============================================
   Hindi Toxicity Bias — interactive bits
   ============================================ */

(function () {
  "use strict";

  // ---- Loader ----
  // Hide once page is fully loaded (with a minimum visible time so it doesn't flash)
  const loader = document.getElementById("loader");
  const minVisible = 700; // ms
  const start = Date.now();
  function hideLoader() {
    if (!loader) return;
    const elapsed = Date.now() - start;
    const wait = Math.max(0, minVisible - elapsed);
    setTimeout(function () {
      loader.classList.add("hidden");
      // Remove from DOM after transition for accessibility
      setTimeout(function () { loader.remove(); }, 600);
    }, wait);
  }
  if (document.readyState === "complete") {
    hideLoader();
  } else {
    window.addEventListener("load", hideLoader);
    // Fallback in case load never fires
    setTimeout(hideLoader, 2500);
  }

  // ---- Scroll-reveal: tag elements and observe ----
  const revealTargets = document.querySelectorAll(
    ".section h2, .section .section-lede, .card, .repro-steps li, .author-card, .delta-card, .stat"
  );
  revealTargets.forEach(function (el) { el.classList.add("fade-up"); });

  if ("IntersectionObserver" in window) {
    const revealObs = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            revealObs.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.05 }
    );
    revealTargets.forEach(function (el) { revealObs.observe(el); });
  } else {
    revealTargets.forEach(function (el) { el.classList.add("visible"); });
  }

  // ---- Mobile nav toggle ----
  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    // close on link click
    links.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  // ---- Active section highlight ----
  const navAnchors = document.querySelectorAll(".nav-links a");
  const sections = Array.from(navAnchors)
    .map(function (a) {
      const id = a.getAttribute("href");
      return id && id.startsWith("#") ? document.querySelector(id) : null;
    })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            const id = "#" + entry.target.id;
            navAnchors.forEach(function (a) {
              if (a.getAttribute("href") === id) {
                a.style.color = "var(--text)";
                a.style.fontWeight = "600";
              } else {
                a.style.color = "";
                a.style.fontWeight = "";
              }
            });
          }
        });
      },
      { rootMargin: "-40% 0px -55% 0px" }
    );
    sections.forEach(function (s) { observer.observe(s); });
  }

  // ---- FPR per-group chart ----
  // Source: results/group_fpr_details.csv
  const fprData = {
    caste:    { Baseline: 0.4211, CDA: 0.3421, Adversarial: 0.3947 },
    gender:   { Baseline: 0.1505, CDA: 0.1465, Adversarial: 0.1386 },
    region:   { Baseline: 0.0000, CDA: 0.1667, Adversarial: 0.1667 },
    religion: { Baseline: 0.2825, CDA: 0.2825, Adversarial: 0.2486 }
  };

  const groupOrder = ["religion", "caste", "gender", "region"];
  const modelOrder = ["Baseline", "CDA", "Adversarial"];
  const modelClass = { Baseline: "baseline", CDA: "cda", Adversarial: "adv" };
  // Find max value across all groups for consistent scaling
  let maxFpr = 0;
  groupOrder.forEach(function (g) {
    modelOrder.forEach(function (m) {
      if (fprData[g][m] > maxFpr) maxFpr = fprData[g][m];
    });
  });
  // Round up for nicer scale
  const scaleMax = Math.ceil(maxFpr * 10) / 10 || 0.5;

  const chartEl = document.getElementById("fprChart");
  if (chartEl) {
    chartEl.innerHTML = "";
    groupOrder.forEach(function (group) {
      const labelDiv = document.createElement("div");
      labelDiv.className = "chart-row-label";
      labelDiv.textContent = group.charAt(0).toUpperCase() + group.slice(1);
      chartEl.appendChild(labelDiv);

      const barsWrap = document.createElement("div");
      barsWrap.className = "chart-bars";
      modelOrder.forEach(function (model) {
        const value = fprData[group][model];
        const row = document.createElement("div");
        row.className = "chart-bar";

        const lab = document.createElement("span");
        lab.className = "chart-bar-label";
        lab.textContent = model;

        const track = document.createElement("div");
        track.className = "chart-bar-track";
        const fill = document.createElement("div");
        fill.className = "chart-bar-fill " + modelClass[model];
        fill.style.width = "0%";
        track.appendChild(fill);

        const val = document.createElement("span");
        val.className = "chart-bar-value";
        val.textContent = value.toFixed(3);

        row.appendChild(lab);
        row.appendChild(track);
        row.appendChild(val);
        barsWrap.appendChild(row);

        // Animate after a tick
        requestAnimationFrame(function () {
          setTimeout(function () {
            fill.style.width = ((value / scaleMax) * 100).toFixed(1) + "%";
          }, 80);
        });
      });
      chartEl.appendChild(barsWrap);
    });
  }
})();
