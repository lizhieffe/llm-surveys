(() => {
  "use strict";

  const categoriesEl = document.getElementById("categories");
  const statsEl = document.getElementById("stats");
  const navEl = document.getElementById("category-nav");
  const searchEl = document.getElementById("search");
  const generatedAtEl = document.getElementById("generated-at");
  const backdrop = document.getElementById("modal-backdrop");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");
  const modalClose = document.getElementById("modal-close");

  const fmtNum = (n) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}k` : String(n);

  function shortName(repoId) {
    return repoId.split("/")[1] || repoId;
  }

  function categorySlugId(slug) {
    return "cat-" + slug.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
  }

  function renderStats(manifest) {
    statsEl.innerHTML = `
      <span><strong>${manifest.num_categories}</strong> categories</span>
      <span><strong>${manifest.num_unique_datasets}</strong> unique datasets</span>
      <span><strong>${manifest.num_sampled_ok}</strong> sampled &middot; 16 rows each</span>
    `;
    if (manifest.generated_at) {
      const d = new Date(manifest.generated_at);
      generatedAtEl.textContent = `Manifest generated ${d.toISOString().slice(0, 10)} from the Hugging Face Hub API.`;
    }
  }

  function datasetCard(cat, ds) {
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.dataset.name = ds.repo_id.toLowerCase();

    const badges = [];
    if (ds.license) badges.push(`<span class="badge">${ds.license}</span>`);
    if (ds.gated) badges.push(`<span class="badge gated">gated</span>`);
    if (ds.downloads) badges.push(`<span class="badge">${fmtNum(ds.downloads)} downloads</span>`);
    if (ds.likes) badges.push(`<span class="badge">${fmtNum(ds.likes)} &hearts;</span>`);

    const canSample = ds.sample_status === "ok";
    const btnLabel = canSample
      ? "View 16 examples"
      : ds.sample_status === "gated"
      ? "Gated (no preview)"
      : "No preview available";

    card.innerHTML = `
      <div class="dataset-card-title"><a href="${ds.url}" target="_blank" rel="noopener">${shortName(ds.repo_id)}</a></div>
      <div class="dataset-owner">${ds.repo_id}</div>
      ${ds.description ? `<div class="dataset-desc">${ds.description}</div>` : ""}
      <div class="badges">${badges.join("")}</div>
      <div class="card-actions">
        <a href="${ds.url}" target="_blank" rel="noopener">View on Hugging Face &#8599;</a>
        <button class="view-samples-btn" ${canSample ? "" : "disabled"}>${btnLabel}</button>
      </div>
    `;

    if (canSample) {
      card.querySelector(".view-samples-btn").addEventListener("click", () => openSamples(ds));
    }
    return card;
  }

  function renderCategories(manifest) {
    categoriesEl.innerHTML = "";
    navEl.innerHTML = "";

    for (const cat of manifest.categories) {
      const id = categorySlugId(cat.slug);

      const navLink = document.createElement("a");
      navLink.href = `#${id}`;
      navLink.textContent = `${cat.title} (${cat.datasets.length})`;
      navEl.appendChild(navLink);

      const section = document.createElement("section");
      section.className = "category";
      section.id = id;
      section.innerHTML = `
        <div class="category-head">
          <h2>${cat.title}</h2>
          <span class="category-count">${cat.datasets.length} dataset${cat.datasets.length === 1 ? "" : "s"}</span>
        </div>
        ${cat.description ? `<p class="category-desc">${cat.description}</p>` : ""}
        <a class="category-link" href="${cat.url}" target="_blank" rel="noopener">View collection on Hugging Face &#8599;</a>
      `;

      const grid = document.createElement("div");
      grid.className = "dataset-grid";
      for (const ds of cat.datasets) grid.appendChild(datasetCard(cat, ds));
      section.appendChild(grid);

      categoriesEl.appendChild(section);
    }
  }

  function applyFilter(query) {
    const q = query.trim().toLowerCase();
    document.querySelectorAll(".category").forEach((section) => {
      let visibleCount = 0;
      section.querySelectorAll(".dataset-card").forEach((card) => {
        const match = !q || card.dataset.name.includes(q);
        card.style.display = match ? "" : "none";
        if (match) visibleCount++;
      });
      section.style.display = visibleCount === 0 ? "none" : "";
    });
  }

  // --- Sample modal -------------------------------------------------------

  function truncatable(str, limit = 320) {
    return str.length > limit;
  }

  function renderScalar(value) {
    if (value === null || value === undefined) return `<span class="field-value">&mdash;</span>`;
    if (typeof value === "string") {
      if (truncatable(value)) {
        const shortText = value.slice(0, 320);
        const id = "t" + Math.random().toString(36).slice(2);
        return `
          <div class="field-value" id="${id}-short">${escapeHtml(shortText)}&hellip;
            <button class="expand-toggle" data-target="${id}">Show more</button>
          </div>
          <div class="field-value" id="${id}-full" style="display:none">${escapeHtml(value)}
            <button class="expand-toggle" data-target="${id}" data-collapse="1">Show less</button>
          </div>
        `;
      }
      return `<div class="field-value">${escapeHtml(value)}</div>`;
    }
    return `<div class="field-value">${escapeHtml(String(value))}</div>`;
  }

  function looksLikeEmbedding(arr) {
    return arr.length > 32 && arr.every((v) => typeof v === "number");
  }

  function renderValue(value, depth = 0) {
    if (value === null || value === undefined) return renderScalar(value);
    if (typeof value !== "object") return renderScalar(value);

    if (Array.isArray(value)) {
      if (value.length === 0) return `<div class="field-value mono">[]</div>`;
      if (looksLikeEmbedding(value)) {
        const preview = value.slice(0, 8).map((v) => v.toFixed(4)).join(", ");
        return `<div class="field-value mono">[${preview}, &hellip;] &nbsp;(${value.length}-dim vector)</div>`;
      }
      if (depth >= 2 || value.length > 6) {
        const shown = value.slice(0, 6);
        const json = JSON.stringify(shown, null, 2);
        const more = value.length > 6 ? `\n  &hellip; (+${value.length - 6} more)` : "";
        return `<div class="field-value mono">${escapeHtml(json)}${more}</div>`;
      }
      return value.map((v, i) => `<div style="margin-bottom:6px"><span class="field-key">[${i}]</span>${renderValue(v, depth + 1)}</div>`).join("");
    }

    // plain object
    const entries = Object.entries(value);
    if (entries.length === 0) return `<div class="field-value mono">{}</div>`;
    if (depth >= 2) {
      return `<div class="field-value mono">${escapeHtml(JSON.stringify(value, null, 2))}</div>`;
    }
    return entries
      .map(([k, v]) => `<div style="margin:4px 0 4px 10px"><span class="field-key">${escapeHtml(k)}</span>${renderValue(v, depth + 1)}</div>`)
      .join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderExample(row, idx) {
    const fields = Object.entries(row)
      .map(
        ([key, value]) => `
        <div class="field">
          <div class="field-key">${escapeHtml(key)}</div>
          ${renderValue(value)}
        </div>`
      )
      .join("");
    return `
      <div class="example">
        <div class="example-head">Example ${idx + 1}</div>
        <div class="example-body">${fields}</div>
      </div>
    `;
  }

  const sampleCache = new Map();

  async function openSamples(ds) {
    modalTitle.textContent = ds.repo_id;
    modalBody.innerHTML = `<div class="loading">Loading sampled rows&hellip;</div>`;
    backdrop.hidden = false;

    try {
      let data = sampleCache.get(ds.sample_file);
      if (!data) {
        const resp = await fetch(`data/samples/${ds.sample_file}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        data = await resp.json();
        sampleCache.set(ds.sample_file, data);
      }

      const meta = `<div class="modal-meta">config: <code>${data.config}</code> &middot; split: <code>${data.split}</code> &middot; showing ${data.rows.length} row${data.rows.length === 1 ? "" : "s"}</div>`;
      const rowsHtml = data.rows.map((row, i) => renderExample(row, i)).join("");
      modalBody.innerHTML = meta + (rowsHtml || `<div class="empty-state">No rows returned.</div>`);

      modalBody.querySelectorAll(".expand-toggle").forEach((btn) => {
        btn.addEventListener("click", () => {
          const id = btn.dataset.target;
          const short = document.getElementById(`${id}-short`);
          const full = document.getElementById(`${id}-full`);
          if (btn.dataset.collapse) {
            short.style.display = "";
            full.style.display = "none";
          } else {
            short.style.display = "none";
            full.style.display = "";
          }
        });
      });
    } catch (err) {
      modalBody.innerHTML = `<div class="empty-state">Couldn't load samples: ${escapeHtml(err.message)}</div>`;
    }
  }

  function closeModal() {
    backdrop.hidden = true;
    modalBody.innerHTML = "";
  }

  modalClose.addEventListener("click", closeModal);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !backdrop.hidden) closeModal();
  });

  searchEl.addEventListener("input", (e) => applyFilter(e.target.value));

  // --- Boot -----------------------------------------------------------------

  fetch("data/manifest.json")
    .then((r) => r.json())
    .then((manifest) => {
      renderStats(manifest);
      renderCategories(manifest);
    })
    .catch((err) => {
      categoriesEl.innerHTML = `<p class="empty-state">Failed to load manifest.json: ${escapeHtml(err.message)}</p>`;
    });
})();
