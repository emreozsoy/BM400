const API_BASE = "http://127.0.0.1:5000";

let selectedGames = [];

const modeSelect = document.getElementById("mode");
const searchInput = document.getElementById("search");
const dropdown = document.getElementById("dropdown");
const chips = document.getElementById("chips");
const recommendBtn = document.getElementById("recommendBtn");
const hint = document.getElementById("hint");

const statusBox = document.getElementById("status");
const selectedTableWrap = document.getElementById("selectedTableWrap");
const recsTableWrap = document.getElementById("recsTableWrap");

function setStatus(text, type = "") {
  statusBox.className = "status" + (type ? ` ${type}` : "");
  statusBox.textContent = text;
}

function updateButton() {
  recommendBtn.disabled = selectedGames.length !== 3;
  hint.textContent = recommendBtn.disabled ? "3 oyun seçince aktif olur." : "Hazır! Öner'e bas.";
}

function renderChips() {
  chips.innerHTML = "";
  selectedGames.forEach((name, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span>${name}</span>`;

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "×";
    removeBtn.onmousedown = (e) => {
      e.preventDefault();
      e.stopPropagation();
      selectedGames.splice(index, 1);
      renderChips();
      updateButton();
      // Seçim değişti -> tabloları temizle (isteğe bağlı)
      selectedTableWrap.innerHTML = "";
      recsTableWrap.innerHTML = "";
      setStatus("Seçim güncellendi. Tekrar öneri alabilirsin.");
    };

    chip.appendChild(removeBtn);
    chips.appendChild(chip);
  });
}

function hideDropdown() {
  dropdown.style.display = "none";
  dropdown.innerHTML = "";
}

function showDropdown(items) {
  dropdown.innerHTML = "";
  if (!items.length) {
    hideDropdown();
    return;
  }

  items.forEach((name) => {
    const div = document.createElement("div");
    div.className = "item";
    div.textContent = name;

    div.onmousedown = (e) => {
      e.preventDefault();
      e.stopPropagation();

      if (selectedGames.length >= 3) return;
      if (selectedGames.includes(name)) return;

      selectedGames.push(name);
      renderChips();
      updateButton();

      searchInput.value = "";
      hideDropdown();
      searchInput.focus();
    };

    dropdown.appendChild(div);
  });

  dropdown.style.display = "block";
}

let debounceTimer = null;

async function searchGames(query) {
  try {
    const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();
    const results = (data.results || []).filter((n) => !selectedGames.includes(n));
    showDropdown(results);
  } catch (err) {
    hideDropdown();
  }
}

searchInput.addEventListener("input", () => {
  const q = searchInput.value.trim();
  if (q.length < 3) {
    hideDropdown();
    return;
  }
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => searchGames(q), 350);
});

document.addEventListener("pointerdown", (e) => {
  if (!e.target.closest(".box")) hideDropdown();
});

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderTable(containerEl, rows, columns) {
  if (!rows || rows.length === 0) {
    containerEl.innerHTML = `<div class="muted">Kayıt bulunamadı.</div>`;
    return;
  }

  const thead = `
    <thead>
      <tr>
        ${columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("")}
      </tr>
    </thead>
  `;

  const tbody = `
    <tbody>
      ${rows
        .map((r) => {
          return `
            <tr>
              ${columns
                .map((c) => {
                  const val = r[c.key];
                  return `<td>${escapeHtml(val)}</td>`;
                })
                .join("")}
            </tr>
          `;
        })
        .join("")}
    </tbody>
  `;

  containerEl.innerHTML = `<table>${thead}${tbody}</table>`;
}

function normalizeNumber(x, digits = 4) {
  const num = Number(x);
  if (Number.isFinite(num)) return num.toFixed(digits);
  return x;
}

function prepareSelectedRows(selectedDetails) {
  // selected_games_details: [{Name, Genres, Metacritic score, Recommendations}, ...]
  return (selectedDetails || []).map((r) => ({
    "Name": r["Name"],
    "Genres": r["Genres"],
    "Metacritic score": r["Metacritic score"],
    "Recommendations": r["Recommendations"],
  }));
}

function prepareRecRows(recs, mode) {
  return (recs || []).map((r) => {
    const base = {
      "Name": r["Name"],
      "Genres": r["Genres"],
      "Metacritic score": r["Metacritic score"],
      "Recommendations": r["Recommendations"],
      "final_score": normalizeNumber(r["final_score"], 6),
    };

    // Cosine modunda API cosine_similarity döndürüyorsa göster
    if (mode === "kmeans_cosine" && r["cosine_similarity"] !== undefined) {
      base["cosine_similarity"] = normalizeNumber(r["cosine_similarity"], 6);
    }
    return base;
  });
}

recommendBtn.addEventListener("click", async () => {
  if (selectedGames.length !== 3) return;

  setStatus("İstek gönderiliyor...");

  const mode = modeSelect.value;

  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ games: selectedGames, mode }),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      const msg = (data && (data.error || data.message)) ? (data.error || data.message) : `HTTP ${res.status}`;
      setStatus("Hata: " + msg, "error");
      selectedTableWrap.innerHTML = "";
      recsTableWrap.innerHTML = "";
      return;
    }

    const selectedDetails = data.selected_by_user_games_details || data.selected_games_details || [];

    const recs = data.recommendations || [];

    // Selected by user
    const selectedCols = [
      { key: "Name", label: "Name" },
      { key: "Genres", label: "Genres" },
      { key: "Metacritic score", label: "Metacritic" },
      { key: "Recommendations", label: "Recommendations" },
    ];
    renderTable(selectedTableWrap, prepareSelectedRows(selectedDetails), selectedCols);

    // Recommendations
    const recCols = [
      { key: "Name", label: "Name" },
      { key: "Genres", label: "Genres" },
      { key: "Metacritic score", label: "Metacritic" },
      { key: "Recommendations", label: "Recommendations" },
    ];

    if (data.mode === "kmeans_cosine") {
      // cosine varsa kolonu ekle (yoksa eklesek de boş kalır, sorun değil)
      recCols.push({ key: "cosine_similarity", label: "Cosine Similarity" });
    }

    recCols.push({ key: "final_score", label: "Final Score" });

    renderTable(recsTableWrap, prepareRecRows(recs, data.mode), recCols);

    setStatus(`Başarılı ✅ | Mode: ${data.mode}`, "ok");
  } catch (err) {
    setStatus("Hata: " + err, "error");
    selectedTableWrap.innerHTML = "";
    recsTableWrap.innerHTML = "";
  }
});

updateButton();
renderChips();
