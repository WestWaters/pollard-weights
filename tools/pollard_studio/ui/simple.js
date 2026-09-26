/* Pollard Studio — SIMPLE mode.
   Device → Model → Install, the same three slabs as pollard.app, with the two things only the desktop
   app can do: read the machine it is running on, and put the build on disk and into chat.
   Advanced mode (app.js) is untouched; this file only talks to it through setMode() and, on
   "Open in chat", the existing R / select() / show() so the downloaded build lands in the chat screen. */
(() => {
  const HEADROOM = 2.0;                                       // GB for KV cache + context
  const LANES = ["GGUF", "MLX", "GPTQ", "EXL3", "MX"];
  const api = () => (window.pywebview && window.pywebview.api) || null;
  const $ = s => document.querySelector(s);
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // the same device list as the site, for the override chips; the detected box is the default
  const CATS = {
    detected: { l: "This machine", d: {} , svg: '<rect x="4" y="6" width="22" height="14" rx="1.6"/><path d="M2 24h26"/>' },
    phone: { l: "Phone", d: { "ph-6": ["6 GB", 3], "ph-8": ["8 GB", 5], "ph-12": ["12 GB", 8], "ph-16": ["16 GB", 11] },
      svg: '<rect x="8" y="3" width="14" height="24" rx="3"/><path d="M13 23h4"/>' },
    pi: { l: "Raspberry Pi · SBC", d: { "pi-4": ["Pi 4 · 4 GB", 2.5], "pi-8": ["Pi 4/5 · 8 GB", 6], "pi-16": ["Pi 5 · 16 GB", 12], "jet-8": ["Jetson · 8 GB", 6], "jet-16": ["Jetson · 16 GB", 13] },
      svg: '<rect x="3" y="7" width="24" height="16" rx="1.6"/><path d="M7 7V4M11 7V4M15 7V4M19 7V4M23 7V4"/><rect x="7" y="12" width="7" height="6" rx="1"/><circle cx="21" cy="15" r="2"/>' },
    mac: { l: "Mac", d: { "mac-8": ["8 GB", 5], "mac-16": ["16 GB", 11], "mac-24": ["24 GB", 17], "mac-32": ["32 GB", 24], "mac-48": ["48 GB", 38], "mac-64": ["64 GB", 52], "mac-96": ["96 GB", 80], "mac-128": ["128 GB", 108], "mac-192": ["192 GB", 168], "mac-512": ["512 GB", 460] },
      svg: '<rect x="4" y="6" width="22" height="14" rx="1.6"/><path d="M2 24h26"/>' },
    gpu: { l: "PC · GPU", d: { "gpu-8": ["8 GB", 7], "gpu-12": ["12 GB", 11], "gpu-16": ["16 GB", 15], "gpu-24": ["24 GB", 23], "gpu-32": ["32 GB", 31], "gpu-48": ["48 GB", 46], "gpu-80": ["80 GB", 78], "gpu-96": ["96 GB", 94] },
      svg: '<rect x="3" y="8" width="24" height="13" rx="1.6"/><circle cx="11" cy="14.5" r="3.2"/><circle cx="20" cy="14.5" r="3.2"/><path d="M5 21v3M25 21v3"/>' },
    box: { l: "AI box · unified", d: { "spark-1": ["DGX Spark · 128 GB", 110], "spark-2": ["2× Spark · 256 GB", 230], "halo-64": ["Strix Halo · 64 GB", 48], "halo-128": ["Strix Halo · 128 GB", 100], "orin-64": ["Jetson AGX Orin · 64 GB", 52], "thor-128": ["Jetson Thor · 128 GB", 108] },
      svg: '<rect x="4" y="9" width="22" height="12" rx="2"/><path d="M8 13h6M8 17h10"/><circle cx="21" cy="15" r="1.4"/><path d="M9 9V6h12v3"/>' },
    srv: { l: "Server · Cluster", d: { "srv-128": ["128 GB", 120], "srv-256": ["256 GB", 240], "srv-512": ["512 GB", 490], "srv-1t": ["1 TB", 980], "srv-2t": ["2 TB", 1960] },
      svg: '<rect x="4" y="4" width="22" height="6" rx="1.4"/><rect x="4" y="12" width="22" height="6" rx="1.4"/><rect x="4" y="20" width="22" height="6" rx="1.4"/><path d="M8 7h.01M8 15h.01M8 23h.01"/>' }
  };

  const S = { hw: null, cat: "detected", dev: null, lane: "GGUF", shelf: [], repo: null, file: null, q: "", dl: null, local: {} };

  const budget = () => S.cat === "detected" ? (S.hw ? S.hw.budget_gb : 11) : CATS[S.cat].d[S.dev][1];
  const devLabel = () => S.cat === "detected" ? (S.hw ? S.hw.label : "This machine") : CATS[S.cat].l + " · " + CATS[S.cat].d[S.dev][0];
  const verdict = gb => gb + HEADROOM <= budget() ? ["fits", "FITS"] : gb <= budget() ? ["tight", "TIGHT"] : ["no", "TOO BIG"];
  const rungs = () => S.shelf.flatMap(r => r.files.map(f => ({ r, f })));
  const laneRungs = () => rungs().filter(x => x.r.lane === (S.lane === "MLX" ? "mlx" : "gguf"));
  const localPath = (r, f) => S.local[r.id + "/" + f.file] || null;

  /* ── the machine ─────────────────────────────────────────────────────── */
  async function readMachine() {
    let hw = null;
    try { if (api()) hw = await api().hardware(); } catch (e) { console.warn(e); }
    if (!hw || !hw.ram_gb) { S.hw = { label: "This machine", budget_gb: 11, ram_gb: null, note: "could not read memory — pick a preset" }; return; }
    const kind = hw.platform === "darwin" ? "Mac" : hw.platform === "win32" ? "PC" : "Linux box";
    // what the model can actually have: what is free right now if we know it, else Studio's 75% suggestion
    const usable = hw.avail_gb != null ? Math.round(Math.min(hw.ram_gb * 0.85, Math.max(hw.avail_gb, hw.ram_gb * 0.5))) : hw.suggest_target_gb;
    const ramGiB = Math.round(hw.ram_gb * 1e9 / 1073741824);        // the number on the box, not the decimal one
    S.hw = { label: `${kind} · ${ramGiB} GB`, budget_gb: usable, ram_gb: hw.ram_gb,
      note: hw.avail_gb != null ? `${hw.avail_gb} GB free right now · ${hw.cpus} CPUs` : `${hw.cpus} CPUs · ${hw.suggest_target_gb} GB suggested` };
  }

  /* ── the shelf ───────────────────────────────────────────────────────── */
  async function readShelf() {
    let data = null;
    try { if (api()) data = await api().shelf(); } catch (e) { console.warn(e); }
    if (!data || !data.shelf) {                            // no bridge (browser preview): straight from the Hub
      try { data = await shelfFromHub(); } catch (e) { data = { shelf: [] }; }
    }
    S.shelf = data.shelf || [];
    S.local = data.local || {};
  }
  async function shelfFromHub() {
    const models = await fetch("https://huggingface.co/api/models?author=PollardWeights&limit=100").then(r => r.json());
    const shelf = [];
    for (const m of models) {
      const tree = await fetch(`https://huggingface.co/api/models/${m.id}/tree/main`).then(r => r.json()).catch(() => []);
      shelf.push(shape(m.id, m.downloads || 0, tree.map(t => ({ name: t.path, bytes: (t.lfs && t.lfs.size) || t.size || 0 }))));
    }
    return { shelf: shelf.filter(r => r.files.length) };
  }
  // one shape for a repo, shared with the Python side: name, lane, vision, moe, files[{q, gb, ik, file}]
  function shape(id, dl, files) {
    const short = id.split("/")[1];
    const out = { id, name: short.replace("-Pollard", ""), dl, vision: false, lane: "gguf", moe: /A\d+B|Ling/i.test(short) ? "MoE" : "", files: [] };
    for (const f of files) {
      if (f.name.startsWith("mmproj")) { out.vision = true; continue; }
      if (f.name.endsWith(".safetensors")) { out.lane = "mlx"; out.files.push({ q: "4bit", gb: +(f.bytes / 1e9).toFixed(2), ik: false, file: f.name }); continue; }
      if (!f.name.endsWith(".gguf")) continue;
      const q = f.name.replace(/\.gguf$/, "").split("-").pop();
      out.files.push({ q, gb: +(f.bytes / 1e9).toFixed(2), ik: q.endsWith("_KT"), file: f.name });
    }
    out.files.sort((a, b) => a.gb - b.gb);
    return out;
  }

  /* ── render ──────────────────────────────────────────────────────────── */
  function render() {
    const root = $("#simple"); if (!root) return;
    if (!root.dataset.built) { root.innerHTML = frame(); root.dataset.built = "1"; }
    // slab 1
    $("#s-rows").innerHTML = Object.entries(CATS).filter(([k]) => k !== "detected").map(([k, c]) => `
      <button class="row" aria-pressed="${k === S.cat}" data-cat="${k}">
        <svg viewBox="0 0 30 30">${c.svg}</svg>${c.l}<span class="ch">&rsaquo;</span></button>`).join("");
    $("#s-rows").querySelectorAll(".row").forEach(b => b.onclick = () => { S.cat = b.dataset.cat; S.dev = S.cat === "detected" ? null : Object.keys(CATS[S.cat].d)[0]; S.file = null; render(); });
    $("#s-detect").innerHTML = S.hw ? `<b>${esc(S.hw.label)}</b><span>${esc(S.hw.note)}</span><small>${S.hw.budget_gb} GB usable for a model</small>` : `<b>Reading this machine…</b>`;
    $("#s-detect").setAttribute("aria-pressed", S.cat === "detected");
    $("#s-detect").onclick = () => { S.cat = "detected"; S.dev = null; S.file = null; render(); };
    $("#s-chips").innerHTML = S.cat === "detected" ? "" : Object.entries(CATS[S.cat].d).map(([id, [l]]) =>
      `<button class="chip" aria-pressed="${id === S.dev}" data-dev="${id}">${l}</button>`).join("");
    $("#s-chips").querySelectorAll(".chip").forEach(b => b.onclick = () => { S.dev = b.dataset.dev; S.file = null; render(); });

    // slab 2
    const q = S.q.toLowerCase();
    const all = laneRungs().filter(x => !q || (x.r.name + " " + x.f.q).toLowerCase().includes(q));
    const ok = all.filter(x => verdict(x.f.gb)[0] !== "no").sort((a, b) => (verdict(a.f.gb)[0] === "fits" ? 0 : 1) - (verdict(b.f.gb)[0] === "fits" ? 0 : 1) || b.f.gb - a.f.gb);
    const cur = S.file && laneRungs().find(x => x.f.file === S.file);
    if (ok.length && (!cur || verdict(cur.f.gb)[0] === "no")) { S.repo = ok[0].r.id; S.file = ok[0].f.file; }
    $("#s-found").innerHTML = S.shelf.length ? `Found in <b>${ok.length} FITS</b>` : `Loading the shelf…`;
    $("#s-fitword").textContent = ok.length ? "FITS" : "TOO BIG";
    $("#s-hits").innerHTML = (ok.length ? ok : all).slice(0, 12).map(x => `
      <button class="hit" aria-pressed="${x.f.file === S.file}" data-repo="${x.r.id}" data-file="${x.f.file}">
        <span>${esc(x.r.name)} <small>${x.f.q}</small>${localPath(x.r, x.f) ? '<span class="own">ON DISK</span>' : ""}</span><small>${x.f.gb.toFixed(1)} GB</small></button>`).join("");
    $("#s-hits").querySelectorAll(".hit").forEach(b => b.onclick = () => { S.repo = b.dataset.repo; S.file = b.dataset.file; render(); });

    // slab 3
    $("#s-lanes").innerHTML = LANES.map(l => `<button aria-pressed="${l === S.lane}" data-lane="${l}">${l}${l === S.lane ? " <span>&#10003;</span>" : ""}</button>`).join("");
    $("#s-lanes").querySelectorAll("button").forEach(b => b.onclick = () => { S.lane = b.dataset.lane; S.file = null; render(); });
    const btn = $("#s-install"), open = $("#s-open"), prog = $("#s-prog"), cmd = $("#s-cmd"), chosen = $("#s-chosen");
    const shelfLane = S.lane === "GGUF" || S.lane === "MLX";
    if (S.file && shelfLane) {
      const r = S.shelf.find(x => x.id === S.repo), f = r.files.find(x => x.file === S.file), [cls, txt] = verdict(f.gb);
      const here = localPath(r, f);
      chosen.innerHTML = `<b>${esc(r.name)} · ${f.q}</b><br>${f.gb.toFixed(2)} GB + ${HEADROOM.toFixed(1)} KV of ${budget()} GB <span class="v ${cls}">${txt}</span>${f.ik ? "<br><span style='color:var(--scold)'>needs the ik_llama runtime</span>" : ""}`;
      if (S.dl && S.dl.active && S.dl.file === f.file) {
        btn.textContent = `${S.dl.pct}%`; btn.className = "install busy"; btn.disabled = true; prog.style.display = ""; prog.firstElementChild.style.width = S.dl.pct + "%";
      } else {
        btn.textContent = here ? "INSTALLED" : "INSTALL"; btn.className = "install"; btn.disabled = !!here || !api(); prog.style.display = "none";
      }
      open.style.display = here ? "" : "none";
      cmd.textContent = S.lane === "GGUF" ? `ollama run hf.co/${r.id}:${f.q}` : `mlx_lm.generate --model ${r.id}`;
      btn.onclick = () => install(r, f);
      open.onclick = () => openInChat(r, f, here);
    } else if (!shelfLane) {
      chosen.innerHTML = `<b>${S.lane}</b><br>not on the shelf yet — build it here in one shot, or have it cooked`;
      btn.textContent = "BUILD IN ADVANCED"; btn.className = "install"; btn.disabled = false; prog.style.display = "none"; open.style.display = "none";
      cmd.textContent = `pollard --hf <model> --format ${S.lane.toLowerCase()} --run`;
      btn.onclick = () => { setMode("advanced"); if (window.show) window.show("build"); };
    } else {
      chosen.textContent = S.shelf.length ? "Pick a rung and this becomes one click." : "Loading the shelf…";
      btn.textContent = "INSTALL"; btn.className = "install"; btn.disabled = true; prog.style.display = "none"; open.style.display = "none"; cmd.textContent = "";
    }
  }

  function frame() {
    return `
    <div class="s-top">
      <div class="s-mark">POLLARD</div>
      <div class="s-right">
        <div class="s-mode"><button aria-pressed="true">Simple</button><button aria-pressed="false" data-adv>Advanced</button></div>
        <div class="s-win">
          <button title="Minimize" data-win="minimize"><svg viewBox="0 0 16 16"><rect x="3.5" y="7.2" width="9" height="1.6" rx=".8"/></svg></button>
          <button title="Full screen" data-win="toggle"><svg viewBox="0 0 16 16"><path d="M3.4 3.4h4v1.5H4.9v2.5H3.4zM12.6 12.6h-4v-1.5h2.5V8.6h1.5z"/></svg></button>
          <button class="x" title="Close" data-win="close"><svg viewBox="0 0 16 16"><path d="M4.6 3.5l3.4 3.4 3.4-3.4 1.1 1.1L9.1 8l3.4 3.4-1.1 1.1L8 9.1l-3.4 3.4-1.1-1.1L6.9 8 3.5 4.6z"/></svg></button>
        </div>
      </div>
    </div>
    <div class="s-wrap">
      <div class="s-head"><p class="s-caps">Fit</p><h2>One question. One tap. One file.</h2>
        <p>This machine is already read. Every rung on the shelf is badged against what it can actually hold.</p></div>
      <div class="s-scene">
        <div class="slab l"><div class="smark">POLLARD<i></i></div><h3>What machine will this run on?</h3>
          <div class="s-detect" id="s-detect"></div><div class="rows" id="s-rows"></div><div class="chips" id="s-chips"></div></div>
        <div class="slab c"><div class="smark">POLLARD<i></i></div>
          <div class="search"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.6-3.6"/></svg><input id="s-q" placeholder="Search models..."></div>
          <div class="found" id="s-found"></div><div class="hits" id="s-hits"></div>
          <div class="fitsrow"><div class="fits" id="s-fitword">FITS</div></div></div>
        <div class="slab r"><div class="smark">POLLARD<i></i></div>
          <div class="lanes" id="s-lanes"></div><div class="chosen" id="s-chosen"></div>
          <button class="install" id="s-install">INSTALL</button>
          <div class="progress" id="s-prog" style="display:none"><i></i></div>
          <p class="micro">1-click &bull; verified &bull; on this machine</p>
          <button class="open" id="s-open" style="display:none">Open in chat &rsaquo;</button>
          <div class="cmd" id="s-cmd"></div>
          <p class="micro link" id="s-request">Need a different model? Request a build &rsaquo;</p></div>
      </div>
      <p class="s-note">Everything else — measure, allocate, verify, publish, cluster — is one click away in Advanced.</p>
    </div>`;
  }

  /* ── install: download into the workspace, then hand to chat ─────────── */
  async function install(r, f) {
    if (!api()) return;
    let res; try { res = await api().download(r.id, f.file); } catch (e) { res = { ok: false, error: String(e) }; }
    if (!res || !res.ok) { alert(res && res.error || "download failed"); return; }
    S.dl = { active: true, pct: 0, file: f.file }; render();
    const tick = async () => {
      let st; try { st = await api().download_status(); } catch (e) { st = { active: false, error: String(e) }; }
      if (st.active) { S.dl = { active: true, pct: st.pct || 0, file: f.file }; render(); setTimeout(tick, 700); return; }
      S.dl = null;
      if (st.error) { alert("download failed: " + st.error); render(); return; }
      if (st.path) S.local[r.id + "/" + f.file] = st.path;
      try { if (window.S && api()) Object.assign(window.S, await api().rescan()); if (window.renderModelPicker) window.renderModelPicker(); } catch (e) {}
      render();
    };
    tick();
  }
  async function openInChat(r, f, path) {
    if (!path) return;
    try {
      if (window.R) { window.R.gguf = path; window.R.source = path; }
      if (api() && window.S) { const st = await api().rescan(); Object.assign(window.S, st);
        const m = (st.models || []).find(m => (m.name || "").toLowerCase() === r.name.toLowerCase() || (m.key || "").includes(r.name));
        if (m) Object.assign(window.S, await api().select(m.key)); }
      if (window.renderModelPicker) window.renderModelPicker();
    } catch (e) { console.warn(e); }
    setMode("advanced"); if (window.show) window.show("chat");
  }

  /* ── mode ────────────────────────────────────────────────────────────── */
  function setMode(mode) {
    const simple = mode === "simple";
    document.body.classList.toggle("mode-simple", simple);
    const root = $("#simple"); if (root) root.hidden = !simple;
    try { localStorage.setItem("pollard.mode", mode); } catch (e) {}
    const adv = $("#modebtn"); if (adv) adv.setAttribute("aria-pressed", simple ? "false" : "true");
    if (simple && !S.hw) start();
  }
  window.setMode = setMode;

  async function start() {
    render();
    $("#simple").addEventListener("input", e => { if (e.target.id === "s-q") { S.q = e.target.value; render(); } });
    $("#simple").addEventListener("click", e => {
      const w = e.target.closest("[data-win]"); if (w && window.win) return window.win(w.dataset.win);
      if (e.target.closest("[data-adv]")) return setMode("advanced");
      if (e.target.id === "s-request") { const u = "https://pollard.app/#request"; if (api() && api().open_url) api().open_url(u); else window.open(u); }
    });
    await readMachine(); render();
    await readShelf(); render();
    try { const st = api() && await api().download_status(); if (st && st.active) { S.dl = { active: true, pct: st.pct || 0, file: st.file }; render(); } } catch (e) {}
  }

  // Simple is the front door; Advanced is remembered once chosen.
  const boot = () => { let m = "simple"; try { m = localStorage.getItem("pollard.mode") || "simple"; } catch (e) {} setMode(m); };
  if (window.pywebview) boot(); else { window.addEventListener("pywebviewready", boot); window.addEventListener("load", () => setTimeout(() => { if (!S.hw) boot(); }, 350)); }
})();
