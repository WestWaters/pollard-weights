/* Pollard Studio — front end.

   Two rules this file follows:
     Every control maps to a flag that exists (see actions.py, which is generated against the
     tools' own argparse definitions).
     Every number is measured or says it is not. There is no fallback to a plausible-looking
     figure -- in a quality panel that is worse than a blank.

   Charts are hand-rolled SVG: no chart library, no CDN, nothing fetched at runtime. */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const ic = n => `<svg class="ic"><use href="#i-${n}"/></svg>`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const gb = b => (b / 1e9).toFixed(2);
/* Absolute paths carry the user's home directory, which has their name in it. Anything on screen
   -- and anything that ends up in a screenshot they paste somewhere -- gets it stripped. The real
   path still goes to the tool; this is display only. */
function short(p, keep = 2) {
  if (!p) return "";
  let t = String(p);
  if (S.home && t.startsWith(S.home)) t = "$POLLARD_HOME" + t.slice(S.home.length);
  else t = t.replace(/^\/(?:Users|home)\/[^/]+/, "~").replace(/^[A-Z]:\\Users\\[^\\]+/i, "~");
  const parts = t.split(/[\\/]/).filter(Boolean);
  return parts.length > keep + 1 ? (t[0] === "~" || t[0] === "$" ? parts[0] + "/…/" : "…/")
    + parts.slice(-keep).join("/") : t;
}
const shortArgs = a => a.map(x => (/[\\/]/.test(String(x)) ? short(x) : x)).join(" ");
const NM = '<span class="nm-none">not measured</span>';

/* ── state, filled from the bridge ──────────────────────────────────────── */
let S = { home: "", home_exists: false, models: [], model: null, tools: [],
          repo: "", repo_exists: false, status: {}, version: "?", downloads: [], hw: {} };

/* the recipe the controls are steering — keys match actions.py */
let R = { source: "", gguf: "", imatrix: "", body: "iq1_s", protect: "iq1_kt", emb: "Q8_0",
          ram: 16, reserve: 3, tier: "", ngl: 0, chunks: 120, lane: "GGUF", protectVision: true,
          planOnly: true, allow1bit: false, allowGrow: false, mixOnly: false,
          noImatrix: false, noGate: false, allowDense: false, speed: true, coherence: false };

/* atom bit-rates. ik_llama trellis atoms are marked so the runtime line can warn. */
const ATOM = { iq1_s: 1.56, iq1_m: 1.75, iq1_kt: 1.75, iq2_xxs: 2.06, iq2_kt: 2.125,
               iq2_s: 2.50, iq3_s: 3.44, iq4_xs: 4.25, q6_K: 6.56 };
const EMB  = { IQ3_S: 3.44, Q4_K: 4.50, Q6_K: 6.56, Q8_0: 8.50 };
const TRELLIS = ["iq1_kt", "iq2_kt", "iq3_kt", "iq4_kt"];
const BODY_ATOMS = ["iq1_s", "iq1_m", "iq1_kt", "iq2_xxs", "iq2_kt", "iq3_s"];
const PROT_ATOMS = ["iq1_kt", "iq2_kt", "iq2_s", "iq3_s", "iq4_xs", "q6_K"];
const EMB_ATOMS  = ["IQ3_S", "Q4_K", "Q6_K", "Q8_0"];
const LANES = ["GGUF", "GPTQ", "MLX", "EXL3", "MX"];

/* Every lane has its own alphabet. iq1_s and q6_K are llama.cpp container types and mean nothing
   to MLX or EXL3 -- showing them on those lanes offers a control the run will ignore. Each entry
   below names the lane's OWN knobs, and every flag here is declared by that lane's tool. */
const LANE_ALLOC = {
  GGUF: { tool: "pollard-automap", fields: [
    ["Body atom", "--body", "sel", BODY_ATOMS, "body", "pollard_automap"],
    ["Protect atom", "--protect", "sel", PROT_ATOMS, "protect", "pollard_automap"],
    ["Embeddings", "--token-embedding-type", "sel", EMB_ATOMS, "emb", "pollard_smoke"]] },
  GPTQ: { tool: "pollard-gptq", fields: [
    ["Bits", "--bits", "num", [2, 8, 4], "bits", "pollard_gptq"],
    ["Group size", "--groupsize", "sel", ["32", "64", "128", "-1"], "groupsize", "pollard_gptq"],
    ["Alphabet", "--qmode", "sel", ["int", "ternary", "binary"], "qmode", "pollard_gptq"]] },
  MLX: { tool: "pollard-mlx", fields: [
    ["Group size", "--group-size", "sel", ["32", "64", "128"], "groupSize", "pollard_mlx"],
    ["Hot fraction", "--hot-frac", "num", [0, 100, 35], "hotFrac", "pollard_mlx"],
    ["Focus layers", "--focus-layers", "text", "e.g. 0,1,30,31", "focusLayers", "pollard_mlx"]] },
  EXL3: { tool: "pollard-exl3", fields: [
    ["Bits per weight", "--bpw", "num", [1, 8, 4], "bpw", "pollard_exl3"],
    ["Head bits", "--head-bits", "num", [0, 16, 0], "headBits", "pollard_exl3"],
    ["CUDA devices", "--devices", "text", "device ids, e.g. 0,1", "devices", "pollard_exl3"]] },
  MX: { tool: "pollard-mx", fields: [
    ["Body scheme", "--scheme", "sel", ["NVFP4", "MXFP4", "W4A16", "W8A16"], "scheme", "pollard_mx"],
    ["Protect scheme", "--protect-scheme", "sel", ["FP8", "FP8_DYNAMIC", "W8A16"], "protectScheme", "pollard_mx"],
    ["Hot fraction", "--hot-frac", "num", [0, 100, 25], "hotFrac", "pollard_mx"]] },
};

/* Render the allocation controls for whichever lane the knob is on. */
function laneAlloc() {
  const want = R.lane || "GGUF";
  const spec = LANE_ALLOC[want] || LANE_ALLOC.GGUF;
  return spec.fields.map(([label, flag, kind, arg, key, owner], i) => {
    const id = `la-${i}`;
    // the hint names the flag AND the tool, because on the GGUF lane these three knobs belong
    // to three different tools -- body/protect to automap, embeddings to smoke
    flag = `${flag} <b>${(owner || "").replace("pollard_", "pollard-")}</b>`;
    if (kind === "sel")
      return field(`${label} <span class='hintr'>${flag}</span>`,
                   sel(id, arg, R[key] != null ? String(R[key]) : undefined));
    if (kind === "num")
      return field(`${label} <span class='hintr'>${flag}</span>`,
                   srow(id, arg[0], arg[1], R[key] != null ? R[key] : arg[2], 1, true));
    return field(`${label} <span class='hintr'>${flag}</span>`,
                 `<input type="text" id="${id}" placeholder="${escAttr(arg)}"
                    value="${escAttr(R[key] || "")}"
                    oninput="R['${key}']=this.value.trim()||undefined;refresh()">`);
  }).join("");
}

function wireLaneAlloc() {
  const spec = LANE_ALLOC[R.lane || "GGUF"] || LANE_ALLOC.GGUF;
  const el = $("#la-tool");
  if (el) el.textContent = spec.tool;
  spec.fields.forEach(([, , kind, , key], i) => {
    if (kind === "sel") bindSel(`la-${i}`, key);
    else if (kind === "num") bindSlider(`la-${i}`, key);
  });
}

const SCREENS_ORDER = [
  ["build", "BUILD", "PIPELINE"], ["monitor", "MONITOR", ""], ["ladder", "LADDER", ""],
  ["convert", "CONVERT", "TOOLING"],
  ["tools", "TOOLS", ""],
  ["advanced", "ADVANCED", ""], ["train", "TRAIN", ""], ["chat", "CHAT", ""],
  ["bench", "BENCH", "CHECKS"], ["eval", "EVAL", ""], ["doctor", "DOCTOR", ""],
  ["brains", "BRAINS", "OUTPUT"], ["publish", "PUBLISH", ""],
];

/* ── model helpers ──────────────────────────────────────────────────────── */
const builds = () => (S.model && S.model.builds) || [];
/* the reference build: biggest one, i.e. the least-quantized thing present */
const refBuild = () => builds().reduce((a, b) => (!a || b.bytes > a.bytes ? b : a), null);
const curBuild = () => builds().find(b => b.path === R.gguf) || refBuild();
/* The build the SOURCE dropdown is pointed at. The projection used refBuild() unconditionally,
   so picking a different rung as the source changed the command and nothing else -- every
   readout kept describing the largest build on disk. */
const srcBuild = () => builds().find(b => b.path === R.source) || refBuild();

/* Projection from REAL per-group parameter counts read out of the reference GGUF.
   Same arithmetic pollard-fit does; the inputs are measured, not assumed. Returns null when
   there is nothing measured to base it on, and the UI then says so. */
/* What a lane's knobs mean in bits per weight. These are not estimates: an NVFP4 element IS
   4 bits, an EXL3 build IS its --bpw target, and a GGUF atom's rate is the container's own.
   A lane whose knobs are not set yet reports its tool's documented default. */
function laneBpw() {
  const n = (v, d) => (v == null || v === "" || isNaN(+v) ? d : +v);
  switch (R.lane || "GGUF") {
    case "GPTQ": {                       // uniform width, plus the group scale/zero overhead
      const b = n(R.bits, 4), gs = n(R.groupsize, 128);
      const over = gs > 0 ? 16 / gs : 0;
      return { body: b + over, protect: b + over, emb: b + over };
    }
    case "EXL3":
      return { body: n(R.bpw, 4), protect: n(R.bpw, 4),
               emb: n(R.headBits, 0) || n(R.bpw, 4) };
    case "MLX": {                        // --hot-frac of the layers stay 8-bit, the rest go 4
      const hot = n(R.hotFrac, 35) / 100, gs = n(R.groupSize, 64);
      const over = gs > 0 ? 16 / gs : 0;
      return { body: 8 * hot + 4 * (1 - hot) + over, protect: 8 + over, emb: 8 + over };
    }
    case "MX": {
      const wid = { NVFP4: 4, MXFP4: 4, W4A16: 4, W8A16: 8 }[R.scheme || "NVFP4"] ?? 4;
      const pro = { FP8: 8, FP8_DYNAMIC: 8, W8A16: 8 }[R.protectScheme || "FP8"] ?? 8;
      const hot = n(R.hotFrac, 25) / 100;
      return { body: pro * hot + wid * (1 - hot), protect: pro, emb: pro };
    }
    default:
      return { body: ATOM[R.body] ?? 2, protect: ATOM[R.protect] ?? 4, emb: EMB[R.emb] ?? 8 };
  }
}

function project() {
  const ref = srcBuild();
  if (!ref || !ref.groups) return null;
  // vision, projector and audio are held high on purpose. They carry modality alignment, and a
  // model that cannot tell red from blue has lost more than its perplexity shows. GGUF sidesteps
  // this by shipping the projector unquantized as a separate mmproj; every other lane has the
  // tower inside the checkpoint, where the allocator would otherwise treat it as ordinary layers.
  const w = laneBpw();      // body / protect / embeddings, in THIS lane's own units
  const bpwFor = g => ({ embeddings: w.emb, mixing: w.protect, ffn: w.body,
                         vision: R.protectVision ? 8.5 : w.protect,
                         projector: 16, audio: R.protectVision ? 8.5 : w.protect,
                         mtp: 2.63, other: 32 })[g] ?? 32;
  let bytes = 0, params = 0;
  const parts = [];
  const colour = { embeddings: "#c9a961", ffn: "#6f9cbb", mixing: "#8e8a80",
                   vision: "#7d9e6b", projector: "#4a7546", audio: "#9e8a6b",
                   mtp: "#a9803f", other: "#b0aaa0" };
  for (const [g, v] of Object.entries(ref.groups)) {
    const b = v.params * bpwFor(g) / 8;
    bytes += b; params += v.params;
    parts.push([g, b, colour[g] || "#b0aaa0", v.params]);
  }
  parts.sort((a, b) => b[1] - a[1]);
  return {
    bytes, params, gb: bytes / 1e9, bpw: bytes * 8 / params, parts,
    ref, fits: bytes / 1e9 + (+R.reserve || 0) <= (+R.ram || 16),
    runtime: (TRELLIS.includes(R.body) || TRELLIS.includes(R.protect)) ? "ik_llama" : "stock",
  };
}

/* ── SVG charts ─────────────────────────────────────────────────────────── */
function spark(data, colour = "#6f9cbb") {
  if (!data || data.length < 2) return `<span class="spark-none">—</span>`;
  const lo = Math.min(...data), hi = Math.max(...data), sp = (hi - lo) || 1;
  const d = data.map((v, i) => `${i ? "L" : "M"}${(3 + i / (data.length - 1) * 90).toFixed(1)} ${
    (23 - (v - lo) / sp * 20).toFixed(1)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 96 26" preserveAspectRatio="none"><path d="${d}"
      fill="none" stroke="${colour}" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
}

function linechart(series, { w = 560, h = 210, ylab = "", xlab = "", xticks = null } = {}) {
  const all = series.flatMap(s => s.data);
  if (!all.length) return empty("nothing measured yet");
  const lo = Math.min(...all), hi = Math.max(...all), sp = (hi - lo) || 1;
  const L = 46, B = 24, pad = 10;
  const n = Math.max(...series.map(s => s.data.length));
  const px = i => L + (i / (n - 1 || 1)) * (w - L - pad);
  const py = v => h - B - ((v - lo) / sp) * (h - B - pad);
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = lo + sp * k / 4, y = py(v);
    g += `<line x1="${L}" y1="${y.toFixed(1)}" x2="${w - pad}" y2="${y.toFixed(1)}"
            stroke="#5d7a4e" stroke-width=".5" opacity=".45"/>
          <text x="${L - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="8.5"
            font-family="ui-monospace,monospace" fill="#7fa86a">${
              Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) < 1 ? 3 : 2)}</text>`;
  }
  if (xticks) g += xticks.map((t, i) => `<text x="${px(i * (n - 1) / (xticks.length - 1)).toFixed(1)}"
      y="${h - 10}" text-anchor="middle" font-size="8" font-family="ui-monospace,monospace"
      fill="#7fa86a">${t}</text>`).join("");
  const lines = series.map(s => `<path d="${s.data.map((v, i) =>
      `${i ? "L" : "M"}${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(" ")}" fill="none"
      stroke="${s.colour}" stroke-width="${s.w || 1.8}" stroke-linejoin="round"
      stroke-linecap="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`).join("");
  const xl = xlab ? `<text x="${(L + w) / 2}" y="${h - 3}" text-anchor="middle" font-size="8.5"
       font-family="ui-monospace,monospace" fill="#7fa86a">${xlab}</text>` : "";
  const yl = ylab ? `<text x="12" y="${h / 2}" text-anchor="middle" transform="rotate(-90 12 ${h / 2})"
       font-size="8.5" font-family="ui-monospace,monospace" fill="#7fa86a">${ylab}</text>` : "";
  return `<svg class="chart" viewBox="0 0 ${w} ${h}">${g}${lines}${xl}${yl}</svg>`;
}

function barchart(rows, { h = 210, unit = "", ylab = "" } = {}) {
  if (!rows.length) return empty("nothing measured yet");
  const w = 560, L = 46, B = 30, pad = 10;
  const hi = Math.max(...rows.map(r => r.a)) * 1.12 || 1;
  const bw = (w - L - pad) / rows.length;
  const y = v => h - B - (v / hi) * (h - B - pad);
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = hi * k / 4, yy = y(v);
    g += `<line x1="${L}" y1="${yy.toFixed(1)}" x2="${w - pad}" y2="${yy.toFixed(1)}"
            stroke="#5d7a4e" stroke-width=".5" opacity=".45"/>
          <text x="${L - 6}" y="${(yy + 3).toFixed(1)}" text-anchor="end" font-size="8.5"
            font-family="ui-monospace,monospace" fill="#7fa86a">${v.toFixed(v < 10 ? 1 : 0)}</text>`;
  }
  const bars = rows.map((r, i) => {
    const x = L + i * bw, iw = bw * 0.55, off = (bw - iw) / 2;
    return `<rect x="${(x + off).toFixed(1)}" y="${y(r.a).toFixed(1)}" width="${iw.toFixed(1)}"
        height="${Math.max(0, h - B - y(r.a)).toFixed(1)}" fill="#a9d18d" rx="1.5"/>
      <text x="${(x + bw / 2).toFixed(1)}" y="${h - 16}" text-anchor="middle" font-size="8"
        font-family="ui-monospace,monospace" fill="#a9d18d">${r.name}</text>
      <text x="${(x + bw / 2).toFixed(1)}" y="${h - 5}" text-anchor="middle" font-size="7.5"
        font-family="ui-monospace,monospace" fill="#7fa86a">${r.a.toFixed(2)}${unit}</text>`;
  }).join("");
  const yl = ylab ? `<text x="12" y="${h / 2}" text-anchor="middle" transform="rotate(-90 12 ${h / 2})"
       font-size="8.5" font-family="ui-monospace,monospace" fill="#7fa86a">${ylab}</text>` : "";
  return `<svg class="chart" viewBox="0 0 ${w} ${h}">${g}${bars}${yl}</svg>`;
}

const empty = msg => `<div class="chart-empty">${ic("info")}<span>${msg}</span></div>`;

function gauge(frac, label, value) {
  const a = -52 + clamp(frac, 0, 1) * 104, r = 30, cx = 46, cy = 40;
  const x = cx + r * Math.sin(a * Math.PI / 180), y = cy - r * Math.cos(a * Math.PI / 180);
  let t = "";
  for (let k = 0; k <= 8; k++) {
    const d = -52 + k / 8 * 104, s = Math.sin(d * Math.PI / 180), c = Math.cos(d * Math.PI / 180);
    t += `<line x1="${cx + 30 * s}" y1="${cy - 30 * c}" x2="${cx + (k % 2 ? 26 : 24) * s}"
      y2="${cy - (k % 2 ? 26 : 24) * c}" stroke="#7a736a" stroke-width="${k % 2 ? .7 : 1.1}"/>`;
  }
  return `<div class="gauge"><div class="gface"><svg viewBox="0 0 92 48">${t}
      <path d="M22 40a24 24 0 0148 0" fill="none" stroke="#b9b2a6" stroke-width=".7"/>
      <line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#3c3b37"
        stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="${cx}" cy="${cy}" r="4.5" fill="#4a463f"/>
      <circle cx="${cx}" cy="${cy}" r="2" fill="#7d7669"/></svg></div>
    <div class="gl">${label}</div><div class="gv">${value}</div></div>`;
}

const ruler = (name, value, frac, marks) => `<div class="ruler">
  <div class="rh"><span class="rn">${name}</span><span class="rv">${value}</span></div>
  <div class="rtrack"><div class="rneedle" style="left:${(clamp(frac, 0, 1) * 100).toFixed(1)}%"></div></div>
  <div class="rmarks">${marks.map(m => `<span>${m}</span>`).join("")}</div></div>`;

/* ── controls ───────────────────────────────────────────────────────────── */
const chip = s => {
  const c = { VERIFIED: "ok", verified: "ok", PASSED: "ok", COMPLETED: "ok", READY: "ok",
              BUILDING: "run", RUNNING: "run", IDLE: "dim", QUEUED: "warn",
              UNVERIFIED: "warn", FAILED: "bad" }[s] || "dim";
  return `<span class="chip ${c}">${s}</span>`;
};
const field = (l, inner) => `<div class="field"><label>${l}</label>${inner}</div>`;
const sel = (id, opts, cur) => `<div class="selwrap"><select id="${id}">${opts.map(o => {
  const [v, t] = Array.isArray(o) ? o : [o, o];
  return `<option value="${escAttr(v)}" ${v === cur ? "selected" : ""}>${esc(String(t))}</option>`;
}).join("")}</select>${ic("chev")}</div>`;
const srow = (id, min, max, val, step = 1, blue) => `<div class="srow">
  <input type="range" class="flat ${blue ? "blue" : ""}" id="${id}" min="${min}" max="${max}"
     step="${step}" value="${val}" style="--p:${((val - min) / (max - min)) * 100}">
  <input type="number" id="${id}-n" min="${min}" max="${max}" step="${step}" value="${val}"></div>`;
const tog = (id, t, sub, on) => `<div class="tog"><div class="t">${t}<small>${sub}</small></div>
  <div class="tswitch${on ? " on" : ""}" id="${id}" role="switch" tabindex="0"></div></div>`;
/* Point Pollard at YOUR file. Typing a path works, but a path you typed is the single most
   likely thing in a run to be wrong, so BROWSE opens the real OS dialog. `key` is the recipe
   field the value lands in -- without it the box is decoration, which is what these were. */
const pathrow = (id, key, kind, placeholder) => `<div class="pathwrap">
  <div class="pathrow">
    <input type="text" id="${id}" placeholder="${escAttr(placeholder)}" value="${escAttr(R[key] || "")}"
       oninput="R['${key}']=this.value.trim()||undefined;refresh()"
       onkeydown="if(event.key==='Enter')findInto('${id}','${key}','${kind}')">
    <button class="btn sm" title="search this machine"
       onclick="findInto('${id}','${key}','${kind}')">${ic("search")}FIND</button>
    <button class="btn sm" title="open the file dialog"
       onclick="browseInto('${id}','${key}','${kind}')">${ic("file")}BROWSE</button>
    <button class="btn sm" title="clear"
       onclick="R['${key}']=undefined;$('#${id}').value='';closeFind('${id}');refresh()">CLEAR</button>
  </div>
  <div class="findpanel" id="${id}-find" hidden></div></div>`;
/* does the reference build carry a vision, audio or projector tower inside it? */
/* does the reference build route through experts? drives which gates are offered */
const isMoE = () => {
  const b = refBuild() || {};
  if ((b.types && Object.keys(b.types).some(t => /expert/i.test(t)))) return true;
  const g = (b.groups || {}).ffn || {};
  return /moe|expert/i.test(b.architecture || "") || (b.blocks && g.tensors > b.blocks * 6);
};
const multimodal = () => {
  const g = (refBuild() || {}).groups || {};
  return ["vision", "projector", "audio"].some(k => (g[k] || {}).params > 0);
};
/* Defaults and ranges come from the MACHINE, not from a number I picked. 16 GB is
   meaningful on a 16 GB laptop and meaningless everywhere else, and a slider that stops
   at 128 is wrong on a workstation. */
/* The live pool survey, or null before the first read. */
let poolCache = null;

/* The ceiling is the pool when boxes are linked -- a fader that stops at this machine's RAM
   cannot express the build the cluster exists to make possible. */
const poolGb = () => (poolCache && poolCache.combined_gb) || (S.hw || {}).ram_gb || 64;
const ramMax = () => Math.max(16, Math.ceil(poolGb() * 1.5 / 8) * 8);
const hwNote = () => {
  const h = S.hw || {};
  return h.ram_gb ? `this machine has ${h.ram_gb} GB` : "";
};
let remoteModels = [];

/* Builds from every box, in one list. A remote entry carries the host that holds it: a path only
   means something on its own machine, and a list that hides which box a file is on invites a run
   that cannot find it. */
const buildOpts = () => [
  ...builds().map(b =>
    [b.path, `${b.lane && b.lane !== "?" ? b.lane + " " : ""}${b.tag} · ${gb(b.bytes)} GB`]),
  // a connectome is a graph to train on, not a build to run -- it belongs in the picker, not
  // in a list of things you can point --gguf at
  ...remoteModels.filter(m => m.lane !== "CONNECTOME").map(m =>
    [m.path, `${m.host} · ${m.lane} ${m.name} · ${gb(m.bytes)} GB`]),
];

/* Is this path on another box? Then a local tool cannot open it, and the UI has to say so
   rather than build a command that fails at read time. */
const remoteOf = p => remoteModels.find(m => m.path === p) || null;
const kindOf = p => (builds().find(b => b.path === p) || {}).kind || "gguf";

/* ── screens ────────────────────────────────────────────────────────────── */
const SCREENS = {};

SCREENS.build = () => {
  if (!S.model) return noWorkspace();
  return `

  <div class="grid g-main">
    <div class="card"><h3>${ic("sliders")}ALLOCATION<span class="rt" id="la-tool"></span></h3>
      ${field("Source <span class='hintr'>--gguf / --model</span>",
              sel("s-src", buildOpts(), R.source || (refBuild() || {}).path))}
      <div class="hint" style="margin:2px 0 10px">One measured allocation, emitted into whichever
        lane you need. The profile below is what Pollard measured on this model — every lane reads
        the same one, so moving lanes never re-solves it.</div>
      ${field("Allocation profile <span class='hintr'>--sensitivity</span>",
              pathrow("s-sens", "sensitivity", "data", "sensitivity.json from probe or sensitivity"))}
      ${(R.lane || "GGUF") === "EXL3" ? field(
          "Per-tensor recipe <span class='hintr'>--recipe, EXL3 reads YAML</span>",
          pathrow("s-recf", "recipeFile", "data", "per-tensor bitrate YAML")) : ""}
      <div class="sep"></div>
      <div id="la-box">${laneAlloc()}</div>
      ${field("imatrix <span class='hintr'>--imatrix</span>",
              pathrow("s-imx", "imatrix", "imatrix", "path to .imatrix / .dat"))}
      ${field("Tier <span class='hintr'>--tier</span>", sel("s-tier", ["", "small", "balanced", "quality"], R.tier))}
      <div class="sep"></div>
      ${tog("t-plan", "Plan only", "--plan-only: solve and report, emit nothing", R.planOnly)}
      ${tog("t-1bit", "Allow 1-bit atoms", "--allow-1bit", R.allow1bit)}
      ${tog("t-grow", "Allow grow", "--allow-grow: may exceed the budget", R.allowGrow)}
      ${tog("t-noimx", "No imatrix", "--no-imatrix: calibration-free", R.noImatrix)}
      ${tog("t-nogate", "Skip gate", "--no-gate: do not coherence-check the result", R.noGate)}
      ${multimodal() ? tog("t-vis", "Protect vision + projector",
          "hold the modality path high — GGUF ships mmproj unquantized anyway", R.protectVision) : ""}
    </div>

    <div class="stackv">
      <div class="card"><h3>${ic("box")}ALLOCATION SUMMARY<span class="rt" id="b-fit"></span></h3>
        <div class="lcd" id="b-lcd">—</div>
        <div class="sp"></div>
        <div class="mstrip" id="b-metrics"></div>
      </div>

      <div class="card"><h3>${ic("cpu")}LANE · MEMORY TARGET</h3>
        <div class="knobbay">
          <div class="knobwrap">
            <svg class="ring" viewBox="-24 -8 164 132" id="lane-ring"></svg>
            <div class="knob" id="k-lane" tabindex="0" role="slider"></div>
            <div class="knoblab">LANE</div></div>
          <div class="faders" id="f-bank">
            <div class="fader"><div class="fslot"><div class="rail"></div>
              <div class="ticks">${"<i></i>".repeat(9)}</div>
              <input type="range" id="f-ram" min="2" max="${ramMax()}" step="1" value="${R.ram}"></div>
              <div class="fv" id="f-ram-v">${R.ram} GB</div><div class="fl">RAM</div></div>
            <div class="fader"><div class="fslot"><div class="rail"></div>
              <div class="ticks">${"<i></i>".repeat(9)}</div>
              <input type="range" id="f-res" min="0" max="${Math.max(4, Math.round(ramMax() / 4))}" step="0.5" value="${R.reserve}"></div>
              <div class="fv" id="f-res-v">${R.reserve} GB</div><div class="fl">RESERVE</div></div>
          </div>
          <div class="clusterbay" id="cl-pool"><div class="hint">reading…</div></div>
        </div>
        <div id="dev-place"></div>
        <div class="sep"></div>
        <div class="keys">${[["CALC", "doctor"], ["FIT", "fit"], ["SENS", "fragile"],
            ["SMOKE", "smoke"], ["VRFY", "verify"], ["EVAL", "eval"], ["LS", "ls"]].map(
          ([k, a]) => `<button class="key" onclick="act('${a}')">${k}</button>`).join("")}</div>
      </div>

      <div class="card"><h3>${ic("layers")}ACROSS MACHINES
          <span class="rt" id="cl-note">one box</span></h3>
        <div class="hint">The faders above are one machine. Pollard is not capped on model size —
          the hardware is. Pool boxes over RPC and the budget becomes the whole pool, so a model
          no single node could hold still gets built and served.</div>
        <div class="sp"></div>
        ${field("RPC pool <span class='hintr'>--rpc, ggml-rpc-server on each node</span>",
                `<input type="text" id="cl-rpc" placeholder="host:port,host:port"
                   value="${escAttr(R.rpc || "")}"
                   oninput="R.rpc=this.value.trim()||undefined;refresh()"
                   onkeydown="if(event.key==='Enter')renderPool()">`)}
        <div class="btns" style="margin-bottom:10px">
          <button class="btn sm" onclick="renderPool()">${ic("activity")}RESCAN POOL</button>
          <button class="btn sm" onclick="discoverPeers()">${ic("search")}FIND ON LAN</button>
          <button class="btn sm" id="rpc-btn" onclick="toggleServing()">${ic("play")}START SERVING</button>
        </div>
        <div id="rpc-state"></div>
        <div id="cl-models"></div>
        <div class="grid g-2" style="margin:0">
          ${field("Tensor parallel <span class='hintr'>--tp</span>",
                  sel("cl-tp", [["", "off"], "2", "4", "8", "16", "32", "64"], R.tp || ""))}
          ${field("VRAM per node GB <span class='hintr'>--vram</span>",
                  `<input type="text" id="cl-vram" placeholder="leave headroom for KV"
                     value="${escAttr(R.vram || "")}"
                     oninput="R.vram=this.value.trim()||undefined;refresh()">`)}
        </div>
        ${field("CUDA devices <span class='hintr'>--devices, EXL3 lane</span>",
                `<input type="text" id="cl-dev" placeholder="device ids, e.g. 0,1"
                   value="${escAttr(R.devices || "")}"
                   oninput="R.devices=this.value.trim()||undefined;refresh()">`)}
        <div class="sp"></div>
        <div class="btns">
          <button class="btn" onclick="act('placement')">${ic("play")}PLACEMENT</button>
          <button class="btn" onclick="act('vllmcheck')">TP CHECK</button>
          <button class="btn" onclick="previewOnly('placement')">PREVIEW</button>
        </div>
        <div class="hint" style="margin-top:8px">PLACEMENT measures where each expert should
          actually live across the pool. TP CHECK asks whether the build serves under vLLM at the
          chosen degree — it exits non-zero if that degree will not load.</div>
      </div>

      <div class="card"><h3>${ic("file")}THE COMMAND THIS IS<span class="rt" id="b-tool"></span></h3>
        <div class="cmd" id="b-cmd">…</div>
        <div class="sp"></div>
        <div class="btns">
          <button class="btn gold" onclick="act('build')">${ic("play")}RUN</button>
          <button class="btn" onclick="act('automap')">AUTOMAP</button>
          <button class="btn" onclick="copyCmd()">COPY</button>
          <button class="btn" onclick="saveArgs()">${ic("save")}SAVE ARGS</button>
          <button class="btn bad" onclick="abortRun()">ABORT</button>
        </div>
      </div>
    </div>

    <div class="stackv">
      <div class="card"><h3>${ic("activity")}MEASURED · REFERENCE BUILD</h3><div id="b-ref"></div></div>
      <div class="card"><h3>${ic("scale")}PROJECTED vs REFERENCE</h3><div id="b-cmp"></div></div>
      <div class="card"><h3>${ic("alert")}RECIPE ON DISK</h3><div id="b-recipe"></div></div>
    </div>
  </div>`;
};

const noWorkspace = () => `<div class="card"><h3>${ic("alert")}NO WORKSPACE</h3>
  <div class="hint">Nothing found under <b>${short(S.home, 1) || "$POLLARD_HOME"}</b>. Point
    <b>POLLARD_HOME</b> at your workspace, or build something with Pollard first —
    Studio only ever shows what is on disk.</div>
  <div class="sp"></div><div class="btns">
    <button class="btn gold" onclick="rescan()">RESCAN</button></div></div>`;

SCREENS.monitor = () => `
  <div class="grid g-4" id="mon-lanes"></div>
  <div class="grid g-mon">
    <div class="card"><h3>${ic("activity")}SIZE BY RUNG<span class="rt">measured</span></h3>
      <div class="chartbox" id="mon-chart"></div>
      <div class="legend"><span><i style="background:#a9d18d"></i>file size on disk</span></div>
    </div>
    <div class="card"><h3>${ic("file")}LIVE OUTPUT<span class="rt" id="mon-state"></span></h3>
      <div class="lcd" style="padding:0"><div class="log" id="mon-log"
        style="padding:11px 13px;max-height:330px"></div></div>
      <div class="sp"></div>
      <div class="btns"><button class="btn bad" onclick="abortRun()">${ic("stop")}ABORT</button>
        <button class="btn" onclick="clearLog()">CLEAR</button></div>
    </div>
    <div class="card"><h3>${ic("folder")}WORKSPACE<span class="rt" id="mon-scan"></span></h3>
      <div style="max-height:330px;overflow-y:auto"><table><thead><tr>
        <th>BUILD</th><th>SIZE</th><th>BPW</th><th>MODIFIED</th></tr></thead>
        <tbody id="mon-mf"></tbody></table></div>
      <div class="sp"></div>
      <div class="btns"><button class="btn" onclick="rescan()">RESCAN</button></div>
    </div>
  </div>`;

SCREENS.ladder = () => `
  <div class="grid g-4" id="lad-cards"></div>
  <div class="grid g-2">
    <div class="card"><h3>${ic("activity")}BITS PER WEIGHT<span class="rt">measured from each file</span></h3>
      <div class="chartbox" id="lad-chart"></div></div>
    <div class="card"><h3>${ic("layers")}ALL BUILDS</h3>
      <table><thead><tr><th>BUILD</th><th>LANE</th><th>ARCH</th><th>SIZE</th><th>BPW</th>
        <th>PPL</th><th>MODIFIED</th></tr></thead><tbody id="lad-rows"></tbody></table></div>
  </div>`;

SCREENS.convert = () => `

  <div class="grid" style="grid-template-columns:320px 1fr">
    <div class="card"><h3>${ic("box")}SOURCE</h3>
      ${field("Build to move <span class='hintr'>from</span>", sel("cv-from", buildOpts(), R.gguf))}
      <div id="cv-quality"></div>
      <div class="sep"></div>
      <h3 style="margin-bottom:11px">${ic("sliders")}ALLOCATION</h3>
      ${field("Profile <span class='hintr'>the portable artifact</span>",
              sel("cv-profiler", [["pollard_probe", "probe — cheap, any box"],
                                  ["pollard_sensitivity", "sensitivity — measured KL"],
                                  ["pollard_palette", "palette — below 2 bits"]], "pollard_probe"))}
      ${tog("cv-reuse", "Reuse an existing profile",
            "if sensitivity.json is already next to the build", true)}
      ${tog("cv-planonly", "Plan only", "show the route, run nothing", true)}
      <div class="sep"></div>
      <div class="hint">A source does not have to be f16. Q8-class is near-lossless and emitting
        from it is what most people should do rather than re-downloading the original.</div>
    </div>

    <div class="stackv">
      <div class="card"><h3>${ic("layers")}TARGET LANE</h3>
        <div class="lanepick" id="cv-lanes"></div>
      </div>
      <div class="card"><h3>${ic("gauge")}WILL IT RUN<span class="rt" id="cv-fit"></span></h3>
        <div class="grid g-2" style="margin:0 0 4px">
          ${field(`Target RAM / VRAM (GB) <span class='hintr'>${hwNote()}</span>`,
                  srow("cv-ram", 2, ramMax(), R.ram, 1))}
          ${field("Reserve for the OS (GB)",
                  srow("cv-res", 0, Math.max(4, Math.round(ramMax() / 4)), R.reserve, 0.5, true))}
        </div>
        <div id="cv-runs"></div>
      </div>

      <div class="card"><h3>${ic("chev")}THE ROUTE<span class="rt" id="cv-count"></span></h3>
        <div id="cv-route"><div class="hint">Pick a target lane above.</div></div>
        <div class="sep"></div>
        <div class="btns">
          <button class="btn gold" onclick="runRoute()">${ic("play")}RUN THE ROUTE</button>
          <button class="btn" onclick="convertAllLanes()">${ic("layers")}PLAN EVERY LANE</button>
          <button class="btn bad" onclick="abortRun()">ABORT</button>
        </div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px"><h3>${ic("file")}OUTPUT<span class="rt" id="cv-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="cv-log"
      style="padding:11px 13px;max-height:220px"></div></div></div>`;

/* One renderer for a tool's flags, used by both the Tools screen and Advanced.

   Advanced used to hand-write its controls, and an audit found 122 knobs across 21 tools that
   never made it onto the screen -- because a hand-written list of 562 flags is a list that is
   wrong the day someone adds one. This renders whatever the tool declares, so a section cannot
   be missing a knob and cannot drift from the source. */
function toolOf(mod) {
  return S.tools.find(t => t.module === mod);
}

function flagForm(mod, prefix) {
  const t = toolOf(mod);
  if (!t) return `<div class="hint">${esc(mod)} is not in this Pollard build.</div>`;
  const seen = new Set();
  const flags = t.flags.filter(f => {
    const k = f.names.join();
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return flags.map((f, i) => {
    const id = `${prefix}-${i}`, flag = f.names[0];
    let ctl;
    if (f.is_flag) {
      ctl = `<div class="tswitch${f.default ? " on" : ""}" id="${id}" data-flag="${flag}"
               data-kind="switch" onclick="this.classList.toggle('on');advCheck('${prefix}','${mod}')"></div>`;
    } else if (f.choices) {
      ctl = `<div class="selwrap"><select id="${id}" data-flag="${flag}" data-kind="value"
               onchange="advCheck('${prefix}','${mod}')"><option value=""></option>${
        f.choices.map(c => `<option ${c === f.default ? "selected" : ""}>${esc(String(c))}</option>`).join("")
      }</select>${ic("chev")}</div>`;
    } else {
      ctl = `<input type="text" id="${id}" data-flag="${flag}" data-kind="value"
               value="${escAttr(f.default == null ? "" : f.default)}"
               placeholder="${esc(f.type || "value")}" oninput="advCheck('${prefix}','${mod}')">`;
    }
    return `<div class="flag"><div><div class="fn">${esc(f.names.join(", "))}</div>
        ${f.required ? '<div class="rq">REQUIRED</div>' : ""}</div>
        <div><div class="fh">${esc(f.help || "")}</div>${ctl}</div></div>`;
  }).join("");
}

function formValues(prefix) {
  const out = {};
  $$(`[id^="${prefix}-"][data-flag]`).forEach(el => {
    const flag = el.dataset.flag;
    if (el.dataset.kind === "switch") { if (el.classList.contains("on")) out[flag] = true; }
    else if (el.value !== "") out[flag] = el.value;
  });
  return out;
}

let advPending = null;
function advCheck(prefix, mod) {
  clearTimeout(advPending);
  advPending = setTimeout(async () => {
    if (!bridge()) return;
    const v = await bridge().check_tool(mod, formValues(prefix));
    const cmd = $(`#${prefix}-cmd`), prob = $(`#${prefix}-prob`), btn = $(`#${prefix}-run`);
    if (cmd) cmd.innerHTML = esc(v.display.split(" ").map(x => (/[\\/]/.test(x) ? short(x) : x)).join(" "))
      .replace(/(--[a-z0-9-]+)/g, '<span class="fl">$1</span>');
    if (prob) prob.innerHTML = v.ok ? "" :
      `<div class="problems">${v.problems.map(x => `<div>${ic("alert")}${esc(x)}</div>`).join("")}</div>`;
    if (btn) { btn.disabled = !v.ok; btn.classList.toggle("off", !v.ok); }
  }, 180);
}

async function advRun(prefix, mod) {
  if (!bridge()) return;
  const vals = formValues(prefix);
  const v = await bridge().check_tool(mod, vals);
  if (!v.ok) { toast(v.problems.join("; ")); return; }
  if (!confirm(`Run this?\n\n${v.display}`)) return;
  const r = await bridge().run_tool(mod, vals);
  if (!r.ok) { toast(r.error); return; }
  logSeq = 0; logLines = []; startPolling();
  toast("running " + mod.replace("pollard_", "pollard-"));
}

function toolBlock(mod, prefix) {
  const t = toolOf(mod);
  return `<div class="toolblock">
    <div class="tbhead"><span class="tbname">${mod.replace("pollard_", "pollard-")}</span>
      <span class="tbsum">${esc((t && t.summary) || "")}</span></div>
    ${flagForm(mod, prefix)}
    <div class="sp"></div>
    <div class="cmd" id="${prefix}-cmd">...</div>
    <div id="${prefix}-prob"></div>
    <div class="sp"></div>
    <div class="btns">
      <button class="btn gold" id="${prefix}-run" onclick="advRun('${prefix}','${mod}')">${ic("play")}RUN</button>
      <button class="btn" onclick="act('help:${mod}')">--help</button>
    </div></div>`;
}

/* Sections are curated -- the ORDER these levers pay in is knowledge, not something the manifest
   knows. What each section CONTAINS is generated. */
const ADV_SECTIONS = [
  { title: "ONE SHOT", icon: "play", open: true,
    why: "pollard on its own is the whole path -- point it at any model and it calibrates, builds "
       + "an imatrix, allocates and emits. Every stage it would decide for you is a flag here, so "
       + "this is also the fastest way to override exactly one of them and leave the rest alone.",
    tools: ["pollard_auto"] },
  { title: "CALIBRATION", icon: "file",
    why: "A calibration set that is all one domain teaches the allocator one domain. The held-out "
       + "split is what keeps an eval honest -- score on text the build never saw.",
    tools: ["pollard_calib"] },
  { title: "ALLOCATION", icon: "sliders",
    why: "Measure where the bits should go before spending them. probe is the cheap any-box "
       + "profile; sensitivity is the ground truth; fit and automap turn a profile into a build. "
       + "fragile, errsrc and errtype answer the three different questions about a bad tensor -- "
       + "which one to protect, how much of the measured error it owns, and WHY it quantizes badly.",
    tools: ["pollard_probe", "pollard_sensitivity", "pollard_fit", "pollard_automap",
            "pollard_fragile", "pollard_errsrc", "pollard_errtype", "pollard_fit_dit"] },
  { title: "PRECONDITIONERS", icon: "activity",
    why: "Which preconditioner wins is a property of the model, not of the method -- rotation is "
       + "right for codebook quants where per-channel smoothing actively hurts, and the reverse "
       + "holds elsewhere. precondition measures it rather than assuming.",
    tools: ["pollard_precondition", "pollard_rotate", "pollard_smooth", "pollard_hf_smooth"] },
  { title: "BELOW TWO BITS", icon: "layers",
    why: "Under the 2-bit floor a different ALPHABET beats a smaller number of bits. palette "
       + "chooses per tensor on top of GPTQ reconstruction; trellis spends fractional bits on "
       + "shape instead of integer bits on magnitude.",
    tools: ["pollard_palette", "pollard_lowbit", "pollard_trellis"] },
  { title: "SOLVER", icon: "wave",
    why: "Full-Hessian error feedback. Pollard trains as well as compresses, and these are the "
       + "solver's own knobs.",
    tools: ["pollard_gptq", "pollard_kl"] },
  { title: "MIXTURE OF EXPERTS", icon: "cpu",
    why: "A MoE has no fixed computation graph. Capture which experts your workload uses, drop "
       + "the cold ones whole rather than crushing every one, and check quantization did not "
       + "change the selection.",
    tools: ["pollard_route", "pollard_experts", "pollard_prune", "pollard_routecheck",
            "pollard_run"] },
  { title: "LANE EMITTERS", icon: "box",
    why: "One solved allocation, emitted into whichever runtime you need. Each lane has its own "
       + "atoms and its own knobs.",
    tools: ["pollard_mlx", "pollard_mx", "pollard_exl3", "pollard_exl3_band",
            "pollard_export", "pollard_vllm"] },
  { title: "NEW ARCHITECTURES", icon: "box",
    why: "Pollard is not a list of supported models. onboard audits an architecture it has never "
       + "seen and writes the profile the allocator then works from, so an unknown model is a "
       + "first step rather than a refusal.",
    tools: ["pollard_onboard"] },
  { title: "MEASUREMENT", icon: "activity",
    why: "The eval and bench screens run the standard passes. These are the manual ones: probes "
       + "scores real task accuracy on the build, serve-eval A/Bs a SERVED endpoint against its "
       + "reference so you measure what users will actually hit.",
    tools: ["pollard_probes", "pollard_serve_eval"] },
  { title: "BEHAVIOUR", icon: "brain",
    why: "Optional, and measured rather than assumed: abliterate reports what it changed, and "
       + "pollard-kl is how you check it changed only that.",
    tools: ["pollard_abliterate"] },
];

SCREENS.advanced = () => `
  <div class="card" style="margin-bottom:14px">
    <h3>${ic("sliders")}ADVANCED<span class="rt" id="adv-count"></span></h3>
    <div class="hint">The other screens pick sensible defaults and get out of the way. This one
      does not. Every flag each tool declares is here -- generated from the tools themselves, so
      a section cannot be missing a knob. The grouping and the order are curated, because which
      lever pays first is knowledge the manifest does not have.</div>
  </div>

  ${ADV_SECTIONS.map((sec, si) => `
    <details class="fold"${sec.open ? " open" : ""}><summary>${ic("chev")}${ic(sec.icon)}${sec.title}
      <span class="rt">${sec.tools.length} tool${sec.tools.length === 1 ? "" : "s"}</span></summary>
      <div class="inner">
        <div class="hint" style="margin-bottom:12px">${esc(sec.why)}</div>
        ${sec.tools.map((m, ti) => toolBlock(m, `adv${si}_${ti}`)).join("")}
      </div></details>`).join("")}

  <div class="card" style="margin-top:14px"><h3>${ic("file")}OUTPUT<span class="rt" id="ad-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="ad-log"
      style="padding:11px 13px;max-height:240px"></div></div></div>`;

SCREENS.tools = () => `
  <div class="card"><h3>${ic("cpu")}TOOLS<span class="rt" id="tool-count"></span></h3>
    <div class="searchwrap">${ic("search")}
      <input type="text" id="tool-search" placeholder="filter tools and flags…"></div>
    <div class="sp"></div>
    <div class="toolgrid"><div class="toollist" id="tool-list"></div>
      <div id="tool-detail"></div></div></div>`;

SCREENS.train = () => `
  <div class="grid" style="grid-template-columns:340px 1fr">
    <div class="card"><h3>${ic("sliders")}CONFIGURATION</h3>
      ${field("Source <span class='hintr'>--model</span>", sel("tr-src", buildOpts(), R.source))}
      ${field("Method <span class='hintr'>--method</span>",
              sel("tr-method", ["rtn", "imatrix", "gptq", "gptq-ao", "gptq-seq", "gptq-seq-ao",
                                "both", "all"], R.method || "gptq-seq"))}
      ${field("Bits <span class='hintr'>--bits</span>", srow("tr-bits", 2, 8, 4))}
      ${field("Group size <span class='hintr'>--groupsize</span>",
              sel("tr-group", ["32", "64", "128", "-1"], "128"))}
      ${field("Calibration samples <span class='hintr'>--nsamples</span>",
              srow("tr-ns", 32, 1024, 128, 32, true))}
      ${field("Train on <span class='hintr'>--calib-file, your own corpus</span>",
              pathrow("tr-calib", "calibFile", "text", "your text file (skips the HF datasets)"))}
      ${field("Score on <span class='hintr'>--eval-file, held out</span>",
              pathrow("tr-evalf", "trainEvalFile", "text", "your eval text (skips the HF datasets)"))}

      <details class="fold" open><summary>${ic("chev")}ALPHABET &amp; MIX<span class="rt">3 flags</span></summary>
        <div class="inner">
          <div class="hint" style="margin-bottom:10px">Below two bits the ALPHABET matters more
            than the bit count. And a protected mix crushes the MLP body while holding attention
            high, which is where the quality at low bits actually comes from.</div>
          ${field("Alphabet <span class='hintr'>--qmode</span>",
                  sel("tr-qmode", [["", "int (asymmetric)"], "ternary", "binary"], R.qmode || ""))}
          ${field("Protected mix <span class='hintr'>--recipe, gptq-seq only</span>",
                  sel("tr-recipe", ["none", "handmix", "aggr"], R.recipe || "none"))}
          ${field("Ablate a protect class <span class='hintr'>--ablate</span>",
                  sel("tr-ablate", ["none", "firstlast", "attn", "attnout", "down"], R.ablate || "none"))}
        </div></details>

      <details class="fold"><summary>${ic("chev")}SOLVER<span class="rt">4 flags</span></summary>
        <div class="inner">
          ${field("Sequence length <span class='hintr'>--seqlen</span>", srow("tr-seq", 512, 8192, 2048, 512, true))}
          ${field("Head bits <span class='hintr'>--head-bits, 0 = leave fp16</span>", srow("tr-head", 0, 16, 0))}
          ${field("Embed bits <span class='hintr'>--embed-bits, 0 = leave fp16</span>", srow("tr-emb", 0, 16, 0))}
          ${field("Eval chunks <span class='hintr'>--eval-chunks</span>", srow("tr-evchunks", 0, 200, 40, 10, true))}
        </div></details>

      <details class="fold"><summary>${ic("cpu")}MACHINE<span class="rt">4 flags</span></summary>
        <div class="inner">
          <div class="hint" style="margin-bottom:10px">A long solve that dies at 90% should not
            start over. --work-dir checkpoints each finished block; --resume refuses to continue
            if the settings changed, rather than mixing two runs into one file.</div>
          ${field("Device <span class='hintr'>--device</span>", sel("tr-dev", ["auto", "cpu", "cuda", "mps"], "auto"))}
          ${field("Threads <span class='hintr'>--threads</span>", srow("tr-threads", 0, 64, 0, 1, true))}
          ${field("Checkpoint dir <span class='hintr'>--work-dir</span>",
                  pathrow("tr-work", "workDir", "dir", "checkpoint each finished block here"))}
          ${tog("tr-resume", "Resume", "--resume: continue from the checkpoint dir", R.resume)}
          ${tog("tr-offload", "Offload", "--offload: model on CPU, one block on the GPU at a time", R.offload)}
        </div></details>
      <div class="sp"></div>
      <div class="btns"><button class="btn gold wide" onclick="act('train')">${ic("play")}START</button></div>
      <div class="sp"></div>
      <div class="btns"><button class="btn" onclick="previewOnly('train')">PREVIEW COMMAND</button>
        <button class="btn bad" onclick="abortRun()">ABORT</button></div>
    </div>
    <div class="stackv">
      <div class="card"><h3>${ic("wave")}LOSS<span class="rt" id="tr-step"></span></h3>
        <div class="chartbox" id="tr-chart"></div></div>
      <div class="card"><h3>${ic("file")}OUTPUT<span class="rt" id="tr-state"></span></h3>
        <div class="lcd" style="padding:0"><div class="log" id="tr-log"
          style="padding:11px 13px;max-height:210px"></div></div></div>
    </div>
  </div>`;

SCREENS.chat = () => `

  <div class="card" style="margin-bottom:16px">
    <h3>${ic("layers")}WHAT THIS BUILD DOES<span class="rt" id="mod-kind"></span></h3>
    <div class="modgrid" id="mod-grid"></div>
  </div>

  <div class="grid" style="grid-template-columns:320px 1fr">
    <div class="card"><h3>${ic("chat")}RUNTIME</h3>
      ${field("Build", sel("c-ckpt", buildOpts(), R.gguf))}
      ${field("Runtime <span class='hintr'>not locked to one</span>",
              `<div class="selwrap"><select id="c-rt"></select>${ic("chev")}</div>`)}
      <div id="c-rtnote" class="hint" style="margin:-8px 0 12px"></div>
      ${field("Max tokens", srow("c-max", 32, 1024, 256, 32, true))}
      ${field("Temperature", srow("c-temp", 0, 20, 7, 1))}
      <div class="sep"></div>
      <div class="btns"><button class="btn gold wide" onclick="coherenceGate()">
        ${ic("shield")}RUN TEXT GATE</button></div>
      <div class="sp"></div><div id="c-gate"></div>
      <div class="sep"></div>
      <div class="hint">Below about three bits a build can hold its perplexity and still cycle a
        phrase until the budget runs out, never halt, or open a reasoning block it never closes.
        The text gate checks those four. The modality checks above cover the rest.</div>
    </div>

    <div class="card"><h3>${ic("chat")}CHAT<span class="rt" id="c-state">${chip("IDLE")}</span></h3>
      <div class="chatlog" id="c-log"></div>
      <div class="sep"></div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:9px">
        <input type="text" id="c-in" placeholder="ask the build something…">
        <button class="btn gold" onclick="sendChat()">SEND</button></div></div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>${ic("file")}CHECK A GENERATED FILE<span class="rt">image or audio the build produced</span></h3>
    <div class="hint">Point this at something the build generated. It catches how low-bit
      generation actually fails — a flat frame where the denoiser collapsed, uniform noise where
      it never converged, silence, a constant tone instead of speech, or clipping. Judging whether
      the image matches the prompt or the speech sounds natural needs CLIP and an ASR round-trip;
      those live in pollard-mmeval and pollard-taskeval.</div>
    <div class="sp"></div>
    <div class="grid" style="grid-template-columns:1fr 200px;margin:0">
      ${field("File", pathrow("art-path", "artifact", "any", "image, audio or video the build produced"))}
      ${field("Volume <span class='hintr'>playback</span>", srow("art-vol", 0, 100, 80))}
    </div>
    <div class="btns"><button class="btn gold" onclick="checkArtifact()">${ic("play")}OPEN IT</button>
      <button class="btn" onclick="stopMedia()">${ic("stop")}STOP</button></div>
    <div class="sp"></div>
    <div id="art-player"></div>
    <div id="art-result"></div>
  </div>`;

SCREENS.bench = () => `
  <div class="grid" style="grid-template-columns:1fr 340px">
    <div class="card"><h3>${ic("activity")}SIZE BY BUILD<span class="rt">measured on disk</span></h3>
      <div class="chartbox" id="bn-chart"></div>
      <div class="sp"></div>
      <div class="hint">Throughput is not shown because it has not been measured on this machine.
        Run the bench and it appears here — Studio will not print a tokens-per-second figure it
        did not observe.</div>
    </div>
    <div class="card"><h3>${ic("cpu")}RUN CONFIGURATION</h3>
      ${field("Build <span class='hintr'>--gguf</span>", sel("bn-gguf", buildOpts(), R.gguf))}
      ${field("Compare against <span class='hintr'>--vs</span>", sel("bn-vs", [["", "none"], ...buildOpts()], R.vs || ""))}
      ${field("…or any GGUF on this machine <span class='hintr'>--vs</span>",
              pathrow("bn-vsfile", "vs", "gguf", "a rival build anywhere on disk"))}
      ${field("Eval corpus <span class='hintr'>--eval</span>",
              pathrow("bn-eval", "evalFile", "text", "your own held-out corpus"))}
      ${field("Chunks <span class='hintr'>--chunks</span>", srow("bn-chunks", 10, 500, R.chunks, 10, true))}
      ${field("GPU layers <span class='hintr'>--ngl</span>", srow("bn-ngl", 0, 99, R.ngl))}
      ${tog("bn-speed", "Speed", "--speed: tokens per second", true)}
      ${tog("bn-coh", "Coherence", "--coherence: does it still make sense", false)}
      ${tog("bn-quick", "Quick", "--quick", false)}
      <div class="sp"></div>
      <div class="btns"><button class="btn gold" onclick="act('bench')">${ic("play")}RUN BENCH</button>
        <button class="btn" onclick="previewOnly('bench')">PREVIEW</button>
        <button class="btn bad" onclick="abortRun()">ABORT</button></div>
      <div class="sep"></div>
      <div class="hint">--ngl stays at 0 unless the box is genuinely free. A full offload that does
        not fit takes the machine down with it.</div>
    </div>
  </div>
  <div class="grid g-2">
    <div class="card"><h3>${ic("layers")}KV CACHE<span class="rt">pollard-bench --kv-sweep</span></h3>
      <div class="hint">At long context the cache, not the weights, is what fills the machine.
        Keys and values are swept independently because their outlier structure differs — keys
        high with values low is usually the row that wins, and a symmetric sweep never tries it.</div>
      <div class="sp"></div>
      ${field("Context for the sweep <span class='hintr'>--kv-ctx</span>", srow("kv-ctx", 512, 32768, 4096, 512, true))}
      ${field("Eval corpus <span class='hintr'>--eval</span>",
              pathrow("kv-eval", "kvEvalFile", "text", "held-out text file"))}
      <div class="btns"><button class="btn gold" onclick="act('kvsweep')">${ic("play")}SWEEP KV</button>
        <button class="btn" onclick="previewOnly('kvsweep')">PREVIEW</button></div>
    </div>

    <div class="card"><h3>${ic("cpu")}TOOL CALLS<span class="rt">pollard-toolcall</span></h3>
      <div class="hint">Single-turn benchmarks report 4-bit as near-lossless. The skill that
        actually breaks is narrow — emitting a well-formed call with the right name and argument
        types — and it lives in a small band of layers, so a drop is usually a few tensors rather
        than the whole model.</div>
      <div class="sp"></div>
      ${field("Reference <span class='hintr'>--ref</span>", sel("tc-ref", buildOpts(), (refBuild() || {}).path))}
      ${field("Minimum validity <span class='hintr'>--min-rate</span>", srow("tc-rate", 0, 100, 75))}
      <div class="btns"><button class="btn gold" onclick="act('toolcall')">${ic("play")}CHECK TOOL CALLS</button>
        <button class="btn" onclick="previewOnly('toolcall')">PREVIEW</button></div>
    </div>
  </div>

  <div class="card"><h3>${ic("file")}OUTPUT<span class="rt" id="bn-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="bn-log"
      style="padding:11px 13px;max-height:230px"></div></div></div>`;

SCREENS.eval = () => `
  <div class="grid" style="grid-template-columns:1fr 340px">
    <div class="card"><h3>${ic("activity")}PERPLEXITY BY BUILD</h3>
      <div class="chartbox" id="ev-chart"></div>
      <div class="sp"></div>
      <div id="ev-note"></div></div>
    <div class="card"><h3>${ic("sliders")}EVAL CONFIGURATION</h3>
      ${field("Reference <span class='hintr'>--ref</span>", sel("ev-ref", buildOpts(), (refBuild() || {}).path))}
      ${field("Quantized <span class='hintr'>--quants</span>", sel("ev-q", buildOpts(), R.gguf))}
      ${field("Eval corpus <span class='hintr'>--eval</span>",
              pathrow("ev-file", "evalFile", "text", "your own held-out corpus"))}
      ${field("Prompts <span class='hintr'>--prompts</span>",
              pathrow("ev-prompts", "promptsFile", "text", "your own prompts, one per line"))}
      ${field("Generated tokens <span class='hintr'>--gen-tokens</span>", srow("ev-gt", 32, 1024, 256, 32, true))}
      ${tog("ev-traj", "Trajectory", "--trajectory", true)}
      ${tog("ev-chart", "Write chart", "--chart", false)}
      <div class="sp"></div>
      <div class="btns"><button class="btn gold" onclick="act('eval')">${ic("play")}RUN EVAL</button>
        <button class="btn" onclick="previewOnly('eval')">PREVIEW</button>
        <button class="btn bad" onclick="abortRun()">ABORT</button></div>
      <div class="sep"></div>
      <div class="hint">The eval corpus must be disjoint from calibration. Pollard refuses the run
        on overlap rather than reporting a number that flatters itself.</div>
    </div>
  </div>
  <div class="card"><h3>${ic("check")}YOUR OWN BENCHMARK
      <span class="rt">pollard-probes · pollard-taskeval</span></h3>
    <div class="hint">Perplexity is a proxy. If you have the benchmark you actually care about —
      an MCQ set you built, a HellaSwag or Winogrande file, a task your users run — score the
      build on that instead. FIND searches this machine; nothing has to be in the workspace.</div>
    <div class="sp"></div>
    <div class="grid g-2" style="margin:0">
      <div>
        ${field("Build <span class='hintr'>--gguf</span>", sel("pb-gguf", buildOpts(), R.gguf))}
        ${field("Benchmark file <span class='hintr'>--hellaswag-data / --winogrande / --multiple-choice</span>",
                pathrow("pb-data", "probeData", "data", "your MCQ / HellaSwag / Winogrande file"))}
        ${field("File format <span class='hintr'>picks which flag is used</span>",
                sel("pb-fmt", [["hellaswag", "HellaSwag"], ["winogrande", "Winogrande"],
                               ["multiple-choice", "MMLU / ARC style"]], R.probeFormat || "hellaswag"))}
        ${field("Tasks <span class='hintr'>--tasks</span>", srow("pb-tasks", 20, 2000, 200, 20, true))}
        <div class="btns"><button class="btn gold" onclick="act('probes')">${ic("play")}RUN BENCHMARK</button>
          <button class="btn" onclick="previewOnly('probes')">PREVIEW</button></div>
      </div>
      <div>
        ${field("Suite <span class='hintr'>--suite</span>",
                sel("te-suite", [["", "default"], "text", "vision", "agentic"], R.suite || ""))}
        ${field("lm-eval tasks <span class='hintr'>--tasks, overrides --suite</span>",
                `<input type="text" id="te-tasks" placeholder="arc_easy,hellaswag,gsm8k"
                   value="${escAttr(R.tasks || "")}" oninput="R.tasks=this.value.trim()||undefined;refresh()">`)}
        ${field("Samples per task <span class='hintr'>--limit, 0 = all</span>",
                srow("te-limit", 0, 1000, 100, 10, true))}
        <div class="btns"><button class="btn gold" onclick="act('taskeval')">${ic("play")}RUN TASK EVAL</button>
          <button class="btn" onclick="previewOnly('taskeval')">PREVIEW</button></div>
        <div class="hint" style="margin-top:8px">taskeval drives lm-eval, so any task name it
          knows works here — name them and --suite is ignored.</div>
      </div>
    </div>
  </div>

  <div class="card"><h3>${ic("layers")}ROUTING CONSISTENCY
      <span class="rt">${isMoE() ? "pollard-routecheck" : "dense model — does not apply"}</span></h3>
    ${isMoE() ? `
      <div class="hint">A MoE has no fixed computation graph: the router picks a subset of experts
        per token. If quantization shifts that choice the model runs different weights than the
        ones you measured, and perplexity absorbs a surprising amount of it. Matching router
        logits is not enough — top-k depends only on order, so a shift too small to move a loss
        can still swap an expert at the selection boundary.</div>
      <div class="sp"></div>
      <div class="grid g-2" style="margin:0">
        ${field("Reference <span class='hintr'>--ref</span>", sel("rc-ref", buildOpts(), (refBuild() || {}).path))}
        ${field("Swap budget % <span class='hintr'>--swap-budget</span>", srow("rc-budget", 0, 20, 2))}
      </div>
      <div class="btns"><button class="btn gold" onclick="act('routecheck')">${ic("play")}CHECK ROUTING</button>
        <button class="btn" onclick="previewOnly('routecheck')">PREVIEW</button></div>`
      : `<div class="hint">Routing consistency applies to mixture-of-experts models. This
          checkpoint has no routers, so there is nothing to preserve here.</div>`}
  </div>

  <div class="card"><h3>${ic("file")}OUTPUT<span class="rt" id="ev-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="ev-log"
      style="padding:11px 13px;max-height:230px"></div></div></div>`;

SCREENS.doctor = () => `
  <div class="grid g-3" id="doc-tiles"></div>
  <div class="grid g-2">
    <div class="card"><h3>${ic("gauge")}TARGET</h3>
      <div class="gauges" id="doc-gauges"></div>
      <div class="sep"></div><div id="doc-rulers"></div></div>
    <div class="card"><h3>${ic("cpu")}ENVIRONMENT</h3><div id="doc-hw"></div>
      <div class="sp"></div>
      <div class="btns"><button class="btn gold" onclick="act('doctor')">RUN DOCTOR</button>
        <button class="btn" onclick="act('smoke')">PREFLIGHT SMOKE</button>
        <button class="btn" onclick="act('ggufcheck')">GGUF CHECK</button></div></div>
  </div>

  <div class="card"><h3>${ic("cpu")}DIAGNOSTICS<span class="rt">is the machine, the runtime and
      the reference actually sound</span></h3>
    <div class="hint">These answer questions about the environment rather than the build. A
      silently throttled accelerator or a llama.cpp too old for the architecture produces numbers
      that look like a bad quantization, and chasing the build instead of the box is the expensive
      way to find out.</div>
    <div class="sp"></div>
    <div class="grid g-4" style="margin:0">
      ${[["health", "Accelerator health", "is it at full speed, or silently degraded"],
         ["runtime", "Runtime currency", "can the builds here still be loaded"],
         ["refcheck", "Reference sanity", "prove the reference forward is sane BEFORE measuring on it"],
         ["modelkind", "Model kind", "what this is, so the rest stops guessing"],
         ["archfp", "Architecture", "structurally, which known one is it a twin of"],
         ["envmatch", "Env match", "a python env matched to this model's transformers"],
         ["calc", "Hardware fit", "what this box can do, before downloading 300 GB"],
         ["stop", "What is running", "report, never kill, from a button"]].map(([a, t, w]) => `
        <div class="tile"><div class="th">${ic("gauge")}${t}</div>
          <div class="ts" style="margin:5px 0 9px">${w}</div>
          <button class="btn" style="width:100%" onclick="act('${a}')">RUN</button></div>`).join("")}
    </div>
  </div>

  <div class="card"><h3>${ic("activity")}IS THE DAMAGE REPAIRABLE?<span class="rt">pollard-failmode</span></h3>
    <div class="hint">Low-bit damage comes in two shapes and they want opposite responses.
      <b>Signal degradation</b> leaves the computation running while precision erodes with depth —
      preconditioning recovers most of it. <b>Computation collapse</b> is a component that stopped
      working: the signal is destroyed early and everything after it processes noise, and
      smoothing does not bring that back. Running a repair on a collapsed build spends a full
      reconvert to learn nothing, and can ship something that looks repaired.</div>
    <div class="sp"></div>
    <div class="grid g-3" style="margin:0">
      ${field("Reference <span class='hintr'>--ref</span>", sel("fm-ref", buildOpts(), (refBuild() || {}).path))}
      ${field("Build <span class='hintr'>--model</span>", sel("fm-model", buildOpts(), R.gguf))}
      ${field("Calibration text <span class='hintr'>--calib</span>",
              pathrow("fm-calib", "calibFile", "text", "text file to run through both"))}
    </div>
    <div class="btns"><button class="btn gold" onclick="act('failmode')">${ic("play")}CLASSIFY</button>
      <button class="btn" onclick="previewOnly('failmode')">PREVIEW</button></div>
    <div class="sep"></div>
    <details class="fold"><summary>${ic("chev")}WAFER CAPACITY
      <span class="rt">capacity and throughput for a wafer target</span></summary>
      <div class="inner">${toolBlock("pollard_pack", "dpack")}</div></details>
  </div>
  <div class="card"><h3>${ic("file")}OUTPUT<span class="rt" id="doc-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="doc-log"
      style="padding:11px 13px;max-height:230px"></div></div></div>

`;

/* In the order a brain is actually made: get a connectome, train on it, attach it to a
   backbone, prove it recalls, then ask which runtimes can host it. */
const BRAIN_STEPS = [
  { tool: "pollard_connectome", title: "1 · CONNECTOME", why: "the wiring to train on" },
  { tool: "pollard_flybrain",   title: "2 · TRAIN",      why: "connectome as a language model" },
  { tool: "pollard_brainattach", title: "3 · ATTACH",    why: "ship it alongside a backbone" },
  { tool: "pollard_brainverify", title: "4 · VERIFY",    why: "does it actually recall" },
  { tool: "pollard_brainlanes",  title: "5 · LANES",     why: "which runtimes can host it" },
];

SCREENS.brains = () => `
  <div class="card"><h3>${ic("brain")}BRAINS<span class="rt">attach to a built model</span></h3>
    <div class="hint">A brain is a small fixed-size state file mounted onto a finished backbone. It
      has its own path end to end — nothing on this screen touches allocation, calibration or any
      emitter. Brains are per-backbone and PyTorch-only.</div>
    <div class="sp"></div>
    ${BRAIN_STEPS.map((s, i) => `
      <details class="fold"${i === 0 ? " open" : ""}><summary>${ic("chev")}${s.title}
        <span class="rt">${esc(s.why)}</span></summary>
        <div class="inner">${toolBlock(s.tool, `br${i}`)}</div></details>`).join("")}
  </div>
  <div class="card"><h3>${ic("file")}OUTPUT<span class="rt" id="br-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="br-log"
      style="padding:11px 13px;max-height:230px"></div></div></div>`;

/* The three things you do AROUND a publish: score it, bring an older repo up to the current
   template, and get the disk back once it is live. */
const PUBLISH_TOOLS = ["pollard_scorecard", "pollard_recard", "pollard_reclaim"];
const PUBLISH_WHY = [
  ["SCORECARD", "the standardized memory-fit low-bit numbers"],
  ["RECARD", "bring an already-published repo onto this template"],
  ["RECLAIM", "free the disk a finished model is still holding"],
];

SCREENS.publish = () => `
  <div class="card"><h3>${ic("upload")}PUBLISH<span class="rt">complete repo, uniform template</span></h3>
    <div class="hint">A repo ships the full ladder, the card with real measured numbers, and every
      sidecar. Bare GGUFs are never published.</div>
    <div class="sp"></div>
    <table><thead><tr><th>ARTIFACT</th><th>TAG</th><th>SIZE</th><th>BPW</th><th>RECIPE</th>
      <th>VERIFIED</th></tr></thead><tbody id="pub-rows"></tbody></table>
    <div class="sp"></div>
    <div class="btns"><button class="btn gold" onclick="act('card')">GENERATE CARD</button>
      <button class="btn" onclick="act('verify')">VERIFY</button>
      <button class="btn" onclick="act('ggufcheck')">RUNTIME CHECK</button></div>
    <div class="sep"></div>
    <div class="grid g-2" style="margin:0">
      ${field("Hugging Face repo <span class='hintr'>--upload</span>",
              `<input type="text" id="pub-repo" placeholder="you/Model-Pollard-GGUF"
                 oninput="R.repo=this.value;R.uploadRepo=this.value">`)}
      ${field("Model name <span class='hintr'>--model</span>",
              `<input type="text" id="pub-model" value="${escAttr((S.model || {}).name || "")}"
                 oninput="R.modelName=this.value">`)}
    </div>
    <div class="outbound">${ic("alert")}<div><b>This one leaves the machine.</b>
      Publishing is public and hard to take back, so it asks twice and needs the repo typed in
      full. Everything above it is local.</div></div>
    <div class="sp"></div>
    <div class="btns"><button class="btn gold" onclick="publishNow()">${ic("upload")}PUBLISH TO HUGGING FACE</button></div>
    <div class="sep"></div>
    ${PUBLISH_TOOLS.map((m, i) => `
      <details class="fold"><summary>${ic("chev")}${PUBLISH_WHY[i][0]}
        <span class="rt">${esc(PUBLISH_WHY[i][1])}</span></summary>
        <div class="inner">${toolBlock(m, `pb${i}`)}</div></details>`).join("")}
  </div>
  <div class="card"><h3>${ic("file")}OUTPUT<span class="rt" id="pub-state"></span></h3>
    <div class="lcd" style="padding:0"><div class="log" id="pub-log"
      style="padding:11px 13px;max-height:230px"></div></div></div>`;

/* ── routing ────────────────────────────────────────────────────────────── */
let screen = "build";

function show(name) {
  screen = SCREENS[name] ? name : "build";
  $$("#wing .pill").forEach(b => b.classList.toggle("on", b.dataset.s === screen));
  $("#stage").innerHTML = SCREENS[screen]();
  ({ build: wireBuild, tools: wireTools, monitor: wireMonitor, ladder: wireLadder,
     train: wireTrain, bench: wireBench, eval: wireEval, doctor: wireDoctor,
     chat: wireChat, convert: wireConvert, advanced: wireAdvanced,
     brains: wireBrains, publish: wirePublish }[screen] || (() => {}))();
  refresh();
  renderLog();
}

/* ── wiring helpers ─────────────────────────────────────────────────────── */
function bindSlider(id, key) {
  const r = $("#" + id), n = $("#" + id + "-n");
  if (!r) return;
  const push = v => {
    v = clamp(+v, +r.min, +r.max);
    r.value = v; n.value = v;
    r.style.setProperty("--p", ((v - r.min) / (r.max - r.min)) * 100);
    if (key) R[key] = v;
    refresh();
  };
  r.addEventListener("input", () => push(r.value));
  n.addEventListener("change", () => push(n.value));
  push(r.value);
}

/* Search the machine from the field itself: whatever is typed becomes the query, and the
   results are clickable. This is the path most people take -- you know the model is called
   something like "qwen", not where it landed six weeks ago. */
async function findInto(id, key, kind) {
  const panel = $("#" + id + "-find"), input = $("#" + id);
  if (!panel || !bridge()) return;
  const q = (input.value || "").trim();
  panel.hidden = false;
  panel.innerHTML = `<div class="findnote">searching this machine…</div>`;
  const got = await bridge().search_files(q, kind || "any", 60);
  if (got.error) { panel.innerHTML = `<div class="findnote">${esc(got.error)}</div>`; return; }
  const rows = got.results || [];
  if (!rows.length) {
    panel.innerHTML = `<div class="findnote">nothing matching
      ${q ? `<b>${esc(q)}</b>` : "that kind"} in the usual places. BROWSE points anywhere.</div>`;
    return;
  }
  panel.innerHTML =
    `<div class="findnote">${rows.length}${got.truncated ? "+" : ""} found${
       q ? ` for "${esc(q)}"` : ""} — click to use it</div>` +
    rows.map(r => `<div class="findrow" onclick="useFound('${id}','${key}',this.dataset.p)"
        data-p="${escAttr(r.path)}" title="${escAttr(r.path)}">
        <div class="fname">${esc(r.name)}</div>
        <div class="fmeta">${gb(r.bytes) >= 0.01 ? gb(r.bytes) + " GB" : "&lt;10 MB"}
          · ${esc(short(r.dir, 1))}</div></div>`).join("");
}

function useFound(id, key, path) {
  R[key] = path;
  const el = $("#" + id);
  if (el) el.value = path;
  closeFind(id);
  refresh();
}

function closeFind(id) { const p = $("#" + id + "-find"); if (p) p.hidden = true; }

/* Open the OS file dialog and drop the result into a path field. */
async function browseInto(id, key, kind) {
  if (!bridge()) { toast("no bridge"); return; }
  const got = await bridge().pick_file(kind || "any");
  if (!got || !got.path) { if (got && got.error) toast(got.error); return; }
  R[key] = got.path;
  const el = $("#" + id);
  if (el) el.value = got.path;
  closeFind(id);
  refresh();
}

function bindToggles(pairs) {
  pairs.forEach(([id, key]) => {
    const sw = $("#" + id);
    if (!sw) return;
    /* Screens render a toggle's starting position as a literal, so without this the recipe and
       the switch disagree until the first click -- and a choice made once is silently thrown
       away the next time the screen re-renders. R wins when it has an opinion. */
    if (key) {
      if (R[key] === undefined) R[key] = sw.classList.contains("on");
      else sw.classList.toggle("on", !!R[key]);
    }
    const flip = () => {
      sw.classList.toggle("on");
      if (key) R[key] = sw.classList.contains("on");
      refresh();
    };
    sw.addEventListener("click", flip);
    sw.addEventListener("keydown", e => {
      if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); } });
  });
}

const bindSel = (id, key) => {
  const el = $("#" + id);
  if (el) el.addEventListener("change", () => { R[key] = el.value; refresh(); });
};

/* Say what the pool actually is, from what was typed -- not a number anyone configured here. */
function clusterNote() {
  const el = $("#cl-note");
  if (!el) return;
  // what is actually ANSWERING, not what was typed into the box
  const live = poolCache && poolCache.nodes ? poolCache.nodes.filter(n => n.reachable).length : 0;
  const tp = +R.tp || 0;
  const bits = [];
  if (live > 1) bits.push(`${live} boxes · ${poolCache.combined_gb} GB`);
  if (tp) bits.push(`tp-${tp}`);
  if (R.devices) bits.push(`devices ${R.devices}`);
  el.textContent = bits.length ? bits.join(" · ") : "one box";
}

/* The pool, read live. "96 GB" means something very different as one box than as six, so the
   combined figure never stands alone -- every box is listed with what IT actually brings. */
async function renderPool() {
  const box = $("#cl-pool");
  if (!box || !bridge()) return;
  box.innerHTML = `<div class="hint">reading the cluster…</div>`;
  const s = await bridge().cluster_survey(R.rpc || "", 1.5);
  poolCache = s;
  if (s.error) { box.innerHTML = `<div class="hint">${esc(s.error)}</div>`; refresh(); return; }

  const nodes = s.nodes || [];
  const g = v => (v == null ? "—" : `${v}<i>GB</i>`);
  // The cluster counted as ONE machine -- that is the budget the faders are spending.
  const totals = `
    <div class="cltop">
      <div class="clhead">${s.pooled ? "CLUSTER" : "THIS MACHINE"}
        <span>${s.boxes} box${s.boxes === 1 ? "" : "es"} · ${s.serving} serving${
          s.devices ? ` · ${s.devices} device${s.devices === 1 ? "" : "s"}` : ""}</span></div>
      <div class="clgrid">
        <div class="clcell"><b>${g(s.total_vram_gb)}</b><span>GPU / UNIFIED</span></div>
        <div class="clcell"><b>${g(s.total_ram_gb)}</b><span>RAM${
          s.ram_partial ? " ·&nbsp;LOCAL" : ""}</span></div>
        <div class="clcell"><b>${g(s.total_disk_gb)}</b><span>STORAGE${
          s.disk_partial ? " ·&nbsp;LOCAL" : ""}</span></div>
      </div>
    </div>`;

  const rows = nodes.map(n => {
    const on = n.reachable, here = n.role === "this machine";
    // a box running pollard-node answers for the MACHINE; without it, RPC gives devices only
    const full = here || n.agent;
    const mem = !on ? "not reachable" : (full
      ? [n.unified ? `${n.ram_gb} unified` : (n.ram_gb ? `${n.ram_gb} ram` : null),
         (!n.unified && (n.total_gb || n.vram_gb)) ? `${n.total_gb || n.vram_gb} gpu` : null,
         n.disk_free_gb ? `${n.disk_free_gb} disk` : null,
         n.cpus ? `${n.cpus} cpu` : null].filter(Boolean).join(" · ")
      : [n.total_gb ? `${n.total_gb} GB accelerator` : "memory unknown",
         n.devices.length ? `${n.devices.length} dev` : null].filter(Boolean).join(" · "));
    const cards = (n.cards || []).map(c => c.name).join(", ");
    return `<div class="node ${on ? "on" : "off"}" title="${escAttr(n.note || n.endpoint)}">
      <div class="nhead"><i></i><b>${esc(n.host || n.endpoint)}</b>
        ${here ? "" : `<span class="tag ${n.serving ? "ok" : "warn"}">${
            n.serving ? (n.agent ? "pooled" : "rpc") : "builds only"}</span>`}
        <span class="dim">${esc(here ? (n.platform || "local") : n.endpoint)}</span></div>
      <div class="nmem">${esc(mem)}${cards ? ` · ${esc(cards)}` : ""}</div></div>`;
  }).join("");

  // the RAM fader's ceiling is baked into the template; a bigger pool has to move it
  const ram = $("#f-ram");
  if (ram) {
    const max = ramMax();
    if (+ram.max !== max) {
      ram.max = max;
      ram.style.setProperty("--p", ((+ram.value - +ram.min) / (max - +ram.min)) * 100);
    }
  }

  renderDevices();          // which devices exist changes with the pool
  box.innerHTML = totals + `<div class="nodes">${rows}</div>` +
    (s.offline && s.offline.length
      ? `<div class="nnote">${s.offline.length} configured peer${
          s.offline.length === 1 ? "" : "s"} did not answer — the build will not see them.</div>`
      : "") +
    (s.serving < s.boxes
      ? `<div class="nnote">${s.boxes - s.serving} box${s.boxes - s.serving === 1 ? "" : "es"}
           build on their own but have no <b>ggml-rpc-server</b> answering, so they cannot yet
           hold part of a single model split across machines. START SERVING sets that up.</div>`
      : "") +
    (s.no_agent && s.no_agent.length
      ? `<div class="nnote">${s.no_agent.length} box${s.no_agent.length === 1 ? "" : "es"}
           report only device memory — RPC has no call for a host's RAM or disk. Run
           <b>pollard-node</b> there and its RAM, cards and storage join these totals.</div>`
      : "");
  refresh();
}

/* ggml-rpc-server is the thing standing between "my box is on" and "my box can hold part of
   this model". Studio finds it, starts it bound so other machines can reach it, and says plainly
   when the binary is too old to pool with. */
async function renderRpc() {
  const el = $("#rpc-state"), btn = $("#rpc-btn");
  if (!el || !bridge()) return;
  const st = await bridge().rpc_status(0);
  rpcCache = st;
  if (btn) btn.innerHTML = (st.running ? ic("stop") + "STOP SERVING" : ic("play") + "START SERVING");
  const bits = [];
  if (st.running) {
    bits.push(`<b>serving</b> on ${st.port}${st.protocol ? ` · rpc ${esc(st.protocol)}` : ""}`);
    if (st.listening_everywhere === false) bits.push("bound to loopback only");
  } else if (st.found) {
    bits.push("not serving — this box builds on its own, but cannot hold part of a split model");
  } else {
    bits.push("ggml-rpc-server not found on this box");
  }
  el.innerHTML = `<div class="hint" style="margin-bottom:8px">${bits.join(" · ")}</div>` +
    (st.problems && st.problems.length && !st.running
      ? `<div class="nnote">${st.problems.map(esc).join("<br>")}</div>` : "");
}

let rpcCache = null;

async function toggleServing() {
  const btn = $("#rpc-btn");
  if (!bridge()) return;
  if (btn) btn.disabled = true;
  try {
    const r = (rpcCache && rpcCache.running) ? await bridge().rpc_stop()
                                             : await bridge().rpc_serve(0);
    if (!r.ok) { toast(r.error || "could not change it"); return; }
    toast(r.already ? (r.note || "already serving")
                    : (rpcCache && rpcCache.running ? "stopped" : "serving — other boxes can reach this one"));
  } finally {
    if (btn) btn.disabled = false;
    await renderRpc();
    renderPool();
  }
}

/* -ngl says how many layers leave the CPU; WHICH device each one lands on is -ts, a share per
   device. So the honest control is a fader per device, with -ngl as the master above them.
   A device left at 0 gets nothing -- that is how you keep a build off a machine someone else
   is using without dropping it from the pool. */
let devOrder = [];

/* A rung's filename carries its quant tag; the model is what is left when you take it off.
   Pollard names builds <model>-Pollard-<TAG>.gguf, and the f16 source <model>-f16.gguf, so the
   same model's rungs group together instead of each looking like a model of its own. */
const QUANT_TAG = /-(?:Pollard-)?(?:FLAGSHIP|IQ\d\w*|Q\d\w*|BF16|F16|f16|bf16|MXFP4|NVFP4|u-iq\w+)$/;

function modelNameOf(name) {
  let stem = String(name || "").replace(/\.(gguf|safetensors|pt)$/i, "");
  for (let i = 0; i < 3 && QUANT_TAG.test(stem); i++) stem = stem.replace(QUANT_TAG, "");
  return stem.replace(/-Pollard-?$/, "") || name;
}

function tagOf(name) {
  const stem = String(name || "").replace(/\.(gguf|safetensors|pt)$/i, "");
  const m = stem.match(QUANT_TAG);
  return m ? m[0].replace(/^-(?:Pollard-)?/, "") : stem;
}

/* The cluster's builds, grouped the way the local workspace is: one entry per model per box. */
function remoteGroups() {
  const by = new Map();
  remoteModels.forEach(m => {
    const key = `remote:${m.host}:${modelNameOf(m.name)}`;
    if (!by.has(key)) {
      by.set(key, { key, host: m.host, name: modelNameOf(m.name), builds: [] });
    }
    by.get(key).builds.push({
      path: m.path, tag: tagOf(m.name), bytes: m.bytes, lane: m.lane,
      remote: true, host: m.host, name: m.name,
    });
  });
  const out = [...by.values()];
  out.forEach(g => g.builds.sort((a, b) => b.bytes - a.bytes));
  return out.sort((a, b) => (a.host + a.name).localeCompare(b.host + b.name));
}

/* Every <select> built from buildOpts(). They are filled when a screen renders, which is before
   the cluster has answered -- so once it does, they have to be refilled in place or the boxes'
   builds never appear in the list. */
const BUILD_SELECTS = ["s-src", "cv-from", "tr-src", "c-ckpt", "bn-gguf", "bn-vs", "tc-ref",
                       "ev-ref", "ev-q", "pb-gguf", "rc-ref", "fm-ref", "fm-model"];

function refillBuildSelects() {
  const opts = buildOpts();
  BUILD_SELECTS.forEach(id => {
    const el = $("#" + id);
    if (!el) return;
    const keep = el.value;
    const blank = id === "bn-vs" ? [["", "none"]] : [];
    el.innerHTML = [...blank, ...opts].map(([v, t]) =>
      `<option value="${escAttr(v)}" ${v === keep ? "selected" : ""}>${esc(String(t))}</option>`
    ).join("");
    if (keep) el.value = keep;               // a choice already made must survive the refill
  });
}

/* Pull in what every linked box holds, so the model list is the cluster's, not this machine's. */
async function loadRemoteModels() {
  if (!bridge()) return;
  const got = await bridge().cluster_models(R.rpc || "");
  remoteModels = (got && got.models) || [];
  refillBuildSelects();
  renderModelPicker();   // the cluster's models belong in the top picker too
  const el = $("#cl-models");
  if (el) {
    el.innerHTML = remoteModels.length
      ? `<div class="hint">${remoteModels.length} build${remoteModels.length === 1 ? "" : "s"}
           on ${(got.hosts || []).length} other box${(got.hosts || []).length === 1 ? "" : "es"} —
           they appear in the source and build lists, tagged with the box that holds them.</div>`
      : `<div class="hint">No builds reported by other boxes. Run <b>pollard-node</b> there and
           its workspace joins this list.</div>`;
  }
  refresh();
}

async function renderDevices() {
  const bank = $("#f-bank"), box = $("#dev-place");
  if (!bank || !bridge()) return;
  const got = await bridge().devices(R.rpc || "");
  devOrder = got.devices || [];

  bank.querySelectorAll(".fader.dev").forEach(el => el.remove());   // rebuild, never stack
  if (!devOrder.length) {
    if (box) box.innerHTML = `<div class="hint">No accelerator to place layers on yet — a box has
      to be serving before a build can put weights there.</div>`;
    return;
  }
  if (!R.shares) R.shares = {};
  devOrder.forEach(d => { if (R.shares[d.key] == null) R.shares[d.key] = 100; });
  if (R.ngl == null) R.ngl = 0;

  /* One gold fader per detected device, in the same bank as RAM and RESERVE — they are the same
     decision, so they are the same hardware. -ngl leads: it sets how many layers leave the CPU
     at all, and the device faders split those between boxes. */
  const fader = (id, label, sub, val, max) => `
    <div class="fader dev"><div class="fslot"><div class="rail"></div>
      <div class="ticks">${"<i></i>".repeat(9)}</div>
      <input type="range" id="${id}" min="0" max="${max}" step="${max > 100 ? 1 : 5}"
         value="${val}"></div>
      <div class="fv" id="${id}-v">${val}${max > 100 ? "" : "%"}</div>
      <div class="fl" title="${escAttr(label + " · " + sub)}">${esc(shortLabel(label))}</div></div>`;

  bank.insertAdjacentHTML("beforeend",
    fader("pl-ngl", "LAYERS OFF CPU", "-ngl", R.ngl, 999) +
    devOrder.map((d, i) => fader(`pl-${i}`, d.label, `${d.detail}${d.gb ? " · " + d.gb + " GB" : ""}`,
                                 R.shares[d.key], 100)).join(""));

  const wire = (id, set) => {
    const r = $("#" + id), v = $("#" + id + "-v");
    if (!r) return;
    r.addEventListener("input", () => {
      v.textContent = r.value + (+r.max > 100 ? "" : "%");
      set(+r.value);
      renderPlacement();
    });
  };
  wire("pl-ngl", n => { R.ngl = n; });
  devOrder.forEach((d, i) => wire(`pl-${i}`, n => { R.shares[d.key] = n; }));
  renderPlacement();
}

/* Fader legends are 8px and uppercase; a hostname has to be cut to fit under one. */
function shortLabel(s) {
  s = String(s || "").split(".")[0].replace(/^NVIDIA GeForce /i, "");
  return s.length > 11 ? s.slice(0, 10) + "…" : s;
}

async function renderPlacement() {
  const el = $("#pl-cmd");
  if (!el || !bridge()) return;
  const got = await bridge().placement_args(R.shares || {}, R.rpc || "",
                                            R.ngl == null ? null : R.ngl);
  R.placement = got.args || [];
  // .fl is scoped to .fader for the legends, and is the flag class every other command uses
  el.innerHTML = got.display
    ? `<div class="cmd">${esc(got.display).replace(/(-[a-z]+)/g, '<span class="fl">$1</span>')}</div>`
    : `<div class="hint">Every device at 0 — the build stays on the CPU.</div>`;
}

async function discoverPeers() {
  const box = $("#cl-pool");
  if (!box || !bridge()) return;
  box.innerHTML = `<div class="hint">sweeping this subnet for ggml-rpc-servers…</div>`;
  const got = await bridge().cluster_discover(50052);
  if (got.error || !got.found || !got.found.length) {
    box.innerHTML = `<div class="hint">nothing answering on ${esc(got.subnet || "this subnet")}
      — start <b>ggml-rpc-server</b> on each box you want to pool, then scan again.
      ${got.note ? esc(got.note) : ""}</div>`;
    return;
  }
  const eps = got.found.map(f => f.endpoint);
  const have = String(R.rpc || "").split(",").map(x => x.trim()).filter(Boolean);
  R.rpc = [...new Set([...have, ...eps])].join(",");
  const el = $("#cl-rpc");
  if (el) el.value = R.rpc;
  toast(`found ${eps.length} peer${eps.length === 1 ? "" : "s"}`);
  renderPool();
}

function wireBuild() {
  bindSel("s-src", "source");   // body/protect/emb now come from laneAlloc(), per lane
  bindSel("s-tier", "tier");
  wireLaneAlloc();
  // s-imx is a pathrow now and binds itself; a second listener here just wrote the same key
  // without clearing it when the box was emptied.
  bindSel("cl-tp", "tp");
  clusterNote();
  renderPool();
  renderRpc();                 // read the pool once when the screen opens
  renderDevices();
  loadRemoteModels();
  bindToggles([["t-plan", "planOnly"], ["t-1bit", "allow1bit"], ["t-grow", "allowGrow"],
               ["t-noimx", "noImatrix"], ["t-nogate", "noGate"], ["t-vis", "protectVision"]]);
  [["f-ram", "ram"], ["f-res", "reserve"]].forEach(([id, key]) => {
    const el = $("#" + id);
    el.addEventListener("input", () => {
      R[key] = +el.value; $("#" + id + "-v").textContent = `${el.value} GB`; refresh(); });
  });

  const k = $("#k-lane");
  const setLane = i => {
    i = clamp(i, 0, LANES.length - 1);
    R.lane = LANES[i]; k.dataset.i = i;
    k.style.setProperty("--deg", (-125 + i / (LANES.length - 1) * 250) + "deg");
    // the allocation card is this lane's own knobs, so it has to be rebuilt, not just refreshed
    const box = $("#la-box");
    if (box) { box.innerHTML = laneAlloc(); wireLaneAlloc(); }
    drawLaneRing(i); refresh();
  };
  let dy = null, di = 0;
  k.addEventListener("pointerdown", e => { dy = e.clientY; di = +k.dataset.i || 0;
                                           k.setPointerCapture(e.pointerId); });
  k.addEventListener("pointermove", e => { if (dy !== null) setLane(di + Math.round((dy - e.clientY) / 18)); });
  k.addEventListener("pointerup", () => { dy = null; });
  k.addEventListener("wheel", e => { e.preventDefault();
    setLane((+k.dataset.i || 0) + (e.deltaY < 0 ? 1 : -1)); }, { passive: false });
  k.addEventListener("keydown", e => {
    if (["ArrowUp", "ArrowRight"].includes(e.key)) { e.preventDefault(); setLane((+k.dataset.i || 0) + 1); }
    if (["ArrowDown", "ArrowLeft"].includes(e.key)) { e.preventDefault(); setLane((+k.dataset.i || 0) - 1); }
  });
  setLane(Math.max(0, LANES.indexOf(R.lane)));
}

function drawLaneRing(active) {
  const svg = $("#lane-ring");
  if (!svg) return;
  const cx = 58, cy = 58, r = 40, lr = 48;
  svg.innerHTML = LANES.map((n, i) => {
    const a = (-125 + i / (LANES.length - 1) * 250) * Math.PI / 180;
    const sx = Math.sin(a), cs = Math.cos(a);
    const anchor = sx < -0.35 ? "end" : sx > 0.35 ? "start" : "middle";
    const pad = anchor === "end" ? -3 : anchor === "start" ? 3 : 0;
    return `<line x1="${(cx + (r - 6) * sx).toFixed(1)}" y1="${(cy - (r - 6) * cs).toFixed(1)}"
        x2="${(cx + r * sx).toFixed(1)}" y2="${(cy - r * cs).toFixed(1)}"
        stroke="${i === active ? "#b8944a" : "#928f87"}" stroke-width="${i === active ? 1.8 : 1}"/>
      <text x="${(cx + lr * sx + pad).toFixed(1)}" y="${(cy - lr * cs + 2.5).toFixed(1)}"
        text-anchor="${anchor}" fill="${i === active ? "#8a6d31" : "#6d6b64"}"
        font-weight="${i === active ? 700 : 600}">${n}</text>`;
  }).join("");
}

/* ── refresh ────────────────────────────────────────────────────────────── */
async function refresh() {
  const p = project();
  const st = S.status || {};
  const ver = $("#ver");
  if (ver) ver.textContent = "v" + (S.version || "?");
  clusterNote();
  $("#d-state").textContent = st.running ? "RUNNING" : (st.returncode === 0 ? "DONE"
                              : st.returncode != null ? "EXIT " + st.returncode : "IDLE");
  $("#d-size").textContent = p ? p.gb.toFixed(2) + " GB" : "—";
  $("#d-bpw").textContent  = p ? p.bpw.toFixed(2) : "—";
  $("#d-rt").textContent   = p ? p.runtime : "—";
  $("#led-gate").classList.toggle("off", !!R.noGate);
  $("#led-imatrix").classList.toggle("off", !!R.noImatrix || !R.imatrix);

  $("#topLcd").innerHTML = S.model
    ? `<span class="dim">model</span> <b>${S.model.name}</b>  <span class="dim">lane</span> ${R.lane}`
      + (p ? `  <span class="dim">fit</span> <span class="${p.fits ? "hi" : ""}">${p.gb.toFixed(1)}/${R.ram}.0 GB</span>`
           + `  <span class="dim">bpw</span> ${p.bpw.toFixed(2)}` : "")
    : `<span class="dim">no workspace at</span> ${short(S.home, 1)}`;

  const ref = refBuild();
  $("#deckLcd").innerHTML = builds().length
    ? builds().slice(0, 3).map(b => `${b.tag.padEnd(9)} <span class="hi">${gb(b.bytes)}G</span>  ${
        b.bpw ? b.bpw.toFixed(2) + " bpw" : "<span class='dim'>bpw ?</span>"}`).join("<br>")
    : "<span class='dim'>no builds found</span>";

  // deck spectrum: real per-group share of the projected build
  const spec = $("#spectrum");
  if (p) {
    const cells = 48;
    if (spec.children.length !== cells) spec.innerHTML = "<i></i>".repeat(cells);
    let i = 0;
    for (const [, bytes, colour] of p.parts) {
      const n = Math.max(1, Math.round(bytes / p.bytes * cells));
      for (let k = 0; k < n && i < cells; k++, i++) {
        const el = spec.children[i];
        el.style.background = colour;
        el.style.height = (30 + Math.random() * 12 + 50 * (bytes / p.bytes)) + "%";
      }
    }
  }

  if (screen === "build") refreshBuild(p, ref);
  if (screen === "monitor") refreshMonitor();
}

/* What this build is ACTUALLY made of.
   The .tensor-types.txt sidecar records the recipe a build was GIVEN, and it is often absent --
   it is not written for every lane and it does not survive being moved. But the types are in the
   file itself, counted when the header was read, so "not recorded" was never the right answer:
   the recipe is a nice-to-have, what the file actually contains is the truth. */
function renderRecipe() {
  const box = $("#b-recipe");
  if (!box) return;
  const b = curBuild() || refBuild() || {};
  const types = b.types || {};
  const names = Object.keys(types);
  if (!names.length) {
    box.innerHTML = `<div class="hint">No build selected, or its header could not be read.</div>`;
    return;
  }
  const total = names.reduce((n, k) => n + types[k], 0);
  const ranked = names.slice().sort((a, c) => types[c] - types[a]);
  const rec = b.recipe;
  box.innerHTML =
    `<div class="hint" style="margin-bottom:8px">${total} tensors across ${names.length}
       type${names.length === 1 ? "" : "s"}, counted in the file itself.</div>`
    + ranked.slice(0, 7).map(t => {
        const pct = types[t] / total * 100;
        const high = /^(F32|F16|BF16|Q8)/.test(t);
        return `<div class="cmp"><span class="nm mono">${esc(t)}</span>
          <span class="num">${types[t]}</span>
          <span class="pc" style="color:var(--ink-lo)">${pct.toFixed(0)}%</span>
          <span class="tr"><i style="width:${pct.toFixed(1)}%;background:${
            high ? "var(--gold)" : "var(--trim-lo)"}"></i></span></div>`;
      }).join("")
    + (ranked.length > 7 ? `<div class="hint" style="margin-top:6px">...and ${ranked.length - 7}
        more</div>` : "")
    + (rec ? `<div class="hint" style="margin-top:9px">Recipe it was given:
        <b>${esc(rec.path.split("/").pop())}</b>, ${rec.rules.length} rules.</div>` : "");
}

function refreshBuild(p, ref) {
  /* The bpw readouts are a property of the GGUF atom table; on another lane the fields are
     different knobs entirely and these elements are not on the page. */
  const bpw = (id, table, key) => {
    const el = $("#" + id);
    if (el) el.textContent = table[key] != null ? table[key].toFixed(2) + " bpw" : "";
  };
  bpw("l-body", ATOM, R.body); bpw("l-prot", ATOM, R.protect); bpw("l-emb", EMB, R.emb);

  if (!p) {
    $("#b-lcd").innerHTML = "<span class='dim'>no reference build to measure against</span>";
    $("#b-metrics").innerHTML = ""; $("#b-ref").innerHTML = ""; $("#b-cmp").innerHTML = "";
  } else {
    $("#b-fit").innerHTML = p.fits ? chip("FITS") : chip("OVER BUDGET");
    $("#b-lcd").innerHTML =
      `<span class="dim">ref</span>     ${p.ref.tag}  ${gb(p.ref.bytes)} GB  `
      + `${(p.params / 1e9).toFixed(2)}B params  ${p.ref.blocks || "?"} blk<br>`
      + `<span class="dim">target</span>  RAM ${R.ram}.0  reserve ${R.reserve}<br>`
      + `<span class="dim">FIT</span>     <b>${p.gb.toFixed(2)} / ${R.ram}.0 GB</b>  `
      + `bpw ${p.bpw.toFixed(2)}  ${(p.ref.bytes / p.bytes).toFixed(1)}x vs ref<br>`
      + `<span class="dim">body</span>    ${R.body}   <span class="dim">protect</span> ${R.protect}`
      + `   <span class="dim">emb</span> ${R.emb}<br>`
      + `<span class="dim">runtime</span> ${p.runtime}`
      + (p.ref.mtp_block != null ? `   <span class="dim">mtp</span> blk.${p.ref.mtp_block} pinned` : "")
      + `<br><span class="dim">VERDICT</span> <b>${p.fits ? "FITS BUDGET" : "OVER BUDGET"}</b>`;

    $("#b-metrics").innerHTML = [
      [ic("box"), p.gb.toFixed(2) + " GB", "Projected size", "from measured params"],
      [ic("wave"), p.bpw.toFixed(2), "Bits / weight", "average"],
      [ic("layers"), (p.params / 1e9).toFixed(2) + "B", "Parameters", "counted in the file"],
      [ic("scale"), (p.ref.bytes / p.bytes).toFixed(1) + "x", "vs reference", p.ref.tag],
    ].map(([i, v, k, s], n) => `<div class="metric" ${n ? 'style="padding-left:15px"' : ""}>${i}
        <div><div class="mv">${v}</div><div class="mk">${k}</div><div class="ms">${s}</div></div></div>`).join("");

    $("#b-ref").innerHTML = Object.entries(ref.groups || {})
      .sort((a, b) => b[1].params - a[1].params)
      .map(([g, v]) => `<div class="kv"><span class="k">${g}</span>
        <span class="v">${(v.params / 1e9).toFixed(2)}B · ${v.bpw.toFixed(2)} bpw</span></div>`).join("")
      + `<div class="kv"><span class="k">file</span><span class="v gold">${gb(ref.bytes)} GB</span></div>`
      + `<div class="kv"><span class="k">measured bpw</span><span class="v gold">${
          ref.bpw ? ref.bpw.toFixed(2) : "?"}</span></div>`;

    $("#b-cmp").innerHTML = p.parts.map(([g, bytes, colour, params]) => `
      <div class="cmp"><span class="nm">${g}</span>
        <span class="num">${(bytes / 1e9).toFixed(2)} GB</span>
        <span class="pc" style="color:#6d6b64">${(bytes / p.bytes * 100).toFixed(0)}%</span>
        <span class="tr"><i style="width:${(bytes / p.bytes * 100).toFixed(1)}%;background:${colour}"></i></span>
      </div>`).join("");
  }

  renderRecipe();

  updateCmd();
}

async function updateCmd() {
  if (!bridge()) { $("#b-cmd").textContent = "(bridge not connected)"; return; }
  const p = await bridge().plan("build", R);
  const el = $("#b-cmd"); if (!el) return;
  el.innerHTML = p.ok
    ? esc(p.display.split(" ").map(t => (/[\\/]/.test(t) ? short(t) : t)).join(" "))
        .replace(/(--[a-z0-9-]+)/g, '<span class="fl">$1</span>')
    : `<span class="err">${p.error || "nothing to run"}</span>`;
  const t = $("#b-tool"); if (t) t.textContent = p.ok ? (p.confirm ? "writes output" : "read-only") : "";
}

/* ── other screens ──────────────────────────────────────────────────────── */
function refreshMonitor() {
  const st = S.status || {};
  const el = $("#mon-state");
  if (el) el.innerHTML = st.running ? chip("RUNNING") + ` <span class="hintr">${st.elapsed}s</span>`
                                    : chip(st.returncode === 0 ? "COMPLETED"
                                         : st.returncode != null ? "FAILED" : "IDLE");
}

function wireMonitor() {
  const bs = builds();
  $("#mon-lanes").innerHTML = LANES.map(l => {
    const has = l === "GGUF" && bs.length;
    return `<div class="card"><h3 style="margin-bottom:8px">${l}
        <span class="rt">${chip(has ? "COMPLETED" : "IDLE")}</span></h3>
      <div class="big" style="font-size:22px">${has ? bs.length : 0}<small>builds</small></div>
      <div class="sp"></div>
      <div class="kv"><span class="k">Largest</span><span class="v">${
        has ? gb(Math.max(...bs.map(b => b.bytes))) + " GB" : "—"}</span></div>
      <div class="kv"><span class="k">Emitter</span><span class="v">${
        { GGUF: "pollard-fit", GPTQ: "pollard-gptq", MLX: "pollard-mlx",
          EXL3: "pollard-exl3", MX: "pollard-mx" }[l]}</span></div>
    </div>`;
  }).join("");
  $("#mon-chart").innerHTML = barchart(
    bs.map(b => ({ name: b.tag.slice(0, 8), a: b.bytes / 1e9 })), { unit: "", ylab: "GB" });
  $("#mon-scan").textContent = S.scanned || "";
  $("#mon-mf").innerHTML = bs.map(b => `<tr><td class="fi">${ic("file")}<span class="w">${b.name}</span></td>
    <td class="m">${gb(b.bytes)} GB</td><td class="m">${b.bpw ? b.bpw.toFixed(2) : "—"}</td>
    <td>${b.modified}</td></tr>`).join("") || `<tr><td colspan="4">${NM}</td></tr>`;
  refreshMonitor();
}

function wireLadder() {
  const bs = [...builds()].sort((a, b) => a.bytes - b.bytes);
  $("#lad-cards").innerHTML = bs.slice(0, 4).map(b => `
    <div class="card"><h3 style="margin-bottom:9px">${b.tag}
      <span class="rt">${chip(b.verified ? "VERIFIED" : "UNVERIFIED")}</span></h3>
      <div class="big" style="font-size:26px">${gb(b.bytes)}<small>GB</small></div>
      <div class="sp"></div>
      <div class="kv"><span class="k">Lane</span><span class="v">${b.lane || "GGUF"}${
        b.shards > 1 ? ` · ${b.shards} shards` : ""}</span></div>
      <div class="kv"><span class="k">Arch</span><span class="v">${b.architecture || "?"}</span></div>
      <div class="kv"><span class="k">BPW</span><span class="v gold">${b.bpw ? b.bpw.toFixed(2) : "?"}</span></div>
      <div class="kv"><span class="k">PPL</span><span class="v">${b.ppl ? b.ppl.toFixed(2) : NM}</span></div>
    </div>`).join("");
  $("#lad-chart").innerHTML = barchart(bs.filter(b => b.bpw).map(
    b => ({ name: b.tag.slice(0, 8), a: b.bpw })), { ylab: "bits / weight" });
  $("#lad-rows").innerHTML = bs.map(b => `<tr><td class="m">${b.tag}</td>
    <td><span class="chip ${b.kind === "gguf" ? "run" : "dim"}">${b.lane || "GGUF"}</span></td>
    <td>${b.architecture || "?"}</td><td class="m">${gb(b.bytes)} GB</td>
    <td class="m">${b.bpw ? b.bpw.toFixed(2) : "—"}</td>
    <td class="m">${b.ppl ? b.ppl.toFixed(2) : NM}</td><td>${b.modified}</td></tr>`).join("");
}

function wireTrain() {
  /* Every one of these was bindSlider(id) with no key, so the solver ran on its own defaults no
     matter what the screen showed. The selects were not bound at all. */
  [["tr-bits", "bits"], ["tr-ns", "nsamples"], ["tr-seq", "seqlen"],
   ["tr-head", "headBits"], ["tr-emb", "embedBits"],
   ["tr-evchunks", "evalChunks"], ["tr-threads", "threads"]]
    .forEach(([id, key]) => bindSlider(id, key));
  [["tr-src", "source"], ["tr-method", "method"], ["tr-group", "groupsize"],
   ["tr-dev", "device"], ["tr-qmode", "qmode"], ["tr-recipe", "recipe"],
   ["tr-ablate", "ablate"]].forEach(([id, key]) => bindSel(id, key));
  bindToggles([["tr-resume", "resume"], ["tr-offload", "offload"]]);
  $("#tr-chart").innerHTML = empty("no training run recorded — start one and the loss appears here");
}

function wireBench() {
  bindSlider("bn-chunks", "chunks"); bindSlider("bn-ngl", "ngl");
  bindSel("bn-gguf", "gguf"); bindSel("bn-vs", "vs");
  bindToggles([["bn-speed", "speed"], ["bn-coh", "coherence"], ["bn-quick", "quick"]]);
  bindSlider("kv-ctx", "kvCtx"); bindSlider("tc-rate", "minRate"); bindSel("tc-ref", "ref");
  /* The KV sweep's corpus used to write R.evalFile, the same key the eval screen sets, so
     whichever screen you touched last silently decided both. It has its own key now. */
  $("#bn-chart").innerHTML = barchart(
    builds().map(b => ({ name: b.tag.slice(0, 8), a: b.bytes / 1e9 })), { ylab: "GB" });
}

function wireEval() {
  bindSlider("ev-gt", "genTokens"); bindSel("ev-q", "gguf"); bindSel("ev-ref", "ref");
  // these two drove nothing: the run always took the action's defaults
  bindToggles([["ev-traj", "trajectory"], ["ev-chart", "chart"]]);
  bindSlider("rc-budget", "swapBudget"); bindSel("rc-ref", "ref");
  // bring your own benchmark
  bindSel("pb-gguf", "gguf"); bindSel("pb-fmt", "probeFormat");
  bindSlider("pb-tasks", "probeTasks");
  bindSel("te-suite", "suite"); bindSlider("te-limit", "limit");
  const withPpl = builds().filter(b => b.ppl);
  $("#ev-chart").innerHTML = withPpl.length
    ? linechart([{ data: withPpl.map(b => b.ppl), colour: "#a9d18d", w: 2 }],
                { ylab: "perplexity", xlab: "build" })
    : empty("no perplexity recorded for these builds");
  $("#ev-note").innerHTML = withPpl.length ? "" :
    `<div class="hint">None of the builds in this workspace have a perplexity recorded in their
      MANIFEST. Run the eval and it lands here — Studio does not carry a number it did not see.</div>`;
}

async function wireChat() {
  bindSel("c-ckpt", "gguf");
  await renderRuntimes();
  await renderModalities();
  bindSlider("c-max"); bindSlider("c-temp");
  $("#c-in").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });
  bindSlider("art-vol");
  const v = $("#art-vol");
  if (v) v.addEventListener("input", () => {
    const el = $("#art-el");
    if (el) el.volume = (+v.value || 0) / 100;      // live, so it is usable while something plays
  });
  if (!$("#c-log").children.length) {
    $("#c-log").innerHTML = `<div class="msg"><div class="av">${ic("bear")}</div>
      <div><div class="who">Pollard</div><div class="tx">Pick a build and ask it something. Every
        reply is checked for the four ways a low-bit build fails: repetitive loops, never halting,
        committing to an answer far too late, and leaving a reasoning block unclosed.</div></div></div>`;
  }
}

async function renderModalities() {
  const box = $("#mod-grid");
  if (!box) return;
  const d = bridge() ? await bridge().modalities(R.gguf || "") : previewModalities();
  $("#mod-kind").textContent = d.kind || "";
  box.innerHTML = Object.entries(d.modalities).map(([k, on]) => {
    const L = (d.labels || {})[k] || { label: k, why: "" };
    return `<div class="modcard ${on ? "on" : "off"}">
      <div class="mh">${ic(on ? "check" : "alert")}<b>${L.label}</b>
        <span class="chip ${on ? "ok" : "dim"}">${on ? "PRESENT" : "not in this build"}</span></div>
      <div class="mw">${L.why}</div>
      ${on ? `<div class="btns"><button class="btn" onclick="act('${MOD_ACTION[k] || "eval"}')">
          ${MOD_LABEL[k] || "CHECK"}</button></div>` : ""}
    </div>`;
  }).join("") + (d.notes || []).map(n => `<div class="modnote">${ic("info")}${esc(n)}</div>`).join("");
}

/* Without the bridge (preview/screenshot), infer from what the workspace scan already read.
   Less thorough than modalities.py -- it cannot see sibling mmproj files or a VAE folder -- so it
   is only ever used when there is no Python side to ask. */
function previewModalities() {
  const g = (refBuild() || {}).groups || {};
  const on = k => (g[k] || {}).params > 0;
  return {
    kind: (curBuild() || {}).kind || "unknown",
    modalities: { text: true, vision_in: on("vision") || on("projector"), audio_in: on("audio"),
                  video_in: false, image_out: false, audio_out: false },
    labels: { text: { label: "Text", why: "does it still answer coherently, without looping" },
              vision_in: { label: "Vision in", why: "can it still resolve colour, count, shape and position" },
              audio_in: { label: "Audio in", why: "can it still transcribe and follow spoken input" },
              video_in: { label: "Video in", why: "does it still track content across frames" },
              image_out: { label: "Image out", why: "does it still draw structure, or only noise" },
              audio_out: { label: "Speech out", why: "does it still speak, or output silence or a buzz" } },
    notes: ["preview: read from the workspace scan, not from the model's own config"],
  };
}

/* each modality routes to the Pollard tool that owns that question */
const MOD_ACTION = { text: "eval", vision_in: "mmeval", audio_in: "mmeval",
                     video_in: "mmeval", image_out: "taskeval", audio_out: "taskeval" };
const MOD_LABEL = { text: "TEXT GATE", vision_in: "pollard-mmeval", audio_in: "pollard-mmeval",
                    video_in: "pollard-mmeval", image_out: "pollard-taskeval",
                    audio_out: "pollard-taskeval" };

/* A verdict about a picture is not the same as looking at it. For a model that draws, speaks or
   renders video the only check that settles it is your own eyes and ears, so the file is played
   here rather than merely graded. */
async function checkArtifact() {
  const path = ($("#art-path") || {}).value;
  if (!path || !bridge()) { toast("give it a file first"); return; }
  const m = await bridge().load_media(path);
  const player = $("#art-player"), result = $("#art-result");
  if (!m.ok) {
    player.innerHTML = "";
    result.innerHTML = `<div class="verdict bad">${ic("alert")}<span>${esc(m.reason)}</span></div>`;
    return;
  }
  const vol = (+($("#art-vol") || {}).value || 80) / 100;
  player.innerHTML =
    m.kind === "image" ? `<div class="mediabox"><img src="${m.data}" alt="${esc(m.name)}"></div>`
    : m.kind === "audio" ? `<div class="mediabox"><audio id="art-el" controls src="${m.data}"></audio></div>`
    : `<div class="mediabox"><video id="art-el" controls src="${m.data}"></video></div>`;
  const el = $("#art-el");
  if (el) el.volume = vol;

  const c = m.check;
  result.innerHTML =
    `<div class="kv"><span class="k">file</span>
       <span class="v">${esc(m.name)} &middot; ${(m.bytes / 1e6).toFixed(2)} MB &middot; ${m.mime}</span></div>`
    + (c ? `<div class="verdict ${c.ok ? "ok" : "bad"}">${ic(c.ok ? "check" : "alert")}
         <span>${esc(c.reason || "")}</span></div>`
         + Object.entries(c).filter(([k]) => !["ok", "reason"].includes(k)).map(([k, v]) =>
             `<div class="kv"><span class="k">${k.replace(/_/g, " ")}</span>
               <span class="v">${v}</span></div>`).join("")
       : `<div class="hint">Played, not graded — the structural checks cover images and WAV.
            For video, watch it.</div>`);
}

function stopMedia() {
  const el = $("#art-el");
  if (el) { el.pause(); el.currentTime = 0; }
}

async function renderRuntimes() {
  const el = $("#c-rt");
  if (!el || !bridge()) return;
  const [info, ranked] = await Promise.all([
    bridge().runtimes(),
    bridge().runtimes_for(R.gguf || "", (curBuild() || {}).tag),
  ]);
  const by = Object.fromEntries(info.available.map(r => [r.name, r]));
  // best-for-this-build first, then everything else, so the right one is already selected
  const order = [...ranked, ...info.available.map(r => r.name).filter(n => !ranked.includes(n))];
  el.innerHTML = order.map(n => {
    const r = by[n] || {};
    return `<option value="${escAttr(n)}" ${n === info.current ? "selected" : ""}>${esc(n)}${
      r.ready ? "" : " — not installed"}</option>`;
  }).join("");
  const note = () => {
    const r = by[el.value] || {};
    $("#c-rtnote").innerHTML = `${r.note || ""}${r.ready ? "" :
      ` <span style="color:var(--warn)">· ${esc(r.why || "")}</span>`}`;
  };
  note();
  el.onchange = async () => {
    const res = await bridge().use_runtime(el.value, R.ngl || 0);
    if (!res.ok) { toast(res.error); return; }
    note();
    toast("runtime: " + el.value);
  };
}

function pushMsg(who, text, cls) {
  $("#c-log").insertAdjacentHTML("beforeend",
    `<div class="msg ${cls || ""}"><div class="av">${ic(who === "You" ? "chat" : "bear")}</div>
      <div><div class="who">${who}</div><div class="tx">${esc(text)}</div></div></div>`);
  $("#c-log").scrollTop = $("#c-log").scrollHeight;
}

async function sendChat() {
  const box = $("#c-in");
  if (!box || !box.value.trim() || !bridge()) return;
  if (!R.gguf) { toast("pick a build first"); return; }
  const q = box.value.trim();
  box.value = "";
  pushMsg("You", q, "u");
  $("#c-state").innerHTML = chip("RUNNING");
  const r = await bridge().chat(R.gguf, q, +($("#c-max") || {}).value || 256,
                                (+($("#c-temp") || {}).value || 7) / 10);
  $("#c-state").innerHTML = chip("IDLE");
  if (!r.ok) { pushMsg("Pollard", r.error, ""); return; }
  pushMsg("Pollard", r.text || "(empty)", "");
  const c = r.coherence || {};
  $("#c-log").insertAdjacentHTML("beforeend",
    `<div class="verdict ${c.ok ? "ok" : "bad"}">${ic(c.ok ? "check" : "alert")}
       <span>${esc(c.verdict || "")}${c.words ? ` · ${c.words} words` : ""}</span></div>`);
  $("#c-log").scrollTop = $("#c-log").scrollHeight;
}

async function coherenceGate() {
  if (!bridge()) return;
  if (!R.gguf) { toast("pick a build first"); return; }
  $("#c-gate").innerHTML = `<div class="hint">generating…</div>`;
  const g = await bridge().coherence_gate(R.gguf, null, 192);
  $("#c-gate").innerHTML = `<div class="verdict ${g.pass ? "ok" : "bad"}">
      ${ic(g.pass ? "check" : "alert")}<span>${g.pass ? "PASS" : "FAIL"} — ${esc(g.reason || "")}</span></div>`
    + (g.n ? `<div class="kv"><span class="k">Coherent</span>
        <span class="v ${g.pass ? "ok" : "warn"}">${Math.round((g.rate || 0) * 100)}% of ${g.n}</span></div>` : "");
}

function wireDoctor() {
  advCheck("dpack", "pollard_pack");
  const p = project(), ref = refBuild();
  $("#doc-tiles").innerHTML = [
    ["Workspace", S.home_exists, short(S.home, 1)],
    ["Repo", S.repo_exists, short(S.repo, 1)],
    ["Builds found", builds().length > 0, `${builds().length} in ${(S.model || {}).name || "—"}`],
    ["Reference build", !!ref, ref ? `${ref.tag} · ${gb(ref.bytes)} GB` : "none"],
    ["Recipe sidecar", !!(ref && ref.recipe), ref && ref.recipe
      ? `${ref.recipe.rules.length} rules` : "not recorded"],
    ["Runtime", p ? p.runtime === "stock" : false,
      p ? (p.runtime === "stock" ? "loads anywhere" : "needs ik_llama") : "—"],
  ].map(([k, ok, v]) => `<div class="tile ${ok ? "ok" : "warn"}">
      <div class="th">${ic(ok ? "check" : "alert")}${k}</div>
      <div class="tv">${ok ? "OK" : "Attention"}</div><div class="ts">${v}</div></div>`).join("");

  $("#doc-gauges").innerHTML = p
    ? gauge(p.gb / R.ram, "FIT", `${p.gb.toFixed(1)}/${R.ram}G`)
      + gauge(p.bpw / 8, "BPW", p.bpw.toFixed(2))
      + gauge(clamp(builds().length / 6, 0, 1), "BUILDS", builds().length)
      + gauge(R.reserve / 16, "RESERVE", R.reserve + "G")
    : empty("no reference build");
  $("#doc-rulers").innerHTML =
      ruler("RAM TARGET", R.ram + " GB", R.ram / 128, ["4", "32", "64", "128"])
    + ruler("RESERVE", R.reserve + " GB", R.reserve / 16, ["0", "4", "8", "16"])
    + ruler("GPU LAYERS", String(R.ngl), R.ngl / 99, ["0", "33", "66", "99"]);
  bindSel("fm-ref", "ref"); bindSel("fm-model", "gguf");
  const fc = $("#fm-calib");
  if (fc) fc.addEventListener("input", () => { R.calibFile = fc.value; });
  $("#doc-hw").innerHTML = [
    ["POLLARD_HOME", short(S.home, 1)], ["Repo", short(S.repo, 1)], ["Pollard version", S.version],
    ["Models", String(S.models.length)],
    ["Architecture", (ref || {}).architecture || "—"],
    ["Blocks", String((ref || {}).blocks || "—")],
    ["Downloads", String(S.downloads.length)],
  ].map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}

/* ── tools ──────────────────────────────────────────────────────────────── */
let toolSel = 0;

function wireTools() {
  $("#tool-count").textContent =
    `${S.tools.length} tools · ${S.tools.reduce((n, t) => n + t.flags.length, 0)} flags · parsed from source`;
  $("#tool-search").addEventListener("input", renderToolList);
  renderToolList();
}

function renderToolList() {
  const q = ($("#tool-search")?.value || "").toLowerCase();
  const hit = S.tools.filter(t => !q || t.module.includes(q)
    || (t.summary || "").toLowerCase().includes(q)
    || t.flags.some(f => f.names.join(" ").toLowerCase().includes(q)));
  $("#tool-list").innerHTML = hit.length ? hit.map(t => `
    <div class="titem ${S.tools.indexOf(t) === toolSel ? "on" : ""}" onclick="pickTool(${S.tools.indexOf(t)})">
      <div class="n">${t.module.replace("pollard_", "pollard-")}</div>
      <div class="s">${t.summary || "&nbsp;"}</div></div>`).join("")
    : `<div class="hint" style="padding:10px">no tool matches &ldquo;${q}&rdquo;</div>`;
  renderToolDetail();
}
function pickTool(i) { toolSel = i; renderToolList(); }

function renderToolDetail() {
  const t = S.tools[toolSel], box = $("#tool-detail");
  if (!box) return;
  if (!t) { box.innerHTML = `<div class="hint" style="padding:10px">No manifest — is
      <b>${short(S.repo, 1)}</b> present?</div>`; return; }
  const seen = new Set();
  box.innerHTML = `<div class="toolname">${t.module.replace("pollard_", "pollard-")}</div>
    <div class="hint" style="margin:4px 0 12px">${t.summary || ""}</div>
    ${t.flags.filter(f => { const k = f.names.join(); if (seen.has(k)) return false;
                            seen.add(k); return true; }).map((f, i) => {
      const id = `tf-${i}`, flag = f.names[0];
      let ctl;
      if (f.is_flag) ctl = `<div class="tswitch${f.default ? " on" : ""}" id="${id}"
          data-flag="${flag}" data-kind="switch" onclick="this.classList.toggle('on');toolCheck()"></div>`;
      else if (f.choices) ctl = `<div class="selwrap"><select id="${id}" data-flag="${flag}"
          data-kind="value" onchange="toolCheck()"><option value=""></option>${f.choices.map(c =>
          `<option ${c === f.default ? "selected" : ""}>${c}</option>`).join("")}</select>${ic("chev")}</div>`;
      else ctl = `<input type="text" id="${id}" data-flag="${flag}" data-kind="value"
          value="${escAttr(f.default ?? "")}" placeholder="${escAttr(f.type || "value")}" oninput="toolCheck()">`;
      return `<div class="flag"><div><div class="fn">${f.names.join(", ")}</div>
        ${f.required ? '<div class="rq">REQUIRED</div>' : ""}</div>
        <div><div class="fh">${f.help || ""}</div>${ctl}</div></div>`;
    }).join("")}
    <div class="sp"></div>
    <div class="cmd" id="tool-cmd">…</div>
    <div id="tool-problems"></div>
    <div class="sp"></div>
    <div class="btns">
      <button class="btn gold" id="tool-run" onclick="runTool()">${ic("play")}RUN</button>
      <button class="btn" onclick="act('help:${t.module}')">--help</button>
      <button class="btn bad" onclick="abortRun()">ABORT</button>
    </div>
    <div class="hint" style="margin-top:8px">Every value is checked against this tool's own
      argparse declaration before anything starts — type, choices, required, and whether a path
      that must exist does.</div>`;
  toolCheck();
}

function toolValues() {
  const out = {};
  $$("#tool-detail [data-flag]").forEach(el => {
    const flag = el.dataset.flag;
    if (el.dataset.kind === "switch") { if (el.classList.contains("on")) out[flag] = true; }
    else if (el.value !== "") out[flag] = el.value;
  });
  return out;
}

let toolCheckPending = null;
function toolCheck() {
  clearTimeout(toolCheckPending);
  toolCheckPending = setTimeout(async () => {
    const t = S.tools[toolSel];
    if (!t || !bridge() || !$("#tool-cmd")) return;
    const v = await bridge().check_tool(t.module, toolValues());
    $("#tool-cmd").innerHTML = esc(v.display.split(" ")
        .map(t => (/[\\/]/.test(t) ? short(t) : t)).join(" "))
      .replace(/(--[a-z0-9-]+)/g, '<span class="fl">$1</span>');
    $("#tool-problems").innerHTML = v.ok ? "" :
      `<div class="problems">${v.problems.map(x => `<div>${ic("alert")}${esc(x)}</div>`).join("")}</div>`;
    const btn = $("#tool-run");
    if (btn) { btn.disabled = !v.ok; btn.classList.toggle("off", !v.ok); }
  }, 180);
}

async function runTool() {
  const t = S.tools[toolSel];
  if (!t || !bridge()) return;
  const v = await bridge().check_tool(t.module, toolValues());
  if (!v.ok) { toast(v.problems.join("; ")); return; }
  if (!confirm(`Run this?\n\n${v.display}`)) return;
  const r = await bridge().run_tool(t.module, toolValues());
  if (!r.ok) { toast(r.error); return; }
  logSeq = 0; logLines = [];
  startPolling();
  toast("running " + t.module.replace("pollard_", "pollard-"));
}

/* ── run control ────────────────────────────────────────────────────────── */
const bridge = () => window.pywebview && window.pywebview.api;
let logSeq = 0, logLines = [], poller = null;

async function act(action) {
  if (!bridge()) { console.log("act", action, R); return; }
  const p = await bridge().plan(action, R);
  if (!p.ok) { toast(p.error || "nothing to run"); return; }
  if (p.confirm && !confirm(`Run this?\n\n${p.display}`)) return;
  const res = await bridge().start(action, R);
  if (!res.ok) { toast(res.error); return; }
  logSeq = 0; logLines = [];
  if (screen !== "monitor" && !$(".log")) show("monitor");
  startPolling();
}

async function previewOnly(action) {
  if (!bridge()) return;
  const p = await bridge().plan(action, R);
  toast(p.ok ? p.display : (p.error || "nothing to run"));
}

async function abortRun() {
  if (!bridge()) return;
  const r = await bridge().abort();
  toast(r.ok ? "aborted" : (r.error || "nothing running"));
}

function clearLog() { logLines = []; renderLog(); }

let cvTarget = "MLX";

function wireAdvanced() {
  const n = ADV_SECTIONS.reduce((a, s) => a + s.tools.length, 0);
  const flags = ADV_SECTIONS.reduce((a, s) => a + s.tools.reduce(
    (b, m) => b + ((toolOf(m) || { flags: [] }).flags.length), 0), 0);
  const el = $("#adv-count");
  if (el) el.textContent = `${n} tools - ${flags} flags - generated from the tools`;
  ADV_SECTIONS.forEach((s, si) => s.tools.forEach((m, ti) => advCheck(`adv${si}_${ti}`, m)));
}

/* Any screen built out of toolBlock()s needs its command lines primed, or they sit on "..."
   until the user happens to touch a field. */
function wireBlocks(pairs) { pairs.forEach(([prefix, mod]) => advCheck(prefix, mod)); }

function wireBrains() {
  wireBlocks(BRAIN_STEPS.map((s, i) => [`br${i}`, s.tool]));
}

function wirePublish() {
  wireBlocks(PUBLISH_TOOLS.map((m, i) => [`pb${i}`, m]));
}

async function wireConvert() {
  bindSel("cv-from", "gguf");
  bindSlider("cv-ram", "ram");
  bindSlider("cv-res", "reserve");
  bindToggles([["cv-reuse", "reuseProfile"], ["cv-planonly", "convertPlanOnly"]]);
  $("#cv-from").addEventListener("change", () => { renderQuality(); planConvert(); });
  $("#cv-profiler").addEventListener("change", planConvert);
  renderLaneCards();
  await renderQuality();
  await planConvert();
  await renderWillItRun();
}

/* What each lane is FOR. Static facts, so the picker is never blank waiting on a call --
   a screen with nothing to click is worse than one showing slightly stale text. */
const LANE_FACTS = {
  GGUF: { runs: "llama.cpp · ik_llama · Ollama · LM Studio",
          best: "runs anywhere, CPU included", atom: "K-quants + IQ / trellis",
          note: "the widest reach. Trellis atoms need the ik_llama fork." },
  MLX:  { runs: "mlx_lm on Apple silicon", best: "Macs, unified memory",
          atom: "group-wise int", note: "fastest path on an M-series machine." },
  MX:   { runs: "vLLM · TensorRT-LLM", best: "Blackwell and newer",
          atom: "NVFP4 / MXFP4", note: "FP4 with hardware support; FP8 protect tier." },
  GPTQ: { runs: "vLLM · SGLang · transformers", best: "server GPUs",
          atom: "int4 / int8 packed", note: "full-Hessian error feedback, Pollard's own solver." },
  EXL3: { runs: "exllamav3", best: "single consumer GPU, any target bpw",
          atom: "trellis (QTIP-class)", note: "heavy to build; any average bitrate 1-8." },
  VLLM: { runs: "vLLM · SGLang", best: "serving at scale",
          atom: "compressed-tensors", note: "emitted from the same measured profile." },
};

function renderLaneCards() {
  const box = $("#cv-lanes");
  if (!box) return;
  box.innerHTML = Object.entries(LANE_FACTS).map(([name, f]) => `
    <button class="lanecard ${name === cvTarget ? "on" : ""}" data-lane="${name}"
            onclick="pickLane('${name}')">
      <div class="ln">${name}</div>
      <div class="lr">${f.runs}</div>
      <div class="lw"><b>${f.best}</b> · ${f.atom}</div>
      <div class="lw">${f.note}</div></button>`).join("");
}

function pickLane(name) {
  cvTarget = name;
  $$("#cv-lanes .lanecard").forEach(b => b.classList.toggle("on", b.dataset.lane === name));
  planConvert();
  renderWillItRun();
}

/* The question underneath every lane change: does the result actually run on the machine it is
   for? Size is only half of it -- a GGUF carrying trellis atoms loads in ik_llama and nowhere
   else, and finding that out after a six-hour build is the pain this is here to remove. */
async function renderWillItRun() {
  const box = $("#cv-runs");
  if (!box) return;
  const p = project();
  const from = curBuild() || refBuild() || {};
  const f = LANE_FACTS[cvTarget] || {};
  const ram = +R.ram || 16, res = +R.reserve || 0;
  const gbNow = (from.bytes || 0) / 1e9;
  const gb = p ? p.gb : gbNow;
  const fits = gb + res <= ram;
  const headroom = ram - res - gb;

  let rts = [];
  if (bridge()) {
    try { rts = await bridge().runtimes_for(from.path || "", from.tag || ""); } catch (e) {}
  }
  const chip2 = $("#cv-fit");
  if (chip2) chip2.innerHTML = fits ? chip("FITS") : chip("OVER BUDGET");

  box.innerHTML = `
    <div class="kv"><span class="k">Projected size on ${cvTarget}</span>
      <span class="v ${fits ? "ok" : "warn"}">${gb.toFixed(2)} GB</span></div>
    <div class="kv"><span class="k">Headroom after reserve</span>
      <span class="v ${headroom > 0 ? "ok" : "warn"}">${headroom.toFixed(2)} GB</span></div>
    <div class="kv"><span class="k">Runs on</span><span class="v">${esc(f.runs || "—")}</span></div>
    <div class="kv"><span class="k">Atoms</span><span class="v">${esc(f.atom || "—")}</span></div>
    ${rts.length ? `<div class="kv"><span class="k">Detected here</span>
      <span class="v ${rts.length ? "ok" : ""}">${esc(rts.slice(0, 3).join(", "))}</span></div>` : ""}
    <div class="hint" style="margin-top:8px">${esc(f.note || "")}
      ${fits ? "" : " This will not fit the budget above — drop a rung, or raise the target."}</div>`;
}

async function runRoute() {
  if (!bridge()) return;
  const from = ($("#cv-from") || {}).value || R.gguf;
  const p = await bridge().convert_plan(from, cvTarget);
  if (!p.ok) { toast(p.guidance); return; }
  const blocking = p.steps.find(s => s.blocking);
  if (blocking) {
    toast(`step 1 needs doing first: ${blocking.tool.replace("pollard_", "pollard-")}`);
    return;
  }
  const first = p.steps[0];
  if (!confirm(`Run the route to ${cvTarget}?\n\nStarting with:\n` +
               `${first.tool.replace("pollard_", "pollard-")} ${first.args.join(" ")}\n\n` +
               `${p.steps.length} step(s). Each is run one at a time.`)) return;
  const r = await bridge().run_tool(first.tool, argvToValues(first.args));
  if (!r.ok) { toast(r.error); return; }
  logSeq = 0; logLines = []; startPolling();
}

/* the planner hands back argv; run_tool validates {flag: value} */
function argvToValues(args) {
  const out = {};
  for (let i = 0; i < args.length; i++) {
    const a = String(args[i]);
    if (!a.startsWith("-")) continue;
    const nxt = args[i + 1];
    if (nxt !== undefined && !String(nxt).startsWith("-")) { out[a] = nxt; i++; }
    else out[a] = true;
  }
  return out;
}

async function convertAllLanes() {
  if (!bridge()) return;
  const from = ($("#cv-from") || {}).value || R.gguf;
  const box = $("#cv-route");
  box.innerHTML = `<div class="hint">planning…</div>`;
  const names = Object.keys(LANE_FACTS);
  const plans = await Promise.all(names.map(n => bridge().convert_plan(from, n)));
  box.innerHTML = `<div class="hint" style="margin-bottom:9px">Every lane from this source.
      The allocation is measured once and re-emitted, so these are not six separate
      quantizations of each other.</div>`
    + plans.map((p, i) => {
        const blocked = (p.steps || []).some(s => s.blocking);
        return `<div class="cmp"><span class="nm"><b>${names[i]}</b> —
            ${esc((LANE_FACTS[names[i]] || {}).best || "")}</span>
          <span class="num">${p.ok ? p.steps.length + " step" + (p.steps.length === 1 ? "" : "s") : "—"}</span>
          <span class="pc">${blocked ? chip("NEEDS SOURCE") : chip("READY")}</span></div>`;
      }).join("");
  $("#cv-count").textContent = "all lanes";
}

async function renderQuality() {
  const box = $("#cv-quality");
  if (!box || !bridge()) return;
  const from = ($("#cv-from") || {}).value || R.gguf;
  const p = await bridge().convert_plan(from, cvTarget);
  const q = p.source_quality || {};
  const good = ["ideal", "excellent"].includes(q.tier);
  box.innerHTML = q.bpw == null ? "" : `
    <div class="verdict ${good ? "ok" : q.tier === "usable" ? "" : "bad"}"
         style="${q.tier === "usable" ? "background:#f4ead3;color:#8a5f18;border:1px solid #d4bb88" : ""}">
      ${ic(good ? "check" : "alert")}<span>${q.bpw} bpw — ${q.tier}</span></div>
    <div class="hint">${esc(q.note || "")}</div>`;
}

async function planConvert() {
  if (!bridge()) return;
  const target = cvTarget;
  const from = ($("#cv-from") || {}).value || R.gguf || (refBuild() || {}).path;
  if (!from) { toast("no build selected"); return; }
  const p = await bridge().convert_plan(from, target);
  const box = $("#cv-route");
  if (!box) return;
  const cnt = $("#cv-count");
  if (cnt) cnt.textContent = p.ok ? `${p.steps.length} step${p.steps.length === 1 ? "" : "s"} to ${target}` : "";
  if (!p.ok) { box.innerHTML = `<div class="hint">${esc(p.guidance)}</div>`; return; }
  box.innerHTML = `<div class="hint" style="margin-bottom:9px">${esc(p.guidance)}</div>`
    + p.steps.map((s, i) => `
      <div class="step ${s.blocking ? "blocking" : ""}">
        <div class="sn">${i + 1}</div>
        <div class="sb">
          <div class="scmd" title="${esc(s.args.join(" "))}">${s.tool.replace("pollard_", "pollard-")}
            <span class="fl">${esc(shortArgs(s.args))}</span></div>
          <div class="swhy">${esc(s.why)}</div>
        </div>
        <button class="btn" onclick="act('run:${s.tool}')">RUN</button>
      </div>`).join("")
    + (p.notes || []).map(n => `<div class="modnote">${ic("info")}${esc(n)}</div>`).join("");
}

async function publishNow() {
  const repo = ($("#pub-repo") || {}).value || "";
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    toast("enter the full repo id first, e.g. you/Model-Pollard-GGUF"); return;
  }
  R.repo = R.uploadRepo = repo;
  R.modelName = ($("#pub-model") || {}).value || R.modelName;
  if (!bridge()) return;
  const p = await bridge().plan("publish", R);
  if (!p.ok) { toast(p.error); return; }
  if (!confirm(`This publishes to Hugging Face and is public.\n\n${p.display}\n\nContinue?`)) return;
  if (prompt(`Type the repo id to confirm:`) !== repo) { toast("repo id did not match — nothing published"); return; }
  const r = await bridge().start("publish", R);
  if (!r.ok) { toast(r.error); return; }
  logSeq = 0; logLines = []; startPolling();
  toast("publishing to " + repo);
}

function startPolling() {
  if (poller) return;
  poller = setInterval(async () => {
    if (!bridge()) return;
    const t = await bridge().tail(logSeq);
    if (t.lines && t.lines.length) { logLines.push(...t.lines); renderLog(); }
    logSeq = t.seq;
    S.status = t;
    refreshMonitor();
    $("#d-state").textContent = t.running ? "RUNNING"
      : t.returncode === 0 ? "DONE" : t.returncode != null ? "EXIT " + t.returncode : "IDLE";
    if (!t.running) { clearInterval(poller); poller = null; rescan(); }
  }, 400);
}

const LEVEL = /\b(ERROR|FATAL|WARN|WARNING|OK|PASS|FAIL|INFO)\b/;

function renderLog() {
  const html = logLines.slice(-500).map(l => {
    const m = l.match(LEVEL);
    const lvl = m ? ({ WARNING: "WARN", PASS: "OK", FATAL: "ERROR" }[m[1]] || m[1]) : null;
    const cls = l.startsWith("$") ? "cmdline" : l.startsWith("!") ? "bd-WARN"
              : l.startsWith("—") ? "dim" : "";
    return `<div class="r nb">${lvl ? `<span class="bd bd-${lvl}">${lvl}</span>` : '<span></span>'}
      <span class="mg ${cls}">${esc(l)}</span></div>`;
  }).join("") || idleReadout();
  $$(".log").forEach(el => { el.innerHTML = html; el.scrollTop = el.scrollHeight; });
}

const esc = s => s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
/* esc() is for TEXT. Inside a quoted attribute a quote character ends the attribute early, and
   filenames are allowed to contain them -- so anything from the filesystem that lands in an
   attribute goes through this instead. */
const escAttr = s => String(s == null ? "" : s).replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* An instrument at rest still reads out. A blank panel looks broken; this says what is loaded and
   what it is waiting for, in the same shape the running log uses. */
function idleReadout() {
  const b = curBuild() || refBuild() || {};
  const st = S.status || {};
  const rows = [
    ["model", S.model ? S.model.name : "no workspace"],
    ["build", b.tag ? `${b.tag}  ${gb(b.bytes || 0)} GB${b.bpw ? `  ${b.bpw.toFixed(2)} bpw` : ""}` : "none selected"],
    ["lane", `${b.lane || R.lane}${b.kind ? `  (${b.kind})` : ""}`],
    ["runtime", (S.runtime || "llama.cpp")],
    ["last job", st.label ? `${st.label}  exit ${st.returncode}` : "none this session"],
    ["state", "READY -- nothing running"],
  ];
  return rows.map(([k, v]) =>
    `<div class="r nb"><span class="bd bd-BUILD">${k.toUpperCase().slice(0, 8)}</span>` +
    `<span class="mg">${esc(String(v))}</span></div>`).join("");
}

function toast(msg) {
  let t = $("#toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("on");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), 4200);
}

function copyCmd() {
  const el = $("#b-cmd");
  if (el) { navigator.clipboard?.writeText(el.textContent).then(() => toast("copied")); }
}
function saveArgs() { toast(JSON.stringify(R)); }

async function rescan() {
  if (!bridge()) return;
  Object.assign(S, await bridge().rescan());
  if (!R.source && refBuild()) R.source = refBuild().path;
  show(screen);
}

function renderModelPicker() {
  const el = $("#modelsel");
  if (!el) return;
  const cur = (S.model || {}).key;
  const grp = (label, rows) => rows.length
    ? `<optgroup label="${escAttr(label)}">` + rows.map(([k, t]) =>
        `<option value="${escAttr(k)}" ${k === cur ? "selected" : ""}>${esc(t)}</option>`).join("")
      + `</optgroup>`
    : "";
  const here = S.models.map(m => [m.key, `${m.name} (${m.builds})`]);
  // one group per box, so a cluster's models are as pickable as this machine's
  const byHost = new Map();
  remoteGroups().forEach(g => {
    if (!byHost.has(g.host)) byHost.set(g.host, []);
    // name the lane: a cluster holds MLX, MX, EXL3, GPTQ and brains as well as GGUFs, and
    // "Qwen3-4B (1)" does not say which of them you are about to select
    const lanes = [...new Set(g.builds.map(b => b.lane).filter(Boolean))];
    const tail = lanes.length ? ` · ${lanes.join("/")}` : "";
    byHost.get(g.host).push([g.key, `${g.name} (${g.builds.length})${tail}`]);
  });
  el.innerHTML = (grp("this machine", here)
    + [...byHost].map(([h, rows]) => grp(h, rows)).join(""))
    || `<option>no models found</option>`;

  el.onchange = async () => {
    const v = el.value;
    if (v.startsWith("remote:")) {
      // a box's model: its builds come from the agent's listing, not from a local scan
      const g = remoteGroups().find(x => x.key === v);
      if (!g) return;
      S.model = { key: g.key, name: `${g.host} · ${g.name}`, builds: g.builds, remote: true };
      const r = refBuild();
      if (r) { R.source = r.path; R.gguf = r.path; }
      show(screen);
      return;
    }
    if (!bridge()) return;
    Object.assign(S, await bridge().select(v));
    const r = refBuild();
    if (r) { R.source = r.path; R.gguf = r.path; }
    show(screen);
  };
}

function win(w) { if (bridge()) bridge().win(w); }

async function boot() {
  $("#wing").innerHTML = SCREENS_ORDER.map(([id, label, group]) =>
    (group ? `<div class="wh">${group}</div>` : "")
    + `<button class="pill" data-s="${id}">${label}</button>`).join("");
  $$("#wing .pill").forEach(b => b.addEventListener("click", () => show(b.dataset.s)));

  if (bridge()) {
    try { Object.assign(S, await bridge().state()); } catch (e) { console.warn(e); }
  } else {
    try { Object.assign(S, await fetch("../preview.json").then(r => r.json())); } catch (e) {}
  }
  if (bridge()) {
    try {
      S.hw = await bridge().hardware();
      if (S.hw.suggest_target_gb) R.ram = S.hw.suggest_target_gb;
      if (S.hw.suggest_reserve_gb) R.reserve = S.hw.suggest_reserve_gb;
      if (S.hw.cpus) R.threads = Math.max(1, Math.round(S.hw.cpus * 0.6));
    } catch (e) { console.warn(e); }
  }
  const ref = refBuild();
  if (ref) { R.source = ref.path; R.gguf = ref.path; }
  renderModelPicker();
  show(location.hash.slice(1) || "build");
  // The cluster's builds belong in EVERY build list, not just the one screen that asked for
  // them, and they arrive after the first render -- so load once at boot and refill the selects.
  loadRemoteModels();
}

if (window.pywebview) boot();
else window.addEventListener("pywebviewready", boot);
window.addEventListener("load", () => setTimeout(() => {
  if (!$("#wing").children.length) boot(); }, 300));
