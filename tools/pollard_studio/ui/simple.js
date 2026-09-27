/* Pollard Studio — SIMPLE mode: the app.
   A rail (Fit · Shelf · Request · Cooks · Docs) and one screen at a time, filling the window. The same
   glass, rows, pills and badges as pollard.app, laid out as an application. What only the desktop app can
   do is wired in: read this machine (hardware), put a build on disk (download) and hand it to chat.
   Advanced (app.js) is untouched; we only call its setMode()/show()/select() seams. */
(() => {
  const HEADROOM = 2.0;                                       // GB for KV cache + context
  const LANES = ["GGUF", "MLX", "GPTQ", "EXL3", "MX"];
  const SITE = "https://pollard.app";
  const GH = "https://github.com/WestWaters/pollard-weights";
  const api = () => (window.pywebview && window.pywebview.api) || null;
  const $ = s => document.querySelector(s);
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ICON = {
    fit: '<path d="M4 7h16M4 12h10M4 17h16"/><circle cx="17" cy="12" r="2.2"/>',
    shelf: '<rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/>',
    request: '<path d="M4 6h16v10H9l-5 4z"/><path d="M8 10h8M8 13h5"/>',
    cooks: '<circle cx="9" cy="8" r="3.2"/><path d="M3 19c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.4"/><path d="M15.5 13.2c2.8.3 5 2.6 5 5.8"/>',
    docs: '<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v4h4M9 12h6M9 16h6"/>'
  };
  const CATS = {
    phone: { l: "Phone", d: { "ph-6": ["6 GB", 3], "ph-8": ["8 GB", 5], "ph-12": ["12 GB", 8], "ph-16": ["16 GB", 11] }, svg: '<rect x="8" y="3" width="14" height="24" rx="3"/><path d="M13 23h4"/>' },
    pi: { l: "Raspberry Pi · SBC", d: { "pi-4": ["Pi 4 · 4 GB", 2.5], "pi-8": ["Pi 4/5 · 8 GB", 6], "pi-16": ["Pi 5 · 16 GB", 12], "jet-8": ["Jetson · 8 GB", 6], "jet-16": ["Jetson · 16 GB", 13] }, svg: '<rect x="3" y="7" width="24" height="16" rx="1.6"/><path d="M7 7V4M11 7V4M15 7V4M19 7V4M23 7V4"/><rect x="7" y="12" width="7" height="6" rx="1"/><circle cx="21" cy="15" r="2"/>' },
    mac: { l: "Mac", d: { "mac-8": ["8 GB", 5], "mac-16": ["16 GB", 11], "mac-24": ["24 GB", 17], "mac-32": ["32 GB", 24], "mac-48": ["48 GB", 38], "mac-64": ["64 GB", 52], "mac-96": ["96 GB", 80], "mac-128": ["128 GB", 108], "mac-192": ["192 GB", 168], "mac-512": ["512 GB", 460] }, svg: '<rect x="4" y="6" width="22" height="14" rx="1.6"/><path d="M2 24h26"/>' },
    gpu: { l: "PC · GPU", d: { "gpu-8": ["8 GB", 7], "gpu-12": ["12 GB", 11], "gpu-16": ["16 GB", 15], "gpu-24": ["24 GB", 23], "gpu-32": ["32 GB", 31], "gpu-48": ["48 GB", 46], "gpu-80": ["80 GB", 78], "gpu-96": ["96 GB", 94] }, svg: '<rect x="3" y="8" width="24" height="13" rx="1.6"/><circle cx="11" cy="14.5" r="3.2"/><circle cx="20" cy="14.5" r="3.2"/><path d="M5 21v3M25 21v3"/>' },
    box: { l: "AI box · unified", d: { "spark-1": ["DGX Spark · 128 GB", 110], "spark-2": ["2× Spark · 256 GB", 230], "halo-64": ["Strix Halo · 64 GB", 48], "halo-128": ["Strix Halo · 128 GB", 100], "orin-64": ["Jetson AGX Orin · 64 GB", 52], "thor-128": ["Jetson Thor · 128 GB", 108] }, svg: '<rect x="4" y="9" width="22" height="12" rx="2"/><path d="M8 13h6M8 17h10"/><circle cx="21" cy="15" r="1.4"/><path d="M9 9V6h12v3"/>' },
    srv: { l: "Server · Cluster", d: { "srv-128": ["128 GB", 120], "srv-256": ["256 GB", 240], "srv-512": ["512 GB", 490], "srv-1t": ["1 TB", 980], "srv-2t": ["2 TB", 1960] }, svg: '<rect x="4" y="4" width="22" height="6" rx="1.4"/><rect x="4" y="12" width="22" height="6" rx="1.4"/><rect x="4" y="20" width="22" height="6" rx="1.4"/><path d="M8 7h.01M8 15h.01M8 23h.01"/>' }
  };
  const DEVLIST = Object.entries(CATS).flatMap(([k, c]) => Object.entries(c.d).map(([id, [l, b]]) => ({ k, id, l: c.l + " · " + l, b })));
  const MODS = [["all", "All"], ["chat", "Chat"], ["code", "Code"], ["vision", "Vision"], ["moe", "MoE"], ["connectome", "Connectome"]];
  const modality = r => /flybrain|humanbrain/i.test(r.id) ? "connectome" : /coder/i.test(r.name) ? "code" : r.vision ? "vision" : "chat";

  const S = { screen: "fit", hw: null, cat: "detected", dev: null, lane: "GGUF", shelf: [], local: {}, repo: null, file: null, q: "",
              dl: null, page: 1, mod: "all", onlyFits: false, cooks: null };
  try { S.screen = localStorage.getItem("pollard.simple.screen") || "fit"; } catch (e) {}

  const budget = () => S.cat === "detected" ? (S.hw ? S.hw.budget_gb : 11) : CATS[S.cat].d[S.dev][1];
  const devLabel = () => S.cat === "detected" ? (S.hw ? S.hw.label : "This machine") : CATS[S.cat].l + " · " + CATS[S.cat].d[S.dev][0];
  const verdict = gb => gb + HEADROOM <= budget() ? ["fits", "FITS"] : gb <= budget() ? ["tight", "TIGHT"] : ["no", "TOO BIG"];
  const rungs = () => S.shelf.flatMap(r => r.files.map(f => ({ r, f })));
  const laneRungs = () => rungs().filter(x => x.r.lane === (S.lane === "MLX" ? "mlx" : "gguf"));
  const localPath = (r, f) => S.local[r.id + "/" + f.file] || null;

  /* ── data ────────────────────────────────────────────────────────────── */
  async function readMachine() {
    let hw = null; try { if (api()) hw = await api().hardware(); } catch (e) { console.warn(e); }
    if (!hw || !hw.ram_gb) { S.hw = { label: "This machine", budget_gb: 11, note: "could not read memory — pick a preset" }; return; }
    const kind = hw.platform === "darwin" ? "Mac" : hw.platform === "win32" ? "PC" : "Linux box";
    const ramGiB = Math.round(hw.ram_gb * 1e9 / 1073741824);
    // free right now, but never below half the box: the OS and other apps give memory back when a model loads
    const usable = hw.avail_gb != null ? Math.round(Math.min(hw.ram_gb * 0.85, Math.max(hw.avail_gb, hw.ram_gb * 0.5))) : hw.suggest_target_gb;
    S.hw = { label: `${kind} · ${ramGiB} GB`, budget_gb: usable, note: hw.avail_gb != null ? `${hw.avail_gb} GB free right now · ${hw.cpus} CPUs` : `${hw.cpus} CPUs` };
  }
  async function readShelf() {
    let data = null; try { if (api()) data = await api().shelf(); } catch (e) { console.warn(e); }
    if (!data || !data.shelf) { try { data = await shelfFromHub(); } catch (e) { data = { shelf: [] }; } }
    S.shelf = data.shelf || []; S.local = data.local || {};
  }
  async function shelfFromHub() {                              // browser preview only (no bridge)
    const models = await fetch("https://huggingface.co/api/models?author=PollardWeights&limit=100").then(r => r.json());
    const shelf = [];
    for (const m of models) {
      const tree = await fetch(`https://huggingface.co/api/models/${m.id}/tree/main`).then(r => r.json()).catch(() => []);
      const short = m.id.split("/")[1], out = { id: m.id, name: short.replace("-Pollard", ""), dl: m.downloads || 0, vision: false, lane: "gguf", moe: /A\d+B|Ling/i.test(short) ? "MoE" : "", files: [] };
      for (const t of tree) { const n = t.path, b = (t.lfs && t.lfs.size) || t.size || 0;
        if (n.startsWith("mmproj")) { out.vision = true; continue; }
        if (n.endsWith(".safetensors")) { out.lane = "mlx"; out.files.push({ q: "4bit", gb: +(b / 1e9).toFixed(2), ik: false, file: n }); continue; }
        if (!n.endsWith(".gguf")) continue; const q = n.replace(/\.gguf$/, "").split("-").pop(); out.files.push({ q, gb: +(b / 1e9).toFixed(2), ik: q.endsWith("_KT"), file: n }); }
      out.files.sort((a, b) => a.gb - b.gb); if (out.files.length) shelf.push(out);
    }
    return { shelf };
  }
  async function getJson(url) {
    if (api() && api().fetch_json) { const r = await api().fetch_json(url); if (r && r.ok) return r.data; throw new Error(r && r.error || "fetch failed"); }
    return fetch(url).then(r => r.json());
  }
  const TIERS = [[1e6, "Executive"], [1e5, "Chef"], [1e4, "Sous"], [1e3, "Line"], [0, "Prep"]];
  const tierOf = d => TIERS.find(([m]) => d >= m)[1];
  const fmt = n => n == null ? "—" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(n);
  async function readCooks() {
    // live reputation (HF downloads / models / followers + builds delivered through Pollard), falling back to the plain roster
    try { const c = await getJson(SITE + "/api/cooks"); S.cooks = c.cooks || []; return; } catch (e) {}
    try { const c = await getJson(SITE + "/cooks.json?ts=" + Date.now()); S.cooks = c.cooks || []; }
    catch (e) { S.cooks = [{ name: "Mario · WestWaters", role: "Founder · Pollard", lanes: LANES, hardware: "Apple Silicon · RTX Blackwell", status: "cooking", blurb: "The method, the shelf, every lane." }]; }
  }


  /* ── the water: the same shader pollard.app runs, behind the shell ─────── */
  const VS = `attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}`;
  const FS = `precision highp float;uniform vec2 u_res;uniform float u_time;uniform vec3 u_cy,u_vi,u_mg,u_am;
vec3 pal(float u){u=fract(u);vec3 c=mix(u_cy,u_vi,smoothstep(0.,.3,u));c=mix(c,u_mg,smoothstep(.3,.6,u));
c=mix(c,u_am,smoothstep(.6,.85,u));return mix(c,u_cy,smoothstep(.88,1.,u));}float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;mat2 m=mat2(1.6,1.2,-1.2,1.6);for(int i=0;i<5;i++){v+=a*noise(p);p=m*p;a*=.5;}return v;}
float water(vec2 p,float t){float h=fbm(p*1.3+vec2(t*.04,-t*.025));h+=.22*fbm(p*2.8-vec2(t*.06,t*.03));
 vec2 c1=vec2(.35,-.25),c2=vec2(-.45,-.4);
 h+=.06*sin(length(p-c1)*38.-t*1.6)*exp(-length(p-c1)*1.6);
 h+=.05*sin(length(p-c2)*30.-t*1.3)*exp(-length(p-c2)*1.4);return h;}
float yc(float x,float t){return .18*sin(x*2.4+t*.35)+.28*sin(x*1.1-t*.2);}
vec2 ribbon(vec2 p,float t){float x=p.x;float y0=yc(x,t);float w=.13+.09*sin(x*1.9+t*.3+1.7);
 float v=(p.y-y0)/w;float body=1.-smoothstep(.85,1.05,abs(v));
 float k=v*5.5+.4*sin(x*6.+t*.8)+.25*sin(x*13.-t*.5);float lines=pow(abs(sin(k*3.14159)),10.);
 lines*=.55+.9*noise(vec2(x*7.+t*.3,v*4.));
 float g=body*(.35+1.2*lines);g*=smoothstep(-1.35,-.7,x)*(1.-smoothstep(.9,1.4,x));
 float y2=-.05+.22*sin(x*1.6-t*.25+2.);float w2=.06+.03*sin(x*3.+t*.4);float v2=(p.y-y2)/w2;
 float b2=1.-smoothstep(.8,1.05,abs(v2));float l2=pow(abs(sin((v2*6.+.4*sin(x*5.-t))*3.14159)),12.);
 return vec2(g,v*.5+.5+x*.2);}
void main(){vec2 uv=gl_FragCoord.xy/u_res;vec2 p=(gl_FragCoord.xy-.5*u_res)/u_res.y;float t=u_time;
 float e=.004;vec2 wp=p*vec2(1.,2.2);float h=water(wp,t);
 float hx=water(wp+vec2(e,0.),t)-h,hy=water(wp+vec2(0.,e),t)-h;vec3 n=normalize(vec3(-hx,-hy,e*3.));
 vec3 vd=vec3(0.,0.,1.);vec3 col=mix(vec3(.004,.007,.016),vec3(.010,.022,.048),smoothstep(.3,-.6,p.y));
 vec3 L1=normalize(vec3(.5,.8,.6)),L2=normalize(vec3(-.6,.3,.7));
 float s1=pow(max(dot(reflect(-L1,n),vd),0.),34.),s2=pow(max(dot(reflect(-L2,n),vd),0.),22.);
 float wm=mix(.05,1.,smoothstep(.35,-.4,p.y));s1*=wm;s2*=wm;
 col+=vec3(.70,.92,1.)*s1*.42+vec3(.30,.55,1.)*s2*.22;
 col+=.035*vec3(.3,.6,1.)*pow(max(dot(n,L2),0.),2.)*wm;
 col*=1.-.9*dot(uv-.5,uv-.5);gl_FragColor=vec4(col,1.);}`;
  let water = null;
  function startWater(cv) {
    if (water || !cv) return;
    const gl = cv.getContext("webgl", { antialias: false, alpha: false }); if (!gl) return;
    const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o);
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) { console.warn(gl.getShaderInfoLog(o)); return null; } return o; };
    const v = sh(gl.VERTEX_SHADER, VS), f = sh(gl.FRAGMENT_SHADER, FS); if (!v || !f) return;
    const pr = gl.createProgram(); gl.attachShader(pr, v); gl.attachShader(pr, f); gl.linkProgram(pr); gl.useProgram(pr);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const a = gl.getAttribLocation(pr, "a"); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    const U = n => gl.getUniformLocation(pr, n);
    const u = { res: U("u_res"), time: U("u_time"), cy: U("u_cy"), vi: U("u_vi"), mg: U("u_mg"), am: U("u_am") };
    gl.uniform3f(u.cy, .24, .88, 1); gl.uniform3f(u.vi, .55, .36, .96); gl.uniform3f(u.mg, 1, .48, .88); gl.uniform3f(u.am, .88, .54, .16);
    const size = () => { const d = .6; cv.width = (cv.clientWidth * d) | 0; cv.height = (cv.clientHeight * d) | 0; gl.viewport(0, 0, cv.width, cv.height); };
    size(); window.addEventListener("resize", size);
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const t0 = performance.now(); let n = 0;
    water = { stop: false };
    (function loop() {
      if (water.stop) return;
      if (!(n++ & 1) && !document.getElementById("simple").hidden) {        // 30 fps, and only while Simple is showing
        gl.uniform2f(u.res, cv.width, cv.height); gl.uniform1f(u.time, (performance.now() - t0) / 1000); gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      if (!still) requestAnimationFrame(loop);
    })();
  }

  /* ── shell ───────────────────────────────────────────────────────────── */
  function frame() {
    const rail = [["fit", "Fit"], ["shelf", "Shelf"], ["request", "Request"], ["cooks", "Cooks"], ["docs", "Docs"]]
      .map(([k, l]) => `<button data-screen="${k}" aria-current="${k === S.screen}"><svg viewBox="0 0 24 24">${ICON[k]}</svg><span>${l}</span></button>`).join("");
    return `
    <canvas class="s-water" id="s-water"></canvas>
    <div class="s-top">
      <div class="s-mark">POLLARD<small>STUDIO</small></div>
      <div class="s-right">
        <div class="s-mode"><button aria-pressed="true">Simple</button><button aria-pressed="false" data-adv>Advanced</button></div>
        <div class="s-win">
          <button title="Minimize" data-win="minimize"><svg viewBox="0 0 16 16"><rect x="3.5" y="7.2" width="9" height="1.6" rx=".8"/></svg></button>
          <button title="Full screen" data-win="toggle"><svg viewBox="0 0 16 16"><path d="M3.4 3.4h4v1.5H4.9v2.5H3.4zM12.6 12.6h-4v-1.5h2.5V8.6h1.5z"/></svg></button>
          <button class="x" title="Close" data-win="close"><svg viewBox="0 0 16 16"><path d="M4.6 3.5l3.4 3.4 3.4-3.4 1.1 1.1L9.1 8l3.4 3.4-1.1 1.1L8 9.1l-3.4 3.4-1.1-1.1L6.9 8 3.5 4.6z"/></svg></button>
        </div>
      </div>
    </div>
    <nav class="s-rail">${rail}<div class="grow"></div></nav>
    <div class="s-view">
      <section class="screen" id="scr-fit">
        <div class="dots" id="s-dots"></div>
        <div class="pane" data-page="1"><div class="pmark">POLLARD<i></i></div><h3>What machine will this run on?</h3>
          <div class="s-detect" id="s-detect"></div><div class="rows" id="s-rows"></div><div class="chips" id="s-chips"></div></div>
        <div class="pane" data-page="2"><div class="pmark">POLLARD<i></i></div>
          <div class="search"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.6-3.6"/></svg><input id="s-q" placeholder="Search models..."></div>
          <div class="found" id="s-found"></div><div class="hits" id="s-hits"></div><div class="fitsrow"><div class="fits" id="s-fitword">FITS</div></div></div>
        <div class="pane" data-page="3"><div class="pmark">POLLARD<i></i></div>
          <div class="lanes" id="s-lanes"></div><div class="chosen" id="s-chosen"></div>
          <button class="install" id="s-install">INSTALL</button>
          <div class="progress" id="s-prog" style="display:none"><i></i></div>
          <p class="micro">1-click &bull; verified &bull; on this machine</p>
          <button class="open" id="s-open" style="display:none">Open in chat &rsaquo;</button>
          <div class="cmd" id="s-cmd"></div><div class="spacer"></div>
          <p class="micro link" data-go="request">Need a different model? Request a build &rsaquo;</p></div>
      </section>
      <section class="screen scroll" id="scr-shelf">
        <div class="s-h"><div><p class="s-caps">Shelf</p><h2>Already built. Already gated.</h2></div><p id="s-shelfnote"></p></div>
        <div class="bar"><span class="s-caps">Badged for</span><select id="s-devsel"></select>
          <button class="chip" id="s-only" aria-pressed="false">Only what fits</button><span class="s-caps" id="s-count"></span></div>
        <div class="bar" id="s-mods"></div>
        <div class="grid" id="s-grid"></div>
      </section>
      <section class="screen" id="scr-request">
        <div class="pane">
          <div class="pmark">POLLARD<i></i></div><h3>Not on the shelf? Have it cooked.</h3>
          <form class="form" id="s-form">
            <div class="full"><label>Model</label><input name="model" required placeholder="Hugging Face id or link — f16/bf16 source preferred"></div>
            <div><label>Lane</label><select name="lane">${LANES.map(l => `<option>${l}</option>`).join("")}</select></div>
            <div><label>Target device</label><select name="device" id="s-rqdev"></select></div>
            <div><label>Visibility</label><select name="visibility"><option value="public">Public shelf (PollardWeights)</option><option value="private">Private delivery</option></select></div>
            <div><label>Turnaround</label><select name="speed"><option value="standard">Standard · ~5 days</option><option value="priority">Priority · 48 h · ×1.5</option><option value="rush">Rush · 24 h · ×2</option></select></div>
            <div class="full"><label>Contact</label><input name="contact" required placeholder="email or handle for the quote"></div>
            <div class="full"><label>Notes</label><textarea name="notes" placeholder="evals you care about, context length, anything the cook should know"></textarea></div>
            <button class="send" id="s-send" type="submit">REQUEST A QUOTE</button>
            <div class="sent" id="s-sent"></div>
          </form>
        </div>
        <div class="pane"><div class="pmark">POLLARD<i></i></div><h3>How it works</h3><div class="how">
          <div class="step"><b>1 · Quote</b><span>GPU-hours × lane rate + platform fee, usually within a day. Valid 7 days.</span></div>
          <div class="step"><b>2 · Accept</b><span>A checkout link. Your card is <em>authorized</em>, not charged.</span></div>
          <div class="step"><b>3 · Cook</b><span>A cook builds it on their hardware — Pollard method, same as everything on the shelf.</span></div>
          <div class="step"><b>4 · Verify</b><span><code>pollard-verify</code> must pass and the standard card ships with it. Fail the gate, no charge.</span></div>
          <div class="step"><b>5 · Deliver</b><span>Public: a PollardWeights repo, and it appears here. Private: a repo you control. Charge captured on delivery.</span></div>
          <div class="step"><b>Enterprise</b><span>Monthly SLA — front of queue, guaranteed turnaround, <code>pollard-taskeval</code> report. Ask in the notes.</span></div>
        </div></div>
      </section>
      <section class="screen scroll" id="scr-cooks">
        <div class="s-h"><div><p class="s-caps">Cooks</p><h2>Who builds, on what.</h2></div><p>Same method, same gate, same card. Their attribution is on every build.</p></div>
        <div class="grid" id="s-cooks"></div>
        <p style="margin-top:14px"><button class="chip" data-url="https://github.com/WestWaters/pollard-weights/issues/new?title=Become+a+cook%3A+%3Cyour+name+%2F+handle%3E&body=%2A%2AHugging+Face+profile%3A%2A%2A+%0A%2A%2ALanes+I+cook+%28GGUF+%2F+MLX+%2F+GPTQ+%2F+EXL3+%2F+MX%29%3A%2A%2A+%0A%2A%2AHardware%3A%2A%2A+%0A%2A%2AOne+verified+build+I%27ve+shipped+%28link%29%3A%2A%2A+%0A%0AI%27ve+read+COOKS.md+%E2%80%94+Pollard+method+only%2C+verify+gate%2C+standard+card%2C+70%2F30+split+when+I+bring+the+hardware.">Become a cook — bring a lane and hardware &rsaquo;</button></p>
      </section>
      <section class="screen scroll" id="scr-docs">
        <div class="s-h"><div><p class="s-caps">Docs</p><h2>One allocation, any lane.</h2></div><p>Measure once; every lane is that allocation emitted for a different runtime.</p></div>
        <div class="grid">
          <div class="pane card"><h4>Quick start</h4><p class="chosen">Know what the machine can run before you download anything, then build for it.</p>
            <div class="cmd"><span class="k">pollard-calc</span> --model Qwen/Qwen3-30B-A3B --ram 16
<span class="k">pollard</span> --hf Qwen/Qwen3-8B --run
llama-cli -m Qwen3-8B-Pollard.gguf</div></div>
          <div class="pane card"><h4>Workflow — in order</h4><div class="dl">
            <div><code>pollard-calc</code><span>fits resident · streaming-viable · too big</span></div>
            <div><code>pollard-smooth</code><span>precondition outliers (low-bit lanes)</span></div>
            <div><code>pollard-probe</code><span>per-tensor sensitivity, any box</span></div>
            <div><code>pollard-automap</code><span>the measured mix, dense or MoE</span></div>
            <div><code>pollard-verify</code><span>reconstruction gate — no pass, no ship</span></div>
            <div><code>pollard-card</code><span>the standard model card</span></div></div></div>
          <div class="pane card"><h4>Lanes</h4><div class="dl">
            <div><code>GGUF</code><span>llama.cpp · Ollama · LM Studio — trellis mix to ~1-bit</span></div>
            <div><code>vLLM · SGLang</code><span>GPTQ 4/8-bit dynamic mix (Marlin), tensor-parallel serving</span></div>
            <div><code>GPTQ</code><span>INT3/INT4 full-Hessian error-feedback</span></div>
            <div><code>MLX</code><span>Apple Silicon, mixed 4/8-bit</span></div>
            <div><code>EXL3</code><span>exllamav3 — Pollard beats EXL3 on its own allocator: 8.670 vs 8.699 @4bpw</span></div>
            <div><code>MX · NVFP4</code><span>Blackwell FP4; W4A16 on any vLLM GPU</span></div></div></div>
          <div class="pane card"><h4>Measure &amp; allocate</h4><div class="dl">
            <div><code>pollard-sensitivity</code><span>each tensor's true KL cost</span></div>
            <div><code>pollard-experts</code><span>measured expert usage (MoE)</span></div>
            <div><code>pollard-prune</code><span>REAP-style expert pruning</span></div>
            <div><code>pollard-rotate</code><span>incoherence rotation (QuIP# / QuaRot)</span></div></div></div>
          <div class="pane card"><h4>Evaluate &amp; verify</h4><div class="dl">
            <div><code>pollard-kl</code><span>KL-to-f16 — the judging metric</span></div>
            <div><code>pollard-eval</code><span>top-1 agreement + KL, with chart</span></div>
            <div><code>pollard-taskeval</code><span>task suites; point it at your own evals</span></div>
            <div><code>pollard-doctor</code><span>diagnose · predict · repair any model, any lane</span></div></div></div>
          <div class="pane card"><h4>Runtime, cluster &amp; agents</h4><div class="dl">
            <div><code>pollard-run</code><span>measured expert placement, RAM-streaming</span></div>
            <div><code>pollard-node</code><span>run on every box so Studio totals a cluster's RAM</span></div>
            <div><code>ggml-rpc</code><span>pool machines' memory over the network</span></div>
            <div><code>skills/pollard</code><span>agent skill — routes any model down the right lane</span></div></div></div>
        </div>
        <p style="margin-top:14px"><button class="chip" data-url="${GH}#readme">Full documentation on GitHub &rsaquo;</button></p>
      </section>
    </div>`;
  }

  /* ── render ──────────────────────────────────────────────────────────── */
  function render() {
    const root = $("#simple"); if (!root) return;
    if (!root.dataset.built) { root.innerHTML = frame(); root.dataset.built = "1"; wire(); startWater($("#s-water")); }
    root.querySelectorAll(".s-rail button").forEach(b => b.setAttribute("aria-current", b.dataset.screen === S.screen));
    root.querySelectorAll(".screen").forEach(s => s.classList.toggle("on", s.id === "scr-" + S.screen));
    renderFit(); renderShelf(); renderCooks();
  }
  function renderFit() {
    $("#s-rows").innerHTML = Object.entries(CATS).map(([k, c]) => `<button class="row" aria-pressed="${k === S.cat}" data-cat="${k}"><svg viewBox="0 0 30 30">${c.svg}</svg>${c.l}<span class="ch">&rsaquo;</span></button>`).join("");
    $("#s-detect").innerHTML = S.hw ? `<b>${esc(S.hw.label)}</b><span>${esc(S.hw.note)}</span><small>${S.hw.budget_gb} GB usable for a model</small>` : `<b>Reading this machine…</b>`;
    $("#s-detect").setAttribute("aria-pressed", S.cat === "detected");
    $("#s-chips").innerHTML = S.cat === "detected" ? "" : Object.entries(CATS[S.cat].d).map(([id, [l]]) => `<button class="chip" aria-pressed="${id === S.dev}" data-dev="${id}">${l}</button>`).join("");
    const on = $("#s-chips .chip[aria-pressed='true']"); if (on) on.scrollIntoView({ block: "nearest" });

    const q = S.q.toLowerCase();
    const all = laneRungs().filter(x => !q || (x.r.name + " " + x.f.q).toLowerCase().includes(q));
    const ok = all.filter(x => verdict(x.f.gb)[0] !== "no").sort((a, b) => (verdict(a.f.gb)[0] === "fits" ? 0 : 1) - (verdict(b.f.gb)[0] === "fits" ? 0 : 1) || b.f.gb - a.f.gb);
    const cur = S.file && laneRungs().find(x => x.f.file === S.file);
    if (ok.length && (!cur || verdict(cur.f.gb)[0] === "no")) { S.repo = ok[0].r.id; S.file = ok[0].f.file; }
    $("#s-found").innerHTML = S.shelf.length ? `Found in <b>${ok.length} FITS</b> · ${esc(devLabel())}` : `Loading the shelf…`;
    $("#s-fitword").textContent = ok.length ? "FITS" : "TOO BIG";
    $("#s-hits").innerHTML = (ok.length ? ok : all).slice(0, 40).map(x => `<button class="hit" aria-pressed="${x.f.file === S.file}" data-repo="${x.r.id}" data-file="${x.f.file}">
        <span>${esc(x.r.name)} <small>${x.f.q}</small>${localPath(x.r, x.f) ? '<span class="own">ON DISK</span>' : ""}</span><small>${x.f.gb.toFixed(1)} GB</small></button>`).join("");

    $("#s-lanes").innerHTML = LANES.map(l => `<button aria-pressed="${l === S.lane}" data-lane="${l}">${l}${l === S.lane ? " &#10003;" : ""}</button>`).join("");
    const btn = $("#s-install"), open = $("#s-open"), prog = $("#s-prog"), cmd = $("#s-cmd"), chosen = $("#s-chosen");
    const shelfLane = S.lane === "GGUF" || S.lane === "MLX";
    if (S.file && shelfLane) {
      const r = S.shelf.find(x => x.id === S.repo), f = r.files.find(x => x.file === S.file), [cls, txt] = verdict(f.gb), here = localPath(r, f);
      chosen.innerHTML = `<b>${esc(r.name)} · ${f.q}</b><br>${f.gb.toFixed(2)} GB + ${HEADROOM.toFixed(1)} KV of ${budget()} GB <span class="v ${cls}">${txt}</span>${f.ik ? "<br><span style='color:var(--scold)'>needs the ik_llama runtime</span>" : ""}`;
      if (S.dl && S.dl.active && S.dl.file === f.file) { btn.textContent = `${S.dl.pct}%`; btn.className = "install busy"; btn.disabled = true; prog.style.display = ""; prog.firstElementChild.style.width = S.dl.pct + "%"; }
      else { btn.textContent = here ? "INSTALLED" : "INSTALL"; btn.className = "install"; btn.disabled = !!here || !api(); prog.style.display = "none"; }
      open.style.display = here ? "" : "none";
      cmd.textContent = S.lane === "GGUF" ? `ollama run hf.co/${r.id}:${f.q}` : `mlx_lm.generate --model ${r.id}`;
      btn.onclick = () => install(r, f); open.onclick = () => openInChat(r, f, here);
    } else if (!shelfLane) {
      chosen.innerHTML = `<b>${S.lane}</b><br>not on the shelf yet — build it here in one shot, or have it cooked`;
      btn.textContent = "BUILD IN ADVANCED"; btn.className = "install"; btn.disabled = false; prog.style.display = "none"; open.style.display = "none";
      cmd.textContent = `pollard --hf <model> --format ${S.lane.toLowerCase()} --run`;
      btn.onclick = () => { setMode("advanced"); if (window.show) window.show("build"); };
    } else {
      chosen.textContent = S.shelf.length ? "Pick a rung and this becomes one click." : "Loading the shelf…";
      btn.textContent = "INSTALL"; btn.className = "install"; btn.disabled = true; prog.style.display = "none"; open.style.display = "none"; cmd.textContent = "";
    }
    // narrow window: one pane at a time
    $("#scr-fit").querySelectorAll(".pane").forEach(p => p.classList.toggle("on", +p.dataset.page === S.page));
    $("#s-dots").innerHTML = [1, 2, 3].map(i => `<button aria-current="${i === S.page}" data-page="${i}"></button>`).join("");
    const rq = $("#s-rqdev"); if (rq && !rq.options.length) rq.innerHTML = DEVLIST.map(d => `<option value="${d.l}" ${d.id === S.dev ? "selected" : ""}>${d.l}</option>`).join("");
  }
  function renderShelf() {
    const sel = $("#s-devsel");
    sel.innerHTML = `<option value="detected" ${S.cat === "detected" ? "selected" : ""}>${esc(S.hw ? S.hw.label : "This machine")}</option>` +
      DEVLIST.map(d => `<option value="${d.id}" ${d.id === S.dev ? "selected" : ""}>${d.l}</option>`).join("");
    $("#s-only").setAttribute("aria-pressed", S.onlyFits);
    $("#s-mods").innerHTML = MODS.map(([m, l]) => { const n = m === "all" ? S.shelf.length : m === "moe" ? S.shelf.filter(r => r.moe).length : S.shelf.filter(r => modality(r) === m).length;
      return `<button class="chip" aria-pressed="${m === S.mod}" ${n ? "" : "disabled"} data-mod="${m}">${l}<b>${n}</b></button>`; }).join("");
    let list = S.shelf;
    if (S.mod === "moe") list = list.filter(r => r.moe); else if (S.mod !== "all") list = list.filter(r => modality(r) === S.mod);
    if (S.onlyFits) list = list.filter(r => r.files.some(f => verdict(f.gb)[0] !== "no"));
    $("#s-count").textContent = `${budget()} GB usable · ${list.length} of ${S.shelf.length} models`;
    $("#s-shelfnote").textContent = S.shelf.length ? "Sizes read from the published repos. Tap a rung to install it." : "Loading the shelf…";
    $("#s-grid").innerHTML = list.map(r => `<div class="pane card">
      <div><h4>${esc(r.name)}</h4><div class="meta"><span>${r.dl.toLocaleString()} downloads</span><span class="tg">${modality(r)}</span>${r.moe ? `<span class="tg moe">MoE</span>` : ""}${r.lane === "mlx" ? `<span class="tg">MLX</span>` : ""}</div></div>
      <div class="rungs">${r.files.map(f => { const [cls, txt] = verdict(f.gb); return `<button class="rung" aria-pressed="${f.file === S.file}" data-repo="${r.id}" data-file="${f.file}" data-lane="${r.lane === "mlx" ? "MLX" : "GGUF"}">
          <span>${f.q}${f.ik ? "<i>ik_llama</i>" : ""}${localPath(r, f) ? "<i style='color:var(--sok)'>on disk</i>" : ""}</span><span class="gb">${f.gb.toFixed(2)} GB</span><span class="v ${cls}">${txt}</span></button>`; }).join("")}</div>
      <div class="cfoot"><button data-url="https://huggingface.co/${r.id}">Model card</button><button data-url="https://huggingface.co/${r.id}/tree/main">All files</button></div></div>`).join("");
  }
  function renderCooks() {
    const g = $("#s-cooks"); if (!g) return;
    g.innerHTML = S.cooks ? S.cooks.map(k => { const st = k.stats || {}, tier = k.tier || tierOf(st.downloads || 0); return `<div class="pane card cook">
      <div class="who"><div><h4>${esc(k.name)}${k.verified ? '<span class="vcheck" title="Stripe-onboarded, gated builds delivered">✓</span>' : ""}</h4><small>${esc(k.role || "")}</small></div>
        <span class="tier t-${tier.toLowerCase()}" title="by total Hugging Face downloads">${tier}</span></div>
      <div class="stats"><span><b>${fmt(st.downloads)}</b> downloads</span><span><b>${fmt(st.models)}</b> models</span><span><b>${fmt(st.followers)}</b> followers</span><span><b>${k.delivered || 0}</b> cooks delivered</span></div>
      <div class="meta">${(k.lanes || []).map(l => `<span class="tg">${esc(l)}</span>`).join("")}<span class="st ${esc(k.status || "")}" style="margin-left:auto">${esc(k.status || "")}</span></div>
      <p>${esc(k.blurb || "")}</p><div class="hw">Hardware · <b>${esc(k.hardware || "—")}</b></div>
      <div class="cfoot">${k.hf ? `<button data-url="${esc(k.hf)}">Hugging Face</button>` : ""}${k.gh ? `<button data-url="${esc(k.gh)}">GitHub</button>` : ""}</div></div>`; }).join("") : `<p class="chosen">Loading…</p>`;
  }

  /* ── events (one delegated listener; the DOM is re-rendered often) ───── */
  function wire() {
    const root = $("#simple");
    root.addEventListener("click", e => {
      const t = e.target, c = s => t.closest(s);
      let el;
      if ((el = c("[data-win]")) && window.win) return window.win(el.dataset.win);
      if (c("[data-adv]")) return setMode("advanced");
      if ((el = c("[data-screen]"))) return go(el.dataset.screen);
      if ((el = c("[data-go]"))) return go(el.dataset.go);
      if ((el = c("[data-url]"))) return openUrl(el.dataset.url);
      if (c(".s-detect")) { S.cat = "detected"; S.dev = null; S.file = null; return render(); }
      if ((el = c("[data-cat]"))) { S.cat = el.dataset.cat; S.dev = Object.keys(CATS[S.cat].d)[0]; S.file = null; return render(); }
      if ((el = c("[data-dev]"))) { S.dev = el.dataset.dev; S.file = null; return render(); }
      if ((el = c("[data-lane]"))) { S.lane = el.dataset.lane; S.file = null; return render(); }
      if ((el = c(".rung"))) { S.repo = el.dataset.repo; S.file = el.dataset.file; S.lane = el.dataset.lane; S.page = 3; return go("fit"); }
      if ((el = c(".hit"))) { S.repo = el.dataset.repo; S.file = el.dataset.file; S.page = 3; return render(); }
      if ((el = c("[data-page]"))) { S.page = +el.dataset.page; return render(); }
      if ((el = c("[data-mod]"))) { S.mod = el.dataset.mod; return render(); }
      if (c("#s-only")) { S.onlyFits = !S.onlyFits; return render(); }
    });
    root.addEventListener("input", e => { if (e.target.id === "s-q") { S.q = e.target.value; renderFit(); } });
    root.addEventListener("change", e => { if (e.target.id === "s-devsel") { const v = e.target.value; if (v === "detected") { S.cat = "detected"; S.dev = null; } else { const d = DEVLIST.find(x => x.id === v); S.cat = d.k; S.dev = d.id; } S.file = null; render(); } });
    $("#s-form").addEventListener("submit", sendRequest);
  }
  function go(screen) { S.screen = screen; try { localStorage.setItem("pollard.simple.screen", screen); } catch (e) {} render(); if (screen === "cooks" && !S.cooks) readCooks().then(render); }
  function openUrl(u) { if (api() && api().open_url) api().open_url(u); else window.open(u); }

  /* ── actions ─────────────────────────────────────────────────────────── */
  async function install(r, f) {
    if (!api()) return;
    let res; try { res = await api().download(r.id, f.file); } catch (e) { res = { ok: false, error: String(e) }; }
    if (!res || !res.ok) { alert(res && res.error || "download failed"); return; }
    S.dl = { active: true, pct: 0, file: f.file }; render();
    const tick = async () => {
      let st; try { st = await api().download_status(); } catch (e) { st = { active: false, error: String(e) }; }
      if (st.active) { S.dl = { active: true, pct: st.pct || 0, file: f.file }; renderFit(); setTimeout(tick, 700); return; }
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
  async function sendRequest(e) {
    e.preventDefault();
    const f = e.target, btn = $("#s-send"), out = $("#s-sent");
    const data = Object.fromEntries(new FormData(f).entries()); btn.disabled = true; out.className = "sent";
    const done = m => { out.innerHTML = m; out.className = "sent on"; btn.disabled = false; };
    try {
      const r = api() && api().post_json ? await api().post_json(SITE + "/api/request", data)
        : await fetch(SITE + "/api/request", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }).then(async x => ({ ok: x.ok, data: await x.json().catch(() => ({})) }));
      if (r && r.ok && r.data && r.data.ticket) { f.reset(); return done(`Request received — ticket <b>#${esc(r.data.ticket)}</b>. You'll get a quote at <b>${esc(data.contact)}</b>. Track it at ${SITE}/?quote=${esc(r.data.ticket)}`); }
      done(`Couldn't reach pollard.app (${esc(r && r.error || "no response")}). Try again, or file it at ${SITE}/#request.`);
    } catch (err) { done(`Couldn't send: ${esc(String(err))}`); }
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
    await readMachine(); render();
    await readShelf(); render();
    readCooks().then(render);
    try { const st = api() && await api().download_status(); if (st && st.active) { S.dl = { active: true, pct: st.pct || 0, file: st.file }; render(); } } catch (e) {}
  }
  const boot = () => { let m = "simple"; try { m = localStorage.getItem("pollard.mode") || "simple"; } catch (e) {} setMode(m); };
  if (window.pywebview) boot(); else { window.addEventListener("pywebviewready", boot); window.addEventListener("load", () => setTimeout(() => { if (!S.hw) boot(); }, 350)); }
})();
