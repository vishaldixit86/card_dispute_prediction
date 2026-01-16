const API = "http://localhost:8000";
const sessionId = (localStorage.getItem("session_id") || crypto.randomUUID());
localStorage.setItem("session_id", sessionId);

document.getElementById("sessionPill").textContent = `Session: ${sessionId.slice(0, 8)}…`;

const state = {
  stage: 0,
  raw_input: null,
  enriched_features: null,
  feature_explanations: null,
  predictions: null,
  shap_top: null,
  explanation: "",
  chat: []
};

const steps = [
  { id: "step0", key: 0, title: "Select", sub: "Choose dispute" },
  { id: "step1", key: 1, title: "Agent 1", sub: "Enrich + review" },
  { id: "step2", key: 2, title: "Agent 2", sub: "Predict + SHAP" },
  { id: "step3", key: 3, title: "Agent 3", sub: "Explain + Q&A" },
  { id: "stepDone", key: 4, title: "Done", sub: "Close case" }
];

let shapChart = null;
let disputesCache = [];
let hasDisputeId = true;

function toast(msg){
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"), 1700);
}

function setOverlay(show, text="Working…"){
  const o = document.getElementById("overlay");
  const t = document.getElementById("overlayText");
  if(!o) return;
  t.textContent = text;
  o.classList.toggle("hidden", !show);
}

async function api(path, opts={}){
  const res = await fetch(`${API}${path}`, {
    headers: {"Content-Type":"application/json"},
    ...opts
  });
  if(!res.ok){
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function canGo(targetStage){
  if(targetStage <= state.stage) return true;
  if(targetStage === 1) return !!state.raw_input;
  if(targetStage === 2) return !!state.enriched_features;
  if(targetStage === 3) return !!state.predictions;
  if(targetStage === 4) return !!state.explanation;
  return false;
}

/* ---------- Roadmap rendering ---------- */
function renderRoadmap(){
  const el = document.getElementById("roadmap");
  el.innerHTML = "";

  steps.forEach((s, i)=>{
    const node = document.createElement("div");
    node.className = "node";
    if(i === state.stage) node.classList.add("active");
    if(i < state.stage) node.classList.add("done");
    if(!canGo(i)) node.classList.add("locked");

    node.onclick = () => {
      if(!canGo(i)) return toast("Complete previous step first.");
      setStage(i);
    };

    const dot = document.createElement("div");
    dot.className = "dot";

    const text = document.createElement("div");
    text.className = "nodeText";
    text.innerHTML = `<div class="nodeTitle">${s.title}</div><div class="nodeSub">${s.sub}</div>`;

    node.appendChild(dot);
    node.appendChild(text);
    el.appendChild(node);
  });
}

function updateProgress(){
  const fill = document.getElementById("progressFill");
  const glow = document.getElementById("progressGlow");
  const track = document.querySelector(".progressTrack");
  if(!fill || !glow || !track) return;

  const pct = (state.stage / (steps.length - 1)) * 100;
  fill.style.width = `${pct}%`;

  // glow moves with the fill head
  const trackWidth = track.getBoundingClientRect().width;
  const x = Math.max(-140, (trackWidth * (pct / 100)) - 60);
  glow.style.transform = `translateX(${x}px)`;

  // pulse glow while "active"
  track.classList.add("active");
  clearTimeout(updateProgress._t);
  updateProgress._t = setTimeout(()=>track.classList.remove("active"), 900);
}

/* ---------- Panel transitions ---------- */
function showPanel(stageIdx){
  const current = steps.find((s)=> !document.getElementById(s.id).classList.contains("hidden"));
  const next = steps[stageIdx];

  if(current && current.id !== next.id){
    const curEl = document.getElementById(current.id);
    curEl.classList.remove("enter");
    curEl.classList.add("exit");

    setTimeout(()=>{
      curEl.classList.add("hidden");
      curEl.classList.remove("exit");

      const nextEl = document.getElementById(next.id);
      nextEl.classList.remove("hidden");
      nextEl.classList.add("enter");
      setTimeout(()=> nextEl.classList.remove("enter"), 360);
    }, 180);
  } else {
    steps.forEach((s, i)=>{
      const el = document.getElementById(s.id);
      el.classList.toggle("hidden", i !== stageIdx);
      if(i === stageIdx){
        el.classList.add("enter");
        setTimeout(()=> el.classList.remove("enter"), 360);
      }
    });
  }
}

function setStage(i){
  state.stage = i;
  renderRoadmap();
  showPanel(i);
  updateProgress();

  // when moving to Step 1, populate override inputs
  if(i === 1) setOverrideInputs();
}

function json(elId, obj){
  document.getElementById(elId).textContent = obj ? JSON.stringify(obj, null, 2) : "—";
}

function renderTable(tableEl, rows, headers){
  const el = document.getElementById(tableEl);
  el.innerHTML = "";

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  headers.forEach(h=>{
    const th = document.createElement("th");
    th.textContent = h;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  el.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((r, idx)=>{
    const tr = document.createElement("tr");
    tr.style.animationDelay = `${Math.min(idx * 18, 200)}ms`;
    headers.forEach(h=>{
      const td = document.createElement("td");
      td.textContent = (r[h] ?? "").toString();
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  el.appendChild(tbody);
}

function renderExplanations(){
  const wrap = document.getElementById("explanations");
  wrap.innerHTML = "";
  if(!state.feature_explanations) return;

  Object.entries(state.feature_explanations).forEach(([k,v])=>{
    const d = document.createElement("details");
    const s = document.createElement("summary");
    s.textContent = k;

    const p = document.createElement("div");
    p.className = "small";
    p.textContent = v;

    d.appendChild(s);
    d.appendChild(p);
    wrap.appendChild(d);
  });
}

function setOverrideInputs(){
  const r = state.raw_input || {};
  document.getElementById("mccInput").value = (r.mcc ?? "");
  document.getElementById("riskInput").value = (r.merchant_risk_tier ?? "");
  document.getElementById("amtInput").value = (r.txn_amount ?? "");
  document.getElementById("ttrInput").value = (r.time_to_raise_days ?? r.time_to_raise ?? "");
}

function diffPatch(oldObj, patch){
  const lines = [];
  Object.keys(patch).forEach(k=>{
    const before = oldObj?.[k];
    const after = patch[k];
    if(String(before) !== String(after)){
      lines.push(`${k}: ${before} → ${after}`);
    }
  });
  return lines;
}

/* ---------- Data loading ---------- */
async function loadDisputes(){
  const data = await api("/disputes");
  hasDisputeId = data.has_dispute_id;
  disputesCache = data.items;

  const sel = document.getElementById("disputeSelect");
  sel.innerHTML = "";
  disputesCache.slice(0, 500).forEach(item=>{
    const opt = document.createElement("option");
    opt.value = hasDisputeId ? item.dispute_id : item.row_index;
    opt.textContent = hasDisputeId ? item.dispute_id : `Row ${item.row_index}`;
    sel.appendChild(opt);
  });
}

function filterDisputes(q){
  const sel = document.getElementById("disputeSelect");
  sel.innerHTML = "";
  const items = disputesCache.filter(it => {
    const v = hasDisputeId ? String(it.dispute_id) : String(it.row_index);
    return v.toLowerCase().includes(q.toLowerCase());
  }).slice(0, 500);

  items.forEach(item=>{
    const opt = document.createElement("option");
    opt.value = hasDisputeId ? item.dispute_id : item.row_index;
    opt.textContent = hasDisputeId ? item.dispute_id : `Row ${item.row_index}`;
    sel.appendChild(opt);
  });
}

/* ---------- Step actions ---------- */
async function selectCurrent(){
  const sel = document.getElementById("disputeSelect");
  const val = sel.value;

  const payload = hasDisputeId
    ? { session_id: sessionId, dispute_id: val }
    : { session_id: sessionId, row_index: Number(val) };

  setOverlay(true, "Loading dispute…");
  try{
    const res = await api("/select", { method:"POST", body: JSON.stringify(payload) });
    state.raw_input = res.raw_input;
    state.enriched_features = null;
    state.feature_explanations = null;
    state.predictions = null;
    state.shap_top = null;
    state.explanation = "";
    state.chat = [];

    json("rawJson", state.raw_input);
    toast("Dispute loaded");
    renderRoadmap();
  }catch(e){
    console.error(e);
    toast("Failed to load dispute");
  }finally{
    setOverlay(false);
  }
}

async function runAgent1(){
  if(!state.raw_input) return toast("Select a dispute first");

  const featTable = document.getElementById("featTable");
  featTable.innerHTML = `<tbody>
    <tr><td colspan="2"><div class="skeleton" style="height:34px"></div></td></tr>
    <tr><td colspan="2"><div class="skeleton" style="height:34px"></div></td></tr>
    <tr><td colspan="2"><div class="skeleton" style="height:34px"></div></td></tr>
  </tbody>`;

  setOverlay(true, "Running enrichment (Agent 1)…");
  try{
    const res = await api("/agent1/run", { method:"POST", body: JSON.stringify({session_id: sessionId}) });
    state.enriched_features = res.enriched_features;
    state.feature_explanations = res.feature_explanations;

    const rows = Object.entries(state.enriched_features).map(([feature,value])=>({feature, value}));
    renderTable("featTable", rows, ["feature","value"]);
    renderExplanations();
    toast("Agent 1 complete");
  }catch(e){
    console.error(e);
    toast("Agent 1 failed");
  }finally{
    setOverlay(false);
  }
}

async function applyOverride(){
  if(!state.raw_input) return toast("Select a dispute first");

  const patch = {
    mcc: Number(document.getElementById("mccInput").value),
    merchant_risk_tier: Number(document.getElementById("riskInput").value),
    txn_amount: Number(document.getElementById("amtInput").value),
    time_to_raise_days: Number(document.getElementById("ttrInput").value),
  };

  if(patch.merchant_risk_tier < 1 || patch.merchant_risk_tier > 4){
    return toast("merchant_risk_tier must be 1–4");
  }

  const diffs = diffPatch(state.raw_input, patch);
  document.getElementById("overrideDiff").textContent =
    diffs.length ? `Changed:\n- ${diffs.join("\n- ")}` : "No changes.";

  setOverlay(true, "Applying overrides + recomputing (Agent 1)…");
  try{
    const res = await api("/agent1/override", { method:"POST", body: JSON.stringify({session_id: sessionId, patch}) });
    state.raw_input = res.raw_input;
    state.enriched_features = res.enriched_features;
    state.feature_explanations = res.feature_explanations;

    json("rawJson", state.raw_input);
    setOverrideInputs();

    const rows = Object.entries(state.enriched_features).map(([feature,value])=>({feature, value}));
    renderTable("featTable", rows, ["feature","value"]);
    renderExplanations();
    toast("Overrides applied");
  }catch(e){
    console.error(e);
    toast("Override failed");
  }finally{
    setOverlay(false);
  }
}

async function runAgent2(){
  if(!state.enriched_features) return toast("Run Agent 1 first");

  const shapTable = document.getElementById("shapTable");
  shapTable.innerHTML = `<tbody>
    <tr><td colspan="4"><div class="skeleton" style="height:34px"></div></td></tr>
    <tr><td colspan="4"><div class="skeleton" style="height:34px"></div></td></tr>
    <tr><td colspan="4"><div class="skeleton" style="height:34px"></div></td></tr>
  </tbody>`;

  setOverlay(true, "Running prediction (Agent 2)…");
  try{
    const res = await api("/agent2/run", { method:"POST", body: JSON.stringify({session_id: sessionId}) });
    state.predictions = res.predictions;
    state.shap_top = res.shap_top;

    document.getElementById("predCategory").textContent = state.predictions.predicted_category ?? "—";
    document.getElementById("predProb").textContent =
      ((state.predictions.predicted_customer_favor_prob ?? 0) * 100).toFixed(2) + "%";
    document.getElementById("catProbs").textContent = JSON.stringify(state.predictions.category_proba ?? {}, null, 2);

    renderTable("shapTable", state.shap_top || [], ["feature","value","abs_value","direction"]);

    const shap = await api("/agent2/shap", { method:"POST", body: JSON.stringify({session_id: sessionId}) });
    const labels = shap.top.map(x=>x.feature).reverse();
    const vals = shap.top.map(x=>x.value).reverse();

    const ctx = document.getElementById("shapChart").getContext("2d");
    if(shapChart) shapChart.destroy();
    shapChart = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets: [{ label: "SHAP value (impact)", data: vals }] },
      options: {
        indexAxis: "y",
        responsive: true,
        animation: { duration: 520, easing: "easeOutQuart" },
        plugins: { legend: { display: false } }
      }
    });

    toast("Agent 2 complete");
  }catch(e){
    console.error(e);
    toast("Agent 2 failed");
  }finally{
    setOverlay(false);
  }
}

function renderChat(){
  const el = document.getElementById("chat");
  el.innerHTML = "";
  state.chat.forEach(m=>{
    const b = document.createElement("div");
    b.className = "bubble " + (m.role === "user" ? "user" : "bot");
    b.textContent = m.text;
    el.appendChild(b);
  });
  el.scrollTop = el.scrollHeight;
}

async function genExplanation(){
  if(!state.predictions) return toast("Run Agent 2 first");

  setOverlay(true, "Generating explanation (Agent 3)…");
  try{
    const res = await api("/agent3/explain", { method:"POST", body: JSON.stringify({session_id: sessionId}) });
    state.explanation = res.explanation;
    document.getElementById("explanationText").textContent = state.explanation;

    state.chat = [{ role:"bot", text: "Explanation generated. Ask a follow-up question anytime." }];
    renderChat();
    toast("Explanation ready");
  }catch(e){
    console.error(e);
    toast("Explanation failed");
  }finally{
    setOverlay(false);
  }
}

async function ask(){
  const q = document.getElementById("questionInput").value.trim();
  if(!q) return toast("Type a question first");
  if(!state.explanation) return toast("Generate explanation first");

  state.chat.push({role:"user", text:q});
  renderChat();
  document.getElementById("questionInput").value = "";

  setOverlay(true, "Thinking…");
  try{
    const res = await api("/agent3/ask", { method:"POST", body: JSON.stringify({session_id: sessionId, question:q}) });
    state.chat.push({role:"bot", text: res.answer});
    state.explanation = res.explanation;
    document.getElementById("explanationText").textContent = state.explanation;
    renderChat();
  }catch(e){
    console.error(e);
    toast("Failed to answer");
  }finally{
    setOverlay(false);
  }
}

/* ---------- Wiring ---------- */
function wire(){
  // Step 0
  document.getElementById("searchInput").addEventListener("input", (e)=>filterDisputes(e.target.value));
  document.getElementById("disputeSelect").addEventListener("change", selectCurrent);
  document.getElementById("reloadRowBtn").onclick = selectCurrent;
  document.getElementById("goAgent1Btn").onclick = ()=>{ if(!state.raw_input) return toast("Select a dispute"); setStage(1); };

  // Step 1
  document.getElementById("back0Btn").onclick = ()=>setStage(0);
  document.getElementById("runA1Btn").onclick = runAgent1;
  document.getElementById("applyOverrideBtn").onclick = applyOverride;
  document.getElementById("goAgent2Btn").onclick = ()=>{ if(!state.enriched_features) return toast("Run Agent 1 first"); setStage(2); };
  document.getElementById("approveEnrichBtn").onclick = ()=>toast("Enrichment approved ✅");
  document.getElementById("rejectEnrichBtn").onclick = ()=>toast("Not approved ❌");

  // Step 2
  document.getElementById("back1Btn").onclick = ()=>setStage(1);
  document.getElementById("runA2Btn").onclick = runAgent2;
  document.getElementById("goAgent3Btn").onclick = ()=>{ if(!state.predictions) return toast("Run Agent 2 first"); setStage(3); };

  // Step 3
  document.getElementById("back2Btn").onclick = ()=>setStage(2);
  document.getElementById("genExplBtn").onclick = genExplanation;
  document.getElementById("askBtn").onclick = ask;
  document.getElementById("resetChatBtn").onclick = ()=>{ state.chat=[]; renderChat(); toast("Chat reset"); };

  document.getElementById("approveCloseBtn").onclick = ()=>{
    if(!state.explanation) return toast("Generate explanation first");

    document.getElementById("finalSummary").textContent =
      `dispute_id: ${state.raw_input?.dispute_id ?? "—"}\n` +
      `predicted_category: ${state.predictions?.predicted_category ?? "—"}\n` +
      `customer_favor_prob: ${((state.predictions?.predicted_customer_favor_prob ?? 0)*100).toFixed(2)}%`;

    document.getElementById("finalExplanation").textContent = state.explanation;
    setStage(4);
  };

  // Done
  document.getElementById("newCaseBtn").onclick = ()=>window.location.reload();
  document.getElementById("reviewA2Btn").onclick = ()=>setStage(2);

  // Global reset
  document.getElementById("resetBtn").onclick = async ()=>{
    setOverlay(true, "Resetting session…");
    try{
      await api("/session/reset", { method:"POST", body: JSON.stringify({session_id: sessionId}) });
      toast("Reset done");
      window.location.reload();
    }catch(e){
      console.error(e);
      toast("Reset failed");
      setOverlay(false);
    }
  };
}

/* ---------- Init ---------- */
(async function init(){
  renderRoadmap();
  showPanel(0);
  updateProgress();
  await loadDisputes();
  wire();
})();
