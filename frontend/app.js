/* ============================================================
   CONFIG + SESSION
============================================================ */
const API = "http://192.168.12.43:8001";
const sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", sessionId);

/* Demo-only admin credentials (UI gate only) */
const DEMO_ADMIN_USER = "admin";
const DEMO_ADMIN_PASS = "admin123";
const ADMIN_FLAG_KEY = "is_admin";

/* ============================================================
   GLOBAL STATE
============================================================ */
const state = {
  stage: 0,
  raw_input: null,

  enriched_features: null,
  enriched_features_original: null,   // backend enrichment snapshot
  enriched_features_edited: null,     // working copy while editing
  enriched_features_dirty: false,     // unsaved edits

  feature_explanations: null,

  predictions: null,
  shap_top: null,

  explanation: "",
  chat: [],

  riskTier: 1,
  timeToRaiseDays: 0,

  evidenceFiles: []
};

const steps = [
  { id: "step0", title: "Select" },
  { id: "step1", title: "Enrichment" },
  { id: "step2", title: "Prediction" },
  { id: "step3", title: "Trust" },
  { id: "stepDone", title: "Done" }
];

let disputesCache = [];
let hasDisputeId = true;
let shapChart = null;

/* ============================================================
   UI HELPERS
============================================================ */
const $ = (id) => document.getElementById(id);

function toast(msg){
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"), 1800);
}

function setOverlay(show, text="Working…"){
  $("overlayText").textContent = text;
  $("overlay").classList.toggle("hidden", !show);
}

function hidePageLoader(){
  const el = $("pageLoader");
  if(!el) return;
  el.style.transition = "opacity .25s ease";
  el.style.opacity = "0";
  setTimeout(()=>el.remove(), 260);
}

async function api(path, opts={}){
  const res = await fetch(`${API}${path}`, {
    headers: {"Content-Type":"application/json"},
    ...opts
  });
  if(!res.ok){
    const txt = await res.text().catch(()=> "");
    throw new Error(txt || `HTTP ${res.status}`);
  }
  return res.json();
}

function clamp(n, min, max){
  return Math.max(min, Math.min(max, n));
}

/* ============================================================
   AUTH (UI GATE ONLY)
============================================================ */
function authInit(){
  const ok = localStorage.getItem(ADMIN_FLAG_KEY)==="true";

  if(ok){
    $("authGate").classList.add("hidden");
    $("appRoot").classList.remove("hidden");
  } else {
    $("authGate").classList.remove("hidden");
    $("appRoot").classList.add("hidden");
  }

  $("loginHelpBtn").onclick = () => {
    toast("Demo creds: admin / admin123");
  };

  $("loginBtn").onclick = ()=>{
    const u = $("loginUser").value.trim();
    const p = $("loginPass").value;
    $("authError").textContent = "";

    if(u===DEMO_ADMIN_USER && p===DEMO_ADMIN_PASS){
      localStorage.setItem(ADMIN_FLAG_KEY,"true");
      $("authGate").classList.add("hidden");
      $("appRoot").classList.remove("hidden");
      toast("Logged in");
    } else {
      $("authError").textContent = "Invalid credentials";
    }
  };

  $("logoutBtn").onclick = ()=>{
    localStorage.removeItem(ADMIN_FLAG_KEY);
    location.reload();
  };
}

/* ============================================================
   ROADMAP / STAGE
============================================================ */
function renderRoadmap(){
  const r = $("roadmap");
  r.innerHTML = "";
  steps.forEach((s,i)=>{
    const n = document.createElement("div");
    n.className = "node";
    if(i===state.stage) n.classList.add("active");
    if(i<state.stage) n.classList.add("done");
    n.onclick = ()=>setStage(i);
    n.innerHTML = `<div class="dot"></div><div class="nodeTitle">${s.title}</div>`;
    r.appendChild(n);
  });
}

function updateProgress(){
  const pct = (state.stage/(steps.length-1))*100;
  $("progressFill").style.width = pct+"%";
}

function showPanel(i){
  steps.forEach((s,idx)=>{
    $(s.id).classList.toggle("hidden", idx!==i);
  });
  // add small enter animation
  const el = $(steps[i].id);
  el.classList.add("enter");
  setTimeout(()=>el.classList.remove("enter"), 340);
}

function setStage(i){
  state.stage = i;
  renderRoadmap();
  updateProgress();
  showPanel(i);
  if(i===1) setOverrideInputs();
}

/* ============================================================
   DUMMY EVIDENCE UPLOAD
============================================================ */
function formatBytes(b){
  if(!b) return "";
  const u=["B","KB","MB","GB"];
  let i=0; let x=b;
  while(x>=1024 && i<u.length-1){ x/=1024; i++; }
  return `${x.toFixed(i?1:0)} ${u[i]}`;
}

function updateEvidenceUI(){
  const meta = $("evidenceMeta");
  const list = $("evidenceList");

  const n = state.evidenceFiles.length;
  meta.textContent = n===0 ? "No files uploaded" : `${n} file${n>1?"s":""} uploaded`;
  list.innerHTML = "";

  state.evidenceFiles.forEach(f=>{
    const d = document.createElement("div");
    d.className = "filePill";
    d.innerHTML = `<div>${f.name}</div><div style="color:var(--muted)">${formatBytes(f.size)}</div>`;
    list.appendChild(d);
  });
}

function resetEvidence(){
  state.evidenceFiles = [];
  updateEvidenceUI();
}

/* ============================================================
   CONTROLS (SEGMENT + SLIDER + INPUT)
============================================================ */
function markOverridesDirty(){
  const box = $("overrideDiff");
  if(box && !box.textContent.includes("Pending")){
    box.textContent = "Pending changes (apply + recompute to update enrichment output).";
  }
}

function setRiskTier(v){
  state.riskTier = Number(v);
  document.querySelectorAll("#riskSegment .segBtn")
    .forEach(b=>b.classList.toggle("active", b.dataset.value==String(v)));
  markOverridesDirty();
}

function setTtr(v){
  const slider = $("ttrSlider");
  const input  = $("ttrInput");
  const min = Number(slider.min ?? 0);
  const max = Number(slider.max ?? 365);

  const num = clamp(Number(v || 0), min, max);
  state.timeToRaiseDays = num;

  slider.value = String(num);
  input.value = String(num);
  $("ttrValue").textContent = String(num);

  markOverridesDirty();
}

function setOverrideInputs(){
  const r = state.raw_input || {};
  $("mccInput").value = (r.mcc ?? "");
  $("amtInput").value = (r.txn_amount ?? "");

  const tier = Number(r.merchant_risk_tier ?? 1);
  setRiskTier(Number.isFinite(tier) ? tier : 1);

  const ttr = Number(r.time_to_raise_days ?? r.time_to_raise ?? 0);
  setTtr(Number.isFinite(ttr) ? ttr : 0);

  $("overrideDiff").textContent = "No changes.";
}

/* ============================================================
   ENRICHED FEATURES (EDITABLE) + BANNERS
============================================================ */
function setDirtyBanner(on){
  $("featDirtyBanner").classList.toggle("hidden", !on);
  if(on){
    $("featSavedBanner").classList.add("hidden");
  }
}

function flashSavedBanner(){
  $("featSavedBanner").classList.remove("hidden");
  setTimeout(()=> $("featSavedBanner").classList.add("hidden"), 2200);
}

function renderEditableFeatures(){
  const t = $("featTable");
  t.innerHTML = "";

  if(!state.enriched_features) return;

  const thead = `<thead><tr><th>Feature</th><th>Value</th></tr></thead>`;
  const tbody = Object.entries(state.enriched_features).map(([k,v])=>`
    <tr>
      <td>${k}</td>
      <td><input class="cellInput" data-k="${k}" value="${String(v)}"/></td>
    </tr>
  `).join("");

  t.innerHTML = thead + `<tbody>${tbody}</tbody>`;

  t.querySelectorAll("input").forEach(inp=>{
    inp.addEventListener("input", ()=>{
      if(!state.enriched_features_edited){
        state.enriched_features_edited = structuredClone(state.enriched_features);
      }
      const key = inp.dataset.k;
      const raw = inp.value;

      const asNum = Number(raw);
      state.enriched_features_edited[key] = (raw !== "" && Number.isFinite(asNum)) ? asNum : raw;

      state.enriched_features_dirty = true;
      setDirtyBanner(true);
    });
  });

  setDirtyBanner(state.enriched_features_dirty);
}

function saveFeatureEdits(){
  if(!state.enriched_features_dirty || !state.enriched_features_edited){
    return toast("No edits to save");
  }
  state.enriched_features = structuredClone(state.enriched_features_edited);
  state.enriched_features_dirty = false;
  state.enriched_features_edited = null;
  setDirtyBanner(false);
  flashSavedBanner();
  toast("Feature edits saved");
}

function resetFeatureEdits(){
  if(!state.enriched_features_original){
    return toast("Nothing to reset");
  }
  state.enriched_features = structuredClone(state.enriched_features_original);
  state.enriched_features_edited = null;
  state.enriched_features_dirty = false;
  setDirtyBanner(false);
  renderEditableFeatures();
  toast("Edits reset");
}

function reapplySavedEditsOnTop(newEnriched){
  if(!state.enriched_features || !state.enriched_features_original) return newEnriched;

  const merged = structuredClone(newEnriched);

  for(const [k, vSaved] of Object.entries(state.enriched_features)){
    const vOrig = state.enriched_features_original?.[k];
    if(String(vSaved) !== String(vOrig)){
      merged[k] = vSaved;
    }
  }
  return merged;
}

/* ============================================================
   DATA LOADING
============================================================ */
async function loadDisputes(){
  const d = await api("/disputes");
  disputesCache = d.items;
  hasDisputeId = d.has_dispute_id;

  const s = $("disputeSelect");
  s.innerHTML="";
  disputesCache.slice(0,500).forEach(it=>{
    const o=document.createElement("option");
    o.value = hasDisputeId ? it.dispute_id : it.row_index;
    o.textContent = hasDisputeId ? it.dispute_id : `Row ${it.row_index}`;
    s.appendChild(o);
  });
}

function filterDisputes(q){
  const s = $("disputeSelect");
  s.innerHTML="";
  const items = disputesCache.filter(it=>{
    const v = hasDisputeId ? String(it.dispute_id) : String(it.row_index);
    return v.toLowerCase().includes(q.toLowerCase());
  }).slice(0,500);

  items.forEach(it=>{
    const o=document.createElement("option");
    o.value = hasDisputeId ? it.dispute_id : it.row_index;
    o.textContent = hasDisputeId ? it.dispute_id : `Row ${it.row_index}`;
    s.appendChild(o);
  });
}

/* ============================================================
   STEP ACTIONS
============================================================ */
async function selectCurrent(){
  const v = $("disputeSelect").value;
  const payload = hasDisputeId
    ? {session_id:sessionId, dispute_id:v}
    : {session_id:sessionId, row_index:Number(v)};

  setOverlay(true,"Loading dispute…");
  try{
    const r = await api("/select",{method:"POST",body:JSON.stringify(payload)});
    state.raw_input = r.raw_input;
    $("rawJson").textContent = JSON.stringify(r.raw_input,null,2);

    state.enriched_features = null;
    state.enriched_features_original = null;
    state.enriched_features_edited = null;
    state.enriched_features_dirty = false;

    state.predictions = null;
    state.shap_top = null;
    state.explanation = "";
    state.chat = [];

    resetEvidence();
    setOverrideInputs();

    $("featTable").innerHTML = "";
    $("shapTable").innerHTML = "";
    $("predCategory").textContent = "—";
    $("predProb").textContent = "—";
    $("catProbs").textContent = "—";
    $("explanationText").textContent = "—";

    toast("Dispute loaded");
  }catch(e){
    console.error(e);
    toast("Failed to load dispute");
  }finally{
    setOverlay(false);
  }
}

async function runAgent1(){
  if(!state.raw_input) return toast("Select a dispute first");
  setOverlay(true,"Running Enrichment…");
  try{
    const r = await api("/agent1/run",{method:"POST",body:JSON.stringify({session_id:sessionId})});
    state.enriched_features = r.enriched_features;
    state.enriched_features_original = structuredClone(r.enriched_features);

    state.enriched_features_edited = null;
    state.enriched_features_dirty = false;
    setDirtyBanner(false);

    renderEditableFeatures();
    toast("Enrichment complete");
  }catch(e){
    console.error(e);
    toast("Enrichment failed");
  }finally{
    setOverlay(false);
  }
}

async function applyOverride(){
  if(!state.raw_input) return toast("Select a dispute first");

  const patch = {
    mcc: Number($("mccInput").value),
    merchant_risk_tier: state.riskTier,
    txn_amount: Number($("amtInput").value),
    time_to_raise_days: state.timeToRaiseDays
  };

  setOverlay(true,"Applying overrides + recomputing…");
  try{
    const r = await api("/agent1/override",{
      method:"POST",
      body:JSON.stringify({session_id:sessionId, patch})
    });

    state.raw_input = r.raw_input;
    $("rawJson").textContent = JSON.stringify(r.raw_input,null,2);

    const newEnriched = r.enriched_features;
    const merged = reapplySavedEditsOnTop(newEnriched);

    state.enriched_features = merged;
    state.enriched_features_original = structuredClone(newEnriched);
    state.enriched_features_edited = null;
    state.enriched_features_dirty = false;
    setDirtyBanner(false);

    renderEditableFeatures();
    $("overrideDiff").textContent = "Overrides applied and enrichment recomputed.";
    toast("Recomputed successfully");
  }catch(e){
    console.error(e);
    toast("Override failed");
  }finally{
    setOverlay(false);
  }
}

function renderTable(tableId, rows, headers){
  const el = $(tableId);
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
  rows.forEach(r=>{
    const tr = document.createElement("tr");
    headers.forEach(h=>{
      const td = document.createElement("td");
      td.textContent = (r[h] ?? "").toString();
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  el.appendChild(tbody);
}

async function runAgent2(){
  if(!state.enriched_features) return toast("Run Enrichment first");
  if(state.enriched_features_dirty) return toast("Save enriched feature edits before predicting");

  setOverlay(true,"Running Prediction…");
  try{
    const r = await api("/agent2/run",{
      method:"POST",
      body:JSON.stringify({
        session_id:sessionId,
        enriched_features_override: state.enriched_features
      })
    });

    state.predictions = r.predictions;
    state.shap_top = r.shap_top;

    $("predCategory").textContent = r.predictions?.predicted_category ?? "—";
    $("predProb").textContent =
      ((r.predictions?.predicted_customer_favor_prob ?? 0)*100).toFixed(2) + "%";
    $("catProbs").textContent = JSON.stringify(r.predictions?.category_proba ?? {}, null, 2);

    renderTable("shapTable", state.shap_top || [], ["feature","value","abs_value","direction"]);

    const shap = await api("/agent2/shap",{
      method:"POST",
      body:JSON.stringify({
        session_id:sessionId,
        enriched_features_override: state.enriched_features
      })
    });

    const labels = (shap.top || []).map(x=>x.feature).reverse();
    const vals   = (shap.top || []).map(x=>x.value).reverse();

    const ctx = $("shapChart").getContext("2d");
    if(shapChart) shapChart.destroy();
    shapChart = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets: [{ label: "SHAP value (impact)", data: vals }] },
      options: {
        indexAxis: "y",
        responsive: true,
        animation: { duration: 700, easing: "easeOutQuart" },
        plugins: { legend: { display: false } }
      }
    });

    toast("Prediction complete");
  }catch(e){
    console.error(e);
    toast("Prediction failed");
  }finally{
    setOverlay(false);
  }
}

function renderChat(){
  const el = $("chat");
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
  if(!state.predictions) return toast("Run Prediction first");

  setOverlay(true,"Generating trust note…");
  try{
    const r = await api("/agent3/explain",{
      method:"POST",
      body:JSON.stringify({session_id:sessionId})
    });
    state.explanation = r.explanation;
    $("explanationText").textContent = r.explanation;

    state.chat = [{ role:"bot", text:"Trust note generated. Ask a follow-up question anytime." }];
    renderChat();
    toast("Trust note ready");
  }catch(e){
    console.error(e);
    toast("Failed to generate");
  }finally{
    setOverlay(false);
  }
}

async function ask(){
  const q = $("questionInput").value.trim();
  if(!q) return toast("Type a question first");
  if(!state.explanation) return toast("Generate trust note first");

  state.chat.push({role:"user", text:q});
  renderChat();
  $("questionInput").value = "";

  setOverlay(true,"Thinking…");
  try{
    const r = await api("/agent3/ask",{
      method:"POST",
      body:JSON.stringify({session_id:sessionId, question:q})
    });
    state.chat.push({role:"bot", text:r.answer});
    state.explanation = r.explanation;
    $("explanationText").textContent = r.explanation;
    renderChat();
  }catch(e){
    console.error(e);
    toast("Failed to answer");
  }finally{
    setOverlay(false);
  }
}

/* ============================================================
   WIRING
============================================================ */
function wire(){
  $("sessionPill").textContent = `Session: ${sessionId.slice(0,8)}…`;

  $("searchInput").addEventListener("input", (e)=>filterDisputes(e.target.value));
  $("disputeSelect").addEventListener("change", selectCurrent);
  $("reloadRowBtn").onclick = selectCurrent;
  $("goAgent1Btn").onclick = ()=>{ if(!state.raw_input) return toast("Select a dispute"); setStage(1); };

  $("back0Btn").onclick = ()=>setStage(0);
  $("runA1Btn").onclick = runAgent1;
  $("applyOverrideBtn").onclick = applyOverride;
  $("goAgent2Btn").onclick = ()=>{ if(!state.enriched_features) return toast("Run Enrichment first"); setStage(2); };

  $("approveEnrichBtn").onclick = ()=>toast("Enrichment approved ✅");
  $("rejectEnrichBtn").onclick = ()=>toast("Not approved ❌");

  document.querySelectorAll("#riskSegment .segBtn")
    .forEach(b=>b.onclick=()=>setRiskTier(b.dataset.value));

  $("ttrSlider").addEventListener("input", (e)=> setTtr(e.target.value));
  $("ttrSlider").addEventListener("change", (e)=> setTtr(e.target.value));

  $("ttrInput").addEventListener("input", (e)=> {
    if(e.target.value === "") return;
    setTtr(e.target.value);
  });
  $("ttrInput").addEventListener("change", (e)=> {
    if(e.target.value === "") setTtr(0);
    else setTtr(e.target.value);
  });

  $("mccInput").addEventListener("input", markOverridesDirty);
  $("amtInput").addEventListener("input", markOverridesDirty);

  $("uploadEvidenceBtn").onclick = ()=>$("evidenceInput").click();
  $("evidenceInput").onchange = ()=>{
    const f = Array.from($("evidenceInput").files||[]);
    state.evidenceFiles.push(...f);
    updateEvidenceUI();
    toast(`${f.length} file(s) added`);
    $("evidenceInput").value="";
  };

  $("saveFeatEditsBtn").onclick = saveFeatureEdits;
  $("resetFeatEditsBtn").onclick = resetFeatureEdits;

  $("back1Btn").onclick = ()=>setStage(1);
  $("runA2Btn").onclick = runAgent2;
  $("goAgent3Btn").onclick = ()=>{ if(!state.predictions) return toast("Run Prediction first"); setStage(3); };

  $("back2Btn").onclick = ()=>setStage(2);
  $("genExplBtn").onclick = genExplanation;
  $("askBtn").onclick = ask;
  $("resetChatBtn").onclick = ()=>{ state.chat=[]; renderChat(); toast("Chat reset"); };

  $("approveCloseBtn").onclick = ()=>{
    if(!state.explanation) return toast("Generate trust note first");

    $("finalSummary").textContent =
      `dispute_id: ${state.raw_input?.dispute_id ?? "—"}\n` +
      `predicted_category: ${state.predictions?.predicted_category ?? "—"}\n` +
      `customer_favor_prob: ${((state.predictions?.predicted_customer_favor_prob ?? 0)*100).toFixed(2)}%`;

    $("finalExplanation").textContent = state.explanation;
    setStage(4);
  };

  $("newCaseBtn").onclick = ()=>window.location.reload();
  $("reviewA2Btn").onclick = ()=>setStage(2);

  $("resetBtn").onclick = async ()=>{
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

/* ============================================================
   INIT
============================================================ */
(async function init(){
  authInit();
  renderRoadmap();
  showPanel(0);
  updateProgress();

  setRiskTier(1);
  setTtr(0);
  resetEvidence();

  try{
    await loadDisputes();
  }catch(e){
    console.error(e);
    toast("Failed to load disputes");
  }finally{
    wire();
    hidePageLoader();
  }
})();