/* ============================================================
   CONFIG + SESSION
============================================================ */
const API = "http://localhost:8000";
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
  enriched_features_original: null,
  enriched_features_edited: null,
  enriched_features_dirty: false,

  feature_explanations: null,
  predictions: null,
  shap_top: null,

  explanation: "",
  chat: [],

  riskTier: 1,
  timeToRaiseDays: 0,

  // dummy evidence upload
  evidenceFiles: []
};

const steps = [
  { id: "step0", title: "Select" },
  { id: "step1", title: "Agent 1" },
  { id: "step2", title: "Agent 2" },
  { id: "step3", title: "Agent 3" },
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
  el.style.opacity = "0";
  setTimeout(()=>el.remove(), 250);
}

function api(path, opts={}){
  return fetch(`${API}${path}`, {
    headers: {"Content-Type":"application/json"},
    ...opts
  }).then(r=>{
    if(!r.ok) throw new Error(r.statusText);
    return r.json();
  });
}

/* ============================================================
   AUTH (UI GATE ONLY)
============================================================ */
function authInit(){
  const ok = localStorage.getItem(ADMIN_FLAG_KEY)==="true";
  if(ok){
    $("authGate").classList.add("hidden");
    $("appRoot").classList.remove("hidden");
  }

  $("loginBtn").onclick = ()=>{
    const u = $("loginUser").value.trim();
    const p = $("loginPass").value;
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
  let i=0; while(b>=1024 && i<u.length-1){b/=1024;i++;}
  return `${b.toFixed(i?1:0)} ${u[i]}`;
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
    d.innerHTML = `<div>${f.name}</div><div class="muted">${formatBytes(f.size)}</div>`;
    list.appendChild(d);
  });
}

function resetEvidence(){
  state.evidenceFiles = [];
  updateEvidenceUI();
}

/* ============================================================
   CONTROLS (SEGMENT + SLIDER)
============================================================ */
function setRiskTier(v){
  state.riskTier = Number(v);
  document.querySelectorAll("#riskSegment .segBtn")
    .forEach(b=>b.classList.toggle("active", b.dataset.value==v));
}

function setTtr(v){
  state.timeToRaiseDays = Number(v);
  $("ttrSlider").value = v;
  $("ttrValue").textContent = v;
}

/* ============================================================
   ENRICHED FEATURES (EDITABLE)
============================================================ */
function renderEditableFeatures(){
  const t = $("featTable");
  t.innerHTML = "";
  if(!state.enriched_features) return;

  const h = `<thead><tr><th>Feature</th><th>Value</th></tr></thead>`;
  const b = Object.entries(state.enriched_features).map(([k,v])=>`
    <tr>
      <td>${k}</td>
      <td><input class="cellInput" data-k="${k}" value="${v}"></td>
    </tr>
  `).join("");

  t.innerHTML = h+`<tbody>${b}</tbody>`;

  t.querySelectorAll("input").forEach(inp=>{
    inp.oninput = ()=>{
      if(!state.enriched_features_edited){
        state.enriched_features_edited = structuredClone(state.enriched_features);
      }
      state.enriched_features_edited[inp.dataset.k] = isNaN(inp.value)?inp.value:Number(inp.value);
      state.enriched_features_dirty = true;
    };
  });
}

function saveFeatureEdits(){
  if(!state.enriched_features_dirty) return toast("No edits");
  state.enriched_features = structuredClone(state.enriched_features_edited);
  state.enriched_features_dirty = false;
  toast("Feature edits saved");
}

function resetFeatureEdits(){
  state.enriched_features = structuredClone(state.enriched_features_original);
  state.enriched_features_edited = null;
  state.enriched_features_dirty = false;
  renderEditableFeatures();
  toast("Edits reset");
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
  disputesCache.forEach(it=>{
    const o=document.createElement("option");
    o.value = hasDisputeId?it.dispute_id:it.row_index;
    o.textContent=o.value;
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
  const r = await api("/select",{method:"POST",body:JSON.stringify(payload)});
  state.raw_input = r.raw_input;
  $("rawJson").textContent = JSON.stringify(r.raw_input,null,2);

  state.enriched_features = null;
  resetEvidence();
  toast("Dispute loaded");
  setOverlay(false);
}

async function runAgent1(){
  setOverlay(true,"Running Agent 1…");
  const r = await api("/agent1/run",{method:"POST",body:JSON.stringify({session_id:sessionId})});
  state.enriched_features = r.enriched_features;
  state.enriched_features_original = structuredClone(r.enriched_features);
  renderEditableFeatures();
  toast("Agent 1 complete");
  setOverlay(false);
}

async function runAgent2(){
  setOverlay(true,"Running Agent 2…");
  const r = await api("/agent2/run",{
    method:"POST",
    body:JSON.stringify({
      session_id:sessionId,
      enriched_features_override: state.enriched_features
    })
  });
  state.predictions = r.predictions;
  $("predCategory").textContent = r.predictions.predicted_category;
  $("predProb").textContent = ((r.predictions.predicted_customer_favor_prob||0)*100).toFixed(2)+"%";
  toast("Agent 2 complete");
  setOverlay(false);
}

async function genExplanation(){
  setOverlay(true,"Generating explanation…");
  const r = await api("/agent3/explain",{method:"POST",body:JSON.stringify({session_id:sessionId})});
  state.explanation = r.explanation;
  $("explanationText").textContent = r.explanation;
  toast("Explanation ready");
  setOverlay(false);
}

/* ============================================================
   WIRING
============================================================ */
function wire(){
  $("disputeSelect").onchange = selectCurrent;
  $("goAgent1Btn").onclick = ()=>setStage(1);
  $("runA1Btn").onclick = runAgent1;
  $("goAgent2Btn").onclick = ()=>setStage(2);
  $("runA2Btn").onclick = runAgent2;
  $("goAgent3Btn").onclick = ()=>setStage(3);
  $("genExplBtn").onclick = genExplanation;

  $("saveFeatEditsBtn").onclick = saveFeatureEdits;
  $("resetFeatEditsBtn").onclick = resetFeatureEdits;

  // segmented + slider
  document.querySelectorAll("#riskSegment .segBtn")
    .forEach(b=>b.onclick=()=>setRiskTier(b.dataset.value));
  $("ttrSlider").oninput = e=>setTtr(e.target.value);

  // evidence upload (dummy)
  $("uploadEvidenceBtn").onclick = ()=>$("evidenceInput").click();
  $("evidenceInput").onchange = ()=>{
    const f = Array.from($("evidenceInput").files||[]);
    state.evidenceFiles.push(...f);
    updateEvidenceUI();
    toast(`${f.length} file(s) added`);
    $("evidenceInput").value="";
  };
}

/* ============================================================
   INIT
============================================================ */
(async function init(){
  $("sessionPill").textContent = `Session: ${sessionId.slice(0,8)}…`;
  authInit();
  renderRoadmap();
  showPanel(0);
  updateProgress();
  await loadDisputes();
  wire();
  hidePageLoader();
})();
