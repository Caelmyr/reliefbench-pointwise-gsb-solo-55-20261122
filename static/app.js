const root = document.getElementById("application");
const modal = document.getElementById("dialog");
const notification = document.getElementById("notice");
const state = {
  catalog: null, item: null, active: "", selected: new Set(), revisions: new Map(), tool: "normals",
  view: "work", region: null, edit: null, corners: [], proposal: null, last: null, opened: null,
};

function el(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "htmlFor") node.htmlFor = value;
    else if (typeof value === "boolean" && key !== "aria-pressed" && key !== "aria-current") node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter(Boolean));
  return node;
}
function button(text, action, attrs = {}) {
  const node = el("button", {type:"button", text, ...attrs});
  node.onclick = () => guarded(action);
  return node;
}
function note(text, error = false) {
  notification.textContent = text;
  notification.hidden = !text;
  notification.className = error ? "error" : "";
}
async function guarded(action) {
  try { await action(); } catch (error) { note(error.message || String(error), true); }
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  if (!type.startsWith("application/json")) throw new Error(`Unexpected workbench response (${response.status}).`);
  const json = await response.json();
  if (!response.ok) throw new Error(json.error || `The workbench could not finish the operation (${response.status}).`);
  return json;
}
function json(method, body) {
  return {method, body:JSON.stringify(body), headers:{"Content-Type":"application/json"}};
}
function query(id) { return document.getElementById(id); }
function nameOf(photo) { return photo.metadata.title; }
function asText(value) {
  if (value === null || value === undefined) return "Unknown";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(asText).join(", ");
  if (typeof value === "object") return Object.entries(value).map(([k, v]) => `${human(k)} ${asText(v)}`).join(" / ");
  return String(value);
}
function human(value) { return value.replaceAll("_"," ").replaceAll("-"," ").replace(/^./, char => char.toUpperCase()); }
function link(text, href, attrs = {}) { return el("a", {text, href, class:"text-link", ...attrs}); }
function formField(label, name, value = "", attrs = {}) {
  const input = el("input", {name, value, ...attrs});
  return el("label", {}, [document.createTextNode(label), input]);
}
function jsonField(label, name, value) {
  const field = el("textarea", {name, rows:4});
  field.value = JSON.stringify(value);
  return el("label", {}, [document.createTextNode(label), field]);
}
function checkField(label, attrs) {
  return el("label", {class:"check"}, [el("input", {type:"checkbox", ...attrs}), document.createTextNode(label)]);
}
function selectField(label, name, values, preferred) {
  const box = el("select", {name, "aria-label":label});
  for (const value of values) {
    const [id, title] = Array.isArray(value) ? value : [value, human(value)];
    box.append(el("option",{value:id, text:title, selected:id === preferred}));
  }
  return el("label",{}, [document.createTextNode(label), box]);
}
function formValues(form) {
  const result = Object.fromEntries(new FormData(form).entries());
  for (const node of form.querySelectorAll("input[type=number]")) result[node.name] = Number(node.value);
  for (const node of form.querySelectorAll("input[type=checkbox]")) result[node.name] = node.checked;
  return result;
}
function dialog(title, children, submit, action = "Save") {
  const head = el("div",{class:"modal-head"},[el("h2",{text:title}),button("Close",()=>modal.close(),{class:"quiet"})]);
  const form = el("form",{},children);
  form.append(el("button",{type:"submit",text:action,class:"solid"}));
  form.onsubmit = event => {
    event.preventDefault();
    guarded(async () => { await submit(form); modal.close(); });
  };
  modal.replaceChildren(head, form);
  modal.showModal();
}
async function refreshCatalog() { state.catalog = await api("/api/catalog"); }
async function openSurface(id, preserve=false) {
  const loaded = await api(`/api/surfaces/${id}`);
  if (!preserve || state.item?.surface.id !== id) {
    state.active = loaded.captures[0]?.id || "";
    state.selected = new Set(loaded.captures.filter(c=>c.metadata.role==="raking").map(c=>c.id).slice(0,12));
    state.tool = "normals"; state.view = "work"; state.region = null; state.revisions.clear();
    state.opened = null; state.last = null; state.loadedInputs = null;
  }
  state.item = loaded;
  state.edit = null;
  renderSurface();
}
async function openLibrary() { await refreshCatalog(); state.item=null; renderLibrary(); }
function renderLibrary() {
  const frame = el("div",{class:"library"});
  frame.append(el("section",{class:"intro"},[el("h1",{text:"Surface library"}),
    el("p",{class:"muted",text:"Bring together photographs lit from different directions to inspect the relief, material and measured detail of an object."})]));
  const collection = el("div",{class:"collection"});
  for (const s of state.catalog.surfaces) {
    collection.append(el("article",{class:"card"},[el("h3",{text:s.title}),
      el("p",{class:"meta",text:`${s.accession || "Local project"} · ${s.width_mm} × ${s.height_mm} mm · ${s.captures} photographs`}),
      s.synthetic && el("small",{text:"Synthetic study"}), button("Open surface",()=>openSurface(s.id))]));
  }
  if (state.catalog.surfaces.length) frame.append(el("h2",{text:"Saved surfaces"}), collection);
  const samples = el("div",{class:"collection"});
  for (const s of state.catalog.samples) {
    samples.append(el("article",{class:"card"},[el("img",{src:`/api/examples/${s.key}/cover`,alt:s.title}),
      el("h3",{text:s.title}),el("p",{class:"meta",text:s.note}),
      button("Load example",async()=>{ const reply=await api(`/api/examples/${s.key}`,{method:"POST"}); await openSurface(reply.surface.id); })]));
  }
  frame.append(el("h2",{text:"Synthetic training examples"}),samples,button("Import capture set",captureSetDialog));
  root.replaceChildren(frame);
}
function surfaceDialog(item) {
  const saved = item?.surface;
  dialog(saved ? "Edit surface" : "New surface",[
    formField("Surface title","title",saved?.title || "",{required:true, maxlength:120}),
    formField("Accession","accession",saved?.accession || "",{maxlength:72}),
    el("div",{class:"split"},[formField("Width (mm)","width_mm",saved?.width_mm || 100,{type:"number",min:2,max:1500,step:"any"}),
      formField("Height (mm)","height_mm",saved?.height_mm || 70,{type:"number",min:2,max:1500,step:"any"})]),
    formField("Notes","note",saved?.note || "",{maxlength:800}),
    checkField("Synthetic study",{name:"synthetic",checked:Boolean(saved?.synthetic)}),
  ],async(form)=>{
    const result=await api(saved ? `/api/surfaces/${saved.id}` : "/api/surfaces",
      saved ? json("PUT",{surface:formValues(form),version:saved.version}) : json("POST",formValues(form)));
    await openSurface(result.surface.id,Boolean(saved)); note("Surface saved.");
  });
}
function captureDialog(photo) {
  const existing = photo?.metadata;
  dialog(photo ? "Edit photograph settings" : "Add photograph",[
    !photo && formField("Photograph","photo","",{type:"file",accept:"image/*",required:true}),
    formField("Capture title","title",existing?.title || "",{required:true,maxlength:120}),
    selectField("Capture role","role",["raking","diffuse","reference","sensor-dark"],existing?.role || "raking"),
    el("div",{class:"split"},[formField("Light azimuth (°)","azimuth",existing?.azimuth ?? 0,{type:"number",step:"any",min:-180,max:360}),
      formField("Light elevation (°)","elevation",existing?.elevation ?? 45,{type:"number",step:"any",min:4,max:89.9})]),
    el("div",{class:"split"},[formField("Relative strength","lamp_power",existing?.lamp_power ?? 1,{type:"number",step:"any",min:.05,max:12}),
      formField("Exposure EV","exposure",existing?.exposure ?? 0,{type:"number",step:"any",min:-6,max:6})]),
    formField("Black code","black_code",existing?.black_code ?? 0,{type:"number",step:"any",min:0,max:.45}),
    formField("Camera","camera",existing?.camera || "",{maxlength:80}), formField("Notes","note",existing?.note || "",{maxlength:700}),
  ],async(form)=>{
    const sid = state.item.surface.id;
    const meta = formValues(form);
    let result;
    if (photo) result=await api(`/api/captures/${photo.id}`,json("PUT",{version:photo.version,metadata:meta}));
    else {
      const upload=new FormData();
      upload.append("photo",meta.photo); delete meta.photo;
      upload.append("metadata",JSON.stringify(meta));
      result=await api(`/api/surfaces/${sid}/captures`,{method:"POST",body:upload});
    }
    state.active=result.capture.id; state.last=null;
    await openSurface(sid,true); note("Photograph settings saved.");
  });
}
function titleBar(s) {
  const heading=el("div",{class:"surface-head"},[el("div",{},[
    el("h1",{text:s.title}),el("p",{class:"meta",text:`${s.width_mm} × ${s.height_mm} mm${s.accession ? " · "+s.accession : ""}${s.synthetic ? " · Synthetic" : ""}`}),
  ]),button("Surface settings",()=>surfaceDialog(state.item),{class:"quiet"})]);
  const tabs=el("nav",{class:"tabs","aria-label":"Surface views"});
  for(const [id,label] of [["work","Inspect"],["archive","Saved evidence"]]) tabs.append(button(label,()=>{
    state.view=id; state.edit=null; renderSurface();
  },{"aria-current":id===state.view}));
  return [heading,tabs];
}
function loadedRegistration(){
  const photo=state.item?.captures.find(c=>c.id===state.active);
  if(!photo) return null;
  const select=state.revisions.get(photo.id);
  return select ? photo.registrations.find(r=>r.id===select) : photo.registrations[0];
}
function project(point,axis){
  const matrix=loadedRegistration()?.metrics?.[axis];
  if(!matrix) return point;
  const [x,y]=point;
  const value=row=>matrix[row][0]*x+matrix[row][1]*y+matrix[row][2];
  const w=value(2);
  return w>1e-5 ? [value(0)/w,value(1)/w] : point;
}
function logicalPoint(event, layer){
  return project(onLayer(event,layer),"inverse_matrix").map(q=>Math.max(0,Math.min(1,q)));
}
function renderSurface() {
  const [head,tabs]=titleBar(state.item.surface);
  root.replaceChildren(head,tabs);
  if(state.view==="archive") renderArchive();
  else renderWorkbench();
}
function sourcePane() {
  const active = state.active;
  const pane = el("section",{class:"sidebar","aria-label":"Source photographs"});
  pane.append(el("h2",{text:"Photographs"}),el("div",{class:"row"},[
    button("Add photo",()=>captureDialog()),button("Select raking set",()=>{
      state.selected=new Set(state.item.captures.filter(c=>c.metadata.role==="raking").map(c=>c.id).slice(0,12)); renderSurface();
    }),
  ]));
  for(const photo of state.item.captures) {
    const check=el("input",{type:"checkbox",checked:state.selected.has(photo.id),"aria-label":`Use ${nameOf(photo)}`});
    check.onchange=()=>check.checked ? state.selected.add(photo.id) : state.selected.delete(photo.id);
    const switcher=button(nameOf(photo),()=>{
      state.active=photo.id; state.edit=null; state.proposal=null; state.corners=[]; renderSurface();
    },{"aria-pressed":photo.id===active});
    switcher.append(el("small",{text:photo.metadata.role}));
    pane.append(el("div",{class:"capture"},[check,switcher]));
  }
  return pane;
}
function stagePane() {
  const photo=state.item.captures.find(c=>c.id===state.active);
  if(!photo) return el("section",{class:"stage"},[el("p",{text:"Add a photograph to begin.",class:"empty"})]);
  const caption=el("section",{class:"stage"},[el("div",{class:"stage-actions"},[
    el("h2",{text:nameOf(photo)}),
    button("Capture settings",()=>captureDialog(photo),{class:"quiet"}),
  ])]);
  const layer=el("div",{class:"photo-stage",id:"camera-stage"},[el("img",{src:`/api/captures/${photo.id}/image/preview`,alt:nameOf(photo),draggable:false})]);
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
  svg.setAttribute("viewBox","0 0 1000 1000"); svg.setAttribute("preserveAspectRatio","none");
  layer.append(svg); caption.append(layer);
  const setMode=(type)=>{
    state.edit=type; state.corners=[]; state.proposal=null; layer.classList.add("editing");
    note(type==="corners" ? "Click all four corners clockwise from the upper left." : "Drag a rectangle on the working coordinate view.");
    drawLayer(svg);
  };
  const revisions=photo.registrations.map(r=>[r.id,`Registration v${r.revision}`]);
  const set=selectField("Source registration","registration",[...revisions,["identity","Whole photograph"]],state.revisions.get(photo.id) || revisions[0]?.[0] || "identity");
  set.querySelector("select").onchange=event=>{ state.revisions.set(photo.id,event.target.value); state.last=null; drawLayer(svg); };
  caption.append(el("div",{class:"row"},[
    button("Mark four corners",()=>setMode("corners")),
    button("Preview registration",()=>runRegistration(photo,false),{id:"preview-registration"}),
    button("Save registration",()=>runRegistration(photo,true),{id:"save-registration"}),
    button("Select region",()=>setMode("region")),
    button("Clear region",()=>{ state.region=null; drawLayer(svg); }),
  ]),set);
  caption.append(el("div",{class:"caption"},[
    el("span",{text:`${photo.image.size.join(" × ")} pixels · ${photo.image.encoding}`}),
    link("Original file",`/api/captures/${photo.id}/image/original`),
  ]));
  layer.onpointerdown=event=>{
    if(state.edit!=="region") return;
    state.anchor=logicalPoint(event,layer); layer.setPointerCapture(event.pointerId);
  };
  layer.onpointermove=event=>{
    if(!state.anchor || state.edit!=="region") return;
    state.region=asRegion(state.anchor,logicalPoint(event,layer)); drawLayer(svg);
  };
  layer.onpointerup=event=>{
    if(state.edit==="corners"){
      if(state.corners.length===4) state.corners=[];
      state.corners.push(onLayer(event,layer)); drawLayer(svg);
    } else if(state.edit==="region" && state.anchor) {
      state.region=asRegion(state.anchor,logicalPoint(event,layer));
      if(state.region[2]-state.region[0]<.012 || state.region[3]-state.region[1]<.012) state.region=null;
      state.anchor=null; state.edit=null; layer.classList.remove("editing"); drawLayer(svg);
    }
  };
  drawLayer(svg);
  return caption;
}
function onLayer(event,node) {
  const rect=node.getBoundingClientRect();
  const trim=x=>Math.max(0,Math.min(1,x));
  return [trim((event.clientX-rect.left)/rect.width),trim((event.clientY-rect.top)/rect.height)].map(v=>Number(v.toFixed(5)));
}
function asRegion(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
function drawLayer(svg) {
  svg.replaceChildren();
  const draw=(type,props)=>{
    const n=document.createElementNS("http://www.w3.org/2000/svg",type);
    for(const [name,val] of Object.entries(props)) n.setAttribute(name,val);
    svg.append(n);
  };
  const rgba=state.corners;
  if(rgba.length>=2) draw(rgba.length===4?"polygon":"polyline",{
    points:rgba.map(c=>c.map(n=>n*1000).join(",")).join(" "),fill:"none",stroke:"#ebc48b","stroke-width":4,
  });
  for(const [x,y] of rgba) draw("circle",{cx:x*1000,cy:y*1000,r:7,fill:"#ebc48b"});
  if(state.region){
    const [x,y,r,b]=state.region;
    const ring=[[x,y],[r,y],[r,b],[x,b]].map(point=>project(point,"forward_matrix"));
    draw("polygon",{points:ring.map(pt=>pt.map(n=>1000*n).join(",")).join(" "),
      fill:"#82c4aa25",stroke:"#8bd4b7","stroke-width":4});
  }
}
async function runRegistration(photo,save) {
  if(state.corners.length!==4) throw new Error("Mark all four corners first.");
  const result=await api(`/api/captures/${photo.id}/registration`,json("POST",{corners:state.corners,note:"",save}));
  if(save) {
    state.revisions.set(photo.id,result.registration.id); state.corners=[];
    await openSurface(state.item.surface.id,true);
  }
  renderReport(query("results"),result);
  note(save ? "Registration saved." : "Registration preview.");
}
function toolPane() {
  const tools=state.catalog.tools;
  const selected=tools.find(t=>t.key===state.tool);
  const field=selectField("Inspection tool","tool",tools.map(t=>[t.key,t.title]),state.tool);
  field.querySelector("select").id="tool-select";
  field.querySelector("select").onchange=event=>{state.tool=event.target.value; state.last=null; state.loadedInputs=null; renderSurface();};
  const inputs=el("form",{class:"inputs",id:"tool-options"});
  for(const row of selected.controls){
    if(selected.input_kind==="retained" && row.key==="records") continue;
    const initial=state.loadedInputs?.options?.[row.key] ?? row.default;
    let n;
    if(row.type==="json") n=jsonField(row.label,row.key,initial);
    else if(row.type==="select") n=selectField(row.label,row.key,row.choices,initial);
    else if(row.type==="checkbox") n=checkField(row.label,{name:row.key,checked:initial});
    else n=formField(row.label,row.key,initial,{type:"number",min:row.min,max:row.max,step:row.step});
    inputs.append(n);
  }
  inputs.onsubmit=event=>{event.preventDefault();guarded(()=>runInspection(false));};
  return el("section",{class:"tools"},[el("h2",{text:"Inspection"}),field,el("p",{class:"meta",text:selected.description}),
    selected.input_kind==="retained" && sessionPicker(),
    inputs,el("div",{class:"row"},[
      selected.input_kind!=="retained" && button("Remember setup",()=>procedureForm()),
      selected.input_kind!=="retained" && button("Saved setups",()=>procedurePicker()),
      button("Run inspection",()=>runInspection(false),{class:"solid",id:"run"}),
      button("Save evidence",()=>saveDialog(),{id:"retain",disabled:!state.last})])]);
}
function sessionPicker(){
  const stored=state.loadedInputs?.options?.records || [];
  const candidates=state.item.inspections.filter(row=>row.kind==="normals");
  const box=el("div",{class:"inputs"});
  if(candidates.length<2){
    box.append(el("p",{class:"meta",text:"Save a normal and reflectance report in each session first. Two retained reports are needed."}));
    return box;
  }
  const choices=candidates.map(row=>[row.id,`${row.title} · ${row.created_at.slice(0,10)}`]);
  const older=candidates[candidates.length-1].id, newer=candidates[0].id;
  const prior=selectField("Earlier retained report","sessions-prior",choices,stored[0] && candidates.some(r=>r.id===stored[0]) ? stored[0] : older);
  const current=selectField("Later retained report","sessions-current",choices,stored[1] && candidates.some(r=>r.id===stored[1]) ? stored[1] : newer);
  prior.querySelector("select").dataset.records=true; current.querySelector("select").dataset.records=true;
  box.append(prior,current);
  return box;
}
function renderWorkbench(){
  const page=el("div",{class:"workspace"},[sourcePane(),stagePane(),toolPane()]);
  const results=el("section",{class:"results",id:"results"});
  root.append(page,results);
  if(state.last) renderReport(results,state.last);
}
function requestForInspection(){
  const doc=query("tool-options");
  if(!doc.reportValidity()) throw new Error("Check the input controls.");
  const options=formValues(doc);
  const picked=[...doc.querySelectorAll("select[data-records]")].map(node=>node.value);
  delete options["sessions-prior"]; delete options["sessions-current"];
  if(picked.length===2){
    if(picked[0]===picked[1]) throw new Error("Choose two different retained reports.");
    options.records=picked;
  }
  for(const f of doc.querySelectorAll("textarea")){
    try{options[f.name]=JSON.parse(f.value);}catch{throw new Error(`Check the coordinate array for ${human(f.name)}.`);}
  }
  const captures=state.item.captures.map(c=>c.id).filter(id=>state.selected.has(id));
  const corrections=Object.fromEntries([...state.revisions].filter(([id])=>captures.includes(id)));
  return {captures, corrections, region:state.region, options};
}
async function runInspection(retain,title){
  const s=state.item.surface.id;
  const kind=state.tool;
  const data=retain ? {...state.last.inputs,retain:true,title,
    expected_preview:state.last.evidence?.data_signature?.identity} : requestForInspection();
  note("Processing the images…"); const target=query("run"); if(target) target.disabled=true;
  try{
    const output=await api(`/api/surfaces/${s}/analysis/${kind}`,json("POST",data));
    if(state.item?.surface.id!==s || state.tool!==kind) return;
    state.last={...output,inputs:data};
    renderReport(query("results"),output); query("retain").disabled=false;
    if(retain) { state.loadedInputs=data; await openSurface(s,true); }
    note(retain ? "Evidence archived." : "Inspection ready.");
  } finally {if(target) target.disabled=false;}
}
function saveDialog(){
  dialog("Save evidence",[formField("Inspection name","title",state.last.report.title,{required:true,maxlength:120})],
    form=>runInspection(true,formValues(form).title), "Archive");
}
function table(rows){
  if(!rows?.length) return null;
  const cols=[...new Set(rows.flatMap(row=>Object.keys(row)))];
  const head=el("thead",{},[el("tr",{},cols.map(k=>el("th",{text:human(k)})))]);
  const all=el("tbody",{},rows.slice(0,800).map(row=>el("tr",{},cols.map(k=>el("td",{text:asText(row[k])})))));
  return el("details",{},[el("summary",{text:`Observations (${rows.length} rows)`}),el("div",{class:"table"},[el("table",{},[head,all])])]);
}
function renderReport(target, output){
  if(!target) return;
  const report=output.report;
  const metrics=el("dl",{class:"metrics"},Object.entries(report.metrics||{}).map(([k,v])=>el("div",{},[
    el("dt",{text:human(k)}),el("dd",{text:asText(v)}),
  ])));
  const figures=el("div",{class:"images"});
  const downloads=el("div",{class:"row"});
  for(const file of output.files||[]){
    const uri=file.url||file.data;
    if(file.mime==="image/png"&&uri){
      figures.append(el("figure",{},[el("img",{src:uri,alt:human(file.key)}),
        el("figcaption",{},[document.createTextNode(human(file.key)),link("Download",uri,{download:file.name})])]));
    } else if(uri) downloads.append(link(file.name,uri,{class:"action",download:file.name}));
  }
  const warnings=report.warnings?.length?el("ul",{class:"warnings"},report.warnings.map(w=>el("li",{text:w}))):null;
  const links=report.links?.length?el("div",{class:"row"},report.links.map(item=>button(item.label,()=>openRetained(item.record),{class:"action","data-record":item.record}))):null;
  target.replaceChildren(...[el("h2",{text:report.title}),el("p",{class:"muted",text:report.summary}),
    metrics,links,figures,downloads,table(report.rows),warnings].filter(Boolean));
}
async function openRetained(id){
  state.view="archive"; state.opened=null; renderSurface();
  await openRecord(id);
}
function renderArchive(){
  const list=el("div",{class:"record-list"});
  for(const row of state.item.inspections) list.append(button(row.title,()=>openRecord(row.id),{"data-record":row.id}));
  if(!state.item.inspections.length) list.append(el("p",{class:"meta",text:"No saved inspections yet."}));
  const left=el("section",{class:"record-card"},[el("h2",{text:"Saved inspections"}),list]);
  const right=el("section",{class:"results",id:"saved-report"});
  const archive=el("div",{class:"archive"},[left,right]);
  left.append(link("Export original capture set",`/api/surfaces/${state.item.surface.id}/capture-set`,{class:"action"}));
  root.append(archive);
  if(state.opened) displayRecord(state.opened);
}
async function openRecord(id){ const reply=await api(`/api/records/${id}`);state.opened=reply.record;displayRecord(reply.record); }
function displayRecord(record){
  const target=query("saved-report");
  renderReport(target,record);
  const bar=el("div",{class:"row"},[el("small",{text:`Review ${human(record.state)} · Revision ${record.revision}`}),
    button("Review evidence",()=>editReview(record)),link("Download evidence packet",`/api/records/${record.id}/packet`,{class:"action"}),
    link("Packet without photos",`/api/records/${record.id}/packet?originals=0`,{class:"action"}),
    link("Print inspection",`/api/records/${record.id}/print`,{class:"action"}),
  ]);
  const sources=record.evidence.sources.map(s=>({photo:s.title,id:s.id,registration:s.registration_id,sha256:s.hash}));
  target.prepend(bar,el("details",{},[el("summary",{text:"Source provenance"}),table(sources)]));
  if(record.events.length) target.append(table(record.events));
}
function editReview(record){
  dialog("Review evidence",[formField("Reviewer name","reviewer","",{required:true,maxlength:80}),
    selectField("Decision","state",["accepted","revisit","draft"].filter(s=>s!==record.state),"accepted"),
    formField("Review note","note","",{maxlength:500})],
  async form=>{
    await api(`/api/records/${record.id}/review`,json("PUT",{revision:record.revision,...formValues(form)}));
    await openRecord(record.id); note("Review saved.");
  });
}
function procedureForm(){
  const form=requestForInspection();
  dialog("Remember current setup",[
    formField("Setup name","name","",{required:true,maxlength:96}),
    selectField("Registration on replay","policy",[["recorded","Use original registration"],["latest","Review latest registration"]],"recorded"),
    formField("Setup note","note","",{maxlength:430}),
  ],async node=>{
    const sid=state.item.surface.id;
    const body={version:state.item.surface.version,procedure:{...formValues(node),kind:state.tool,inputs:form}};
    await api(`/api/surfaces/${sid}/procedures`,json("POST",body));
    await openSurface(sid,true); note("Inspection setup saved.");
  });
}
function procedurePicker(){
  const items=state.item.procedures || [];
  if(!items.length) throw new Error("This surface has no saved setups yet.");
  modal.replaceChildren(el("div",{class:"modal-head"},[el("h2",{text:"Saved setups"}),
    button("Close",()=>modal.close(),{class:"quiet"})]));
  for(const item of items){
    modal.append(el("div",{class:"row"},[button(item.name,()=>showProcedure(item.id)),el("span",{class:"meta",text:item.note})]));
  }
  modal.showModal();
}
async function showProcedure(id){
  const reply=await api(`/api/surfaces/${state.item.surface.id}/procedures/${id}`);
  const detail=reply.preview;
  const intro=detail.needs_reconsent ? "Some material settings changed. Review before applying." : "The photos, physical size and registrations match.";
  const headers=el("div",{class:"modal-head"},[el("h2",{text:reply.procedure.name}),button("Close",()=>modal.close(),{class:"quiet"})]);
  const summary=el("ul",{},detail.photos.map(photo=>el("li",{text:`${photo.file || "Photo"} · ${photo.issues.length ? photo.issues.join(" ") : "Matches"}`})));
  const changes=detail.changed_surface.map(field=>el("p",{class:"meta",text:`${human(field.field)}: ${field.recorded} → ${field.current}`}));
  const replay=button("Apply inspected setup",()=>{
    state.tool=detail.kind; state.selected=new Set(detail.inputs.captures);
    state.revisions=new Map(Object.entries(detail.inputs.corrections)); state.region=detail.inputs.region;
    state.edit=null; state.active=detail.inputs.captures[0];
    state.loadedInputs=detail.inputs; state.last=null;
    modal.close(); renderSurface(); note("Setup applied. Run the inspection to see the result.");
  },{class:"solid",disabled:detail.status!=="ready"});
  modal.replaceChildren(headers,el("p",{text:intro}),...changes,summary,replay);
}
function captureSetDialog(){
  dialog("Inspect a capture set",[formField("ReliefBench capture ZIP","packet","",{type:"file",accept:".zip",required:true})],async form=>{
    const content=new FormData(form);
    const plan=await api("/api/capture-sets",{method:"POST",body:content});
    queueMicrotask(()=>setTimeout(()=>captureSetConfirm(content,plan),0));
  },"Inspect ZIP");
}
function captureSetConfirm(content,plan){
  dialog("Import this photograph set",[
    el("p",{text:`${plan.capture_count} original images · ${plan.surface.width_mm} × ${plan.surface.height_mm} mm`}),
    formField("Imported surface title","title",plan.surface.title,{required:true,maxlength:120}),
    table(plan.photos.map(p=>({photo:p.title,role:p.role,registrations:p.registrations}))),
  ],async form=>{
    const args=formValues(form);
    content.set("apply","1");content.set("title",args.title);content.set("preview_id",plan.preview_id);
    const saved=await api("/api/capture-sets",{method:"POST",body:content});
    await openSurface(saved.surface.id); note("Originals and registration versions imported.");
  },"Import as new surface");
}
function verifyPacket(){
  dialog("Verify evidence packet",[formField("ReliefBench ZIP","packet","",{type:"file",accept:".zip",required:true})],async form=>{
    const checked=await api("/api/packets/check",{method:"POST",body:new FormData(form)});
    note(checked.valid?`${checked.checked_member_count} packet members verified.`:checked.findings.join(" "),!checked.valid);
  },"Check package");
}

query("new-surface").onclick=()=>surfaceDialog();
query("library-button").onclick=()=>guarded(openLibrary);
query("verify-packet").onclick=verifyPacket;
guarded(openLibrary);
