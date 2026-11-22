const root = document.getElementById("application");
const modal = document.getElementById("dialog");
const notification = document.getElementById("notice");
const state = {
  catalog: null, item: null, active: "", selected: new Set(), revisions: new Map(), tool: "normals",
  view: "work", region: null, edit: null, corners: [], proposal: null, last: null, opened: null,
  seal: null,
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
    state.opened = null; state.last = null; state.loadedInputs = null; state.seal = null;
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
  for(const [id,label] of [["work","Inspect"],["seal","Fragment finder"],["archive","Saved evidence"]]) tabs.append(button(label,()=>{
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
  else if(state.view==="seal") renderSeal();
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
    inputs,el("div",{class:"row"},[
      button("Remember setup",()=>procedureForm()),button("Saved setups",()=>procedurePicker()),
      button("Run inspection",()=>runInspection(false),{class:"solid",id:"run"}),
      button("Save evidence",()=>saveDialog(),{id:"retain",disabled:!state.last})])]);
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
  const metricsData=report.metrics||{};
  const comparisons=Array.isArray(metricsData.comparisons)?metricsData.comparisons:null;
  const ranking=Array.isArray(metricsData.ranking)?metricsData.ranking:null;
  const hidden=new Set(["comparisons","ranking"]);
  const fileMap=new Map((output.files||[]).map(file=>[file.key,file]));
  const compareBar=comparisons?el("div",{class:"compare-list"},comparisons.map(entry=>{
    const file=fileMap.get(entry.file);
    const uri=file?.url||file?.data;
    return el("div",{class:"compare-row"},[
      el("div",{},[el("strong",{text:entry.title||entry.file}),el("small",{text:`${human(entry.status)}${entry.group?" · group "+entry.group:""}`})]),
      uri?link("Open side by side",uri,{target:"_blank",rel:"noopener",class:"action"}):null,
    ]);
  })):null;
  const rankingTable=ranking?el("table",{},[
    el("thead",{},[el("tr",{},["Rank","Record","Score","Status","Group"].map(h=>el("th",{text:h})))]),
    el("tbody",{},ranking.map((row,index)=>el("tr",{},[
      el("td",{text:String(index+1)}),el("td",{text:row.title||row.record}),
      el("td",{text:row.score===null||row.score===undefined?"—":String(row.score)}),
      el("td",{text:human(row.status||"")}),el("td",{text:row.group||"—"}),
    ]))),
  ]):null;
  const metrics=el("dl",{class:"metrics"},Object.entries(metricsData).filter(([k])=>!hidden.has(k)).map(([k,v])=>el("div",{},[
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
  target.replaceChildren(...[el("h2",{text:report.title}),el("p",{class:"muted",text:report.summary}),
    compareBar,metrics,rankingTable?el("details",{open:true},[el("summary",{text:"Candidate ranking"}),el("div",{class:"table"},[rankingTable])]):null,
    figures,downloads,table(report.rows),warnings].filter(Boolean));
}
function renderSeal(){
  const item=state.item;
  if(!state.seal) state.seal={strokes:[],centre:null,rim:[],mode:"mark",candidates:new Set(),records:[],last:null,inputs:null};
  const seal=state.seal;
  const shorthand=item.captures.filter(photo=>["raking","diffuse"].includes(photo.metadata.role));
  if(!shorthand.some(photo=>photo.id===state.active) && shorthand.length) state.active=shorthand[0].id;
  const page=el("div",{class:"seal-workspace"});
  const left=el("section",{class:"sidebar","aria-label":"Fragment photographs"});
  left.append(el("h2",{text:"Fragment photographs"}));
  for(const photo of (shorthand.length?shorthand:item.captures)){
    const switcher=button(nameOf(photo),()=>{ state.active=photo.id; seal.strokes=[]; seal.centre=null; seal.rim=[]; seal.last=null; renderSurface(); },{"aria-pressed":photo.id===state.active});
    switcher.append(el("small",{text:photo.metadata.role}));
    left.append(el("div",{class:"capture"},[switcher]));
  }
  const photo=item.captures.find(c=>c.id===state.active);
  const middle=el("section",{class:"stage"});
  const right=el("section",{class:"tools"});
  const results=el("section",{class:"results",id:"seal-results"});
  if(!photo){
    middle.append(el("p",{class:"empty",text:"Add a photograph to outline the fragment."}));
    page.append(left,middle,right); root.append(page,results); return;
  }
  state.active=photo.id;
  const layer=el("div",{class:"photo-stage seal-stage",id:"seal-stage"},[el("img",{src:`/api/captures/${photo.id}/image/preview`,alt:nameOf(photo),draggable:false})]);
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
  svg.setAttribute("viewBox","0 0 1000 1000"); svg.setAttribute("preserveAspectRatio","none");
  layer.append(svg);
  const projectPoint=point=>project(point,"forward_matrix");
  const revisionSelect=photo.registrations.map(r=>[r.id,`Registration v${r.revision}`]);
  const registration=state.revisions.get(photo.id) || revisionSelect[0]?.[0] || "identity";
  const registered=selectField("Source registration","registration",[...revisionSelect,["identity","Whole photograph"]],registration);
  registered.querySelector("select").onchange=event=>{
    state.revisions.set(photo.id,event.target.value);
    seal.strokes=seal.strokes.map(stroke=>stroke.map(projectPoint));
    if(seal.centre) seal.centre=projectPoint(seal.centre);
    seal.rim=seal.rim.map(projectPoint);
    seal.last=null; renderSurface();
  };
  const modeLabel={mark:"Tap the strokes you can read. Each tap adds a point.",centre:"Tap the middle of the impression.",rim:"Tap two points on the surviving rim."};
  const drawSeal=()=>{
    svg.replaceChildren();
    const draw=(type,props)=>{const n=document.createElementNS("http://www.w3.org/2000/svg",type);for(const [name,val] of Object.entries(props))n.setAttribute(name,val);svg.append(n);};
    for(const stroke of seal.strokes){
      if(stroke.length>1){
        const line=stroke.map(point=>projectPoint(point).map(n=>n*1000).join(",")).join(" ");
        draw("polyline",{points:line,fill:"none",stroke:"#8bd4b7","stroke-width":4});
      }
      for(const point of stroke){const [x,y]=projectPoint(point);draw("circle",{cx:x*1000,cy:y*1000,r:5,fill:"#8bd4b7"});}
    }
    if(seal.centre){const [x,y]=projectPoint(seal.centre);draw("circle",{cx:x*1000,cy:y*1000,r:9,fill:"none",stroke:"#ebc48b","stroke-width":4});}
    for(const point of seal.rim){const [x,y]=projectPoint(point);draw("circle",{cx:x*1000,cy:y*1000,r:8,fill:"none",stroke:"#ef7765","stroke-width":4});}
    if(seal.rim.length===2 && seal.centre){
      const [cx,cy]=projectPoint(seal.centre); const [rx,ry]=projectPoint(seal.rim[0]);
      const radius=Math.hypot((rx-cx)*1000,(ry-cy)*1000);
      draw("circle",{cx:cx*1000,cy:cy*1000,r:radius,fill:"none",stroke:"#ebc48b","stroke-width":2,"stroke-dasharray":"8 8"});
    }
  };
  const slots={
    mark:()=>{
      if(!seal.strokes.length||seal.newStroke){seal.strokes.push([]);seal.newStroke=false;}
      const point=[0,0];
      seal.strokes[seal.strokes.length-1].push(point);
      return point;
    },
    centre:()=>{
      seal.centre=[0,0];
      return seal.centre;
    },
    rim:()=>{
      if(seal.rim.length<2)seal.rim.push([0,0]);
      else seal.rim=[seal.rim[1],[0,0]];
      return seal.rim[seal.rim.length-1];
    },
  };
  layer.addEventListener("pointerup",event=>{
    const point=logicalPoint(event,layer);
    const slot=slots[seal.mode]();
    slot[0]=point[0];
    slot[1]=point[1];
    seal.last=null;
    drawSeal();
  });
  drawSeal();
  middle.append(el("div",{class:"stage-actions"},[el("h2",{text:nameOf(photo)})]),layer,
    el("p",{class:"meta",text:modeLabel[seal.mode]||""}),
    el("div",{class:"row"},[
      button("Mark strokes",()=>{seal.mode="mark"; seal.newStroke=false; renderSurface();}),
      button("New stroke",()=>{seal.newStroke=true; note("The next tap starts a new stroke.");}),
      button("Set centre",()=>{seal.mode="centre"; renderSurface();}),
      button("Set rim",()=>{seal.mode="rim"; renderSurface();}),
      button("Undo last tap",()=>{
        if(seal.mode==="mark" && seal.strokes.length) seal.strokes[seal.strokes.length-1].pop();
        else if(seal.mode==="rim" && seal.rim.length) seal.rim.pop();
        seal.strokes=seal.strokes.filter(stroke=>stroke.length);
        seal.last=null; drawSeal();
      }),
      button("Clear marks",()=>{seal.strokes=[]; seal.centre=null; seal.rim=[]; seal.last=null; renderSurface();}),
    ]),registered);
  const field=(label,key,value,attrs={})=>{
    const input=el("input",{type:"number",value,...attrs});
    input.oninput=()=>{const v=Number(input.value); if(Number.isFinite(v)&&v>0){seal[key]=v; seal.last=null;}};
    seal[key]=value;
    return el("label",{},[document.createTextNode(label),input]);
  };
  seal.radius_mm=seal.radius_mm||60; seal.stroke_width_mm=seal.stroke_width_mm||1.2; seal.samples=seal.samples||192;
  seal.angle_step=seal.angle_step||2; seal.min_visible=seal.min_visible||.55; seal.tolerance_mm=seal.tolerance_mm||.35;
  const keep=checkField("Keep unresolved pairs",{checked:seal.keep_unresolved!==false});
  keep.querySelector("input").onchange=event=>{seal.keep_unresolved=event.target.checked; seal.last=null;};
  right.append(el("h2",{text:"Fragment finder"}),
    el("p",{class:"meta",text:"Name a few visible strokes, place the centre and click the surviving rim. Sizes are measured on the plate, not guessed from the picture."}),
    el("div",{class:"inputs"},[
      field("Largest dial radius (mm)","radius_mm",seal.radius_mm,{min:4,max:120,step:"any"}),
      field("Stroke width (mm)","stroke_width_mm",seal.stroke_width_mm,{min:.25,max:6,step:"any"}),
      field("Rotation step (°)","angle_step",seal.angle_step,{min:1,max:30,step:"any"}),
      field("Minimum visible share","min_visible",seal.min_visible,{min:.2,max:1,step:"any"}),
      field("Depth tolerance (mm)","tolerance_mm",seal.tolerance_mm,{min:.02,max:4,step:"any"}),
      keep,
    ]),
    el("h2",{text:"Old records"}),
    el("p",{class:"meta",text:"Choose saved relative-height records to search."}),
    el("div",{class:"record-list",id:"seal-records"}));
  const candidateList=right.querySelector("#seal-records");
  if(!seal.records.length) candidateList.append(el("p",{class:"meta",text:"Loading saved height records…"}));
  for(const record of seal.records){
    const box=el("input",{type:"checkbox",checked:seal.candidates.has(record.record),"aria-label":`Search ${record.title}`});
    box.onchange=()=>{
      box.checked?seal.candidates.add(record.record):seal.candidates.delete(record.record);
      seal.last=null; seal.inputs=null;
      const save=query("seal-save"); if(save) save.disabled=true;
    };
    candidateList.append(el("label",{class:"candidate-check"},[box,el("span",{text:`${record.title} · ${record.surface_title} · ${record.width_mm}×${record.height_mm} mm`})]));
  }
  right.append(el("div",{class:"row"},[
    button("Search records",()=>runSeal(false),{id:"seal-run"}),
    button("Save this search",()=>saveSeal(),{class:"solid",id:"seal-save",disabled:!seal.last}),
  ]));
  page.append(left,middle,right);
  root.append(page,results);
  if(seal.last) renderReport(results,seal.last);
  if(!seal.records.length) guarded(async()=>{
    const reply=await api("/api/height-records");
    seal.records=reply.records;
    if(state.view==="seal") renderSurface();
  });
}
function sealPayload(){
  const seal=state.seal; const photo=state.item.captures.find(c=>c.id===state.active);
  if(!photo) throw new Error("Choose the fragment photograph first.");
  if(seal.strokes.reduce((sum,stroke)=>sum+stroke.length,0)<2) throw new Error("Tap at least two points on the visible strokes.");
  if(!seal.centre) throw new Error("Place the centre of the impression.");
  if(seal.rim.length!==2) throw new Error("Tap two points on the surviving rim.");
  if(!seal.candidates.size) throw new Error("Choose at least one old record to search.");
  const registration=state.revisions.get(photo.id) || photo.registrations[0]?.id || "identity";
  const records=seal.records.filter(record=>seal.candidates.has(record.record));
  return {surface:state.item.surface.id,capture:photo.id,registration,
    strokes:seal.strokes.filter(stroke=>stroke.length),
    centre:seal.centre,radius_mm:seal.radius_mm,stroke_width_mm:seal.stroke_width_mm,
    rim_points:seal.rim,samples:seal.samples||192,angle_step:seal.angle_step,
    min_visible:seal.min_visible,tolerance_mm:seal.tolerance_mm,
    keep_unresolved:seal.keep_unresolved!==false,
    candidates:records.map(record=>({record:record.record,surface:record.surface}))};
}
async function runSeal(retain){
  const seal=state.seal;
  const data=retain ? {...seal.inputs,retain:true,title:seal.pendingTitle,
    expected_preview:seal.last?.evidence?.data_signature?.identity} : sealPayload();
  note("Comparing the fragments…");
  const runButton=query("seal-run"); if(runButton) runButton.disabled=true;
  let output;
  try { output=await api("/api/seal-match",json("POST",data)); }
  finally { const buttonNow=query("seal-run"); if(buttonNow) buttonNow.disabled=false; }
  if(state.view!=="seal") return;
  seal.last={...output,inputs:data};
  seal.inputs=data;
  renderReport(query("seal-results"),output);
  const save=query("seal-save");
  if(save) save.disabled=false;
  if(retain){ state.loadedInputs=null; await openSurface(state.item.surface.id,true); state.view="seal"; note("Fragment search archived."); }
  else note("Candidate list ready.");
}
function saveSeal(){
  if(!state.seal?.last) throw new Error("Run the search before saving it.");
  dialog("Save fragment search",[
    formField("Search name","title",state.seal.last.report.title,{required:true,maxlength:120}),
    el("p",{class:"meta",text:"The drawn strokes, rim clicks, chosen records and side-by-side comparisons are kept together."}),
  ],async form=>{
    state.seal.inputs=state.seal.inputs||sealPayload();
    state.seal.pendingTitle=formValues(form).title;
    await runSeal(true);
  },"Archive search");
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
