 function openImage(src){
  const win = window.open("");
  win.document.write(`
    <title>Hand Sign Guide</title>
    <body style="margin:0;background:#111;display:flex;align-items:center;justify-content:center;height:100vh;">
      <img src="${src}" style="max-width:95%;max-height:95%;border-radius:10px;">
    </body>
  `);
}


function showSaveMessage(){

  const name = document.getElementById("profile-name").value
  const email = document.getElementById("profile-email").value
  const lang = document.getElementById("profile-lang").value

  const profile = { name, email, lang }

  localStorage.setItem("gestureProfile", JSON.stringify(profile))

  const msg = document.getElementById("saveMsg")
  msg.style.display = "block"

  setTimeout(() => {
    msg.style.display = "none"
  }, 2500)
}

function loadProfile(){

  const saved = localStorage.getItem("gestureProfile")
  if(!saved) return

  const profile = JSON.parse(saved)

  document.getElementById("profile-name").value = profile.name || ""
  document.getElementById("profile-email").value = profile.email || ""
  document.getElementById("profile-lang").value = profile.lang || "English (ASL)"
}

document.addEventListener("DOMContentLoaded", loadProfile)


function buildWeeklyChart(){

  const sessions = document.querySelectorAll(".session-item")
  const chart = document.getElementById("bar-chart")

  chart.innerHTML = ""

  if(sessions.length === 0){
      chart.innerHTML = "<span style='font-size:12px;color:var(--muted)'>No session data</span>"
      return
  }

  sessions.forEach(session => {

      const accText = session.querySelector(".s-acc").innerText
      const acc = parseInt(accText.replace("%",""))

      const wrap = document.createElement("div")
      wrap.className = "bar-wrap"

      const bar = document.createElement("div")
      bar.className = "bar filled"   // IMPORTANT
      bar.style.height = acc + "%"

      const label = document.createElement("div")
      label.className = "bar-label"
      label.innerText = acc + "%"

      wrap.appendChild(bar)
      wrap.appendChild(label)

      chart.appendChild(wrap)

  })
}

document.addEventListener("DOMContentLoaded", buildWeeklyChart);


// ════════════════════════════════════════
//  CONFIG  (mirrors your Python settings)
// ════════════════════════════════════════
const CFG = {
  apiUrl:          'http://localhost:8000/predict-letter',
  stableThreshold: 8,      // same as STABLE_THRESHOLD in live_camera.py
  confGate:        0.6,    // same as confidence gate
  bufSize:         15,     // pred_buffer maxlen
  minHandSize:     0.03,   // distance guard
  ttsRate:         1.0,
};

// ════════════════════════════════════════
//  STATE
// ════════════════════════════════════════
// ---------- SESSION TRACKING ***************************************************************************************************************
// ---------- SESSION TRACKING ----------
let gestureCooldown = 0
let sessionStart = null
let wordsThisSession = 0
let correctPredictions = 0
let totalPredictions = 0
let confidenceSum = 0
function getSessionStats(){
  return JSON.parse(localStorage.getItem("gesture_sessions") || "[]")
}

function saveSession(words, accuracy){

  const sessions = getSessionStats()

  sessions.unshift({
    date: new Date().toLocaleString(),
    words: words,
    accuracy: accuracy
  })

  if(sessions.length > 10) sessions.pop()

  localStorage.setItem("gesture_sessions", JSON.stringify(sessions))

  updateDashboard()
}

function updateDashboard(){

  const sessions = getSessionStats()

  const totalSessions = sessions.length
  const totalWords = sessions.reduce((a,b)=>a+b.words,0)

  const avgAcc = sessions.length
      ? Math.round(sessions.reduce((a,b)=>a+b.accuracy,0)/sessions.length)
      : 0

  const statCards = document.querySelectorAll(".stat-card .value")

  if(statCards.length >=3){
    statCards[0].textContent = totalSessions
    statCards[1].textContent = totalWords
    statCards[2].textContent = avgAcc + "%"
  }

  const list = document.querySelector(".session-list")
  if(!list) return

  list.innerHTML = ""

  sessions.slice(0,4).forEach(s=>{
    const el = document.createElement("div")
    el.className = "session-item"

    el.innerHTML = `
      <div class="s-left">
        <span class="s-date">${s.date}</span>
        <span class="s-words">${s.words} words</span>
      </div>
      <span class="s-acc">${s.accuracy}%</span>
    `

    list.appendChild(el)
  })
}
//*******************************************************************************************************************************************
let sentence      = '';
let predBuffer    = [];
let stableCount   = 0;
let stableLetter  = '';
let lastAdded     = '';
let camActive     = false;
let mpCamera      = null;
let mpHands       = null;
let videoStream   = null;
let apiOk         = null;     // null=unknown, true=ok, false=error
let frameCount    = 0;
let apiBusy = false;

// word list (subset — expand or load from API)
const WORD_LIST = [
  'apple','able','above','about','across','act','add','after','again','age',
  'ago','agree','air','all','also','always','am','among','and','any','are',
  'art','ask','back','bad','ball','be','because','been','before','big','blue',
  'book','both','box','boy','but','call','came','can','car','cat','change',
  'come','could','day','did','do','dog','done','door','down','draw','each',
  'eat','end','even','ever','every','face','fall','far','feel','few','find',
  'first','fly','for','form','found','four','free','from','full','game','gave',
  'get','girl','give','glad','go','good','got','great','green','grow','had',
  'hand','happy','hard','has','have','he','head','hello','her','here','high',
  'him','his','home','house','how','idea','if','important','in','into','is',
  'it','its','just','keep','kind','know','land','large','last','later','left',
  'let','life','light','like','little','live','long','look','made','make','man',
  'many','may','me','mean','meet','men','more','most','mother','move','much',
  'must','my','name','near','need','never','new','next','night','no','now',
  'number','of','off','often','old','on','once','one','only','open','or',
  'other','our','out','over','own','part','people','place','play','point',
  'put','read','red','right','room','round','run','said','same','saw','say',
  'school','see','seem','set','she','short','should','side','since','small',
  'so','some','soon','still','stop','study','such','sun','take','tell','than',
  'that','the','their','them','then','there','these','they','thing','think',
  'this','those','though','thought','three','through','time','to','today',
  'together','too','took','top','toward','town','tree','true','try','turn',
  'two','under','until','up','us','use','very','walk','want','was','water',
  'way','we','well','went','were','what','when','where','which','while','white',
  'who','why','will','with','without','word','world','would','write','year',
  'you','young','your',
];
//********************************************************************************************************************************************
function getSuggestions(partial, limit=3) {

  if (!partial) return []

  const p = partial.toLowerCase()

  const seen = new Set()

  const exact = WORD_LIST.filter(w => w === p)

  const prefix = WORD_LIST.filter(w =>
      w.startsWith(p) && w !== p
  )

  const combined = [...exact, ...prefix]

  const result = []

  for(const w of combined){
      if(!seen.has(w)){
          seen.add(w)
          result.push(w)
      }
  }

  return result.slice(0,limit)
}
//**********************************************************************************************************************************************
// ════════════════════════════════════════
//  NAVIGATION
// ════════════════════════════════════════
function showPage(id, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if (el && el.classList && el.classList.contains('nav-item')) el.classList.add('active');
  if (id !== 'live') stopCamera();
}

// ════════════════════════════════════════
//  MEDIAPIPE  →  FASTAPI PIPELINE
// ════════════════════════════════════════
function normalise(raw) {
  // mirror of normalize_landmarks() in predict.py
  const pts = [];
  for (let i = 0; i < 21; i++) pts.push([raw[i*3], raw[i*3+1], raw[i*3+2]]);
  const wx = pts[0][0], wy = pts[0][1], wz = pts[0][2];
  const centred = pts.map(([x,y,z]) => [x-wx, y-wy, z-wz]);
  const norms = centred.map(([x,y,z]) => Math.sqrt(x*x+y*y+z*z));
  const maxN = Math.max(...norms) || 1;
  return centred.flat().map(v => v / maxN);
}

async function predictLetter(normLandmarks) {
  try {
    const res = await fetch(CFG.apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: normLandmarks }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    setApiStatus(true);
    return data; // { letter, confidence? }
  } catch(e) {
    setApiStatus(false);
    return null;
  }
}

function setApiStatus(ok) {
  if (apiOk === ok) return;
  apiOk = ok;
  const el = document.getElementById('api-status');
  const txt = document.getElementById('api-status-text');
  el.className = 'api-status show ' + (ok ? 'ok' : 'err');
  txt.textContent = ok ? 'API connected' : 'API unreachable';
}
// ---------- GESTURE COMMANDS ----------

function isOpenPalm(lms){

  const tips = [8,12,16,20]   // finger tips
  const bases = [6,10,14,18]  // finger base joints

  let extended = 0

  for(let i=0;i<tips.length;i++){
    if(lms[tips[i]].y < lms[bases[i]].y){
      extended++
    }
  }

  return extended >= 4
}

function isThumbUp(lms){

  const thumbTip = lms[4]
  const thumbBase = lms[2]

  const indexTip = lms[8]

  const thumbUp = thumbTip.y < thumbBase.y
  const fingersClosed = indexTip.y > lms[6].y

  return thumbUp && fingersClosed
}
// Called by MediaPipe every frame
async function onHandResults(results) {

  const video   = document.getElementById('input-video');
  const canvas  = document.getElementById('output-canvas');
  const ctx     = canvas.getContext('2d');

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  // draw mirrored video
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  const badge   = document.getElementById('detected-badge');
  const warn    = document.getElementById('warn-badge');
  const ringEl  = document.getElementById('stable-ring');
  const ringFill = document.getElementById('ring-fill');

  // no hand
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    predBuffer = [];
    badge.classList.remove('show');
    warn.classList.remove('show');
    ringEl.classList.remove('show');
    return;
  }

  const lms = results.multiHandLandmarks[0];
  // ---------- GESTURE COMMAND CHECK ----------
// ---------- GESTURE COMMAND CHECK ----------

if(gestureCooldown > 0){
  gestureCooldown--
}else{

  if(isOpenPalm(lms)){
    console.log("GESTURE: SPACE")

    addSpace()

    predBuffer = []
    stableCount = 0

    gestureCooldown = 40
    return
  }

  if(isThumbUp(lms)){
    console.log("GESTURE: SPEAK")

    speakSentence()

    gestureCooldown = 40
    return
  }

}

  // draw landmarks
  if (window.drawConnectors && window.HAND_CONNECTIONS) {
    drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: 'rgba(168,245,200,0.6)', lineWidth: 1.5 });
    drawLandmarks(ctx, lms, { color: '#a8f5c8', lineWidth: 1, radius: 3 });
  }

  // raw landmarks
  const raw = [];
  for (const lm of lms) raw.push(lm.x, lm.y, lm.z);

  // distance guard
  const pts = [];
  for (let i = 0; i < 21; i++) pts.push([raw[i*3], raw[i*3+1], raw[i*3+2]]);
  const wrist = pts[0];

  const handSize = Math.max(...pts.map(([x,y,z]) =>
      Math.sqrt((x-wrist[0])**2+(y-wrist[1])**2+(z-wrist[2])**2)
  ));

  if (handSize < CFG.minHandSize) {
    predBuffer = [];
    warn.classList.add('show');
    badge.classList.remove('show');
    ringEl.classList.remove('show');
    return;
  }

  warn.classList.remove('show');

  // limit predictions
  frameCount++;
  if (frameCount % 4 !== 0) return;

  const norm = normalise(raw);

  // ===== API LOCK (prevents freeze) =====
  if(apiBusy) return;
  apiBusy = true;

  const result = await predictLetter(norm);
totalPredictions++
  apiBusy = false;

  if (!result) return;

  const letter = result.letter;
  const conf   = result.confidence ?? 1.0;
confidenceSum += conf
totalPredictions++ 

  if (conf > 0.05) {
    predBuffer.push(letter);
    if (predBuffer.length > CFG.bufSize) predBuffer.shift();
  }

  if (predBuffer.length > 2) {

    // majority vote
    const freq = {};
    predBuffer.forEach(l => freq[l] = (freq[l] || 0) + 1);

    const top = Object.entries(freq)
      .sort((a,b) => b[1]-a[1])[0][0];

    badge.classList.add('show');
    document.getElementById('detected-letter').textContent = top;
    document.getElementById('detected-conf').textContent =
      `Confidence: ${(conf*100).toFixed(0)}%`;

    // stability logic
    if (top === stableLetter) {
      stableCount++;
    } else {
      stableLetter = top;
      stableCount = 0;
    }

    // ring progress
    ringEl.classList.add('show');
    const progress = stableCount / CFG.stableThreshold;
    const circ = 113;

    ringFill.style.strokeDashoffset =
      circ - (circ * Math.min(progress,1));

    // commit letter
    if (stableCount >= 4) {

  sentence += top;
  lastAdded = top;

  stableCount = 0;
  predBuffer = [];
if (stableCount >= 4) {

  if(top === lastAdded) return;

  sentence += top;
  lastAdded = top;

  stableCount = 0;
  predBuffer = [];


  updateSentenceDisplay();
}
  console.log("LETTER COMMITTED:", top);

  updateSentenceDisplay();
}
  }

  console.log("Landmarks detected");
}

// ════════════════════════════════════════
//  CAMERA
// ════════════════════════════════════════
async function toggleCamera() {
  if (!camActive) {
    await startCamera();
  } else {
    stopCamera();
  }
}
//**********************************************************************************************************************************************
async function startCamera() {
  const video  = document.getElementById('input-video');
  const canvas = document.getElementById('output-canvas');
  const ph     = document.getElementById('cam-placeholder');
  const btn    = document.getElementById('start-btn');
   confidenceSum = 0
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: { width:1280, height:720 } });
  } catch(e) {
    alert('Camera permission denied. Please allow camera access and try again.');
    return;
  }

  video.srcObject = videoStream;
  video.style.display = 'block';
  canvas.style.display = 'block';
  ph.style.display = 'none';
  btn.textContent = '⏹ Stop Camera';
  btn.classList.remove('primary'); 
  btn.classList.add('danger');

  camActive = true;
  predBuffer = [];
  stableCount = 0;
  stableLetter = '';

  // ---------- SESSION START ----------
  sessionStart = Date.now();
  wordsThisSession = 0;
  correctPredictions = 0;
  totalPredictions = 0;
  // -----------------------------------

  // init MediaPipe Hands
  mpHands = new Hands({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  mpHands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  mpHands.onResults(onHandResults);

  mpCamera = new Camera(video, {
    onFrame: async () => { 
      await mpHands.send({ image: video }); 
    },
    width: 1280,
    height: 720,
  });

  mpCamera.start();
}

function stopCamera() {

  if (mpCamera) { 
    mpCamera.stop(); 
    mpCamera = null; 
  }

  if (mpHands)  { 
    mpHands.close(); 
    mpHands = null; 
  }

  if (videoStream) { 
    videoStream.getTracks().forEach(t => t.stop()); 
    videoStream = null; 
  }

  const video  = document.getElementById('input-video');
  const canvas = document.getElementById('output-canvas');
  const ph     = document.getElementById('cam-placeholder');
  const btn    = document.getElementById('start-btn');
  const badge  = document.getElementById('detected-badge');
  const ring   = document.getElementById('stable-ring');
  const status = document.getElementById('api-status');

  video.srcObject = null;
  video.style.display = 'none';

  canvas.style.display = 'none';
  ph.style.display = 'flex';

  badge.classList.remove('show');
  ring.classList.remove('show');

  status.className = 'api-status';

  btn.textContent = '▶ Start Camera';
  btn.classList.remove('danger');
  btn.classList.add('primary');

  // ---------- SAVE SESSION ----------
  if(sessionStart){

    const accuracy =
    totalPredictions === 0
    ? 0
    : Math.round((confidenceSum / totalPredictions) * 100)

    saveSession(wordsThisSession, accuracy);

    sessionStart = null;
  }
  // ----------------------------------

  camActive = false;
  apiOk = null;
}
//**********************************************************************************************************************************************
// ════════════════════════════════════════
//  SENTENCE / SUGGESTIONS
// ════════════════════════════════════════
function updateSentenceDisplay() {
  const el = document.getElementById('sentence-output');
  el.innerHTML = sentence + '<span class="cursor-blink"></span>';
  renderSuggestions();
}

function renderSuggestions() {
  const chips = document.getElementById('sugg-chips');
  const words = sentence.trimEnd().split(' ');
  const last  = (words[words.length - 1] || '').toUpperCase();
  const suggs = getSuggestions(last);

  if (suggs.length === 0) {
    chips.innerHTML = '<span style="font-size:13px;color:var(--muted);">No suggestions yet…</span>';
    return;
  }
  chips.innerHTML = suggs.map(w =>
    `<span class="sugg-chip" onclick="acceptSuggestion('${w}')">${w.toUpperCase()}</span>`
  ).join('');
}

function acceptSuggestion(word) {
  const parts = sentence.trimEnd().split(' ');
  parts[parts.length - 1] = word.toUpperCase();
  sentence = parts.join(' ') + ' ';
  stableCount = 0;
  updateSentenceDisplay();
}

function clearSentence() { sentence = ''; updateSentenceDisplay(); }
function addSpace(){

  if(sentence.length === 0) return

  if(!sentence.endsWith(" ")){
      sentence += " "
      wordsThisSession++
  }

  updateSentenceDisplay()
}
function deleteLast()    { sentence = sentence.slice(0, -1); updateSentenceDisplay(); }

function speakSentence() {
  const text = sentence.trim();
  if (!text) return;
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = CFG.ttsRate;
    window.speechSynthesis.speak(utt);
  }
}

// ════════════════════════════════════════
//  SETTINGS
// ════════════════════════════════════════
function applySettings() {

  // Existing config updates
  CFG.apiUrl          = document.getElementById('cfg-api').value.trim();
  CFG.stableThreshold = parseInt(document.getElementById('cfg-stable').value);
  CFG.confGate        = parseFloat(document.getElementById('cfg-conf').value);
  CFG.bufSize         = parseInt(document.getElementById('cfg-buf').value);
  CFG.ttsRate         = parseFloat(document.getElementById('cfg-rate').value);

  document.getElementById('api-url-hint').textContent = CFG.apiUrl;

  // User feature settings
  const autospeak   = document.getElementById("cfg-autospeak").checked;
  const suggestions = document.getElementById("cfg-suggestions").checked;
  const distance    = document.getElementById("cfg-distance").checked;
  const history     = document.getElementById("cfg-history").checked;

  const settings = {
    apiUrl: CFG.apiUrl,
    stableThreshold: CFG.stableThreshold,
    confGate: CFG.confGate,
    bufSize: CFG.bufSize,
    ttsRate: CFG.ttsRate,
    autospeak,
    suggestions,
    distance,
    history
  };

  // Save settings
  localStorage.setItem("gestureSettings", JSON.stringify(settings));

  // Success message instead of alert
  const msg = document.getElementById("settings-msg");
  if (msg) {
    msg.style.display = "block";

    setTimeout(() => {
      msg.style.display = "none";
    }, 2500);
  }
}

// ════════════════════════════════════════
//  CALENDAR
// ════════════════════════════════════════
let calDate = new Date(2026, 2, 1);
const sessionDays = [1, 3, 4, 5, 6];
function renderCalendar() {
  const months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  document.getElementById('cal-month').textContent = months[calDate.getMonth()] + ' ' + calDate.getFullYear();
  const grid = document.getElementById('cal-grid');
  grid.innerHTML = '';
  ['Su','Mo','Tu','We','Th','Fr','Sa'].forEach(d => {
    const el = document.createElement('div'); el.className = 'cal-day-name'; el.textContent = d; grid.appendChild(el);
  });
  const first = new Date(calDate.getFullYear(), calDate.getMonth(), 1).getDay();
  const total = new Date(calDate.getFullYear(), calDate.getMonth()+1, 0).getDate();
  const today = new Date();
  for (let i = 0; i < first; i++) { const el=document.createElement('div'); el.className='cal-day empty'; grid.appendChild(el); }
  for (let d = 1; d <= total; d++) {
    const el = document.createElement('div'); el.className = 'cal-day'; el.textContent = d;
    if (calDate.getFullYear()===today.getFullYear() && calDate.getMonth()===today.getMonth() && d===today.getDate()) el.classList.add('today');
    if (sessionDays.includes(d)) el.classList.add('has-session');
    grid.appendChild(el);
  }
}
function changeMonth(d) { calDate.setMonth(calDate.getMonth()+d); renderCalendar(); }
renderCalendar();

// ACCURACY CHART
const accData=[{day:'Mon',val:85},{day:'Tue',val:88},{day:'Wed',val:91},{day:'Thu',val:87},{day:'Fri',val:94},{day:'Sat',val:90},{day:'Sun',val:92}];
const chart=document.getElementById('bar-chart');
const maxV=Math.max(...accData.map(d=>d.val));
accData.forEach(d=>{
  const wrap=document.createElement('div'); wrap.className='bar-wrap';
  const bar=document.createElement('div'); bar.className='bar filled'; bar.style.height=(d.val/maxV*72)+'px'; bar.title=d.val+'%';
  const lbl=document.createElement('div'); lbl.className='bar-label'; lbl.textContent=d.day;
  wrap.appendChild(bar); wrap.appendChild(lbl); chart.appendChild(wrap);
});

// FAQ
function toggleFaq(el) {
  const a=el.nextElementSibling; const s=el.querySelector('span');
  a.classList.toggle('open'); s.textContent=a.classList.contains('open')?'−':'+';
}

// KEYBOARD shortcuts (live page only)
document.addEventListener('keydown', e => {
  if (!document.getElementById('page-live').classList.contains('active')) return;
  if (e.target.tagName === 'INPUT') return;
  if (e.key===' ')            { e.preventDefault(); addSpace(); }
  if (e.key==='Backspace')    deleteLast();
  if (e.key.toLowerCase()==='c') clearSentence();
  if (e.key.toLowerCase()==='s') speakSentence();
});

// init display
updateSentenceDisplay();
updateDashboard();

document.addEventListener("DOMContentLoaded", () => {
  updateDashboard();
  buildWeeklyChart();
  renderCalendar();
});