// ===============================
// NeuroStride — Frontend Logic
// ===============================

// DOM Elements
const startCamBtn = document.getElementById('startCamBtn');
const fileInput = document.getElementById('fileInput');
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const analyzeBtn = document.getElementById('analyzeBtn');
const captureBtn = document.getElementById('captureBtn');
const matchConfidence = document.getElementById('matchConfidence');
const suspectId = document.getElementById('suspectId');
const suspiciousList = document.getElementById('suspiciousList');
const suspiciousPercent = document.getElementById('suspiciousPercent'); // percentage box
const saveReportBtn = document.getElementById('saveReportBtn');
const clearBtn = document.getElementById('clearBtn');
const behaviorMetrics = document.getElementById('behaviorMetrics'); // <-- needed globally

const ctx = overlay.getContext('2d');
let stream = null;

// ===============================
// Utility
// ===============================
function setStatus(text) { statusEl.textContent = text; }
function resizeCanvas() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
}

// ===============================
// Webcam Start / Stop
// ===============================
startCamBtn.addEventListener('click', async () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    startCamBtn.textContent = 'Start Webcam';
    setStatus('Stopped');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false
    });
    video.srcObject = stream;
    startCamBtn.textContent = 'Stop Webcam';
    setStatus('Webcam started');
    video.play();
    setTimeout(resizeCanvas, 200);
    requestAnimationFrame(drawLoop);
  } catch (err) {
    console.error(err);
    setStatus('Camera permission denied or not available');
  }
});

// ===============================
// File Upload & Video Preview
// ===============================
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    startCamBtn.textContent = 'Start Webcam';
  }
  video.srcObject = null;
  video.src = url;
  video.play();
  setStatus('Playing uploaded video');
  setTimeout(resizeCanvas, 200);
  requestAnimationFrame(drawLoop);
});

// ===============================
// Capture Frame (popup image)
// ===============================
captureBtn.addEventListener('click', () => {
  if (video.readyState < 2) {
    setStatus('No video frame yet');
    return;
  }
  const tmp = document.createElement('canvas');
  tmp.width = video.videoWidth || 640;
  tmp.height = video.videoHeight || 360;
  tmp.getContext('2d').drawImage(video, 0, 0, tmp.width, tmp.height);
  const data = tmp.toDataURL('image/png');
  const w = window.open('');
  w.document.write(`<img src="${data}" style="max-width:100%"/>`);
});

// ===============================
// Analyze Button — Mock
// ===============================
analyzeBtn.addEventListener('click', async () => {
  setStatus('Analyzing…');
  if (video.readyState < 2) {
    setStatus('Video not ready');
    return;
  }
  drawSampleOverlayMock();

  const result = await analyzeWithModelMock();
  displayResult(result);
  setStatus('Analysis complete');
});

// ===============================
// Mock Analysis Function
// ===============================
async function analyzeWithModelMock() {
  await new Promise(r => setTimeout(r, 800));

  const movementDetected = Math.random() > 0.3;

  if (!movementDetected) {
    return {
      movementDetected: false,
      matched: false,
      confidence: 0,
      suspectId: '—',
      suspiciousBehaviors: [],
      eyeMovement: '—',
      nervousness: '—',
      faceTension: '—',
      suspiciousPercentage: 0
    };
  }

  return {
    movementDetected: true,
    matched: Math.random() > 0.4,
    confidence: (Math.random() * 0.5 + 0.5).toFixed(2), // 0.50–1.00
    suspectId: 'SUS-' + Math.floor(Math.random() * 9000 + 1000),
    suspiciousBehaviors: ['Loitering near restricted area', 'Abrupt change in walking speed'],
    eyeMovement: ['Normal','Rapid','Blinking'][Math.floor(Math.random()*3)],
    nervousness: ['Low','Moderate','High'][Math.floor(Math.random()*3)],
    faceTension: ['Relaxed','Neutral','Tensed'][Math.floor(Math.random()*3)],
    suspiciousPercentage: Math.floor(Math.random() * 41 + 60) // 60–100
  };
}

// ===============================
// Display Results (red if >75 else green)
// ===============================
function displayResult(res) {
  if (!res.movementDetected) {
    matchConfidence.textContent = '—';
    suspectId.textContent = '—';
    suspiciousList.innerHTML = '<li>No suspicious</li>';
    behaviorMetrics.innerHTML = `
      <li>Eye Movement: —</li>
      <li>Nervousness: —</li>
      <li>Facial Tension: —</li>
    `;
    suspiciousPercent.textContent = '—';
    suspiciousPercent.className = 'percentage-badge';
    return;
  }

  matchConfidence.textContent = res.matched
    ? (res.confidence * 100).toFixed(0) + '%'
    : 'No match';
  suspectId.textContent = res.matched ? res.suspectId : 'Unknown';

  // Suspicious Level badge (keep base class)
  const isRed = res.suspiciousPercentage > 75; // requirement: > 75 => red
  suspiciousPercent.textContent = `${res.suspiciousPercentage}%`;
  suspiciousPercent.className = `percentage-badge ${isRed ? 'red-label' : 'green-label'}`;

  // Suspicious behavior simplified
  suspiciousList.innerHTML = isRed ? '<li>Suspect</li>' : '<li>No suspicious</li>';

  // Behavior metrics
  behaviorMetrics.innerHTML = `
    <li>Eye Movement: ${res.eyeMovement}</li>
    <li>Nervousness: ${res.nervousness}</li>
    <li>Facial Tension: ${res.faceTension}</li>
  `;
}

// ===============================
// Drawing Overlay
// ===============================
function drawLoop() {
  if (video.readyState >= 2) {
    resizeCanvas();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 2;
    ctx.strokeRect(8, 8, overlay.width - 16, overlay.height - 16);
  }
  requestAnimationFrame(drawLoop);
}

function drawSampleOverlayMock() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.strokeStyle = 'rgba(11,79,108,0.95)';
  ctx.fillStyle = 'rgba(11,79,108,0.12)';
  ctx.lineWidth = 3;

  const w = overlay.width, h = overlay.height;
  const cx = w * 0.5, cy = h * 0.4;

  // head
  ctx.beginPath();
  ctx.arc(cx, cy - 30, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // spine
  ctx.beginPath();
  ctx.moveTo(cx, cy - 14);
  ctx.lineTo(cx, cy + 60);
  ctx.stroke();

  // arms
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx - 34, cy + 18);
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + 36, cy + 18);
  ctx.stroke();

  // legs
  ctx.beginPath();
  ctx.moveTo(cx, cy + 60);
  ctx.lineTo(cx - 18, cy + 110);
  ctx.moveTo(cx, cy + 60);
  ctx.lineTo(cx + 28, cy + 100);
  ctx.stroke();
}

// ===============================
// Save Report — PDF with table + live frame
// ===============================
saveReportBtn.addEventListener('click', () => {
  const { jsPDF } = window.jspdf;
  if (!jsPDF || !window.jspdf?.jsPDF) {
    alert('jsPDF not loaded. Add the jsPDF and AutoTable scripts before app.js.');
    return;
  }
  const doc = new jsPDF();

  // Title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('NeuroStride Forensic Report', 14, 20);

  // Capture current frame from video
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = video.videoWidth || 640;
  captureCanvas.height = video.videoHeight || 360;
  const ctxCap = captureCanvas.getContext('2d');
  ctxCap.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  const imgData = captureCanvas.toDataURL('image/jpeg', 0.9);

  // Add image
  doc.addImage(imgData, 'JPEG', 14, 28, 180, 90);

  // Table
  const rows = [
    ['Date', new Date().toLocaleString()],
    ['Match Confidence', matchConfidence.textContent],
    ['Suspect ID', suspectId.textContent],
    ['Suspicious Level', suspiciousPercent.textContent],
    ['Suspicious Behavior', Array.from(suspiciousList.children).map(li => li.textContent).join(', ')],
    ['Eye Movement', behaviorMetrics.children[0]?.textContent?.replace('Eye Movement: ', '') || '—'],
    ['Nervousness', behaviorMetrics.children[1]?.textContent?.replace('Nervousness: ', '') || '—'],
    ['Facial Tension', behaviorMetrics.children[2]?.textContent?.replace('Facial Tension: ', '') || '—']
  ];

  doc.autoTable({
    startY: 125,
    head: [['Field', 'Value']],
    body: rows,
    theme: 'grid',
    styles: { fontSize: 10, cellPadding: 3, valign: 'middle' },
    headStyles: { fillColor: [0, 229, 255], textColor: 0 },
    columnStyles: { 0: { cellWidth: 60 }, 1: { cellWidth: 120 } }
  });

  doc.save(`NeuroStride_Report_${Date.now()}.pdf`);
});

// ===============================
// Clear UI
// ===============================
clearBtn.addEventListener('click', () => {
  matchConfidence.textContent = '—';
  suspectId.textContent = '—';
  suspiciousList.innerHTML = '<li>No alerts yet</li>';
  suspiciousPercent.textContent = '—';
  suspiciousPercent.className = 'percentage-badge';
  behaviorMetrics.innerHTML = `
    <li>Eye Movement: —</li>
    <li>Nervousness: —</li>
    <li>Facial Tension: —</li>
  `;
  setStatus('Idle');
});

// Footer Year
document.getElementById('year').textContent = new Date().getFullYear();

// ===============================
// DNA + Binary Background Animation
// ===============================
function createDNA() {
  const dnaBg = document.createElement('div');
  dnaBg.classList.add('dna-bg');
  document.body.appendChild(dnaBg);

  for (let i = 0; i < 50; i++) {
    const dot = document.createElement('div');
    dot.classList.add('dna-strand');
    dot.style.left = Math.random() * window.innerWidth + 'px';
    dot.style.top = Math.random() * window.innerHeight + 'px';
    const duration = (Math.random() * 5 + 3).toFixed(1) + 's';
    dot.style.animationDuration = duration;
    dot.style.animationDelay = (Math.random() * 5).toFixed(1) + 's';
    dnaBg.appendChild(dot);
  }
}

function createBinaryRain() {
  const rainContainer = document.createElement('div');
  rainContainer.classList.add('binary-rain');

  const columns = Math.floor(window.innerWidth / 20);
  for (let i = 0; i < columns; i++) {
    const col = document.createElement('div');
    col.classList.add('binary-column');
    col.textContent = Array(100).fill(0).map(() => (Math.random() > 0.5 ? '0' : '1')).join('\n');
    col.style.left = `${i * 20}px`;
    col.style.animationDuration = (Math.random() * 5 + 5).toFixed(1) + 's';
    col.style.animationDelay = (Math.random() * 5).toFixed(1) + 's';
    rainContainer.appendChild(col);
  }

  document.body.appendChild(rainContainer);
}

// Call animations
createDNA();
createBinaryRain();