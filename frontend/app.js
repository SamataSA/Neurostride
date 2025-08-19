// ===============================
// NeuroStride — Frontend Logic
// ===============================

// -------------------------------
// DOM Elements
// -------------------------------
const startCamBtn       = document.getElementById('startCamBtn');
const fileInput         = document.getElementById('fileInput');
const imageInput        = document.getElementById('imageInput');
const video             = document.getElementById('video');
const uploadedImage     = document.getElementById('uploadedImage');
const overlay           = document.getElementById('overlay');
const statusEl          = document.getElementById('status');
const analyzeBtn        = document.getElementById('analyzeBtn');
const captureBtn        = document.getElementById('captureBtn');
const matchConfidence   = document.getElementById('matchConfidence');
const suspectId         = document.getElementById('suspectId');
const suspiciousList    = document.getElementById('suspiciousList');
const suspiciousPercent = document.getElementById('suspiciousPercent'); 
const saveReportBtn     = document.getElementById('saveReportBtn');
const clearBtn          = document.getElementById('clearBtn');
const behaviorMetrics   = document.getElementById('behaviorMetrics'); 
const ctx               = overlay.getContext('2d');

let stream = null;
let activeSource = null; // "webcam" | "video" | "image"

// ===============================
// Utility Functions
// ===============================
function setStatus(text) { statusEl.textContent = text; }

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
    activeSource = null;
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false
    });
    video.style.display = "block";
    uploadedImage.style.display = "none";

    video.srcObject = stream;
    startCamBtn.textContent = 'Stop Webcam';
    setStatus('Webcam started');
    video.play();
    activeSource = "webcam";
  } catch (err) {
    console.error(err);
    setStatus('Camera permission denied or not available');
  }
});

// ===============================
// Video Upload
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

  uploadedImage.style.display = "none";
  video.style.display = "block";
  video.srcObject = null;
  video.src = url;
  video.play();
  activeSource = "video";
  setStatus('Playing uploaded video');
});

// ===============================
// Image Upload
// ===============================
imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    startCamBtn.textContent = 'Start Webcam';
  }

  video.style.display = "none";
  uploadedImage.style.display = "block";
  uploadedImage.src = url;
  activeSource = "image";
  setStatus('Image loaded');
});

// ===============================
// Capture Frame
// ===============================
captureBtn.addEventListener('click', () => {
  if (activeSource === "image") {
    if (!uploadedImage.src) {
      setStatus('No image loaded');
      return;
    }
    const tmp = document.createElement('canvas');
    tmp.width = uploadedImage.naturalWidth;
    tmp.height = uploadedImage.naturalHeight;
    tmp.getContext('2d').drawImage(uploadedImage, 0, 0);
    const data = tmp.toDataURL('image/png');
    const w = window.open('');
    w.document.write(`<img src="${data}" style="max-width:100%"/>`);
    return;
  }

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
// Analyze (Mock)
// ===============================
analyzeBtn.addEventListener('click', async () => {
  setStatus('Analyzing…');
  if (activeSource === "image" && !uploadedImage.src) {
    setStatus('No image loaded');
    return;
  }
  if ((activeSource === "webcam" || activeSource === "video") && video.readyState < 2) {
    setStatus('Video not ready');
    return;
  }

  const result = await analyzeWithModelMock();
  displayResult(result);
  setStatus('Analysis complete');
});

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
    confidence: (Math.random() * 0.5 + 0.5).toFixed(2),
    suspectId: 'SUS-' + Math.floor(Math.random() * 9000 + 1000),
    suspiciousBehaviors: ['Loitering near restricted area'],
    eyeMovement: ['Normal','Rapid','Blinking'][Math.floor(Math.random()*3)],
    nervousness: ['Low','Moderate','High'][Math.floor(Math.random()*3)],
    faceTension: ['Relaxed','Neutral','Tensed'][Math.floor(Math.random()*3)],
    suspiciousPercentage: Math.floor(Math.random() * 41 + 60)
  };
}

// ===============================
// Display Results
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

  const isRed = res.suspiciousPercentage > 75;
  suspiciousPercent.textContent = `${res.suspiciousPercentage}%`;
  suspiciousPercent.className = `percentage-badge ${isRed ? 'red-label' : 'green-label'}`;
  suspiciousList.innerHTML = isRed ? '<li>Suspect</li>' : '<li>No suspicious</li>';

  behaviorMetrics.innerHTML = `
    <li>Eye Movement: ${res.eyeMovement}</li>
    <li>Nervousness: ${res.nervousness}</li>
    <li>Facial Tension: ${res.faceTension}</li>
  `;
}

// ===============================
// Save Report
// ===============================
saveReportBtn.addEventListener('click', () => {
  const { jsPDF } = window.jspdf;
  if (!jsPDF || !window.jspdf?.jsPDF) {
    alert('jsPDF not loaded.');
    return;
  }
  const doc = new jsPDF();

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('NeuroStride Forensic Report', 14, 20);

  const captureCanvas = document.createElement('canvas');
  if (activeSource === "image" && uploadedImage.src) {
    captureCanvas.width = uploadedImage.naturalWidth;
    captureCanvas.height = uploadedImage.naturalHeight;
    captureCanvas.getContext('2d').drawImage(uploadedImage, 0, 0);
  } else {
    captureCanvas.width = video.videoWidth || 640;
    captureCanvas.height = video.videoHeight || 360;
    captureCanvas.getContext('2d').drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  }
  const imgData = captureCanvas.toDataURL('image/jpeg', 0.9);
  doc.addImage(imgData, 'JPEG', 14, 28, 180, 90);

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
  suspiciousList.innerHTML = '<li>No suspicious</li>';
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
// Background Animations
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
    dot.style.animationDuration = (Math.random() * 5 + 3).toFixed(1) + 's';
    dot.style.animationDelay    = (Math.random() * 5).toFixed(1) + 's';
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
    col.style.animationDelay    = (Math.random() * 5).toFixed(1) + 's';
    rainContainer.appendChild(col);
  }

  document.body.appendChild(rainContainer);
}

createDNA();
createBinaryRain();
