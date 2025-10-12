// ===============================
// NeuroStride — Frontend Logic (Auto Analysis Added)
// ===============================

// -------------------------------
// DOM Elements
// -------------------------------
const startCamBtn       = document.getElementById('startCamBtn');
const stopAnalysisBtn   = document.getElementById('stopAnalysisBtn'); // optional stop button
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
const saveReportBtn     = document.getElementById('saveReportBtn');
const clearBtn          = document.getElementById('clearBtn');
const behaviorMetrics   = document.getElementById('behaviorMetrics');

const movementLevelEl   = document.getElementById('movementLevel');
const tensionLevelEl    = document.getElementById('tensionLevel');
const behaviorLevelEl   = document.getElementById('behaviorLevel');
const finalResultEl     = document.getElementById('finalResult');

const ctx               = overlay.getContext('2d');

let stream = null;
let activeSource = null; // "webcam" | "video" | "image"
let autoAnalysisInterval = null;
analyzeBtn.disabled = true;

// -------------------------------
// Utility
function setStatus(text) { statusEl.textContent = text; }

// -------------------------------
// Webcam Start / Stop
startCamBtn.addEventListener('click', async () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    startCamBtn.textContent = 'Start Webcam';
    setStatus('Stopped');
    activeSource = null;
    analyzeBtn.disabled = true;
    stopAutoAnalysis();
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
    analyzeBtn.disabled = false;

    startAutoAnalysis(); // start auto-analysis for webcam
  } catch (err) {
    console.error(err);
    setStatus('Camera permission denied or not available');
  }
});

// -------------------------------
// Video Upload
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
  analyzeBtn.disabled = false;

  startAutoAnalysis(); // auto-analysis for uploaded video

  video.onended = () => stopAutoAnalysis(); // stop when video ends
});

// -------------------------------
// Image Upload
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
  analyzeBtn.disabled = false;

  // Auto-analyze once for image
  setTimeout(async () => {
    const result = await analyzeWithModelMock();
    displayResult(result);
    setStatus('Image analysis complete');
  }, 800);
});
// -------------------------------
// Start Auto Analysis (Webcam: 5-6 sec, Video: full duration, Stop on Suspect + Auto Report)
function startAutoAnalysis() {
  if (autoAnalysisInterval) return;
  setStatus('Auto analysis started');

  let startTime = Date.now();

  autoAnalysisInterval = setInterval(async () => {

    // ---------- Webcam Analysis ----------
    if (activeSource === "webcam" && video.readyState >= 2) {
      const result = await analyzeWithModelMock();
      displayResult(result);

      // Stop immediately if suspect detected
      if (finalResultEl.textContent.includes('Suspect')) {
        stopAutoAnalysis();
        setStatus('⚠ Suspect detected! Auto analysis stopped.');
        saveReport();
        return;
      }

      // Stop automatically after 6 seconds for webcam
      if (Date.now() - startTime >= 6000) {
        stopAutoAnalysis();
        setStatus('Webcam analysis completed.');
      }

    // ---------- Video Analysis ----------
    } else if (activeSource === "video" && video.readyState >= 2) {
      const result = await analyzeWithModelMock();
      displayResult(result);

      // Stop immediately if suspect detected
      if (finalResultEl.textContent.includes('Suspect')) {
        stopAutoAnalysis();
        setStatus('⚠ Suspect detected! Auto analysis stopped.');
        saveReport();
        return;
      }

      // Optional: stop automatically when video ends
      if (video.ended) {
        stopAutoAnalysis();
        setStatus('Video analysis completed.');
      }
    }

    // ---------- Image Analysis (3 sec) ----------
    else if (activeSource === "image" && uploadedImage.src) {
      const result = await analyzeWithModelMock();
      displayResult(result);

      // Stop immediately if suspect detected
      if (finalResultEl.textContent.includes('Suspect')) {
        stopAutoAnalysis();
        setStatus('⚠ Suspect detected! Auto analysis stopped.');
        saveReport();
        return;
      }

      // Stop automatically after 3 seconds for image
      if (Date.now() - startTime >= 3000) {
        stopAutoAnalysis();
        setStatus('Image analysis completed.');
      }
    }

  }, 800); // Run analysis roughly every 0.9 sec
}

// -------------------------------
// Stop Auto Analysis
function stopAutoAnalysis() {
  if (autoAnalysisInterval) {
    clearInterval(autoAnalysisInterval);
    autoAnalysisInterval = null;
    if (!finalResultEl.textContent.includes('Suspect')) {
      setStatus('Auto analysis stopped');
    }
  }
}

// -------------------------------
// Save Report (use existing values)
function saveReport() {
  const { jsPDF } = window.jspdf;
  if (!jsPDF) { alert('jsPDF not loaded.'); return; }

  const doc = new jsPDF();

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('NeuroStride Forensic Report', 14, 20);

  // Capture current frame (webcam/video/image)
  const captureCanvas = document.createElement('canvas');
  if (activeSource === "image" && uploadedImage.src) {
    captureCanvas.width = uploadedImage.naturalWidth;
    captureCanvas.height = uploadedImage.naturalHeight;
    captureCanvas.getContext('2d').drawImage(uploadedImage, 0, 0);
  } else if ((activeSource === "webcam" || activeSource === "video") && video.readyState >= 2) {
    captureCanvas.width = video.videoWidth || 640;
    captureCanvas.height = video.videoHeight || 360;
    captureCanvas.getContext('2d').drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  }

  const imgData = captureCanvas.toDataURL('image/jpeg', 0.9);
  doc.addImage(imgData, 'JPEG', 14, 28, 180, 90);

  // Table with percentages
  const rows = [
    ['Date', new Date().toLocaleString()],
    ['Match Status', matchConfidence.textContent],
    ['Suspect ID', suspectId.textContent],
    ['Movement Level', movementLevelEl.textContent],
    ['Tension Level', tensionLevelEl.textContent],
    ['Behavior Level', behaviorLevelEl.textContent],
    ['Final Result', finalResultEl.textContent],
    ['Eye Movement', behaviorMetrics.children[0]?.textContent.replace('Eye Movement: ', '') || '—'],
    ['Nervousness', behaviorMetrics.children[1]?.textContent.replace('Nervousness: ', '') || '—'],
    ['Facial Tension', behaviorMetrics.children[2]?.textContent.replace('Facial Tension: ', '') || '—']
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
}

// -------------------------------
// Manual Analyze Button
analyzeBtn.addEventListener('click', async () => {
  setStatus('Analyzing…');
  const result = await analyzeWithModelMock();
  displayResult(result);
  setStatus('Manual analysis complete');
});

// -------------------------------
// Mock Analysis
async function analyzeWithModelMock() {
  await new Promise(r => setTimeout(r, 500));

  const movementDetected = Math.random() > 0.2;
  if (!movementDetected) return { movementDetected: false };

  return {
    movementDetected: true,
    matched: true,
    confidence: +(Math.random() * 0.5 + 0.5).toFixed(2),
    suspectId: 'SUS-' + Math.floor(Math.random() * 9000 + 1000),
    eyeMovement: ['Normal','Rapid','Blinking'][Math.floor(Math.random()*3)],
    nervousness: ['Low','Moderate','High'][Math.floor(Math.random()*3)],
    faceTension: ['Relaxed','Neutral','Tensed'][Math.floor(Math.random()*3)],
    movementLevel: Math.floor(Math.random() * 40 + 60),
    tensionLevel: Math.floor(Math.random() * 40 + 60),
    behaviorLevel: Math.floor(Math.random() * 40 + 60),
  };
}


function displayResult(res) {
  if (!res.movementDetected) {
    matchConfidence.textContent = 'Not Matched';
    suspectId.textContent = '—';
    suspiciousList.innerHTML = '<li>Normal</li>';
    behaviorMetrics.innerHTML = `
      <li>Eye Movement: —</li>
      <li>Nervousness: —</li>
      <li>Facial Tension: —</li>
    `;
    movementLevelEl.textContent = '0%';
    tensionLevelEl.textContent = '0%';
    behaviorLevelEl.textContent = '0%';
    finalResultEl.textContent = 'Normal';
    movementLevelEl.className = tensionLevelEl.className = behaviorLevelEl.className = 'percentage-badge green-label';
    finalResultEl.className = 'final-result green-label';
    return;
  }

  behaviorMetrics.innerHTML = `
    <li>Eye Movement: ${res.eyeMovement}</li>
    <li>Nervousness: ${res.nervousness}</li>
    <li>Facial Tension: ${res.faceTension}</li>
  `;

  let movement = res.movementLevel || 0;
  let tension  = res.tensionLevel || 0;
  let expression = Math.floor(Math.random() * 20 + 80);

  movementLevelEl.textContent = `${movement}%`;
  tensionLevelEl.textContent  = `${tension}%`;
  behaviorLevelEl.textContent = `${expression}%`;

  let isSuspect = false;

  if (activeSource === "image") {
    movement = 0;
    movementLevelEl.textContent = '0%';
    tensionLevelEl.className  = `percentage-badge ${tension > 90 ? 'orange-label' : 'green-label'}`;
    behaviorLevelEl.className = `percentage-badge ${expression > 90 ? 'orange-label' : 'green-label'}`;
    if (tension > 90 && expression > 90) isSuspect = true;
  } else {
    movementLevelEl.className = `percentage-badge ${movement > 87 ? 'orange-label' : 'green-label'}`;
    tensionLevelEl.className  = `percentage-badge ${tension > 80 ? 'orange-label' : 'green-label'}`;
    behaviorLevelEl.className = `percentage-badge ${expression > 80 ? 'orange-label' : 'green-label'}`;
    if (movement > 87 && expression > 80 && tension > 80) isSuspect = true;
  }

  // Only suspect shows Matched
  finalResultEl.textContent = isSuspect ? '⚠ Suspect' : 'Normal';
  finalResultEl.className = `final-result ${isSuspect ? 'red-label' : 'green-label'}`;
  matchConfidence.textContent = isSuspect ? 'Matched' : 'Not Matched';
  suspectId.textContent = isSuspect ? res.suspectId : '—';
  suspiciousList.innerHTML = `<li>${isSuspect ? 'Suspect' : 'Normal'}</li>`;
}

// -------------------------------
// Clear UI
// -------------------------------
clearBtn.addEventListener('click', () => {
  matchConfidence.textContent = '—';
  suspectId.textContent = '—';
  suspiciousList.innerHTML = '<li>Normal</li>';
  behaviorMetrics.innerHTML = `
    <li>Eye Movement: —</li>
    <li>Nervousness: —</li>
    <li>Facial Tension: —</li>
  `;
  movementLevelEl.textContent = '—';
  tensionLevelEl.textContent = '—';
  behaviorLevelEl.textContent = '—';
  finalResultEl.textContent = '—';
  setStatus('Idle');
  analyzeBtn.disabled = true;
});

// -------------------------------
// Save Report (Improved)
// -------------------------------
saveReportBtn.addEventListener('click', () => {
  const { jsPDF } = window.jspdf;
  if (!jsPDF) { alert('jsPDF not loaded.'); return; }
  const doc = new jsPDF();

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('NeuroStride Forensic Report', 14, 20);

  // Capture Image
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

  // Table
  const rows = [
    ['Date', new Date().toLocaleString()],
    ['Match Status', matchConfidence.textContent],
    ['Suspect ID', suspectId.textContent],
    ['Movement Level', movementLevelEl.textContent],
    ['Tension Level', tensionLevelEl.textContent],
    ['Behavior Level', behaviorLevelEl.textContent],
    ['Final Result', finalResultEl.textContent],
    ['Eye Movement', behaviorMetrics.children[0]?.textContent.replace('Eye Movement: ', '') || '—'],
    ['Nervousness', behaviorMetrics.children[1]?.textContent.replace('Nervousness: ', '') || '—'],
    ['Facial Tension', behaviorMetrics.children[2]?.textContent.replace('Facial Tension: ', '') || '—']
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
