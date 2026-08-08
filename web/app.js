/* ── Visual Voice — fully client-side ISL translator ──────────────── */

// DOM handles
const intro = document.getElementById("intro");
const translator = document.getElementById("translator");
const startButton = document.getElementById("start");
const video = document.getElementById("video");
const canvas = document.getElementById("capture");
const prediction = document.getElementById("prediction");
const stable = document.getElementById("stable");
const letterEl = document.getElementById("letter");
const confidenceEl = document.getElementById("confidence");
const handsEl = document.getElementById("hands");
const statusText = document.getElementById("status");
const voiceToggle = document.getElementById("voiceToggle");
const zoom = document.getElementById("zoom");
const zoomValue = document.getElementById("zoomValue");

// ── Stable-letter voting ────────────────────────────────────────────
const history = [];
const windowSize = 16;
const minVotes = 12;
const transitionHoldMs = 650;
const stableHoldMs = 900;

let displayedLetter = "";
let pendingLetter = "";
let pendingSince = 0;
let lastCompletedStable = "";

// ── Voice (Web Speech API) ──────────────────────────────────────────
let voiceEnabled = false;
const speechCooldownMs = 1200;
const speechHoldMs = 2000;
let speechCandidate = "";
let speechCandidateSince = 0;
let spokenCandidate = "";
let lastSpoken = "";
let lastSpokenAt = 0;

// ── Inference state ─────────────────────────────────────────────────
let busy = false;
let timer = null;
let zoomLevel = 1;

// ML handles (loaded at startup)
let handLandmarker = null;
let onnxSession = null;
let classNames = [];
let modelClasses = [];
let centroids = null;
let distanceThresholds = null;

const CONFIDENCE_THRESHOLD = 0.70;
const NOT_DETECTABLE_THRESHOLD = 0.60;
const MARGIN_THRESHOLD = 0.10;
const DISTANCE_SLACK = 1.3;
const MIN_HAND_AREA = 0.015;

// ── Load ML models ──────────────────────────────────────────────────

async function loadModels() {
  statusText.textContent = "Loading hand detection model…";

  // 1. MediaPipe HandLandmarker
  const { HandLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm"
  );

  const wasmFileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 4,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  statusText.textContent = "Loading sign classifier…";

  // 2. ONNX Runtime — SVM model
  const ort = await import(
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/+esm"
  );

  // Configure Wasm paths
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

  onnxSession = await ort.InferenceSession.create("./model/isl_az_live_svm.onnx", {
    executionProviders: ["wasm"],
  });

  // Store ort reference for later use
  window.__ort = ort;

  // 3. JSON metadata
  const [cnRes, mcRes, ctRes, dtRes] = await Promise.all([
    fetch("./model/class_names.json").then((r) => r.json()),
    fetch("./model/model_classes.json").then((r) => r.json()),
    fetch("./model/centroids.json")
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
    fetch("./model/distance_thresholds.json")
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
  ]);

  classNames = cnRes;
  modelClasses = mcRes;
  centroids = ctRes;
  distanceThresholds = dtRes;

  statusText.textContent = "Models loaded — ready!";
}

// ── Feature engineering (mirrors Python exactly) ────────────────────

function normalizedHandVector(landmarks) {
  // landmarks = array of {x,y,z} with 21 entries
  const wrist = landmarks[0];
  const shifted = landmarks.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));

  let maxNorm = 0;
  for (const p of shifted) {
    const n = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (n > maxNorm) maxNorm = n;
  }

  const result = new Float32Array(63); // 21 × 3
  for (let i = 0; i < 21; i++) {
    const p = shifted[i];
    result[i * 3] = maxNorm > 0 ? p.x / maxNorm : 0;
    result[i * 3 + 1] = maxNorm > 0 ? p.y / maxNorm : 0;
    result[i * 3 + 2] = maxNorm > 0 ? p.z / maxNorm : 0;
  }
  return result;
}

function handBoxArea(landmarks) {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
}

function selectPrimaryHands(allHands, maxHands = 2, minArea = MIN_HAND_AREA) {
  const filtered = allHands.filter((h) => handBoxArea(h) >= minArea);
  filtered.sort((a, b) => handBoxArea(b) - handBoxArea(a));
  return filtered.slice(0, maxHands);
}

function handsToFeature(hands, maxHands = 2) {
  // Sort left-to-right by mean X
  const ordered = [...hands].sort((a, b) => {
    const ax = a.reduce((s, p) => s + p.x, 0) / a.length;
    const bx = b.reduce((s, p) => s + p.x, 0) / b.length;
    return ax - bx;
  });

  const parts = [];
  for (let i = 0; i < maxHands; i++) {
    if (i < ordered.length) {
      parts.push(1.0); // hand-present flag
      parts.push(...normalizedHandVector(ordered[i]));
    } else {
      // Missing hand: 1 flag + 63 zeros = 64 zeros
      for (let j = 0; j < 64; j++) parts.push(0.0);
    }
  }
  return new Float32Array(parts);
}

function passesDistanceGate(feature, predictedLabelId, slack = DISTANCE_SLACK) {
  if (!centroids || !distanceThresholds) return [true, 0.0, 0.0];

  const centroid = centroids[String(predictedLabelId)];
  const threshold = distanceThresholds[String(predictedLabelId)];
  if (!centroid || threshold == null) return [true, 0.0, 0.0];

  let sum = 0;
  for (let i = 0; i < feature.length; i++) {
    const d = feature[i] - centroid[i];
    sum += d * d;
  }
  const distance = Math.sqrt(sum);
  const slackedThreshold = threshold * slack;

  return [distance <= slackedThreshold, distance, slackedThreshold];
}

// ── ONNX inference ──────────────────────────────────────────────────

async function predict(feature) {
  const ort = window.__ort;
  const inputTensor = new ort.Tensor("float32", feature, [1, 128]);
  const feeds = {};

  // The input name might vary — use the first input name from the session
  const inputName = onnxSession.inputNames[0];
  feeds[inputName] = inputTensor;

  const results = await onnxSession.run(feeds);

  // Get predicted label and probabilities
  const labelOutput = results[onnxSession.outputNames[0]];
  const probOutput = results[onnxSession.outputNames[1]];

  const predictedClassIdx = Number(labelOutput.data[0]);

  // probOutput is a sequence of maps — extract probabilities
  // For sklearn SVMs converted with zipmap=False, output[1] is a tensor of shape [1, n_classes]
  const probs = Array.from(probOutput.data);

  // Find best and second-best
  const indexed = probs.map((p, i) => [p, i]);
  indexed.sort((a, b) => b[0] - a[0]);

  const bestProb = indexed[0][0];
  const bestIdx = indexed[0][1];
  const secondProb = indexed.length > 1 ? indexed[1][0] : bestProb;
  const gap = bestProb - secondProb;

  // Map from model class (which may be a label ID int) to class name
  const predictedLabelId = modelClasses[bestIdx];
  const label = classNames[predictedLabelId];

  const [passesDist, distance, threshold] = passesDistanceGate(
    feature,
    predictedLabelId,
    DISTANCE_SLACK
  );

  const accepted =
    bestProb >= CONFIDENCE_THRESHOLD &&
    gap >= MARGIN_THRESHOLD &&
    passesDist;

  const notDetectable = bestProb < NOT_DETECTABLE_THRESHOLD;

  return {
    ok: true,
    accepted,
    not_detectable: notDetectable,
    label,
    confidence: bestProb,
    gap,
    distance,
    threshold,
  };
}

// ── Camera ──────────────────────────────────────────────────────────

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720 },
    audio: false,
  });
  video.srcObject = stream;
}

// ── Stable letter logic ─────────────────────────────────────────────

function mostCommon(values) {
  const counts = new Map();
  for (const v of values) counts.set(v, (counts.get(v) || 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0] || ["", 0];
}

function updateStable(label) {
  history.push(label);
  while (history.length > windowSize) history.shift();
  const [topLabel, count] = mostCommon(history);
  const stableCandidate =
    history.length >= minVotes && count >= minVotes ? topLabel : "";

  if (stableCandidate && stable.dataset.pending !== stableCandidate) {
    stable.dataset.pending = stableCandidate;
    stable.dataset.pendingSince = String(Date.now());
  }
  if (!stableCandidate) {
    stable.dataset.pending = "";
    stable.dataset.pendingSince = "";
  }

  const isStable =
    stableCandidate &&
    Date.now() - Number(stable.dataset.pendingSince || 0) >= stableHoldMs;
  stable.textContent = isStable ? `Stable: ${stableCandidate}` : "";
}

// ── Voice via Web Speech API ────────────────────────────────────────

function speakLetter(text) {
  if (!voiceEnabled || !("speechSynthesis" in window)) return;
  const now = Date.now();
  if (text === lastSpoken && now - lastSpokenAt < speechCooldownMs) return;

  const utter = new SpeechSynthesisUtterance(`Letter ${text}`);
  utter.rate = 0.9;
  utter.pitch = 1;
  window.speechSynthesis.speak(utter);
  lastSpoken = text;
  lastSpokenAt = now;
  statusText.textContent = `Spoken ${text}`;
}

function updateSpeechCandidate(label) {
  if (!voiceEnabled) return;
  const now = Date.now();
  if (speechCandidate !== label) {
    speechCandidate = label;
    speechCandidateSince = now;
    spokenCandidate = "";
    return;
  }
  if (now - speechCandidateSince >= speechHoldMs && spokenCandidate !== label) {
    statusText.textContent = `Speaking ${label}`;
    speakLetter(label);
    spokenCandidate = label;
  }
}

function resetSpeechCandidate() {
  speechCandidate = "";
  speechCandidateSince = 0;
  spokenCandidate = "";
}

// ── Display letter with transition hold ─────────────────────────────

function updateDisplayedLetter(nextText, accepted) {
  prediction.classList.toggle("not-detectable", nextText === "Not detectable");
  if (!accepted || nextText === "Not detectable") {
    pendingLetter = "";
    pendingSince = 0;
    prediction.textContent = nextText;
    displayedLetter = nextText;
    return;
  }
  if (!displayedLetter || displayedLetter === "Not detectable") {
    prediction.textContent = nextText;
    displayedLetter = nextText;
    return;
  }
  if (nextText === displayedLetter) {
    pendingLetter = "";
    pendingSince = 0;
    prediction.textContent = nextText;
    return;
  }
  const now = Date.now();
  if (pendingLetter !== nextText) {
    pendingLetter = nextText;
    pendingSince = now;
    return;
  }
  if (now - pendingSince >= transitionHoldMs) {
    prediction.textContent = nextText;
    displayedLetter = nextText;
    pendingLetter = "";
    pendingSince = 0;
  }
}

// ── Main predict loop ───────────────────────────────────────────────

let lastVideoTime = -1;

async function predictFrame() {
  if (busy || video.readyState < 2 || !handLandmarker || !onnxSession) return;
  busy = true;

  try {
    const nowMs = performance.now();
    if (video.currentTime === lastVideoTime) {
      busy = false;
      return;
    }
    lastVideoTime = video.currentTime;

    // Run MediaPipe hand detection
    const mpResult = handLandmarker.detectForVideo(video, nowMs);
    const allHands = (mpResult.landmarks || []);

    const selected = selectPrimaryHands(allHands, 2, MIN_HAND_AREA);
    const detectedHands = selected.length;

    if (detectedHands < 1) {
      prediction.classList.toggle("not-detectable", true);
      prediction.textContent = "Not detectable";
      stable.textContent = "";
      statusText.textContent = "Waiting for a clear sign";
      history.length = 0;
      resetSpeechCandidate();
      pendingLetter = "";
      pendingSince = 0;
      letterEl.textContent = "-";
      confidenceEl.textContent = "-";
      handsEl.textContent = "0";
      return;
    }

    // Build feature vector and run ONNX
    const feature = handsToFeature(selected);
    const result = await predict(feature);

    const visibleLetter = result.not_detectable
      ? "Not detectable"
      : result.label;

    updateDisplayedLetter(
      visibleLetter,
      result.accepted && !result.not_detectable
    );

    letterEl.textContent = result.label;
    confidenceEl.textContent = result.confidence.toFixed(2);
    handsEl.textContent = String(detectedHands);
    statusText.textContent = result.accepted ? "Recognizing" : "Checking sign";

    if (!result.not_detectable) {
      updateSpeechCandidate(result.label);
    } else {
      resetSpeechCandidate();
    }

    if (result.accepted && !result.not_detectable) {
      updateStable(result.label);
    } else {
      stable.textContent = "";
      history.length = 0;
    }
  } finally {
    busy = false;
  }
}

// ── Event listeners ─────────────────────────────────────────────────

startButton.addEventListener("click", async () => {
  intro.classList.add("hidden");
  translator.classList.remove("hidden");
  prediction.textContent = "Loading models…";
  statusText.textContent = "Initializing…";

  try {
    await loadModels();
    statusText.textContent = "Camera permission needed";
    await startCamera();
    statusText.textContent = "Camera active";
    timer = setInterval(predictFrame, 280);
  } catch (error) {
    prediction.textContent = "Error";
    statusText.textContent = error.message || error.name;
    console.error(error);
    if (timer) clearInterval(timer);
  }
});

voiceToggle.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  voiceToggle.textContent = `Voice: ${voiceEnabled ? "On" : "Off"}`;
  voiceToggle.classList.toggle("active", voiceEnabled);
  statusText.textContent = voiceEnabled ? "Voice enabled" : "Voice off";
  if (voiceEnabled && "speechSynthesis" in window) {
    const utter = new SpeechSynthesisUtterance("Voice on");
    utter.rate = 0.9;
    window.speechSynthesis.speak(utter);
  }
  resetSpeechCandidate();
});

zoom.addEventListener("input", () => {
  zoomLevel = Number(zoom.value);
  zoomValue.textContent = `${zoomLevel.toFixed(1)}x`;
  video.style.transform = `scale(${zoomLevel})`;
});
