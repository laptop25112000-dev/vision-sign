const intro = document.getElementById("intro");
const translator = document.getElementById("translator");
const analysis = document.getElementById("analysis");
const startButton = document.getElementById("start");
const video = document.getElementById("video");
const canvas = document.getElementById("capture");
const prediction = document.getElementById("prediction");
const stable = document.getElementById("stable");
const letter = document.getElementById("letter");
const confidence = document.getElementById("confidence");
const hands = document.getElementById("hands");
const statusText = document.getElementById("status");
const voiceToggle = document.getElementById("voiceToggle");
const nextPanel = document.getElementById("nextPanel");
const backToDetection = document.getElementById("backToDetection");
const zoom = document.getElementById("zoom");
const zoomValue = document.getElementById("zoomValue");

const history = [];
const windowSize = 16;
const minVotes = 12;
let busy = false;
let timer = null;
let displayedLetter = "";
let pendingLetter = "";
let pendingSince = 0;
let voiceEnabled = false;
let lastSpoken = "";
let lastSpokenAt = 0;
let speechCandidate = "";
let speechCandidateSince = 0;
let spokenCandidate = "";
let zoomLevel = 1;
let completedTranslations = 0;
let lastCompletedStable = "";
const transitionHoldMs = 650;
const stableHoldMs = 900;
const speechCooldownMs = 1200;
const speechHoldMs = 2000;

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720 },
    audio: false,
  });
  video.srcObject = stream;
}

function mostCommon(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
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
  if (isStable && stableCandidate !== lastCompletedStable) {
    completedTranslations += 1;
    lastCompletedStable = stableCandidate;
    if (completedTranslations >= 2) {
      nextPanel.classList.remove("hidden");
    }
  }
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

async function speakLetter(label) {
  if (!voiceEnabled) return;
  const now = Date.now();
  if (label === lastSpoken && now - lastSpokenAt < speechCooldownMs) return;
  try {
    const response = await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: `Letter ${label}` }),
    });
    const result = await response.json();
    statusText.textContent = result.ok ? `Spoken ${label}` : "Voice unavailable";
    if (result.ok) {
      lastSpoken = label;
      lastSpokenAt = now;
    }
  } catch (error) {
    statusText.textContent = "Voice server error";
  }
}

async function speakNow(text) {
  try {
    const response = await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const result = await response.json();
    statusText.textContent = result.ok ? "Voice ready" : "Voice unavailable";
  } catch (error) {
    statusText.textContent = "Voice server error";
  }
}

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

async function predictFrame() {
  if (busy || video.readyState < 2) return;
  busy = true;

  canvas.width = 480;
  canvas.height = Math.round((video.videoHeight / video.videoWidth) * canvas.width) || 360;
  const ctx = canvas.getContext("2d");
  const scaledWidth = canvas.width * zoomLevel;
  const scaledHeight = canvas.height * zoomLevel;
  const offsetX = (canvas.width - scaledWidth) / 2;
  const offsetY = (canvas.height - scaledHeight) / 2;
  ctx.fillStyle = "#030506";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, offsetX, offsetY, scaledWidth, scaledHeight);

  try {
    const image = canvas.toDataURL("image/jpeg", 0.72);
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const result = await response.json();

    if (!result.ok) {
      prediction.classList.toggle("not-detectable", result.reason === "no_hand_detected");
      prediction.textContent = result.reason === "no_hand_detected" ? "Not detectable" : "";
      stable.textContent = "";
      statusText.textContent = "Waiting for a clear sign";
      history.length = 0;
      resetSpeechCandidate();
      pendingLetter = "";
      pendingSince = 0;
      letter.textContent = "-";
      confidence.textContent = "-";
      hands.textContent = String(result.hands || 0);
      return;
    }

    const visibleLetter = result.not_detectable ? "Not detectable" : result.label;
    updateDisplayedLetter(visibleLetter, result.accepted && !result.not_detectable);
    letter.textContent = result.label;
    confidence.textContent = result.confidence.toFixed(2);
    hands.textContent = String(result.hands);
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

startButton.addEventListener("click", async () => {
  intro.classList.add("hidden");
  translator.classList.remove("hidden");
  prediction.textContent = "Starting";
  statusText.textContent = "Camera permission needed";

  try {
    await startCamera();
    statusText.textContent = "Camera active";
    timer = setInterval(predictFrame, 280);
  } catch (error) {
    prediction.textContent = "Camera blocked";
    statusText.textContent = error.name;
    if (timer) clearInterval(timer);
  }
});

voiceToggle.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  voiceToggle.textContent = `Voice: ${voiceEnabled ? "On" : "Off"}`;
  voiceToggle.classList.toggle("active", voiceEnabled);
  statusText.textContent = voiceEnabled ? "Voice starting" : "Voice off";
  if (voiceEnabled) {
    speakNow("Voice on");
  }
  resetSpeechCandidate();
});

zoom.addEventListener("input", () => {
  zoomLevel = Number(zoom.value);
  zoomValue.textContent = `${zoomLevel.toFixed(1)}x`;
  video.style.transform = `scale(${zoomLevel})`;
});

nextPanel.addEventListener("click", () => {
  window.location.href = "/visual_voice_analysis.html";
});

backToDetection.addEventListener("click", () => {
  analysis.classList.add("hidden");
  translator.classList.remove("hidden");
});
