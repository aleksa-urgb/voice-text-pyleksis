/* ==========================================================
   app.js — BISINDO Sign Language App
   ========================================================== */

// ── Element references ──────────────────────────────────────────
const modeEl = document.getElementById("mode");
const modeHint = document.getElementById("modeHint");

const signAudio = document.getElementById("signAudio");
const ttsAudio = document.getElementById("ttsAudio");

const video = document.getElementById("video");
const camOverlay = document.getElementById("camOverlay");
const detectBadge = document.getElementById("detectBadge");
const signResult = document.getElementById("signResult");
const signConf = document.getElementById("signConf");
const signStatus = document.getElementById("signStatus");
const historyTape = document.getElementById("historyTape");
const chkSpeak = document.getElementById("chkSpeak");

const btnStartCam = document.getElementById("btnStartCam");
const btnStopCam = document.getElementById("btnStopCam");
const btnClearHistory = document.getElementById("btnClearHistory");

const btnStartSTT = document.getElementById("btnStartSTT");
const btnStopSTT = document.getElementById("btnStopSTT");
const btnReadSTT = document.getElementById("btnReadSTT");
const sttResult = document.getElementById("sttResult");
const micRing = document.getElementById("micRing");
const voiceState = document.getElementById("voiceState");

const btnSpeak = document.getElementById("btnSpeak");
const ttsInput = document.getElementById("ttsInput");

const dotLandmarker = document.getElementById("dotLandmarker");
const dotModel = document.getElementById("dotModel");

// ── Panel navigation ────────────────────────────────────────────
document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`panel-${btn.dataset.panel}`).classList.add("active");
    });
});

// ── Mode hint ────────────────────────────────────────────────────
function updateModeHint() {
    modeHint.textContent = modeEl.value === "tunanetra"
        ? "Suara aktif: huruf diucapkan otomatis."
        : "Teks aktif: hasil ditampilkan di layar.";
}
modeEl.addEventListener("change", updateModeHint);
updateModeHint();

// ── Health check (status indicator) ─────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();

        dotLandmarker.className = "dot " + (data.hand_landmarker ? "ok" : "err");
        dotModel.className = "dot " + (data.az_model ? "ok" : "err");
    } catch {
        dotLandmarker.className = "dot err";
        dotModel.className = "dot err";
    }
}
checkHealth();
setInterval(checkHealth, 10_000);

// ── TTS helper ───────────────────────────────────────────────────
async function speakText(text, audioEl = signAudio) {
    if (!text || !text.trim()) return;
    try {
        const res = await fetch("/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        const blob = await res.blob();
        audioEl.src = URL.createObjectURL(blob);
        audioEl.play().catch(() => { });
    } catch (e) {
        console.warn("TTS error:", e);
    }
}

// ── Camera + Sign Detection ──────────────────────────────────────
let stream = null;
let frameTimer = null;
let isSending = false;

// Stabilizer
let lastLabel = "";
let stableCount = 0;
const STABLE_NEED = 3;       // berapa kali berturut-turut sama
const MIN_SPEAK_MS = 1500;    // jeda min antar TTS
let lastSpeakAt = 0;

function toJpegDataUrl() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.65);
}

function addHistoryChip(label) {
    const chip = document.createElement("div");
    chip.className = "history-chip";
    chip.textContent = label;
    historyTape.appendChild(chip);

    // Hapus oldest kalau > 30 karakter
    const chips = historyTape.querySelectorAll(".history-chip");
    if (chips.length > 30) chips[0].remove();
}

async function sendFrame() {
    if (isSending) return;          // skip kalau request sebelumnya masih jalan
    isSending = true;

    try {
        const dataUrl = toJpegDataUrl();
        const res = await fetch("/sign/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: dataUrl }),
        });

        const data = await res.json();
        const label = (data.label || "").trim();
        const conf = Number(data.confidence || 0);

        if (!label) {
            signResult.textContent = "—";
            signConf.textContent = "";
            signStatus.textContent = data.error || "Tangan tidak terdeteksi";
            detectBadge.classList.remove("active");
            lastLabel = "";
            stableCount = 0;
            return;
        }

        // Tampilkan
        signResult.textContent = label;
        signConf.textContent = `${Math.round(conf * 100)}%`;
        signStatus.textContent = "Terdeteksi";
        detectBadge.classList.add("active");

        // Stabilisasi
        if (label === lastLabel) {
            stableCount++;
        } else {
            lastLabel = label;
            stableCount = 1;
        }

        const now = Date.now();
        if (stableCount >= STABLE_NEED && (now - lastSpeakAt) > MIN_SPEAK_MS) {
            lastSpeakAt = now;
            addHistoryChip(label);

            if (chkSpeak.checked || modeEl.value === "tunanetra") {
                await speakText(`Huruf ${label}`);
            }
        }
    } catch (e) {
        signStatus.textContent = "Error: " + e.message;
    } finally {
        isSending = false;
    }
}

btnStartCam.addEventListener("click", async () => {
    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            alert("Browser tidak mendukung akses kamera.");
            return;
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false,
        });

        video.srcObject = stream;
        await new Promise(res => { video.onloadedmetadata = res; });
        await video.play();

        camOverlay.classList.add("hidden");
        btnStartCam.disabled = true;
        btnStopCam.disabled = false;
        signStatus.textContent = "Kamera aktif...";

        frameTimer = setInterval(() => {
            if (video.videoWidth > 0) sendFrame();
        }, 250);   // ~4 FPS — cukup responsif, tidak terlalu berat

    } catch (err) {
        console.error(err);
        signStatus.textContent = `Kamera gagal: ${err.name} — ${err.message}`;
    }
});

btnStopCam.addEventListener("click", () => {
    clearInterval(frameTimer);
    frameTimer = null;

    stream?.getTracks().forEach(t => t.stop());
    stream = null;

    video.srcObject = null;
    camOverlay.classList.remove("hidden");
    btnStartCam.disabled = false;
    btnStopCam.disabled = true;
    signResult.textContent = "—";
    signConf.textContent = "";
    signStatus.textContent = "Kamera dimatikan";
    detectBadge.classList.remove("active");
});

btnClearHistory.addEventListener("click", () => {
    historyTape.innerHTML = "";
});

// ── STT (Speech-to-Text) ─────────────────────────────────────────
let recognition = null;

btnStartSTT.addEventListener("click", () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Browser ini tidak mendukung Web Speech API.\nGunakan Chrome atau Edge.");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "id-ID";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;

    btnStartSTT.disabled = true;
    btnStopSTT.disabled = false;
    micRing.classList.add("recording");
    voiceState.textContent = "Merekam...";

    recognition.onresult = async event => {
        let final = "", interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const t = event.results[i][0].transcript;
            if (event.results[i].isFinal) final += t;
            else interim += t;
        }

        const display = (sttResult.value + final).trim() || interim;
        sttResult.value = display;

        // Mode tunanetra: bacakan otomatis setiap kalimat final selesai
        if (final && modeEl.value === "tunanetra") {
            await speakText(final, ttsAudio);
        }
    };

    recognition.onerror = e => {
        console.warn("STT error:", e.error);
        voiceState.textContent = "Error: " + e.error;
    };

    recognition.onend = () => {
        btnStartSTT.disabled = false;
        btnStopSTT.disabled = true;
        micRing.classList.remove("recording");
        voiceState.textContent = "Rekaman selesai";
    };

    recognition.start();
});

btnStopSTT.addEventListener("click", () => {
    recognition?.stop();
});

btnReadSTT.addEventListener("click", () => {
    speakText(sttResult.value, ttsAudio);
});

// ── TTS Panel ─────────────────────────────────────────────────────
btnSpeak.addEventListener("click", () => {
    speakText(ttsInput.value, ttsAudio);
});