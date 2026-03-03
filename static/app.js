// ====== MODE + STT ======
const modeEl = document.getElementById("mode");
const modeHint = document.getElementById("modeHint");
const sttResult = document.getElementById("sttResult");
const ttsAudio = document.getElementById("ttsAudio");

const btnStartSTT = document.getElementById("btnStartSTT");
const btnStopSTT = document.getElementById("btnStopSTT");

function updateModeHint() {
    const m = modeEl.value;
    modeHint.textContent =
        m === "tunanetra"
            ? "Mode Tunanetra: sistem akan membacakan hasil (TTS)."
            : "Mode Tunarungu: sistem menampilkan hasil dalam teks.";
}
modeEl.addEventListener("change", updateModeHint);
updateModeHint();

let recognition = null;

btnStartSTT.addEventListener("click", async () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Browser kamu tidak mendukung Web Speech API. Gunakan Chrome/Edge atau pakai Whisper backend.");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "id-ID";
    recognition.interimResults = true;
    recognition.continuous = true;

    btnStartSTT.disabled = true;
    btnStopSTT.disabled = false;

    recognition.onresult = async (event) => {
        let finalText = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            finalText += event.results[i][0].transcript;
        }
        sttResult.value = finalText.trim();

        // Mode tunanetra: bacakan hasil
        if (modeEl.value === "tunanetra" && sttResult.value.length > 0) {
            const resp = await fetch("/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: sttResult.value })
            });
            const blob = await resp.blob();
            ttsAudio.src = URL.createObjectURL(blob);
            ttsAudio.play().catch(() => { });
        }
    };

    recognition.onerror = (e) => console.log("STT error:", e);
    recognition.onend = () => {
        btnStartSTT.disabled = false;
        btnStopSTT.disabled = true;
    };

    recognition.start();
});

btnStopSTT.addEventListener("click", () => {
    if (recognition) recognition.stop();
});

// ====== CAMERA REALTIME -> TEXT + VOICE ======
const video = document.getElementById("video");
const btnStartCam = document.getElementById("btnStartCam");
const btnStopCam = document.getElementById("btnStopCam");
const signResult = document.getElementById("signResult");
const signConf = document.getElementById("signConf");

let stream = null;
let timer = null;

// Stabilization settings
let lastLabel = "";
let stableCount = 0;
const STABLE_NEED = 3;       // butuh 3 prediksi sama berturut-turut
const MIN_SPEAK_MS = 1200;   // jeda minimal antar TTS
let lastSpeakAt = 0;

function toJpegDataUrl() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.6);
}

async function speak(text) {
    const resp = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });
    const blob = await resp.blob();
    ttsAudio.src = URL.createObjectURL(blob);
    ttsAudio.play().catch(() => { });
}

async function sendFrame() {
    const dataUrl = toJpegDataUrl();
    const resp = await fetch("/sign/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: dataUrl })
    });

    const data = await resp.json();
    const label = data.label || "";
    const confidence = Number(data.confidence || 0);

    if (!label) {
        signResult.textContent = "-";
        signConf.textContent = "";
        lastLabel = "";
        stableCount = 0;
        return;
    }

    signResult.textContent = label;
    signConf.textContent = ` (${Math.round(confidence * 100)}%)`;

    // Stabilisasi: label harus sama beberapa kali
    if (label === lastLabel) stableCount++;
    else {
        lastLabel = label;
        stableCount = 1;
    }

    const now = Date.now();

    // Kalau stabil dan tidak spam, bacakan
    if (stableCount >= STABLE_NEED && (now - lastSpeakAt) > MIN_SPEAK_MS) {
        lastSpeakAt = now;

        // Kamu minta: langsung muncul suara + teks.
        // Ini bacakan untuk semua mode. Kalau mau mode-based, tinggal kita kondisikan.
        await speak(`Huruf ${label}`);
    }
}

btnStartCam.addEventListener("click", async () => {
    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            signResult.textContent = "Browser tidak mendukung kamera.";
            return;
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false
        });

        video.srcObject = stream;
        await new Promise((resolve) => (video.onloadedmetadata = resolve));
        await video.play();

        btnStartCam.disabled = true;
        btnStopCam.disabled = false;

        timer = setInterval(() => {
            if (video.videoWidth > 0) sendFrame().catch(console.warn);
        }, 250); // 4 fps, terasa realtime tapi tidak berat
    } catch (err) {
        console.error(err);
        signResult.textContent = `Kamera gagal: ${err.name || ""} ${err.message || err}`;
    }
});

btnStopCam.addEventListener("click", () => {
    if (timer) clearInterval(timer);
    timer = null;

    if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
    }

    video.srcObject = null;
    btnStartCam.disabled = false;
    btnStopCam.disabled = true;
    signResult.textContent = "-";
    signConf.textContent = "";
});