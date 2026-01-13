const video = document.getElementById('webcam');
const canvas = document.getElementById('process-canvas');
const ctx = canvas.getContext('2d', { alpha: false });

const elAr = document.getElementById('prediction-text-ar');
const elEn = document.getElementById('prediction-text-en');
const elConfVal = document.getElementById('confidence-val');
const elConfBar = document.getElementById('confidence-bar');
const elStatus = document.getElementById('connection-status');
const elStatusText = document.getElementById('status-text');

const sentenceOutput = document.getElementById('sentence-output');
const btnSpeak = document.getElementById('btn-speak');
const btnClear = document.getElementById('btn-clear');

const settingsOverlay = document.getElementById('settings-overlay');
const btnSettings = document.getElementById('btn-settings');
const btnCloseSettings = document.getElementById('btn-close-settings');
const langSelect = document.getElementById('lang-select');

const CONFIG = {
    wsUrl: `ws://${location.host}/live-signs`,
    fps: 30,
    jpgQuality: 0.7,
    processWidth: 320,
    STABILITY_THRESHOLD: 15,
    CONFIDENCE_THRESHOLD: 0.4,
    theme: 'light',
    lang: 'ar-EG'
};

let state = {
    isSocketOpen: false,
    isSending: false,
    lastFrameTime: 0,
    lastFrameWord: "",
    sentenceBuffer: [],
    stabilityCounter: 0
};

const socket = new WebSocket(CONFIG.wsUrl);

socket.onopen = () => {
    updateStatus('connected');
    state.isSocketOpen = true;
};

socket.onclose = () => {
    updateStatus('disconnected');
    state.isSocketOpen = false;
};

socket.onerror = (err) => {
    console.error("WS Error", err);
    updateStatus('error');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateUI(data);
};

function initConfig() {
    const sysDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        CONFIG.theme = savedTheme;
    } else if (sysDark) {
        CONFIG.theme = 'dark';
    }
    setTheme(CONFIG.theme);

    const savedLang = localStorage.getItem('speechLang');
    if (savedLang) {
        CONFIG.lang = savedLang;
        langSelect.value = savedLang;
    }
}

window.setTheme = function (themeName) {
    document.documentElement.setAttribute('data-theme', themeName);
    CONFIG.theme = themeName;
    localStorage.setItem('theme', themeName);
    document.getElementById('theme-btn-light').classList.toggle('active', themeName === 'light');
    document.getElementById('theme-btn-dark').classList.toggle('active', themeName === 'dark');
};

langSelect.addEventListener('change', (e) => {
    CONFIG.lang = e.target.value;
    localStorage.setItem('speechLang', CONFIG.lang);
    speakText(CONFIG.lang.startsWith('ar') ? "مرحباً" : "Hello");
});

btnSettings.addEventListener('click', () => {
    settingsOverlay.classList.remove('hidden');
});

btnCloseSettings.addEventListener('click', () => {
    settingsOverlay.classList.add('hidden');
});

settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) {
        settingsOverlay.classList.add('hidden');
    }
});

function updateStatus(status) {
    elStatus.className = `status-pill ${status}`;
    if (status === 'connected') elStatusText.textContent = "Live";
    else if (status === 'disconnected') elStatusText.textContent = "Offline";
    else elStatusText.textContent = "Error";
}

function updateUI(data) {
    if (data.status === "idle" || !data.detected_word) {
        // elAr.textContent = "...";
        // elEn.textContent = "...";
        state.stabilityCounter = 0;
        state.lastFrameWord = "";

        elConfVal.textContent = "0%";
        elConfBar.style.width = "0%";
        return;
    }

    const { sign_ar, sign_en } = data.detected_word;
    const confidencePct = Math.round(data.confidence * 100);

    elAr.textContent = sign_ar;
    elEn.textContent = sign_en;
    elConfVal.textContent = `${confidencePct}%`;
    elConfBar.style.width = `${confidencePct}%`;

    if (sign_ar === state.lastFrameWord && data.confidence > CONFIG.CONFIDENCE_THRESHOLD) {
        state.stabilityCounter++;
    } else {
        state.stabilityCounter = 0;
        state.lastFrameWord = sign_ar;
    }

    if (state.stabilityCounter === CONFIG.STABILITY_THRESHOLD) {
        addWordToSentence(sign_ar);

        elAr.style.transform = "scale(1.2)";
        setTimeout(() => elAr.style.transform = "scale(1)", 200);

        speakText(sign_ar);
    }

    if (confidencePct > 80) elConfBar.style.backgroundColor = 'var(--success-color)';
    else if (confidencePct > 50) elConfBar.style.backgroundColor = 'var(--accent-color)';
    else elConfBar.style.backgroundColor = 'var(--error-color)';
}

function addWordToSentence(word) {
    const lastWord = state.sentenceBuffer[state.sentenceBuffer.length - 1];
    if (lastWord !== word) {
        state.sentenceBuffer.push(word);
        renderSentence();

        sentenceOutput.scrollLeft = sentenceOutput.scrollWidth;
    }
}

function renderSentence() {
    sentenceOutput.textContent = state.sentenceBuffer.join(" ");
}

function speakText(text) {
    if (!text) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);

    utterance.lang = CONFIG.lang;
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
}

btnSpeak.addEventListener('click', () => {
    const text = state.sentenceBuffer.join(" ");
    speakText(text);
});

btnClear.addEventListener('click', () => {
    state.sentenceBuffer = [];
    renderSentence();
});

async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;

        video.onloadedmetadata = () => {
            const ratio = video.videoWidth / video.videoHeight;
            canvas.width = CONFIG.processWidth;
            canvas.height = CONFIG.processWidth / ratio;
            requestAnimationFrame(loop);
        };
    } catch (err) {
        console.error("Camera access denied:", err);
        alert("Please allow camera access to use this app.");
    }
}

function loop(timestamp) {
    requestAnimationFrame(loop);

    const interval = 1000 / CONFIG.fps;
    const elapsed = timestamp - state.lastFrameTime;

    if (elapsed > interval) {
        state.lastFrameTime = timestamp - (elapsed % interval);

        if (state.isSocketOpen && !state.isSending && video.readyState === 4) {
            processFrame();
        }
    }
}

function processFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    state.isSending = true;

    canvas.toBlob((blob) => {
        if (blob && state.isSocketOpen) {
            socket.send(blob);
        }
        state.isSending = false;
    }, 'image/jpeg', CONFIG.jpgQuality);
}

initConfig();
setupWebcam();
