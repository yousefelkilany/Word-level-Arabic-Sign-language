const video = document.getElementById('webcam');
const canvas = document.getElementById('process-canvas');
const ctx = canvas.getContext('2d', { alpha: false });

const elAr = document.getElementById('prediction-text-ar');
const elEn = document.getElementById('prediction-text-en');
const elConfVal = document.getElementById('confidence-val');
const elConfBar = document.getElementById('confidence-bar');
const elStatus = document.getElementById('connection-status');
const elStatusText = document.getElementById('status-text');

const CONFIG = {
    wsUrl: `ws://${location.host}/live-signs`,
    fps: 3,
    jpgQuality: 0.7,
    processWidth: 320
};

let state = {
    isSocketOpen: false,
    isSending: false,
    lastFrameTime: 0
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

    if (confidencePct > 80) elConfBar.style.backgroundColor = 'var(--success-color)';
    else if (confidencePct > 50) elConfBar.style.backgroundColor = 'var(--accent-color)';
    else elConfBar.style.backgroundColor = 'var(--error-color)';
}

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

function initTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    const htmlEl = document.documentElement;

    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
        htmlEl.setAttribute('data-theme', 'dark');
    } else {
        htmlEl.setAttribute('data-theme', 'light');
    }

    function toggleTheme() {
        const currentTheme = htmlEl.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        htmlEl.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }
    themeToggleBtn.addEventListener('click', toggleTheme);
}

initTheme();
setupWebcam();
