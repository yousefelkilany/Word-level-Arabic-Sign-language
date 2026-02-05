const elVideo = document.getElementById('webcam');
const elCanvas = document.getElementById('process-canvas');
const elCtx = elCanvas.getContext('2d', { alpha: false });

const elOverlayCanvas = document.getElementById('overlay-canvas');
const elOverlayCtx = elOverlayCanvas.getContext('2d');
const chkDrawPoints = document.getElementById('chk-draw-points');
const chkDrawLines = document.getElementById('chk-draw-lines');

const elAr = document.getElementById('prediction-text-ar');
const elEn = document.getElementById('prediction-text-en');
const elConfVal = document.getElementById('confidence-val');
const elConfBar = document.getElementById('confidence-bar');
const elStatus = document.getElementById('connection-status');
const elStatusText = document.getElementById('status-text');

const elSentenceOutput = document.getElementById('sentence-output');
const btnSpeak = document.getElementById('btn-speak');
const btnClear = document.getElementById('btn-clear');

const elSettingsOverlay = document.getElementById('settings-overlay');
const btnSettings = document.getElementById('btn-settings');
const btnCloseSettings = document.getElementById('btn-close-settings');
const elLangSelect = document.getElementById('lang-select');

const elHistorySidebar = document.getElementById('history-sidebar');
const elHistoryList = document.getElementById('history-list');
const btnHistoryToggle = document.getElementById('history-toggle');
const btnCloseHistory = document.getElementById('close-history');
const btnClearHistory = document.getElementById('clear-history');

const btnArchive = document.getElementById('btn-archive');
const elArchiveModal = document.getElementById('archive-modal');
const btnCloseArchive = document.getElementById('btn-close-archive');
const elSessionList = document.getElementById('session-list');
const elSessionDetail = document.getElementById('session-detail');
const btnBackList = document.getElementById('btn-back-list');

let currentSessionLog = [];
let POSE_KPS = [];
let FACE_KPS = [];
let HAND_KPS = [];
let POSE_CONNECTIONS = [];
let FACE_CONNECTIONS = [];
let HAND_CONNECTIONS = [];
let KPS_MAPPING = {};

const CONFIG = {
    wsUrl: `ws://${location.host}/live-signs`,
    fps: 30,
    jpgQuality: 0.7,
    processWidth: 320,
    STABILITY_THRESHOLD: 0,
    theme: 'light',
    lang: 'ar-EG',
    viz: {
        face: false,
        pose: true,
        hands: false,
        lines: false,
        points: true
    }
};
let needLandmarks = false;

let state = {
    sessionId: new Date().toISOString(),
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

function setupDrawingToggle(element, draw_mode) {
    const configKey = `chk-draw-${draw_mode}`;
    let saved = localStorage.getItem(configKey);
    if (saved)
        saved = saved === 'true';
    else
        saved = CONFIG.viz[draw_mode];

    element.checked = saved;
    CONFIG.viz[draw_mode] = saved;

    element.addEventListener('change', (e) => {
        CONFIG.viz[draw_mode] = e.target.checked;
        localStorage.setItem(configKey, CONFIG.viz[draw_mode]);
    });
}

window.toggleViz = function (region) {
    CONFIG.viz[region] = !CONFIG.viz[region];
    const btn = document.getElementById(`btn-viz-${region}`);
    btn.classList.toggle('active', CONFIG.viz[region]);
    localStorage.setItem(`viz_${region}`, CONFIG.viz[region]);
};

function initVizSettings() {
    ['face', 'pose', 'hands'].forEach(region => {
        const saved = localStorage.getItem(`viz_${region}`);
        if (saved !== null) {
            CONFIG.viz[region] = (saved === 'true');
            document.getElementById(`btn-viz-${region}`).classList.toggle('active', CONFIG.viz[region]);
        }
    });
}

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
        elLangSelect.value = savedLang;
    }

    loadSkeletonConfig();

    initVizSettings();
    setupDrawingToggle(chkDrawLines, 'lines');
    setupDrawingToggle(chkDrawPoints, 'points');
}

window.setTheme = function (themeName) {
    document.documentElement.setAttribute('data-theme', themeName);
    CONFIG.theme = themeName;
    localStorage.setItem('theme', themeName);
    document.getElementById('theme-btn-light').classList.toggle('active', themeName === 'light');
    document.getElementById('theme-btn-dark').classList.toggle('active', themeName === 'dark');
};

btnHistoryToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    elHistorySidebar.classList.toggle('open');
});

btnCloseHistory.addEventListener('click', () => {
    elHistorySidebar.classList.remove('open');
});

elHistorySidebar.addEventListener('click', (e) => {
    e.stopPropagation();
});

function addToHistoryLog(word) {
    const emptyMsg = elHistoryList.querySelector('.history-empty');
    if (emptyMsg) emptyMsg.remove();

    const li = document.createElement('li');
    li.className = 'history-item';

    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    li.innerHTML = `
        <span class="history-word">${word}</span>
        <span class="history-time">${timeString}</span>
    `;

    elHistoryList.prepend(li);
    if (elHistoryList.children.length > 50) {
        elHistoryList.lastElementChild.remove();
    }
}

document.addEventListener('click', (e) => {
    if (elHistorySidebar.classList.contains('open')) {
        const isClickInside = elHistorySidebar.contains(e.target);
        const isClickOnToggle = btnHistoryToggle.contains(e.target);
        if (!isClickInside && !isClickOnToggle) {
            elHistorySidebar.classList.remove('open');
        }
    }
});

elLangSelect.addEventListener('change', (e) => {
    CONFIG.lang = e.target.value;
    localStorage.setItem('speechLang', CONFIG.lang);
    speakText(CONFIG.lang.startsWith('ar') ? "مرحباً" : "Hello");
});

btnSettings.addEventListener('click', () => {
    elSettingsOverlay.classList.remove('hidden');
});

btnCloseSettings.addEventListener('click', () => {
    elSettingsOverlay.classList.add('hidden');
});

elSettingsOverlay.addEventListener('click', (e) => {
    if (e.target === elSettingsOverlay) {
        elSettingsOverlay.classList.add('hidden');
    }
});

function updateStatus(status) {
    elStatus.className = `status-pill ${status}`;
    if (status === 'connected') elStatusText.textContent = "Live";
    else if (status === 'disconnected') elStatusText.textContent = "Offline";
    else elStatusText.textContent = "Error";
}

function updateUI(data) {
    console.log(data);
    drawSkeleton(data.landmarks);

    if (data.status === "idle" || !data.detected_sign) {
        elAr.textContent = "...";
        elEn.textContent = "...";
        state.stabilityCounter = 0;
        state.lastFrameWord = "";
        state.sentenceBuffer = [];
        renderSentence();
        elConfVal.textContent = "0%";
        elConfBar.style.width = "0%";
        return;
    }

    const { sign_ar, sign_en } = data.detected_sign;
    const confidencePct = Math.round(data.confidence * 100);

    if (sign_ar === state.lastFrameWord) {
        state.stabilityCounter++;
    } else {
        state.stabilityCounter = 0;
        state.lastFrameWord = sign_ar;
    }

    // TODO: there's now this guard for stability on client-side
    // this is suppressed in favor of server-side
    if (state.stabilityCounter === CONFIG.STABILITY_THRESHOLD) {
        elAr.textContent = sign_ar;
        elEn.textContent = sign_en;
        elConfVal.textContent = `${confidencePct}%`;
        elConfBar.style.width = `${confidencePct}%`;

        addWordToSentence(sign_ar);
        speakText(sign_ar);
        saveCurrentSession();

        elAr.style.transform = "scale(1.2)";
        setTimeout(() => elAr.style.transform = "scale(1)", 200);
    }

    if (confidencePct > 80) elConfBar.style.backgroundColor = 'var(--success-color)';
    else if (confidencePct > 50) elConfBar.style.backgroundColor = 'var(--accent-color)';
    else elConfBar.style.backgroundColor = 'var(--error-color)';
}

async function loadSkeletonConfig() {
    try {
        const response = await fetch('/static/simplified_kps_connections.json');
        if (!response.ok)
            throw new Error(`HTTP error! status: ${response.status}`);

        const kps = await response.json();
        if (!kps)
            throw new Error(`connections JSON error!`);

        POSE_KPS = kps['pose_kps'];
        FACE_KPS = kps['face_kps'];
        HAND_KPS = kps['hand_kps'];
        POSE_CONNECTIONS = kps['pose_connections'];
        FACE_CONNECTIONS = kps['face_contours'];
        HAND_CONNECTIONS = kps['hand_connections'];
        KPS_MAPPING = kps['mp_idx_to_kps_idx'];
        if (!POSE_CONNECTIONS || !FACE_CONNECTIONS || !HAND_CONNECTIONS || !KPS_MAPPING)
            throw new Error(`Failed to load connections and kyepoints mapping!`);
        console.log(`[ArSL] Loaded skeleton connections.`);
    } catch (err) {
        console.error("[ArSL] Failed to load skeleton config:", err);
        POSE_CONNECTIONS = [];
        FACE_CONNECTIONS = [];
        HAND_CONNECTIONS = [];
        KPS_MAPPING = {};
    }
}

function drawSkeleton(landmarks) {
    elOverlayCtx.clearRect(0, 0, elOverlayCanvas.width, elOverlayCanvas.height);
    if (!landmarks || landmarks.length === 0 || (!CONFIG.viz.lines && !CONFIG.viz.points))
        return;

    const w = elOverlayCanvas.width;
    const h = elOverlayCanvas.height;

    // landmarks are normalized 0-1, so we need to scale them to the canvas size
    const scaledLandmarks = landmarks.map(p => ({ x: p[0] * w, y: p[1] * h }));

    if (CONFIG.viz.points) {
        const drawRegion = (mp_idx, mp_idx_map_kps, color) => {
            elOverlayCtx.beginPath();
            elOverlayCtx.fillStyle = color;
            const point_r = 2;
            mp_idx.forEach((idx) => {
                const p = scaledLandmarks[mp_idx_map_kps[idx]];
                if (p && (p.x !== 0 || p.y !== 0)) {
                    elOverlayCtx.moveTo(p.x + point_r, p.y);
                    elOverlayCtx.arc(p.x, p.y, point_r, 0, 2 * Math.PI);
                }
            });
            elOverlayCtx.fill();
        };

        if (CONFIG.viz.pose) drawRegion(POSE_KPS, KPS_MAPPING['pose'], '#0d9165');
        if (CONFIG.viz.face) drawRegion(FACE_KPS, KPS_MAPPING['face'], '#d1880a');
        if (CONFIG.viz.hands) {
            drawRegion(HAND_KPS, KPS_MAPPING['rh'], '#ca3a3a');
            drawRegion(HAND_KPS, KPS_MAPPING['lh'], '#ca3a3a');
        }
    }

    if (CONFIG.viz.lines) {
        const drawRegion = (connections, mp_idx_map_kps, color) => {
            if (connections.length == 0)
                return;

            elOverlayCtx.beginPath();
            elOverlayCtx.lineWidth = 2;
            elOverlayCtx.lineJoin = 'round';

            connections.forEach(([startIdx, endIdx]) => {
                const p1 = scaledLandmarks[mp_idx_map_kps[startIdx]];
                const p2 = scaledLandmarks[mp_idx_map_kps[endIdx]];
                if (p1 && p2 && (p1.x || p1.y) && (p2.x || p2.y)) {
                    elOverlayCtx.moveTo(p1.x, p1.y);
                    elOverlayCtx.lineTo(p2.x, p2.y);
                }
            });

            elOverlayCtx.strokeStyle = color;
            elOverlayCtx.stroke();
        };

        if (CONFIG.viz.pose) drawRegion(POSE_CONNECTIONS, KPS_MAPPING['pose'], '#10b981');
        if (CONFIG.viz.face)
            Object.values(FACE_CONNECTIONS).forEach(face_subregion =>
                drawRegion(face_subregion, KPS_MAPPING['face'], '#f59e0b')
            );
        if (CONFIG.viz.hands) {
            drawRegion(HAND_CONNECTIONS, KPS_MAPPING['rh'], '#ef4444');
            drawRegion(HAND_CONNECTIONS, KPS_MAPPING['lh'], '#ef4444');
        }
    }
}

function addWordToSentence(word) {
    const lastWord = state.sentenceBuffer[state.sentenceBuffer.length - 1];
    if (lastWord !== word) {
        state.sentenceBuffer.push(word);
        renderSentence();
        addToHistoryLog(word);
        currentSessionLog.push({
            word: word,
            time: (new Date()).toLocaleTimeString()
        });
    }
}

function saveCurrentSession(renew_id = false) {
    if (state.sentenceBuffer.length === 0 && currentSessionLog.length === 0) return;

    const sessionData = {
        id: state.sessionId,
        date: new Date().toLocaleString(),
        sentence: state.sentenceBuffer.join(" "),
        log: currentSessionLog
    };

    if (renew_id) state.sessionId = new Date().toISOString();

    const savedSessions = JSON.parse(localStorage.getItem('slr_sessions') || '[]');
    if (savedSessions.length > 0 && savedSessions[0].id == state.sessionId) {
        savedSessions.shift();
    }
    savedSessions.unshift(sessionData);
    if (savedSessions.length > 20) savedSessions.pop();

    localStorage.setItem('slr_sessions', JSON.stringify(savedSessions));
}

function resetRecognizedWords() {
    saveCurrentSession(true);
    elHistoryList.innerHTML = '<li class="history-empty">No signs detected yet.</li>';
    currentSessionLog = [];
    updateUI({});
}

btnClearHistory.addEventListener('click', () => {
    resetRecognizedWords();
});

function renderSessions() {
    const savedSessions = JSON.parse(localStorage.getItem('slr_sessions') || '[]');
    elSessionList.innerHTML = '';

    if (savedSessions.length === 0) {
        elSessionList.innerHTML = '<li style="text-align:center; color:var(--text-sub); margin-top:2rem;">No saved sessions.</li>';
        return;
    }

    savedSessions.forEach(session => {
        const li = document.createElement('li');
        li.className = 'session-item';
        const preview = session.sentence || (session.log.length + " signs detected");

        li.innerHTML = `
            <div>
                <span class="session-date">${session.date}</span>
                <span class="session-preview">${preview}</span>
            </div>
            <div style="color:var(--text-sub)">›</div>
        `;

        li.addEventListener('click', () => showSessionDetail(session));
        elSessionList.appendChild(li);
    });
}

function showSessionDetail(session) {
    elSessionList.style.display = 'none';
    elSessionDetail.classList.remove('hidden');

    document.getElementById('detail-date').textContent = session.date;
    document.getElementById('detail-sentence').textContent = session.sentence || "(No sentence formed)";

    const elLog = document.getElementById('detail-log');
    elLog.innerHTML = '';
    session.log.forEach(item => {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `
            <span class="history-word">${item.word}</span>
            <span class="history-time">${item.time}</span>
        `;
        elLog.appendChild(li);
    });
}

btnArchive.addEventListener('click', () => {
    renderSessions();
    elSessionList.style.display = 'block';
    elSessionDetail.classList.add('hidden');
    elArchiveModal.classList.remove('hidden');
});

btnCloseArchive.addEventListener('click', () => {
    elArchiveModal.classList.add('hidden');
});

btnBackList.addEventListener('click', () => {
    elSessionDetail.classList.add('hidden');
    elSessionList.style.display = 'block';
});

elArchiveModal.addEventListener('click', (e) => {
    if (e.target === elArchiveModal) elArchiveModal.classList.add('hidden');
});

function renderSentence() {
    elSentenceOutput.textContent = state.sentenceBuffer.join(" ");
    elSentenceOutput.scrollLeft = elSentenceOutput.scrollWidth;
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
    resetRecognizedWords();
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

        elVideo.srcObject = stream;

        elVideo.onloadedmetadata = () => {
            const ratio = elVideo.videoWidth / elVideo.videoHeight;
            elCanvas.width = CONFIG.processWidth;
            elCanvas.height = CONFIG.processWidth / ratio;
            elOverlayCanvas.width = elCanvas.width;
            elOverlayCanvas.height = elCanvas.height;
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
        if (state.isSocketOpen && !state.isSending && elVideo.readyState === 4) {
            processFrame();
        }
    }
}

function processFrame() {
    needLandmarks = (CONFIG.viz.lines || CONFIG.viz.points) &&
        (CONFIG.viz.face || CONFIG.viz.pose || CONFIG.viz.hands);
    state.isSending = true;
    elCtx.drawImage(elVideo, 0, 0, elCanvas.width, elCanvas.height);
    elCanvas.toBlob((blob) => {
        if (blob && state.isSocketOpen) {
            const header = new Uint8Array(1);
            header[0] = needLandmarks ? 1 : 0;
            const finalBlob = new Blob([header, blob], { type: 'image/jpeg' });
            socket.send(finalBlob);
        }
    }, 'image/jpeg', CONFIG.jpgQuality);
    state.isSending = false;
}

initConfig();
setupWebcam();
