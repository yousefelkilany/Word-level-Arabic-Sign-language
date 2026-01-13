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

const historySidebar = document.getElementById('history-sidebar');
const historyList = document.getElementById('history-list');
const btnHistoryToggle = document.getElementById('history-toggle');
const btnCloseHistory = document.getElementById('close-history');
const btnClearHistory = document.getElementById('clear-history');

const btnArchive = document.getElementById('btn-archive');
const archiveModal = document.getElementById('archive-modal');
const btnCloseArchive = document.getElementById('btn-close-archive');
const sessionListEl = document.getElementById('session-list');
const sessionDetailEl = document.getElementById('session-detail');
const btnBackList = document.getElementById('btn-back-list');

let currentSessionLog = [];

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

btnHistoryToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    historySidebar.classList.toggle('open');
});

btnCloseHistory.addEventListener('click', () => {
    historySidebar.classList.remove('open');
});

historySidebar.addEventListener('click', (e) => {
    e.stopPropagation();
});

function addToHistoryLog(word) {
    const emptyMsg = historyList.querySelector('.history-empty');
    if (emptyMsg) emptyMsg.remove();

    const li = document.createElement('li');
    li.className = 'history-item';

    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    li.innerHTML = `
        <span class="history-word">${word}</span>
        <span class="history-time">${timeString}</span>
    `;

    historyList.prepend(li);
    if (historyList.children.length > 50) {
        historyList.lastElementChild.remove();
    }
}

document.addEventListener('click', (e) => {
    if (historySidebar.classList.contains('open')) {
        const isClickInside = historySidebar.contains(e.target);
        const isClickOnToggle = btnHistoryToggle.contains(e.target);
        if (!isClickInside && !isClickOnToggle) {
            historySidebar.classList.remove('open');
        }
    }
});

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
        speakText(sign_ar);
        saveCurrentSession();

        elAr.style.transform = "scale(1.2)";
        setTimeout(() => elAr.style.transform = "scale(1)", 200);
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
    console.log(sessionData);
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
    state.sentenceBuffer = [];
    renderSentence();
    historyList.innerHTML = '<li class="history-empty">No signs detected yet.</li>';
}

btnClearHistory.addEventListener('click', () => {
    resetRecognizedWords();
    currentSessionLog = [];
});

function renderSessions() {
    const savedSessions = JSON.parse(localStorage.getItem('slr_sessions') || '[]');
    sessionListEl.innerHTML = '';

    if (savedSessions.length === 0) {
        sessionListEl.innerHTML = '<li style="text-align:center; color:var(--text-sub); margin-top:2rem;">No saved sessions.</li>';
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
        sessionListEl.appendChild(li);
    });
}

function showSessionDetail(session) {
    sessionListEl.style.display = 'none';
    sessionDetailEl.classList.remove('hidden');

    document.getElementById('detail-date').textContent = session.date;
    document.getElementById('detail-sentence').textContent = session.sentence || "(No sentence formed)";

    const logEl = document.getElementById('detail-log');
    logEl.innerHTML = '';

    session.log.forEach(item => {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `
            <span class="history-word">${item.word}</span>
            <span class="history-time">${item.time}</span>
        `;
        logEl.appendChild(li);
    });
}

btnArchive.addEventListener('click', () => {
    renderSessions();
    sessionListEl.style.display = 'block';
    sessionDetailEl.classList.add('hidden');
    archiveModal.classList.remove('hidden');
});

btnCloseArchive.addEventListener('click', () => {
    archiveModal.classList.add('hidden');
});

btnBackList.addEventListener('click', () => {
    sessionDetailEl.classList.add('hidden');
    sessionListEl.style.display = 'block';
});

archiveModal.addEventListener('click', (e) => {
    if (e.target === archiveModal) archiveModal.classList.add('hidden');
});

function renderSentence() {
    sentenceOutput.textContent = state.sentenceBuffer.join(" ");
    sentenceOutput.scrollLeft = sentenceOutput.scrollWidth;
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
