class AppConfig {
    constructor() {
        this.wsUrl = `ws://${location.host}/live-signs`;
        this.fps = 30;
        this.jpgQuality = 0.7;
        this.processWidth = 320;
        this.STABILITY_THRESHOLD = 0;
        this.theme = 'light';
        this.lang = localStorage.getItem('speechLang') || 'ar-EG';
        this.viz = {
            face: false,
            pose: true,
            hands: false,
            lines: false,
            points: true
        };

        this.init();
    }

    init() {
        const sysDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            this.theme = savedTheme;
        } else if (sysDark) {
            this.theme = 'dark';
        }

        ['face', 'pose', 'hands'].forEach(region => {
            const saved = localStorage.getItem(`viz_${region}`);
            if (saved !== null) {
                this.viz[region] = (saved === 'true');
            }
        });

        const savedLines = localStorage.getItem('chk-draw-lines');
        if (savedLines !== null) this.viz.lines = (savedLines === 'true');

        const savedPoints = localStorage.getItem('chk-draw-points');
        if (savedPoints !== null) this.viz.points = (savedPoints === 'true');
    }

    setTheme(themeName) {
        this.theme = themeName;
        localStorage.setItem('theme', themeName);
        document.documentElement.setAttribute('data-theme', themeName);
    }

    setLang(lang) {
        this.lang = lang;
        localStorage.setItem('speechLang', lang);
    }

    setViz(region, value) {
        this.viz[region] = value;
        localStorage.setItem(`viz_${region}`, value);
    }

    setDrawMode(mode, value) {
        this.viz[mode] = value;
        localStorage.setItem(`chk-draw-${mode}`, value);
    }
}

class SessionManager {
    constructor(config) {
        this.config = config;
        this.sessionId = new Date().toISOString();
        this.sentenceBuffer = [];
        this.currentSessionLog = [];
    }

    addWord(word, onUpdate) {
        const lastWord = this.sentenceBuffer[this.sentenceBuffer.length - 1];
        if (lastWord !== word) {
            this.sentenceBuffer.push(word);
            this.currentSessionLog.push({
                word: word,
                time: (new Date()).toLocaleTimeString()
            });
            this.save();
            if (onUpdate) onUpdate(word, this.sentenceBuffer);
        }
    }

    save(renewId = false) {
        if (this.sentenceBuffer.length === 0 && this.currentSessionLog.length === 0) return;

        const sessionData = {
            id: this.sessionId,
            date: new Date().toLocaleString(),
            sentence: this.sentenceBuffer.join(" "),
            log: this.currentSessionLog
        };

        if (renewId) this.sessionId = new Date().toISOString();

        const savedSessions = JSON.parse(localStorage.getItem('slr_sessions') || '[]');
        if (savedSessions.length > 0 && savedSessions[0].id == this.sessionId) {
            savedSessions.shift();
        }
        savedSessions.unshift(sessionData);
        if (savedSessions.length > 20) savedSessions.pop();

        localStorage.setItem('slr_sessions', JSON.stringify(savedSessions));
    }

    reset() {
        this.save(true);
        this.sentenceBuffer = [];
        this.currentSessionLog = [];
    }

    getHistory() {
        return JSON.parse(localStorage.getItem('slr_sessions') || '[]');
    }
}

class UIManager {
    constructor(app) {
        this.app = app;
        this.elements = {
            video: document.getElementById('webcam'),
            canvas: document.getElementById('process-canvas'),
            overlayCanvas: document.getElementById('overlay-canvas'),
            arText: document.getElementById('prediction-text-ar'),
            enText: document.getElementById('prediction-text-en'),
            confVal: document.getElementById('confidence-val'),
            confBar: document.getElementById('confidence-bar'),
            statusPill: document.getElementById('connection-status'),
            statusText: document.getElementById('status-text'),
            sentenceOutput: document.getElementById('sentence-output'),
            historySidebar: document.getElementById('history-sidebar'),
            historyList: document.getElementById('history-list'),
            langSelect: document.getElementById('lang-select'),
            settingsOverlay: document.getElementById('settings-overlay'),
            archiveModal: document.getElementById('archive-modal'),
            sessionList: document.getElementById('session-list'),
            sessionDetail: document.getElementById('session-detail'),
            chkDrawPoints: document.getElementById('chk-draw-points'),
            chkDrawLines: document.getElementById('chk-draw-lines'),
            themeBtnLight: document.getElementById('theme-btn-light'),
            themeBtnDark: document.getElementById('theme-btn-dark')
        };

        this.ctx = this.elements.canvas.getContext('2d', { alpha: false });
        this.overlayCtx = this.elements.overlayCanvas.getContext('2d');

        this.initEventListeners();
        this.syncWithConfig();
    }

    initEventListeners() {
        const { elements, app } = this;

        // Buttons
        document.getElementById('btn-speak').onclick = () => app.speakSentence();
        document.getElementById('btn-clear').onclick = () => app.clearSession();
        document.getElementById('btn-settings').onclick = () => this.toggleSettings(true);
        document.getElementById('btn-close-settings').onclick = () => this.toggleSettings(false);
        document.getElementById('history-toggle').onclick = (e) => { e.stopPropagation(); this.toggleHistory(true); };
        document.getElementById('close-history').onclick = () => this.toggleHistory(false);
        document.getElementById('clear-history').onclick = () => app.clearSession();
        document.getElementById('btn-archive').onclick = () => this.showArchive();
        document.getElementById('btn-close-archive').onclick = () => this.toggleArchive(false);
        document.getElementById('btn-back-list').onclick = () => this.showSessionList();

        // Toggles
        ['face', 'pose', 'hands'].forEach(region => {
            const btn = document.getElementById(`btn-viz-${region}`);
            btn.onclick = () => {
                app.config.setViz(region, !app.config.viz[region]);
                btn.classList.toggle('active', app.config.viz[region]);
            };
        });

        elements.chkDrawLines.onchange = (e) => app.config.setDrawMode('lines', e.target.checked);
        elements.chkDrawPoints.onchange = (e) => app.config.setDrawMode('points', e.target.checked);

        // Theme
        elements.themeBtnLight.onclick = () => this.updateTheme('light');
        elements.themeBtnDark.onclick = () => this.updateTheme('dark');

        // Lang
        elements.langSelect.onchange = (e) => {
            app.config.setLang(e.target.value);
            app.speakText(app.config.lang.startsWith('ar') ? "مرحباً" : "Hello");
        };

        // Modal/Sidebar clicks
        elements.settingsOverlay.onclick = (e) => { if (e.target === elements.settingsOverlay) this.toggleSettings(false); };
        elements.archiveModal.onclick = (e) => { if (e.target === elements.archiveModal) this.toggleArchive(false); };
        elements.historySidebar.onclick = (e) => e.stopPropagation();
        document.onclick = (e) => { if (elements.historySidebar.classList.contains('open') && !elements.historySidebar.contains(e.target) && !document.getElementById('history-toggle').contains(e.target)) this.toggleHistory(false); };
    }

    syncWithConfig() {
        const { config } = this.app;
        this.updateTheme(config.theme);
        this.elements.langSelect.value = config.lang;
        this.elements.chkDrawLines.checked = config.viz.lines;
        this.elements.chkDrawPoints.checked = config.viz.points;

        ['face', 'pose', 'hands'].forEach(region => {
            const btn = document.getElementById(`btn-viz-${region}`);
            if (btn) btn.classList.toggle('active', config.viz[region]);
        });
    }

    updateTheme(theme) {
        this.app.config.setTheme(theme);
        this.elements.themeBtnLight.classList.toggle('active', theme === 'light');
        this.elements.themeBtnDark.classList.toggle('active', theme === 'dark');
    }

    updateStatus(status) {
        this.elements.statusPill.className = `status-pill ${status}`;
        const texts = { connected: "Live", disconnected: "Offline", error: "Error" };
        this.elements.statusText.textContent = texts[status] || "Unknown";
    }

    updatePrediction(data, stabilityCounter) {
        if (data.status === "idle") {
            this.elements.arText.textContent = this.elements.enText.textContent = "...";
            this.elements.confVal.textContent = this.elements.confBar.style.width = "0%";
            return;
        }

        if (!data.detected_sign) return;

        const { sign_ar, sign_en } = data.detected_sign;
        const confidencePct = Math.round(data.confidence * 100);

        if (stabilityCounter === this.app.config.STABILITY_THRESHOLD) {
            this.elements.arText.textContent = sign_ar;
            this.elements.enText.textContent = sign_en;
            this.elements.confVal.textContent = this.elements.confBar.style.width = `${confidencePct}%`;

            this.elements.arText.style.transform = "scale(1.2)";
            setTimeout(() => this.elements.arText.style.transform = "scale(1)", 200);
        }

        const barColor = confidencePct > 80 ? 'var(--success-color)' : (confidencePct > 50 ? 'var(--accent-color)' : 'var(--error-color)');
        this.elements.confBar.style.backgroundColor = barColor;
    }

    renderSentence(buffer) {
        this.elements.sentenceOutput.textContent = buffer.join(" ");
        this.elements.sentenceOutput.scrollLeft = this.elements.sentenceOutput.scrollWidth;
    }

    addToHistoryLog(word) {
        const emptyMsg = this.elements.historyList.querySelector('.history-empty');
        if (emptyMsg) emptyMsg.remove();

        const li = document.createElement('li');
        li.className = 'history-item';
        const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        li.innerHTML = `<span class="history-word">${word}</span><span class="history-time">${timeString}</span>`;
        this.elements.historyList.prepend(li);
        if (this.elements.historyList.children.length > 50) this.elements.historyList.lastElementChild.remove();
    }

    toggleSettings(show) {
        this.elements.settingsOverlay.classList.toggle('hidden', !show);
    }

    toggleHistory(show) {
        this.elements.historySidebar.classList.toggle('open', show);
    }

    toggleArchive(show) {
        this.elements.archiveModal.classList.toggle('hidden', !show);
    }

    showArchive() {
        this.toggleArchive(true);
        this.showSessionList();
    }

    showSessionList() {
        this.elements.sessionList.style.display = 'block';
        this.elements.sessionDetail.classList.add('hidden');
        this.renderSessionList();
    }

    renderSessionList() {
        const sessions = this.app.session.getHistory();
        this.elements.sessionList.innerHTML = sessions.length ? '' : '<li style="text-align:center; color:var(--text-sub); margin-top:2rem;">No saved sessions.</li>';

        sessions.forEach(session => {
            const li = document.createElement('li');
            li.className = 'session-item';
            li.innerHTML = `<div><span class="session-date">${session.date}</span><span class="session-preview">${session.sentence || (session.log.length + " signs detected")}</span></div><div style="color:var(--text-sub)">›</div>`;
            li.onclick = () => this.showSessionDetail(session);
            this.elements.sessionList.appendChild(li);
        });
    }

    showSessionDetail(session) {
        this.elements.sessionList.style.display = 'none';
        this.elements.sessionDetail.classList.remove('hidden');
        document.getElementById('detail-date').textContent = session.date;
        document.getElementById('detail-sentence').textContent = session.sentence || "(No sentence formed)";
        const elLog = document.getElementById('detail-log');
        elLog.innerHTML = '';
        session.log.forEach(item => {
            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `<span class="history-word">${item.word}</span><span class="history-time">${item.time}</span>`;
            elLog.appendChild(li);
        });
    }

    clearHistoryUI() {
        this.elements.historyList.innerHTML = '<li class="history-empty">No signs detected yet.</li>';
    }
}

class SkeletonDrawer {
    constructor(App) {
        this.app = App;
        this.ui = App.ui;
        this.POSE_KPS = [];
        this.FACE_KPS = [];
        this.HAND_KPS = [];
        this.POSE_CONNECTIONS = [];
        this.FACE_CONNECTIONS = [];
        this.HAND_CONNECTIONS = [];
        this.KPS_MAPPING = {};
        this.loadConfig();
    }

    async loadConfig() {
        try {
            const response = await fetch('/static/simplified_kps_connections.json');
            const kps = await response.json();
            this.POSE_KPS = kps['pose_kps'];
            this.FACE_KPS = kps['face_kps'];
            this.HAND_KPS = kps['hand_kps'];
            this.POSE_CONNECTIONS = kps['pose_connections'];
            this.FACE_CONNECTIONS = kps['face_contours'];
            this.HAND_CONNECTIONS = kps['hand_connections'];
            this.KPS_MAPPING = kps['mp_idx_to_kps_idx'];
            console.log(`[ArSL] Loaded skeleton connections.`);
        } catch (err) {
            console.error("[ArSL] Failed to load skeleton config:", err);
        }
    }

    draw(landmarks, vizConfig) {
        const { overlayCtx } = this.ui;
        const { overlayCanvas } = this.app.ui.elements;

        const w = overlayCanvas.width;
        const h = overlayCanvas.height;
        overlayCtx.clearRect(0, 0, w, h);
        if (!landmarks || landmarks.length === 0 || (!vizConfig.lines && !vizConfig.points)) return;

        const { sx, sy, ratio } = this.ui.video_rect;
        const scaled = landmarks.map(p => ({ x: p[0] * w / ratio + sx / 2, y: p[1] * h + sy / 2 }));

        if (vizConfig.points) {
            const drawPoints = (mp_idx, mp_idx_map_kps, color) => {
                overlayCtx.beginPath();
                overlayCtx.fillStyle = color;
                mp_idx.forEach(idx => {
                    const p = scaled[mp_idx_map_kps[idx]];
                    if (p && (p.x !== 0 || p.y !== 0)) {
                        overlayCtx.moveTo(p.x + 2, p.y);
                        overlayCtx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
                    }
                });
                overlayCtx.fill();
            };
            if (vizConfig.pose) drawPoints(this.POSE_KPS, this.KPS_MAPPING['pose'], '#0d9165');
            if (vizConfig.face) drawPoints(this.FACE_KPS, this.KPS_MAPPING['face'], '#d1880a');
            if (vizConfig.hands) {
                drawPoints(this.HAND_KPS, this.KPS_MAPPING['rh'], '#ca3a3a');
                drawPoints(this.HAND_KPS, this.KPS_MAPPING['lh'], '#ca3a3a');
            }
        }

        if (vizConfig.lines) {
            const drawLines = (connections, mp_idx_map_kps, color) => {
                if (!connections.length) return;
                overlayCtx.beginPath();
                overlayCtx.lineWidth = 2;
                overlayCtx.lineJoin = 'round';
                overlayCtx.strokeStyle = color;
                connections.forEach(([s, e]) => {
                    const p1 = scaled[mp_idx_map_kps[s]], p2 = scaled[mp_idx_map_kps[e]];
                    if (p1 && p2 && (p1.x || p1.y) && (p2.x || p2.y)) {
                        overlayCtx.moveTo(p1.x, p1.y);
                        overlayCtx.lineTo(p2.x, p2.y);
                    }
                });
                overlayCtx.stroke();
            };
            if (vizConfig.pose) drawLines(this.POSE_CONNECTIONS, this.KPS_MAPPING['pose'], '#10b981');
            if (vizConfig.face) Object.values(this.FACE_CONNECTIONS).forEach(sub => drawLines(sub, this.KPS_MAPPING['face'], '#f59e0b'));
            if (vizConfig.hands) {
                drawLines(this.HAND_CONNECTIONS, this.KPS_MAPPING['rh'], '#ef4444');
                drawLines(this.HAND_CONNECTIONS, this.KPS_MAPPING['lh'], '#ef4444');
            }
        }
    }
}

class SignSocket {
    constructor(app) {
        this.app = app;
        this.isOpen = false;
        this.connect();
    }

    connect() {
        this.socket = new WebSocket(this.app.config.wsUrl);
        this.socket.onopen = () => { this.isOpen = true; this.app.ui.updateStatus('connected'); };
        this.socket.onclose = () => { this.isOpen = false; this.app.ui.updateStatus('disconnected'); };
        this.socket.onerror = (e) => { console.error("WS Error", e); this.app.ui.updateStatus('error'); };
        this.socket.onmessage = (e) => this.app.handleMessage(JSON.parse(e.data));
    }

    send(blob, needLandmarks) {
        if (!this.isOpen) return;
        const header = new Uint8Array([needLandmarks ? 1 : 0]);
        this.socket.send(new Blob([header, blob], { type: 'image/jpeg' }));
    }
}

class WebcamHandler {
    constructor(app) {
        this.app = app;
        this.lastFrameTime = 0;
        this.isSending = false;
    }

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false
            });
            const { video, canvas, overlayCanvas } = this.app.ui.elements;
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                const sw = video.videoWidth;
                const sh = video.videoHeight;
                const ratio = sw / sh;
                canvas.width = this.app.config.processWidth;
                canvas.height = this.app.config.processWidth / ratio;
                overlayCanvas.width = canvas.width;
                overlayCanvas.height = canvas.height;
                const minDim = Math.min(sw, sh);
                const sx = (sw - minDim) / 2;
                const sy = (sh - minDim) / 2;
                this.app.ui.video_rect = {
                    sx, sy, ratio, minDim
                };

                requestAnimationFrame((t) => this.loop(t));
            };
        } catch (err) {
            console.error("Camera error:", err);
            alert("Please allow camera access.");
        }
    }

    loop(timestamp) {
        requestAnimationFrame((t) => this.loop(t));
        const interval = 1000 / this.app.config.fps;
        const elapsed = timestamp - this.lastFrameTime;
        if (elapsed > interval) {
            this.lastFrameTime = timestamp - (elapsed % interval);
            if (this.app.socket.isOpen && !this.isSending && this.app.ui.elements.video.readyState === 4) {
                this.processFrame();
            }
        }
    }

    processFrame() {
        const { video, canvas } = this.app.ui.elements;
        const { sx, sy, minDim } = this.app.ui.video_rect;

        this.app.ui.ctx.drawImage(
            video,
            sx, sy, minDim, minDim,
            0, 0, canvas.width, canvas.height
        );

        const { viz } = this.app.config;
        const needLandmarks = (viz.lines || viz.points) && (viz.face || viz.pose || viz.hands);
        this.isSending = true;

        canvas.toBlob((blob) => {
            if (blob) this.app.socket.send(blob, needLandmarks);
            this.isSending = false;
        }, 'image/jpeg', this.app.config.jpgQuality);
    }
}

class ArSLApp {
    constructor() {
        this.config = new AppConfig();
        this.session = new SessionManager(this.config);
        this.ui = new UIManager(this);
        this.drawer = new SkeletonDrawer(this);
        this.socket = new SignSocket(this);
        this.webcam = new WebcamHandler(this);

        this.state = {
            lastFrameWord: "",
            stabilityCounter: 0
        };

        this.init();
    }

    init() {
        this.webcam.start();
    }

    handleMessage(data) {
        if (data.landmarks)
            this.drawer.draw(data.landmarks, this.config.viz);

        if (data.status === "idle") {
            this.state.stabilityCounter = 0;
            this.state.lastFrameWord = "";
            this.ui.updatePrediction(data, 0);
            return;
        }

        if (!data.detected_sign) return;

        const { sign_ar } = data.detected_sign;
        if (sign_ar === this.state.lastFrameWord) {
            this.state.stabilityCounter++;
        } else {
            this.state.stabilityCounter = 0;
            this.state.lastFrameWord = sign_ar;
        }

        if (this.state.stabilityCounter === this.config.STABILITY_THRESHOLD) {
            this.session.addWord(sign_ar, (word, buffer) => {
                this.ui.addToHistoryLog(word);
                this.ui.renderSentence(buffer);
                this.speakText(word);
            });
        }
        this.ui.updatePrediction(data, this.state.stabilityCounter);
    }

    speakText(text) {
        if (!text) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = this.config.lang;
        utterance.rate = 0.9;
        window.speechSynthesis.speak(utterance);
    }

    speakSentence() {
        this.speakText(this.session.sentenceBuffer.join(" "));
    }

    clearSession() {
        this.session.reset();
        this.ui.renderSentence([]);
        this.ui.clearHistoryUI();
        this.handleMessage({ status: "idle" });
    }
}

// Global initialization
window.addEventListener('DOMContentLoaded', () => {
    window.app = new ArSLApp();
});
