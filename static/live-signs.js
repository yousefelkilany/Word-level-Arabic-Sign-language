const video = document.getElementById('webcam');
const cam_canvas = document.getElementById('cam-canvas');
const cam_context = cam_canvas.getContext('2d');

const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');

const socket = new WebSocket(`ws://${location.host}/live-signs`); // localhost:8000
const CANVAS_WIDTH = 320;
const CANVAS_HEIGHT = 240;

let isSocketOpen = false;
let isSending = false;
let lastSentTimstamp = 0;

const FPS = 30;
const MS_FPS_INT = parseInt(1000 / FPS);
const JPG_QUALITY = 0.7;

// --- WebSocket Connection ---  
socket.onopen = function (event) {
    console.log("WebSocket connection established.");
    isSocketOpen = true;
};

socket.onmessage = function (event) {
    const data = JSON.parse(event.data);

    if (data.status == "idle") {
        predictionText.textContent = "---";
        confidenceText.textContent = "Idle...";
    } else if (data.detected_word) {
        let ar_sign = data.detected_word.ar_sign;
        let en_sign = data.detected_word.en_sign;
        predictionText.textContent = `word: ${en_sign}<br>الكلمة: ${ar_sign}`;
        // predictionText.textContent = `word: ${en_sign}\nالكلمة: ${ar_sign}`;
        confidenceText.textContent = `confidence: ${(data.confidence * 100).toFixed(1)}%`;
    }
};

socket.onclose = function (event) {
    console.log("WebSocket connection closed.");
    isSocketOpen = false;
};

socket.onerror = function (error) {
    console.error("WebSocket Error:", error);
};

async function setupWebcam() {
    video.setAttribute('width', CANVAS_WIDTH);
    video.setAttribute('height', CANVAS_HEIGHT);

    cam_canvas.setAttribute('width', CANVAS_WIDTH);
    cam_canvas.setAttribute('height', CANVAS_HEIGHT);

    return new Promise((resolve, reject) => {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            reject(new Error("getUserMedia not supported"));
            return;
        }

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.addEventListener('loadeddata', () => resolve(), false);
            }).catch(err => {
                reject(err);
            });
    });
}

async function cameraLoop(timestamp) {
    if (!lastSentTimstamp)
        lastSentTimstamp = timestamp;
    const elapsed = timestamp - lastSentTimstamp;

    if (elapsed > MS_FPS_INT) {
        lastSentTimstamp = timestamp - (elapsed % MS_FPS_INT);

        if (isSocketOpen && video.readyState === 4 && !isSending) {
            cam_context.drawImage(video, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

            isSending = true;
            cam_canvas.toBlob((blob) => {
                if (blob && isSocketOpen) socket.send(blob);
                isSending = false;
            }, 'image/jpeg', JPG_QUALITY);
        }
    }
}

async function main() {
    await setupWebcam();
    requestAnimationFrame(cameraLoop);
}

main();  
