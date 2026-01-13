const video = document.getElementById('webcam');
const cam_canvas = document.createElement('canvas');
const cam_context = cam_canvas.getContext('2d', { alpha: false });

const canvas_container = document.getElementById('canvas-container');

const predictionTextAr = document.getElementById('prediction-text-ar');
const predictionTextEn = document.getElementById('prediction-text-en');


const confidenceText = document.getElementById('confidence-text');

const socket = new WebSocket(`ws://${location.host}/live-signs`); // localhost:8000
const CANVAS_WIDTH = 320;
const CANVAS_HEIGHT = 240;

let isSocketOpen = false;
let isSending = false;
let lastSentTimstamp = 0;

const FPS = 10;
const MS_FPS_INT = parseInt(1000 / FPS);
const JPG_QUALITY = 0.7;

// --- WebSocket Connection ---  
socket.onopen = function (event) {
    console.log("WebSocket connection established.");
    isSocketOpen = true;
};

socket.onmessage = function (event) {
    const data = JSON.parse(event.data);

    if (data.status === "idle") {
        predictionTextAr.textContent = "---";
        predictionTextEn.textContent = "";
        confidenceText.textContent = "Idle...";
    } else if (data.detected_word) {
        predictionTextAr.textContent = `الكلمة: ${data.detected_word.sign_ar}`;
        predictionTextEn.textContent = `word: ${data.detected_word.sign_en}`;
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
    return new Promise((resolve, reject) => {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            reject(new Error("getUserMedia not supported"));
            return;
        }

        navigator.mediaDevices.getUserMedia({
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 },
            video: true,
            audio: false,
        }).then(stream => {
            video.srcObject = stream;
            video.play();

            video.onloadedmetadata = () => {
                const ratio = video.videoWidth / video.videoHeight;
                canvas_container.width = cam_canvas.width = CANVAS_HEIGHT * ratio;
                canvas_container.height = cam_canvas.height = CANVAS_HEIGHT;

                requestAnimationFrame(cameraLoop);
                resolve();
            };
        }).catch(err => {
            reject(err);
        });
    });
}

async function cameraLoop(timestamp) {
    requestAnimationFrame(cameraLoop);

    if (!lastSentTimstamp)
        lastSentTimstamp = timestamp;
    const elapsed = timestamp - lastSentTimstamp;

    if (elapsed > MS_FPS_INT) {
        lastSentTimstamp = timestamp - (elapsed % MS_FPS_INT);

        if (isSocketOpen && video.readyState === 4 && !isSending) {
            cam_context.drawImage(video, 0, 0, cam_canvas.width, cam_canvas.height);

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
}

main();  
