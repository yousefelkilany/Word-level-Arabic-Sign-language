const video = document.getElementById('webcam');
const cam_canvas = document.getElementById('cam-canvas');
const cam_context = cam_canvas.getContext('2d');
const predictionText = document.getElementById('prediction-text');

const recieved_canvas = document.getElementById('recieved-canvas');
const recieved_context = recieved_canvas.getContext('2d');

const offscreen_canvas = document.createElement('canvas');
const offscreen_context = offscreen_canvas.getContext('2d');

const socket = new WebSocket('ws://localhost:8000/live-signs');
const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 480;

let latestFrame = null;
let isSocketOpen = false;

const FPS = 30;
const MS_FPS_INT = 1000 / FPS;

// --- WebSocket Connection ---  
socket.onopen = function (event) {
    console.log("WebSocket connection established.");
    isSocketOpen = true;
};

socket.onmessage = function (event) {
    latestFrame = event.data;

    // predictionText.textContent = event.data.predicted_word;
};

socket.onclose = function (event) {
    console.log("WebSocket connection closed.");
    isSocketOpen = false;
};

socket.onerror = function (error) {
    console.error("WebSocket Error:", error);
};


async function renderLoop() {
    try {
        if (!latestFrame)
            return;
        createImageBitmap(latestFrame).then(
            img_bmp => {
                recieved_context.drawImage(img_bmp, 0, 0);
                // offscreen_context.drawImage(img_bmp, 0, 0);
                // recieved_context.drawImage(offscreen_canvas, 0, 0);
                img_bmp.close();
            }
        )
    } catch (e) {
        console.log("Error:", e);
    } finally {
        latestFrame = null;
        requestAnimationFrame(renderLoop);
    }
}

// --- Webcam and Frame Sending Logic ---  
async function setupWebcam() {
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

async function main() {
    await setupWebcam();

    requestAnimationFrame(renderLoop);

    recieved_canvas.width = offscreen_canvas.width = CANVAS_WIDTH;
    recieved_canvas.height = offscreen_canvas.height = CANVAS_HEIGHT;

    // Adjust interval for performance vs. real-time feel
    setInterval(() => {
        if (isSocketOpen) {
            cam_context.drawImage(video, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            cam_canvas.toBlob((blob) => socket.send(blob), 'image/jpeg', 0.8);
        }
    }, MS_FPS_INT);
}

main();  
