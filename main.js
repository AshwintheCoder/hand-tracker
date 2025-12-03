import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");
let handLandmarker = undefined;
let runningMode = "VIDEO";
let enableWebcamButton;
let webcamRunning = false;
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const statusElement = document.getElementById("status");
const overlayControls = document.getElementById("overlayControls");
const confidenceSlider = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const clearButton = document.getElementById("clearButton");

// Drawing state
let canvasPaths = [];
let currentPath = [];
let isDrawing = false;
const PINCH_THRESHOLD = 0.05;

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
    statusElement.innerText = "Webcam not supported by this browser.";
}

// Create the HandLandmarker class.
const createHandLandmarker = async () => {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: runningMode,
            numHands: 2
        });
        statusElement.innerText = "System Ready. Enable camera to start.";
        enableWebcamButton.disabled = false;
        enableWebcamButton.classList.add("ready");
    } catch (error) {
        console.error(error);
        statusElement.innerText = "Error loading model: " + error.message;
    }
};
createHandLandmarker();

// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE CAMERA";
        overlayControls.classList.remove("hidden");
        video.pause();
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
        overlayControls.classList.add("hidden");

        const constraints = {
            video: {
                width: 1280,
                height: 720
            }
        };

        // Activate the webcam stream.
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
        });
    }
}

let lastVideoTime = -1;
let results = undefined;

async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            // Check for pinch gesture (Thumb Tip #4 and Index Tip #8)
            const thumbTip = landmarks[4];
            const indexTip = landmarks[8];
            const distance = Math.sqrt(
                Math.pow(thumbTip.x - indexTip.x, 2) +
                Math.pow(thumbTip.y - indexTip.y, 2)
            );

            // Midpoint for drawing
            const midpoint = {
                x: (thumbTip.x + indexTip.x) / 2,
                y: (thumbTip.y + indexTip.y) / 2
            };

            if (distance < PINCH_THRESHOLD) {
                if (!isDrawing) {
                    isDrawing = true;
                    currentPath = [];
                }
                currentPath.push(midpoint);
            } else {
                if (isDrawing) {
                    isDrawing = false;
                    if (currentPath.length > 1) {
                        canvasPaths.push([...currentPath]);
                    }
                    currentPath = [];
                }
            }

            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#38bdf8",
                lineWidth: 3
            });
            drawLandmarks(canvasCtx, landmarks, {
                color: "#22d3ee",
                lineWidth: 1,
                radius: 3
            });
        }
    }

    // Draw all paths
    drawPaths(canvasCtx, canvasPaths, currentPath);

    canvasCtx.restore();

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Drawing utilities (simplified version of MediaPipe drawing utils to avoid extra heavy dependency)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [5, 9], [9, 10], [10, 11], [11, 12], // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [0, 17] // Palm
];

function drawConnectors(ctx, landmarks, connections, style) {
    ctx.strokeStyle = style.color;
    ctx.lineWidth = style.lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Add glow effect
    ctx.shadowColor = style.color;
    ctx.shadowBlur = 10;

    for (const connection of connections) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];

        ctx.beginPath();
        ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
        ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
        ctx.stroke();
    }

    // Reset shadow
    ctx.shadowBlur = 0;
}

function drawLandmarks(ctx, landmarks, style) {
    ctx.fillStyle = style.color;

    // Add glow effect
    ctx.shadowColor = style.color;
    ctx.shadowBlur = 5;

    for (const landmark of landmarks) {
        const x = landmark.x * ctx.canvas.width;
        const y = landmark.y * ctx.canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, style.radius, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Reset shadow
    ctx.shadowBlur = 0;
}

function drawPaths(ctx, paths, current) {
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#f472b6"; // Pink/Magenta for drawing
    ctx.shadowColor = "#f472b6";
    ctx.shadowBlur = 10;

    const allPaths = [...paths];
    if (current.length > 1) {
        allPaths.push(current);
    }

    for (const path of allPaths) {
        if (path.length < 2) continue;

        ctx.beginPath();
        ctx.moveTo(path[0].x * ctx.canvas.width, path[0].y * ctx.canvas.height);

        for (let i = 1; i < path.length; i++) {
            ctx.lineTo(path[i].x * ctx.canvas.width, path[i].y * ctx.canvas.height);
        }
        ctx.stroke();
    }

    ctx.shadowBlur = 0;
}

// Clear drawing
clearButton.addEventListener('click', () => {
    canvasPaths = [];
    currentPath = [];
});

// Update confidence threshold
confidenceSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    confidenceValue.innerText = `${Math.round(value * 100)}%`;
    if (handLandmarker) {
        handLandmarker.setOptions({
            minHandDetectionConfidence: Number(value),
            minHandPresenceConfidence: Number(value),
            minTrackingConfidence: Number(value)
        });
    }
});
