import {
    GestureRecognizer,
    FilesetResolver,
    DrawingUtils,
    GestureRecognizerResult,
    BoundingBox,
    Category
} from "@mediapipe/tasks-vision"

import Game from "./game";
import { AxesHelper, OrthographicCamera, Scene, WebGLRenderer } from "three";

import Stats from "three/examples/jsm/libs/stats.module";
import {GUI} from "three/examples/jsm/libs/lil-gui.module.min";

const gameCanvasElement = document.getElementById("game_canvas") as HTMLCanvasElement;
const renderer = new WebGLRenderer({ 
    alpha: true, 
    canvas: gameCanvasElement, 
    antialias: true 
});
const scene = new Scene();
const camera = new OrthographicCamera(0, 1, 1, 0, 0, 1000);
camera.position.z = 500;

const helper = new AxesHelper(100);
helper.position.set(1, 1, 0);
scene.add(helper);

const stats = new Stats();
document.body.appendChild(stats.dom);

const gui = new GUI();
gui.add({ mirrored: true }, "mirrored")
    .name("Mirror Camera")
    .onChange(cameraFlip);

let gestureRecognizer: GestureRecognizer;

const createGestureRecognizer = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
};
await createGestureRecognizer();

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById("output_canvas") as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d") as CanvasRenderingContext2D;

const gestureOutput = document.getElementById("gesture_output") as HTMLElement;

function cameraFlip(bool: boolean) {
    video.classList.toggle("mirrored", bool);
    canvasElement.classList.toggle("mirrored", bool);
    gameCanvasElement.classList.toggle("mirrored", bool);
}

navigator.mediaDevices.getUserMedia({ video: true})
    .then(function (stream) {
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            renderer.setSize(video.videoWidth, video.videoHeight);
            camera.right = video.videoWidth
            // Camera needs to be flipped vertically
            // Video elements is rendered in 2D, so the origin is at the top left corner
            // In Three.s origin is bottom left corner
            camera.top = 0;
            camera.bottom = video.videoHeight;
            camera.updateProjectionMatrix();

            const widthString = (video.videoWidth).toString();
            const heightString = (video.videoHeight).toString();
        
            gameCanvasElement.setAttribute("width", widthString);
            gameCanvasElement.setAttribute("height", heightString);

            canvasElement.setAttribute("width", widthString);
            canvasElement.setAttribute("height", heightString);

            predictWebcam();
        });
    });

let lastVideoTime = -1;
let results: GestureRecognizerResult;

const game = new Game(scene, renderer);
game.addEventListener("scoreChanged", (event) => {
    console.log("Score: " + event.detail);
});
await game.loadModel("/apple_2.glb");

async function predictWebcam() {
    let nowInMs = Date.now();

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        results = gestureRecognizer.recognizeForVideo(video, nowInMs);
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    const drawingUtils = new DrawingUtils(canvasCtx);

    let boundingBoxes: BoundingBox[] = [];

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            // drawingUtils.drawConnectors(
            //     landmarks,
            //     GestureRecognizer.HAND_CONNECTIONS,
            //     {
            //         color: "#00FF00",
            //         lineWidth: 5
            //     }
            // );
            
            // drawingUtils.drawLandmarks(landmarks, {
            //     color: "#FF0000",
            //     lineWidth: 2
            // });

            let minX = 1;
            let maxX = 0;
            let minY = 1;
            let maxY = 0;

            for(const landmark of landmarks) {
                const { x, y } = landmark;

                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            const width = maxX - minX;
            const height = maxY - minY;

            const boundingBox: BoundingBox = {
                originX: minX * canvasElement.width,
                originY: minY * canvasElement.height,
                height: height * canvasElement.height,
                width: width * canvasElement.width,
                angle: 0
            }

            boundingBoxes.push(boundingBox);

            drawingUtils.drawBoundingBox(boundingBox, {
                    color: "#0000FF",
                    fillColor: "transparent",
                    lineWidth: 4,
                }
            );
        }
    }
    canvasCtx.restore();

    stats.update();
    renderer.render(scene, camera);

    game.target.mesh.rotateY(0.01);

    let html = "";
    for(let i=0; i<results.gestures.length; i++) {
        const result = results.gestures[i];
        
        html += generateHTML(result, i);

        if (result[0].categoryName === "Closed_Fist" && game.lastHandStates[i] !== "Closed_Fist") {
            const box = boundingBoxes[i];            
            game.target.hit(box);
        }

        game.lastHandStates[i] = result[0].categoryName;
    }
    gestureOutput.innerHTML = html;    

    window.requestAnimationFrame(predictWebcam);
}

function generateHTML(result: Category[], i: number) {
    const categoryName = result[0].categoryName;
    const categoryScore = parseFloat((result[0].score * 100).toString()).toFixed(2);       
    const handedness = results.handedness[i][0].displayName;

    return `<div>
        <p>GestureRecognizer: ${categoryName}</p>
        <p>Confidence: ${categoryScore} %</p>
        <p>Handedness: ${handedness}</p>
    </div>`;
}