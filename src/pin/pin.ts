import { DrawingUtils, FaceLandmarker, FaceLandmarkerResult, FilesetResolver } from "@mediapipe/tasks-vision";
import { AmbientLight, Matrix4, Object3D, OrthographicCamera, PerspectiveCamera, Scene, Vector3, VideoTexture, WebGLRenderer } from "three";

import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { AxesHelper } from "three";

import { Font, FontLoader } from "three/examples/jsm/loaders/FontLoader";

import { GUI } from "three/examples/jsm/libs/lil-gui.module.min";
import Stats from "three/examples/jsm/libs/stats.module";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

const CONFIGS = {
  "DRAW_CONNECTORS": false,
  "AXIS_HELPER": false,
  "ENABLE_CAMERA": false,
}

const MODIFICATIONS = {
}

let faceLandmarker: FaceLandmarker;
let runningMode: "IMAGE" | "VIDEO" = "VIDEO";

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
  faceLandmarker = await FaceLandmarker.createFromOptions(
    filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    runningMode,
    numFaces: 1
  });
}
createFaceLandmarker();

const video = document.getElementById("webcam") as HTMLVideoElement;
const meshOutput = document.getElementById("mesh-output") as HTMLCanvasElement;
const imageOutput = document.getElementById("image-output") as HTMLCanvasElement;

const meshOutputCtx = meshOutput.getContext("2d");
if(!meshOutputCtx) throw new Error("Mesh Canvas context is null");

const renderer = new WebGLRenderer({
  canvas: imageOutput,
  antialias: true,
  alpha: true,
});

const scene = new Scene();
const camera = new PerspectiveCamera(75, 480 / 360, 0.1, 1000);
camera.position.z = 3;
scene.add(camera);

const light = new AmbientLight(0xffffff, 1);
scene.add(light);

// Get video texture
const texture = new VideoTexture(video);
texture.flipY = false;

const axesHelper = new AxesHelper(1);
scene.add(axesHelper);

const fontLoader = new FontLoader();
let font: Font;
fontLoader.load("/font.json", (f) => { font = f });

const controls = new OrbitControls(camera, imageOutput);
controls.enableDamping = true;
controls.dampingFactor = 0.25;
controls.enabled = CONFIGS.ENABLE_CAMERA;

const stats = new Stats();
document.body.appendChild(stats.dom);

// GUI
const gui = new GUI();
const folder = gui.addFolder("Configs");
folder.add(CONFIGS, "DRAW_CONNECTORS").name("Draw Connectors");
folder.add(CONFIGS, "AXIS_HELPER").name("Axis Helper");
folder.add(CONFIGS, "ENABLE_CAMERA").name("Enable Camera");
folder.open();

const modFolder = gui.addFolder("Modifications");

function drawLoop() {  
  requestAnimationFrame(drawLoop);
  stats.update();

  axesHelper.visible = CONFIGS.AXIS_HELPER;
  controls.enabled = CONFIGS.ENABLE_CAMERA;

  renderer.render(scene, camera);

  gui.controllers.forEach((controller) => {
    controller.updateDisplay();
  });
}
drawLoop();


// Enable the live webcam view and start detection.
navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      predictWebcam();
      renderer.setSize(video.videoWidth, video.videoHeight);

      meshOutput.width = video.videoWidth;
      meshOutput.height = video.videoHeight;
    });
  });

let lastVideoTime = -1;
let results: FaceLandmarkerResult;

const meshDrawingUtils = new DrawingUtils(meshOutputCtx);
// const imageDrawingUtils = new DrawingUtils(imageCanvasCtx);

const modelLoader = new GLTFLoader();
let model: Object3D
modelLoader.load("/monke.glb", (gltf) => {
  model = gltf.scene;
  scene.add(model);
});

// Based on
// https://i.sstatic.net/5Mohl.jpg
const ANCHOR_POINT_ID = 9;
async function predictWebcam() {
  let startTimeMs = performance.now();

  if(!meshOutputCtx) throw new Error("Mesh Canvas context is null");
  meshOutputCtx.clearRect(0, 0, meshOutput.width, meshOutput.height);

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = faceLandmarker.detectForVideo(video, startTimeMs);
  }
  
  if (results.faceLandmarks) {
    if(CONFIGS.DRAW_CONNECTORS) {
      for (const landmarks of results.faceLandmarks) {
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION,   { color: "#C0C0C070", lineWidth: 1 } );
  
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,     { color: "#E0E0E0" } );
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS,          { color: "#E0E0E0" } );
  
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,     { color: "#FF3030" } );
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" } );
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,    { color: "#FF3030" } );
  
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,      { color: "#30FF30" } );
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,  { color: "#30FF30" } );
        meshDrawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,     { color: "#30FF30" } );
      }
    }

    if(results.facialTransformationMatrixes.length) {
      const matrix = new Matrix4();
      matrix.fromArray(results.facialTransformationMatrixes[0].data);

      const position = results.faceLandmarks[0][ANCHOR_POINT_ID];
      
      model.position.set(position.x, position.y, position.z);
      model.setRotationFromMatrix(matrix);
    }
  }
  
  // Call this function again to keep predicting when the browser is ready.
  window.requestAnimationFrame(predictWebcam);
}

function mapMediapipeCoordsToThree(coords: Vector3) {
  return new Vector3(coords.x, coords.y, coords.z);
}
