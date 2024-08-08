import { Category, DrawingUtils, FaceLandmarker, FaceLandmarkerResult, FilesetResolver, NormalizedLandmark } from "@mediapipe/tasks-vision";
import { BufferAttribute, BufferGeometry, DoubleSide, Float32BufferAttribute, LineBasicMaterial, Mesh, MeshBasicMaterial, OrthographicCamera, RepeatWrapping, Scene, TextureLoader, Vector3, VideoTexture, WebGLRenderer, WebGLRenderTarget } from "three";

import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { AxesHelper } from "three";

import { TextGeometry } from "three/examples/jsm/geometries/TextGeometry";
import { Font, FontLoader } from "three/examples/jsm/loaders/FontLoader";

import { GUI } from "three/examples/jsm/libs/lil-gui.module.min";
import Stats from "three/examples/jsm/libs/stats.module";

const CONFIGS = {
  "DRAW_NUMBERS": false,
  "DRAW_CONNECTORS": false,
  "SEPARATE_MOUTH_RENDER": false,
  "AXIS_HELPER": false,
  "ENABLE_CAMERA": false,
  "RESET_CAMERA": () => {
    console.log("Resetting camera");
    
    controls.enabled = false;
    CONFIGS.ENABLE_CAMERA = false;

    controls.reset();
    const vec = new Vector3(0, 0, 0);
    controls.target = vec;

    camera.position.z = -1;
    camera.position.y = 1;
    camera.position.x = 1;
    camera.rotation.set(0, Math.PI, Math.PI);
  }
}

const MODIFICATIONS = {
  scale: 1,
  flipX: false,
  flipY: false,
  curvature: 0,
}

const videoBlendShapes = document.getElementById("video-blend-shapes");

let faceLandmarker: FaceLandmarker;
let runningMode: "IMAGE" | "VIDEO" = "VIDEO";
const videoWidth = 480;

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
const mouthOverlay = document.getElementById("mouth-overlay") as HTMLCanvasElement;
const imageOutput = document.getElementById("image-output") as HTMLCanvasElement;

const meshCanvasCtx = meshOutput.getContext("2d");
if(!meshCanvasCtx) throw new Error("Mesh Canvas context is null");
const mouthOverlayCtx = mouthOverlay.getContext("2d");
if(!mouthOverlayCtx) throw new Error("Mouth Overlay context is null");

const renderer = new WebGLRenderer({
  canvas: imageOutput,
  antialias: true,
  alpha: true,
});
renderer.setSize(480, 360);

const scene = new Scene();
const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
camera.position.z = -1;
camera.position.y = 1;
camera.position.x = 1;
camera.rotateZ(Math.PI);
camera.rotateY(Math.PI);
scene.add(camera);

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
folder.add(CONFIGS, "DRAW_NUMBERS").name("Draw Numbers");
folder.add(CONFIGS, "DRAW_CONNECTORS").name("Draw Connectors");
folder.add(CONFIGS, "SEPARATE_MOUTH_RENDER").name("Separate Mouth Render");
folder.add(CONFIGS, "AXIS_HELPER").name("Axis Helper");
folder.add(CONFIGS, "ENABLE_CAMERA").name("Enable Camera");
folder.add(CONFIGS, "RESET_CAMERA").name("Reset Camera");
folder.open();

const modFolder = gui.addFolder("Modifications");
modFolder.add(MODIFICATIONS, "scale").min(0.1).max(10).step(0.1).name("Scale");
modFolder.add(MODIFICATIONS, "flipX").name("Flip X");
modFolder.add(MODIFICATIONS, "flipY").name("Flip Y");
modFolder.add(MODIFICATIONS, "curvature").min(-1).max(1).step(0.1).name("Curvature");

function drawLoop() {  
  requestAnimationFrame(drawLoop);
  stats.update();

  axesHelper.visible = CONFIGS.AXIS_HELPER;
  controls.enabled = CONFIGS.ENABLE_CAMERA;

  renderer.render(scene, camera);

  if(CONFIGS.SEPARATE_MOUTH_RENDER && mouthOverlayCtx) {
    mouthOverlayCtx.clearRect(0, 0, mouthOverlay.width, mouthOverlay.height);
    mouthOverlayCtx.drawImage(renderer.domElement, 0, 0);
  }

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
    video.addEventListener("loadeddata", predictWebcam);
  });

let lastVideoTime = -1;
let results: FaceLandmarkerResult;

const meshDrawingUtils = new DrawingUtils(meshCanvasCtx);
// const imageDrawingUtils = new DrawingUtils(imageCanvasCtx);

let mouthMesh: Mesh;

const MOUTH_POINTS = [
  { start: 0, end: 267 },
  { start: 267, end: 269 },
  { start: 269, end: 270 },
  { start: 270, end: 409 },
  { start: 409, end: 291 },
  { start: 291, end: 375 },
  { start: 375, end: 321 },
  { start: 321, end: 405 },
  { start: 405, end: 314 },
  { start: 314, end: 17 },
  { start: 17, end: 84 },
  { start: 84, end: 181 },
  { start: 181, end: 91 },
  { start: 91, end: 146 },
  { start: 146, end: 61 },
  { start: 61, end: 185 },
  { start: 185, end: 40 },
  { start: 40, end: 39 },
  { start: 39, end: 37 },
  { start: 37, end: 0 },
];

const MOUTH_INDICES = new Set<number>();
MOUTH_POINTS.forEach(({ start, end }) => {
  MOUTH_INDICES.add(start);
  MOUTH_INDICES.add(end);
});

const MOUTH_INDICES_SORTED = [ 15, 16, 14, 17, 13, 18, 12, 19, 11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5 ];

const lineMaterial = new LineBasicMaterial({ color: 0xffffff });

async function predictWebcam() {
  const ratio = video.videoHeight / video.videoWidth;
  video.style.width = videoWidth + "px";
  video.style.height = videoWidth * ratio + "px";

  meshOutput.style.width = videoWidth + "px";
  meshOutput.style.height = videoWidth * ratio + "px";
  meshOutput.width = video.videoWidth;
  meshOutput.height = video.videoHeight;

  mouthOverlay.style.width = videoWidth + "px";
  mouthOverlay.style.height = videoWidth * ratio + "px";
  mouthOverlay.width = video.videoWidth;
  mouthOverlay.height = video.videoHeight;

  // Now let's start detecting the stream.

  let startTimeMs = performance.now();

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
  
    const mouthLandmarks = [];
    if(results.faceLandmarks[0]) {
      for(const index of MOUTH_INDICES) {
        mouthLandmarks.push(results.faceLandmarks[0][index]);
      }
    }
    
    const mesh = await landmarksToMesh(mouthLandmarks);

    if(mesh) {
      scene.remove(mouthMesh);

      mesh.scale.set(MODIFICATIONS.scale, MODIFICATIONS.scale, MODIFICATIONS.scale);
      if(MODIFICATIONS.flipX) mesh.scale.x *= -1;
      if(MODIFICATIONS.flipY) mesh.scale.y *= -1;

      if(MODIFICATIONS.curvature && "array" in mesh.geometry.attributes.position) {
        const vertices = mesh.geometry.attributes.position.array as number[];

        for(let i = 0; i < vertices.length; i+=3) {
          const [x, y, z] = [vertices[i], vertices[i+1], vertices[i+2]];

          const lineGeometry = new BufferGeometry().setFromPoints([
            new Vector3(x, y, z),
            mesh.position,
          ]);
          const line = new Mesh(lineGeometry, lineMaterial);
          
          scene.add(line);

          const distX = Math.abs(x - mesh.position.x);

          vertices[i+1] = y + Math.sin(distX * MODIFICATIONS.curvature) * 0.1;

        }
      }

      scene.add(mesh);
      mouthMesh = mesh;
    };    
  }
  
  if(!videoBlendShapes) throw new Error("videoBlendShapes is null");;
  drawBlendShapes(videoBlendShapes, results.faceBlendshapes);
  
  // Call this function again to keep predicting when the browser is ready.
  window.requestAnimationFrame(predictWebcam);
}


function drawBlendShapes(el: HTMLElement, blendShapes: any[]) {
  if (!blendShapes.length) {
    return;
  }
  
  let htmlMaker = "";
  blendShapes[0].categories.map((shape: Category) => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${
          shape.displayName || shape.categoryName
        }</span>
        <span class="blend-shapes-value" style="width: calc(${
          +shape.score * 100
        }% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
  });

  el.innerHTML = htmlMaker;
}

function drawClippingRegion(landmarks: NormalizedLandmark[], connections: typeof FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, image: HTMLCanvasElement) {
  const ctx = image.getContext("2d");
  if(!ctx) throw new Error("Canvas context is null");

  ctx.save(); 

  const path = connections.map((connection) => {
    return [landmarks[connection.start], landmarks[connection.end]];
  });

  const averagePoint = path.reduce((acc, [from, to]) => {
    return { x: acc.x + (from.x + to.x) / 2, y: acc.y + (from.y + to.y) / 2 };
  }, { x: 0, y: 0 });

  ctx.beginPath();
  path.forEach(([from, to], i) => {
    if (i === 0) {
      ctx.moveTo(from.x * image.width, from.y * image.height);
    }
    ctx.lineTo(to.x * image.width, to.y * image.height);
  });
  ctx.closePath();

  ctx.clip();
  ctx.drawImage(video, 0, 0, image.width, image.height);

  ctx.translate(averagePoint.x * image.width, averagePoint.y * image.height);
  ctx.rotate(Math.PI);
  ctx.drawImage(video, 0, 0, image.width, image.height);
  
  ctx.restore();
}

async function landmarksToMesh(landmarks: NormalizedLandmark[]): Promise<Mesh | undefined> {
  if(!landmarks?.length) return;

  const mesh = new Mesh();

  let landmarksCopy = [...landmarks];

  let vertices = [];
  let uvs = [];

  const sum = { x: 0, y: 0, z: 0 };

  // Setup vertices
  for(let i = 0; i < MOUTH_INDICES_SORTED.length-2; i++) {
    let [i1, i2, i3] = [MOUTH_INDICES_SORTED[i], MOUTH_INDICES_SORTED[i+1], MOUTH_INDICES_SORTED[i+2]];
    let [x1, y1, z1] = [landmarksCopy[i1].x * 2, landmarksCopy[i1].y * 2, landmarksCopy[i1].z * 2];
    let [x2, y2, z2] = [landmarksCopy[i2].x * 2, landmarksCopy[i2].y * 2, landmarksCopy[i2].z * 2];
    let [x3, y3, z3] = [landmarksCopy[i3].x * 2, landmarksCopy[i3].y * 2, landmarksCopy[i3].z * 2];
    vertices.push([x1, y1, z1]);
    vertices.push([x2, y2, z2]);
    vertices.push([x3, y3, z3]);

    uvs.push([landmarksCopy[i1].x, landmarksCopy[i1].y]);
    uvs.push([landmarksCopy[i2].x, landmarksCopy[i2].y]);
    uvs.push([landmarksCopy[i3].x, landmarksCopy[i3].y]);

    if(i===0) {
      sum.x += x1;
      sum.y += y1;
      sum.z += z1;
    }
    if(i===MOUTH_INDICES_SORTED.length-3) {
      sum.x += x3;
      sum.y += y3;
      sum.z += z3;
    }

    sum.x += x2;
    sum.y += y2;
    sum.z += z2;
  }

  const averagePoint = { 
    x: sum.x / MOUTH_INDICES_SORTED.length, 
    y: sum.y / MOUTH_INDICES_SORTED.length, 
    z: sum.z / MOUTH_INDICES_SORTED.length 
  };
  
  // Draw numbers
  let textMeshes = scene.children.filter((child) => child instanceof Mesh);
  for(const textMesh of textMeshes) scene.remove(textMesh);
  if(CONFIGS.DRAW_NUMBERS) {
    for(let i = 0; i < landmarksCopy.length; i++) {
      const landmark = landmarksCopy[i];
      let text = new TextGeometry(`${i}`, {
        font,
        size: 0.03,
        depth: 0.1,
        curveSegments: 12,
      });

      let textMesh = new Mesh(text, new MeshBasicMaterial({ color: 0x000000 }));
      textMesh.position.set(landmark.x * 2, landmark.y * 2, landmark.z * 2);
      text.rotateZ(Math.PI);
      text.scale(-1, 1, 1);
      scene.add(textMesh);
    }
  }

  // Create geometry
  const geometry = new BufferGeometry();
  
  // Set attributes
  geometry.setAttribute("position", new Float32BufferAttribute(vertices.flat(), 3));
  geometry.setAttribute("uv", new Float32BufferAttribute(uvs.flat(), 2));
  geometry.center();

  // Create final mesh
  const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
  mesh.geometry = geometry;
  mesh.material = material;
  mesh.position.set(averagePoint.x, averagePoint.y, averagePoint.z);

  return mesh;
}
