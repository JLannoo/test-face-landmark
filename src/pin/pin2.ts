import * as THREE from "three";
import { GLTFLoader, GLTF } from "three/examples/jsm/loaders/GLTFLoader";
import { FilesetResolver, FaceLandmarker, DrawingUtils} from "@mediapipe/tasks-vision";

import { GUI } from "three/examples/jsm/libs/lil-gui.module.min";
import Stats from "three/examples/jsm/libs/stats.module";

import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

const imageOutput = document.getElementById("image-output") as HTMLCanvasElement;
const meshOutput = document.getElementById("mesh-output") as HTMLCanvasElement;
const meshOutputCtx = meshOutput.getContext("2d");
if(!meshOutputCtx) throw new Error("Failed to get 2D context from the mesh output canvas.");

/**
 * Returns the world-space dimensions of the viewport at `depth` units away from
 * the camera.
 */
function getViewportSizeAtDepth(
  camera: THREE.PerspectiveCamera,
  depth: number
): THREE.Vector2 {
  const viewportHeightAtDepth =
    2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * camera.fov));
  const viewportWidthAtDepth = viewportHeightAtDepth * camera.aspect;
  return new THREE.Vector2(viewportWidthAtDepth, viewportHeightAtDepth);
}

/**
 * Creates a `THREE.Mesh` which fully covers the `camera` viewport, is `depth`
 * units away from the camera and uses `material`.
 */
function createCameraPlaneMesh(
  camera: THREE.PerspectiveCamera,
  depth: number,
  material: THREE.Material
): THREE.Mesh {
  if (camera.near > depth || depth > camera.far) {
    console.warn("Camera plane geometry will be clipped by the `camera`!");
  }
  const viewportSize = getViewportSizeAtDepth(camera, depth);
  const cameraPlaneGeometry = new THREE.PlaneGeometry(viewportSize.width, viewportSize.height);
  cameraPlaneGeometry.translate(0, 0, -depth);

  return new THREE.Mesh(cameraPlaneGeometry, material);
}

function setupGUI(config: { [key: string]: any }): GUI {
    const gui = new GUI();

    const debugFolder = gui.addFolder("Debug");
    debugFolder.add(config, "drawFaceGrid").name("Draw Face Grid");
    debugFolder.add(config, "axesHelper").name("Show Axes Helper");

    return gui;
}

type RenderCallback = (delta: number) => void;

class BasicScene {
  scene: THREE.Scene;
  width: number;
  height: number;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  lastTime: number = 0;
  callbacks: RenderCallback[] = [];
  debug: { gui: GUI, stats: Stats, axesHelper: THREE.AxesHelper, config: { [key: string]: any } } | null = null;

  constructor({ debug = false } = {}) {
    // Initialize the canvas with the same aspect ratio as the video input
    const ratio = 4/3;
    this.width = window.innerWidth;
    this.height = window.innerWidth / ratio;
    // Set up the Three.js scene, camera, and renderer
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, this.width / this.height, 0.01, 5000);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, canvas: imageOutput });
    this.renderer.setSize(this.width, this.height);
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.renderer.shadowMap.enabled = true;

    // Set up the basic lighting for the scene
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    this.scene.add(directionalLight);

    // Set up the camera position and controls
    this.camera.position.z = 10;
    let orbitTarget = this.camera.position.clone();
    orbitTarget.z -= 5;

    // Add a video background
    const video = document.getElementById("webcam") as HTMLVideoElement;
    const inputFrameTexture = new THREE.VideoTexture(video);
    if (!inputFrameTexture) {
      throw new Error("Failed to get the 'input_frame' texture!");
    }
    inputFrameTexture.encoding = THREE.sRGBEncoding;
    const inputFramesDepth = 500;
    const inputFramesPlane = createCameraPlaneMesh(
      this.camera,
      inputFramesDepth,
      new THREE.MeshBasicMaterial({ map: inputFrameTexture })
    );
    this.scene.add(inputFramesPlane);

    if(debug) {
        // Set up the orbit controls
        const controls = new OrbitControls(this.camera, this.renderer.domElement);
        controls.target = orbitTarget;
        controls.update();

        // Add axes helper
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);

        const config = {
          drawFaceGrid: false,
          axesHelper: false,
        }

        const stats = new Stats();
        document.body.appendChild(stats.dom);

        const gui = setupGUI(config);

        this.debug = { gui, stats, axesHelper, config };
    }

    // Render the scene
    this.render();

    window.addEventListener("resize", this.resize.bind(this));
  }

  resize() {
    this.width = video.videoWidth;
    this.height = video.videoHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    meshOutput.width = this.width;
    meshOutput.height = this.height;

    this.renderer.render(this.scene, this.camera);
  }

  render(time: number = this.lastTime): void {
    const delta = (time - this.lastTime) / 1000;
    this.lastTime = time;
    // Call all registered callbacks with deltaTime parameter
    for (const callback of this.callbacks) {
      callback(delta);
    }

    // Render the scene
    this.renderer.render(this.scene, this.camera);

    // Debug
    if (this.debug) {
        this.debug.stats.update();
        this.debug.gui.controllers.forEach((controller) => controller.updateDisplay());
        this.debug.axesHelper.visible = this.debug.config.axesHelper;
    }
    // Request next frame
    requestAnimationFrame((t) => this.render(t));
  }
}

interface MatrixRetargetOptions {
  decompose?: boolean;
  scale?: number;
}

class PinnedObject {
  scene: THREE.Scene;
  loader: GLTFLoader = new GLTFLoader();
  gltf: GLTF | null;
  root: THREE.Bone | null;
  url: string;
  vertexToPin: number;

  constructor(url: string, scene: THREE.Scene, vertexToPin: number = 0) {
    this.url = url;
    this.scene = scene;
    this.loadModel(this.url);

    this.gltf = null;
    this.root = null;
    this.vertexToPin = vertexToPin;
  }

  loadModel(url: string) {
    this.url = url;
    this.loader.load(url, (gltf) => {
        if (this.gltf) this.gltf.scene.remove();

        this.gltf = gltf;

        this.gltf.scene.traverse((object) => {
          object.visible = false;
        });

        this.scene.add(gltf.scene);
        this.init(gltf);

        const folder = scene.debug?.gui.addFolder("Object");
        folder?.add(this, "vertexToPin", 0, 468).step(1).name("Vertex to Pin");
      },

      (progress) => console.log("Loading model...", 100.0 * (progress.loaded / progress.total),"%"),
      (error) => console.error(error)
    );
  }

  init(gltf: GLTF) {
    gltf.scene.traverse((object) => {
      // Register first bone found as the root
      if ((object as THREE.Bone).isBone && !this.root) {
        this.root = object as THREE.Bone;
      }
      // Return early if no mesh is found.
      if (!(object as THREE.Mesh).isMesh) return;

      const mesh = object as THREE.Mesh;
      // Reduce clipping when model is close to camera.
      mesh.frustumCulled = false;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });
  }

  /**
   * Apply a position, rotation, scale matrix to current GLTF.scene
   * @param matrix
   * @param matrixRetargetOptions
   * @returns
   */
  applyMatrix(
    matrix: THREE.Matrix4,
    matrixRetargetOptions?: MatrixRetargetOptions
  ): void {
    const { decompose = false, scale = 1 } = matrixRetargetOptions || {};
    if (!this.gltf) return;

    // Three.js will update the object matrix when it render the page
    // according the object position, scale, rotation.
    // To manually set the object matrix, you have to set autoupdate to false.
    matrix.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.matrix.copy(matrix);
    
  }
}

let faceLandmarker: FaceLandmarker;
let drawUtils: DrawingUtils;
let video: HTMLVideoElement;

const scene = new BasicScene({ debug: true });
const object = new PinnedObject("/monke.glb", scene.scene, 10);

function detectFaceLandmarks(time: DOMHighResTimeStamp): void {
  if (!faceLandmarker) return;
  const landmarks = faceLandmarker.detectForVideo(video, time);

  // Hide object if no face landmarks are detected
  if(object.gltf) {
    object.gltf.scene.traverse((object) => {
      object.visible = landmarks.faceLandmarks.length > 0;
    });
  }

  // Apply facial transformation matrix to object
  const transformationMatrices = landmarks.facialTransformationMatrixes;
  if (transformationMatrices && transformationMatrices.length > 0) {
    let matrix = new THREE.Matrix4().fromArray(transformationMatrices[0].data);

    // Get displacemente from center of the face to the pin vertex
    const vertex = landmarks.faceLandmarks[0][object.vertexToPin];
    const center = landmarks.faceLandmarks[0][1]; // Tip of the nose

    // There's probably a better way to do this whole 'translate to vertex' thing
    // but I'm not sure how to do it yet
    const faceWidth = Math.abs(landmarks.faceLandmarks[0][234].x - landmarks.faceLandmarks[0][454].x) * 3;
    const faceHeight = Math.abs(landmarks.faceLandmarks[0][10].y - landmarks.faceLandmarks[0][152].y) * 1.25;
    const faceDepth = Math.abs(landmarks.faceLandmarks[0][10].z - landmarks.faceLandmarks[0][152].z);
    const faceDimensions = new THREE.Vector3(faceWidth, faceHeight, faceDepth);

    const displacement = new THREE.Vector3(
      vertex.x - center.x, 
      - vertex.y + center.y, 
      vertex.z - center.z
    )
    .multiply(faceDimensions)
    .multiplyScalar(120); // 120 is a magic number to scale the displacement

    // Convert vertex coordinates to NDC
    const ndcX = (vertex.x * 2) - 1;
    const ndcY = 1 - (vertex.y * 2); // Invert Y axis
    const ndc = new THREE.Vector3(ndcX, ndcY, vertex.z);

    // Unproject to world space
    ndc.unproject(scene.camera);

    const vertexTransformationMatrix = new THREE.Matrix4().makeTranslation(ndc.x, ndc.y, ndc.z);
    matrix.multiply(vertexTransformationMatrix);

    const translationMatrix = new THREE.Matrix4().makeTranslation(displacement.x, displacement.y, displacement.z);
    matrix.multiply(translationMatrix);
    
    object.applyMatrix(matrix, { scale: 5 });
  }

  // Draw face landmarks
  meshOutputCtx?.clearRect(0, 0, meshOutput.width, meshOutput.height);
  if(drawUtils && scene.debug?.config.drawFaceGrid) {
    if(landmarks.faceLandmarks.length > 0) {
      drawUtils.drawConnectors(landmarks.faceLandmarks[0], FaceLandmarker.FACE_LANDMARKS_TESSELATION,   { color: "#C0C0C070", lineWidth: 1 } );
      drawUtils.drawConnectors(landmarks.faceLandmarks[0], FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);

      drawUtils.drawLandmarks([landmarks.faceLandmarks[0][object.vertexToPin]], { color: "#FF3030", lineWidth: 2 });
    }
  }

}

function onVideoFrame(time: DOMHighResTimeStamp): void {
  detectFaceLandmarks(time);
  video.requestVideoFrameCallback(onVideoFrame);
}

// Stream webcam into landmarker loop (and also make video visible)
async function streamWebcamThroughFaceLandmarker(): Promise<void> {
  video = document.getElementById("webcam") as HTMLVideoElement;

  function onAcquiredUserMedia(stream: MediaStream): void {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      video.hidden = true;

      scene.resize();

      meshOutput.width = video.videoWidth;
      meshOutput.height = video.videoHeight;
    };
  }

  try {
    const evt = await navigator.mediaDevices.getUserMedia({ video: {
      facingMode: "user",
    } });
    onAcquiredUserMedia(evt);
    video.requestVideoFrameCallback(onVideoFrame);
  } catch (e: unknown) {
    console.error(`Failed to acquire camera feed: ${e}`);
  }
}

async function runDemo() {
  await streamWebcamThroughFaceLandmarker();
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.1.0-alpha-16/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromModelPath(
    vision,
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
  );
  await faceLandmarker.setOptions({
    baseOptions: {
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true
  });

  if(!meshOutputCtx) throw new Error("Failed to get 2D context from the mesh output canvas.");
  drawUtils = new DrawingUtils(meshOutputCtx);

  console.log("Finished Loading MediaPipe Model.");
}

runDemo();
