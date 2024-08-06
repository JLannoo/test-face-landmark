/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

import * as THREE from "three";
import {
  GLTFLoader,
  GLTF
} from "three/examples/jsm/loaders/GLTFLoader";
import {
  FilesetResolver,
  FaceLandmarker,
  Classifications,
} from "@mediapipe/tasks-vision";

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
  const cameraPlaneGeometry = new THREE.PlaneGeometry(
    viewportSize.width,
    viewportSize.height
  );
  cameraPlaneGeometry.translate(0, 0, -depth);

  return new THREE.Mesh(cameraPlaneGeometry, material);
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

  constructor() {
    // Initialize the canvas with the same aspect ratio as the video input
    const ratio = 4/3;
    this.width = window.innerWidth;
    this.height = window.innerWidth / ratio;
    // Set up the Three.js scene, camera, and renderer
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      60,
      this.width / this.height,
      0.01,
      5000
    );

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    document.body.appendChild(this.renderer.domElement);

    // Set up the basic lighting for the scene
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    this.scene.add(directionalLight);

    // Set up the camera position and controls
    this.camera.position.z = 0;
    let orbitTarget = this.camera.position.clone();
    orbitTarget.z -= 5;

    // Add a video background
    const video = document.getElementById("video") as HTMLVideoElement;
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

    // Render the scene
    this.render();

    window.addEventListener("resize", this.resize.bind(this));
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

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
    // Request next frame
    requestAnimationFrame((t) => this.render(t));
  }
}

interface MatrixRetargetOptions {
  decompose?: boolean;
  scale?: number;
}

class Avatar {
  scene: THREE.Scene;
  loader: GLTFLoader = new GLTFLoader();
  gltf: GLTF | null;
  root: THREE.Bone | null;
  morphTargetMeshes: THREE.Mesh[] = [];
  url: string;

  constructor(url: string, scene: THREE.Scene) {
    this.url = url;
    this.scene = scene;
    this.loadModel(this.url);

    this.gltf = null;
    this.root = null;
  }

  loadModel(url: string) {
    this.url = url;
    this.loader.load(
      // URL of the model you want to load
      url,
      // Callback when the resource is loaded
      (gltf) => {
        if (this.gltf) {
          // Reset GLTF and morphTargetMeshes if a previous model was loaded.
          this.gltf.scene.remove();
          this.morphTargetMeshes = [];
        }
        this.gltf = gltf;
        console.log();
        this.scene.add(gltf.scene);
        this.init(gltf);
      },

      // Called while loading is progressing
      (progress) =>
        console.log(
          "Loading model...",
          100.0 * (progress.loaded / progress.total),
          "%"
        ),
      // Called when loading has errors
      (error) => console.error(error)
    );
  }

  init(gltf: GLTF) {
    gltf.scene.traverse((object) => {
      // Register first bone found as the root
      if ((object as THREE.Bone).isBone && !this.root) {
        this.root = object as THREE.Bone;
        console.log(object);
      }
      // Return early if no mesh is found.
      if (!(object as THREE.Mesh).isMesh) {
        // console.warn(`No mesh found`);
        return;
      }

      const mesh = object as THREE.Mesh;
      // Reduce clipping when model is close to camera.
      mesh.frustumCulled = false;

      // Return early if mesh doesn't include morphable targets
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) {
        // console.warn(`Mesh ${mesh.name} does not have morphable targets`);
        return;
      }
      this.morphTargetMeshes.push(mesh);
    });
  }

  updateBlendshapes(blendshapes: Map<string, number>) {
    for (const mesh of this.morphTargetMeshes) {
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) {
        // console.warn(`Mesh ${mesh.name} does not have morphable targets`);
        continue;
      }
      for (const [name, value] of blendshapes) {
        if (!Object.keys(mesh.morphTargetDictionary).includes(name)) {
          // console.warn(`Model morphable target ${name} not found`);
          continue;
        }

        const idx = mesh.morphTargetDictionary[name];
        mesh.morphTargetInfluences[idx] = value;
      }
    }
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
    if (!this.gltf) {
      return;
    }
    // Three.js will update the object matrix when it render the page
    // according the object position, scale, rotation.
    // To manually set the object matrix, you have to set autoupdate to false.
    matrix.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    // Set new position and rotation from matrix
    this.gltf.scene.matrix.copy(matrix);
  }

  /**
   * Takes the root object in the avatar and offsets its position for retargetting.
   * @param offset
   * @param rotation
   */
  offsetRoot(offset: THREE.Vector3, rotation?: THREE.Vector3): void {
    if (this.root) {
      this.root.position.copy(offset);
      if (rotation) {
        let offsetQuat = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(rotation.x, rotation.y, rotation.z)
        );
        this.root.quaternion.copy(offsetQuat);
      }
    }
  }
}

let faceLandmarker: FaceLandmarker;
let video: HTMLVideoElement;

const scene = new BasicScene();
const avatar = new Avatar(
  "/monke.glb",
  scene.scene
);

function detectFaceLandmarks(time: DOMHighResTimeStamp): void {
  if (!faceLandmarker) {
    return;
  }
  const landmarks = faceLandmarker.detectForVideo(video, time);

  // Apply transformation
  const transformationMatrices = landmarks.facialTransformationMatrixes;
  
  if (transformationMatrices && transformationMatrices.length > 0) {
    let matrix = new THREE.Matrix4().fromArray(transformationMatrices[0].data);
    
    // Example of applying matrix directly to the avatar
    avatar.applyMatrix(matrix, { scale: 15 });
  }

  // Apply Blendshapes
  const blendshapes = landmarks.faceBlendshapes;
  if (blendshapes && blendshapes.length > 0) {
    const coefsMap = retarget(blendshapes);
    avatar.updateBlendshapes(coefsMap);
  }
}

function retarget(blendshapes: Classifications[]) {
  const categories = blendshapes[0].categories;
  let coefsMap = new Map<string, number>();
  for (let i = 0; i < categories.length; ++i) {
    const blendshape = categories[i];
    // Adjust certain blendshape values to be less prominent.
    switch (blendshape.categoryName) {
      case "browOuterUpLeft":
        blendshape.score *= 1.2;
        break;
      case "browOuterUpRight":
        blendshape.score *= 1.2;
        break;
      case "eyeBlinkLeft":
        blendshape.score *= 1.2;
        break;
      case "eyeBlinkRight":
        blendshape.score *= 1.2;
        break;
      default:
    }
    coefsMap.set(categories[i].categoryName, categories[i].score);
  }
  return coefsMap;
}

function onVideoFrame(time: DOMHighResTimeStamp): void {
  // Do something with the frame.
  detectFaceLandmarks(time);
  // Re-register the callback to be notified about the next frame.
  video.requestVideoFrameCallback(onVideoFrame);
}

// Stream webcam into landmarker loop (and also make video visible)
async function streamWebcamThroughFaceLandmarker(): Promise<void> {
  video = document.getElementById("video") as HTMLVideoElement;

  function onAcquiredUserMedia(stream: MediaStream): void {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      video.hidden = true;
    };
  }

  try {
    const evt = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
      }
    });
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

  console.log("Finished Loading MediaPipe Model.");
}

runDemo();
