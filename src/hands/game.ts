import { BoundingBox } from "@mediapipe/tasks-vision";
import { Mesh, MeshBasicMaterial, Renderer, Scene, SphereGeometry } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

export type CustomEventListener<T = Event> = (this: T, ev: T) => any;
export type ScoreChangedEvent = CustomEvent<number>;

export declare interface Game {
    score: number;

    addScore(points: number): void;

    addEventListener(type: "scoreChanged", listener: CustomEventListener<ScoreChangedEvent>): void;
    addEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
}

export class Game extends EventTarget implements Game {
    score: number;
    lastHandStates: string[] = [];

    private scene: Scene;
    private renderer: Renderer;
    
    target: Target;
    targetModel: Mesh | undefined;

    constructor(scene: Scene, renderer: Renderer) {
        super();
        this.score = 0;

        this.scene = scene;
        this.renderer = renderer;

        this.target = this.generateNewTarget();
    }

    addScore(points: number) {
        this.score += points;
        this.dispatchEvent(new CustomEvent("scoreChanged", { detail: this.score }));
    }

    async loadModel(url: string) {
        const loader = new GLTFLoader();
        const model = await new Promise((resolve, reject) => {
            loader.load(url, resolve, undefined, reject);
        }) as any;



        this.targetModel = model.scene.children[0] as Mesh;
        
        this.scene.remove(this.target.mesh);
        this.generateNewTarget();
    }

    private generateNewTarget() {
        const canvas = this.renderer.domElement;
        const radius = 5;
        const margin = 20;

        let x = Math.random() * canvas.width;
        let y = Math.random() * canvas.height;

        x = Math.max(radius + margin, Math.min(x, canvas.width - radius - margin));
        y = Math.max(radius + margin, Math.min(y, canvas.height - radius - margin));


        this.target = new Target(x, y, radius, this.targetModel);

        this.scene.add(this.target.mesh);

        this.target.addEventListener("hit", this.onHit.bind(this));

        return this.target;
    }

    private onHit() {
        this.scene.remove(this.target.mesh);
        this.addScore(1);
        this.generateNewTarget();
    }
}

export declare interface Target {
    hit(box: BoundingBox): boolean;

    addEventListener(type: "hit", listener: CustomEventListener): void;
    addEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
}

export class Target extends EventTarget implements Target {
    x: number;
    y: number;
    radius: number;

    mesh: Mesh;

    constructor(x: number, y: number, radius: number, model?: Mesh) {
        super();
        this.x = x;
        this.y = y;
        this.radius = radius;        

        if(model) {
            this.mesh = model.clone();
            this.mesh.position.set(x, y, 0);
            this.mesh.scale.set(radius, -radius, radius);
        } else {
            const geometry = new SphereGeometry(radius, 32, 32);
            const material = new MeshBasicMaterial({ color: 0xff0000 });
            this.mesh = new Mesh(geometry, material);
            this.mesh.position.set(x, y, 0);
        }
    }

    hit(box: BoundingBox) {
        const { originX: x, originY: y, width, height } = box;     

        if(
            this.x >= x && this.x <= x + width && 
            this.y >= y && this.y <= y + height
        ) {            
            this.dispatchEvent(new CustomEvent("hit"));
            return true;
        }

        return false;
    }
}

export default Game;