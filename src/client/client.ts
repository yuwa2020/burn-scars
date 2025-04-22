import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'
import * as TWEEN from '@tweenjs/tween.js'
import { terrainShader } from './shaders/terrain-shader'
import { GUI } from 'dat.gui'
import { Mesh } from 'three'
import axios from 'axios'
import {
    metaState,
    init,
    sessionData,
    initVis,
    gameState,
    logMyState,
    getLocalCordinate,
    readstateFile,
    toggleAnnoation,
} from './util'
import { terrainDimensions } from './constants'
import './styles/style.css'
import * as tiff from 'tiff'
import Stats from 'three/examples/jsm/libs/stats.module'
import { ajax } from 'jquery'
import { url } from 'inspector'
const UPNG = require('upng-js');

// import { loadImage } from 'canvas'

// -------------------------------unzip ---------------------------

// import { Archive } from 'libarchive.js/main.js'
// Archive.init({
//     workerUrl: 'libarchive.js/dist/worker-bundle.js',
// })
// ;(document.getElementById('file') as HTMLInputElement).addEventListener('change', async (e) => {
//     let file = (e.currentTarget as HTMLInputElement).files?[0] as any
//     let archive:any = await Archive.open(file)
//     let obj = await archive.extractFiles()
//     console.log(obj)
// })

// const worker = new Worker('.../src/client/worker.js')

// worker.postMessage('i am in worker')
let Developer = false
let overRideControl = false
var data : { [x : number]: Array<number> } = {}

var regionBounds : Array<number> = [0, 0, 0, 0]
var regionDimensions : Array<number> = [0, 0]

let _fetchData: any
let mesh: THREE.Mesh

let isSegmentationDone = true
let isSTLDone = true
let isModelLoaded = false
let isSatelliteImageLoaded = false

const scene = new THREE.Scene()
// const blurs = [0, 1, 2];
// const zs = [100, 200, 300, 400, 500];

const pers = [0, 0.02, 0.04, 0.08, 0.16, 0.32]
// const pers = [0]
var meshes: { [key: string]: Mesh } = {}

var forestArray: Uint8Array

interface PixelDict {
    [position: number]: number;
  }

const pixelDict: PixelDict = {};
var confidenceJSON: PixelDict

let host = ''
if (location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '172.28.200.135') {
    host = ''
} else {
    host = 'https://floodmap.b-cdn.net/'
}

const stats_mb = Stats()
stats_mb.showPanel(2)
stats_mb.domElement.style.cssText = 'position:absolute;top:250px;right:50px;'
document.body.appendChild(stats_mb.domElement)

// Get the URL parameters
const urlParams = new URLSearchParams(window.location.search);

// Retrieve the value of the 'region_id' parameter
// const testRegion = urlParams.get('region');
// const student_id = urlParams.get('student_id');

// // Use the parameter value in your code
// if (testRegion) {
//   console.log('Region ID:', testRegion);
//   // Perform actions based on the testRegion
// } else {
//   console.log('No region ID specified.');
// }


const thresholdSlider = document.getElementById('thresholdSlider') as HTMLInputElement;
const sliderValue = document.getElementById('sliderValue') as HTMLInputElement;

var predDataOrig: ImageData;

function clearPixelsBelowThreshold(thresholdValue: any) {
    // const thresholdValue = parseInt(thresholdSlider.value, 10);

    // Get the pixel data from the entire canvas
    const imageData = predContext.getImageData(0, 0, predCanvas.width, predCanvas.height);

    const predDataCurr = imageData.data;
    const predData = predDataOrig.data;

    // Iterate through pixel data
    for (let i = 0; i < predData.length; i += 4) {
        const my_val = predData[i + 1];

        if (my_val < thresholdValue) {
            // Set the alpha channel (transparency) to 0 for pixels below the threshold
            predDataCurr[i + 3] = 0;
        }
        else{
            predDataCurr[i] = 0;
            predDataCurr[i + 1] = 127;
            predDataCurr[i + 2] = 0;
            predDataCurr[i + 3] = 255;
        }
    }

    // Put the modified pixel data back to the canvas
    predContext.putImageData(imageData, 0, 0);
    predictionTexture.needsUpdate = true;
    
    context.putImageData(imageData, 0, 0);
    forestMapTexture.needsUpdate = true;
}

// Event listener for the threshold slider
thresholdSlider.addEventListener('input', function() {
    // Get the current threshold value from the slider
    const thresholdValue = parseInt(thresholdSlider.value, 10);
    sliderValue.textContent = String((Number.parseFloat(thresholdSlider.value)/128).toFixed(2));

    clearPixelsBelowThreshold(thresholdValue);
});

function showLoadingScreen(){
    ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'block'
    // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'block'
    ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
}

function hideLoadingScreen(){
    ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
    // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
    ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
}

// Function to poll the backend
async function pollBackendTask(taskId: string) {
    const response = await fetch(`http://127.0.0.1:5005/check-status?taskId=${taskId}&testRegion=${sessionData.testRegion}`);
    const data = await response.json();

    console.log("data: ", data)
  
    if (data.status === 'completed') {
        // Backend task is completed, handle the response
        console.log('Backend task completed:', data.result);

        // context!.clearRect(0, 0, annCanvas.width, annCanvas.height);
        // forestMapTexture.needsUpdate = true;
  
        // Continue with other actions on the frontend
        const superpixelBuffer = await fetch(`http://127.0.0.1:5005/superpixel?recommend=${0}&testRegion=${sessionData.testRegion}`).then(response => response.arrayBuffer());
        console.log("superpixelBuffer: ", superpixelBuffer)

        // Convert ArrayBuffer to base64
        const base64ImageSuperpixel = arrayBufferToBase64(superpixelBuffer)

        // Create an Image element
        const imgSuperpixel = new Image();

        // Set the source of the Image to the base64-encoded PNG data
        imgSuperpixel.src = 'data:image/png;base64,' + base64ImageSuperpixel;

        await new Promise(resolve => {
            imgSuperpixel.onload = resolve;
        });

        // Set canvas dimensions to match the image dimensions
        superpixelCanvas.width = imgSuperpixel.width;
        superpixelCanvas.height = imgSuperpixel.height;

        console.log("height: ", superpixelCanvas.height)
        console.log("width: ", superpixelCanvas.width)

        // Draw the image on the canvas
        superpixelContext!.drawImage(imgSuperpixel, 0, 0);
        superpixelTexture.needsUpdate = true // saugat

        const predBuffer = await fetch(`http://127.0.0.1:5005/pred?taskId=${taskId}&testRegion=${sessionData.testRegion}`).then(response => response.arrayBuffer());
        console.log("arraybuffer: ", predBuffer)

        // Convert ArrayBuffer to base64
        const base64ImagePred = arrayBufferToBase64(predBuffer)

        // Create an Image element
        const imgPred = new Image();

        // Set the source of the Image to the base64-encoded PNG data
        imgPred.src = 'data:image/png;base64,' + base64ImagePred;

        // Wait for the image to load
        imgPred.onload = () => {

            // Set canvas dimensions to match the image dimensions
            predCanvas.width = imgPred.width;
            predCanvas.height = imgPred.height;

            console.log("height: ", predCanvas.height)
            console.log("width: ", predCanvas.width)

            // Draw the image on the canvas
            predContext!.drawImage(imgPred, 0, 0);
            predictionTexture.needsUpdate = true // saugat
        };

        ;(document.getElementById('exploration') as HTMLElement).style.display = 'block'
        ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
        // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
    
        // Hide the loading screen
        //   hideLoadingScreen();
    } else {
      // Backend task is still in progress, continue polling
      setTimeout(() => pollBackendTask(taskId), 120000); // Poll every 2 mins
    }
  }

function dataURItoBlob(dataURI: string) {
    // convert base64 to raw binary data held in a string
    // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
    var byteString = atob(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to an ArrayBuffer
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    //Old Code
    //write the ArrayBuffer to a blob, and you're done
    //var bb = new BlobBuilder();
    //bb.append(ab);
    //return bb.getBlob(mimeString);

    //New Code
    return new Blob([ab], {type: mimeString});
}

// saugat
async function retrainSession(event: Event) {
    event.stopPropagation();
    
    // TODO: check what this does???
    // disposeUniform()
    console.log("Retrain session")

    var dataURL = labelsCanvas.toDataURL()
    var dataURL_2 = predCanvas.toDataURL()

    // Create a FormData object and append the image data
    var formData = new FormData();
    const dataURLFile = dataURItoBlob(dataURL);

    var formData_2 = new FormData();
    const dataURLFile_2 = dataURItoBlob(dataURL_2);

    formData.append('image', dataURLFile);
    formData.append('image_2', dataURLFile_2);

    const taskId = encodeURIComponent(sessionData.name);
    const testRegion = encodeURIComponent(sessionData.testRegion);

    // Send a POST request to the Flask backend
    showLoadingScreen();
    const response = await fetch(`http://127.0.0.1:5005/retrain?taskId=${taskId}&testRegion=${testRegion}`, {
        method: 'POST',
        body: formData,
    });
    const data = await response.json();

    // Check if the task was successfully started
    if (data.status === 'success') {
        // Start polling the backend for task status
        // pollBackendTask(data.taskId);

        const predBuffer = await fetch(`http://127.0.0.1:5005/pred?taskId=${taskId}&predict=${0}&testRegion=${testRegion}`).then(response => response.arrayBuffer());
        console.log("arraybuffer: ", predBuffer)

        // Convert ArrayBuffer to base64
        const base64ImagePred = arrayBufferToBase64(predBuffer)

        // Create an Image element
        const imgPred = new Image();
        const imgForest = new Image();

        // Set the source of the Image to the base64-encoded PNG data
        imgPred.src = 'data:image/png;base64,' + base64ImagePred;
        imgForest.src = 'data:image/png;base64,' + base64ImagePred;

        // Wait for the image to load
        imgPred.onload = () => {

            // Set canvas dimensions to match the image dimensions
            predCanvas.width = imgPred.width;
            predCanvas.height = imgPred.height;

            console.log("height: ", predCanvas.height)
            console.log("width: ", predCanvas.width)

            // Draw the image on the canvas
            predContext!.drawImage(imgPred, 0, 0);
            predictionTexture.needsUpdate = true // saugat

            predDataOrig = predContext.getImageData(0, 0, predCanvas.width, predCanvas.height);
        };

        // Wait for the image to load
        imgForest.onload = () => {

            // Set canvas dimensions to match the image dimensions
            annCanvas.width = imgForest.width;
            annCanvas.height = imgForest.height;

            console.log("height: ", annCanvas.height)
            console.log("width: ", annCanvas.width)

            context!.drawImage(imgForest, 0, 0);
            forestMapTexture.needsUpdate = true
        };

        ;(document.getElementById('exploration') as HTMLElement).style.display = 'block'
        ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
        // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'

    } else {
        // Handle the case where the task couldn't be started
        console.error('Failed to start backend task');
        hideLoadingScreen();
    }
}



let eventFunction: { [key: string]: any } = {
    BFS: (x: number, y: number, flood: boolean, clear: boolean) => BFSHandler(x, y, flood, clear),
    brush: (x: number, y: number, flood: boolean, clear: boolean) =>
        brushHandler('t', x, y, flood, clear),
    brushLine: (x: number, y: number, flood: boolean, clear: boolean, linePoints: Array<number>) =>
        brushLineHandler(linePoints, flood, clear),
    polygonSelector: (x: number, y: number, flood: boolean, clear: boolean) =>
        polygonSelectionHandler(x, y, flood, clear),
    polygonFill: (
        x: number,
        y: number,
        flood: boolean,
        clear: boolean,
        linePoints: Array<number>
    ) => polygonFillHandler(flood, clear, linePoints),
    segmentation: (x: number, y: number, flood: boolean, clear: boolean) =>
        segAnnotationHandler('s', x, y, flood, clear),
    connectedSegmentation: (x: number, y: number, flood: boolean, clear: boolean) =>
        connectedSegAnnotationHandler('s', x, y, flood, clear),
}

function delay(time: number) {
    return new Promise((resolve) => setTimeout(resolve, time))
}

let time: Date | undefined = undefined

let _readstateFile = async (array: any[]) => {
    sessionData.sessionStart = new Date(array[0].start.time)
    for (let i = 0; i < array.length; i++) {
        if (array[i].start) {
            gameState.push({ start: array[i].start })
            continue
        }
        let event = array[i].mouseEvent
        // if (event.label != "brush") {
        //     await delay(50)
        // }
        // if (i % 1000 == 0) {
        //     console.log(i / array.length)
        // }
        // let _cameraPosition = event.cameraPosition
        // let _target = event.targetPosition
        // camera.position.set(_cameraPosition.x, _cameraPosition.y, _cameraPosition.z)
        // controls.target.set(_target.x, _target.y, _target.z)
        // controls.update()
        let x, y, flood, clear
        if (event.x == undefined) {
            x = 0
            y = 0
        } else {
            x = event.x
            y = event.y
        }
        flood = event.flood
        clear = event.clear
        if (event.brushSize) {
            params.brushSize = event.brushSize
        }
        if (event.persistanceThreshold) {
            params.pers = event.persistanceThreshold
        }
        time = event.time
        eventFunction[event.label](x, y, flood, clear, event.linePoints)
    }
    time = undefined
}
const persLoader = new THREE.TextureLoader()


// Load Annotation from checkpoint or Image
// ;(document.getElementById('upload') as HTMLElement).oninput = () => {
//     if ((document.getElementById('upload') as HTMLInputElement).files) {
//         let file = (document.getElementById('upload') as HTMLInputElement).files![0]
//         ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
//         ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
//         var fr = new FileReader()

//         if (file.type == "application/json") {
//             fr.onload = async function (e) {
//                 var result = JSON.parse(e.target!.result as string)
//                 // console.log(result)
//                 await _readstateFile(result)
//                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
//                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
//             }
//             fr.readAsText(file)
            
//         } else if (file.type == "image/png") {
//             fr.onload = async function (e) {
//                 let image = document.createElement('img')
//                 image.src = e.target!.result as string
//                 // console.log(regionDimensions[0], regionDimensions[1], image.width, image.height)
//                 image.onload = function() {
//                     if (image.width == regionDimensions[0] && image.height == regionDimensions[1]) {
//                         context!.drawImage(image, 0, 0)
//                         forestMapTexture.needsUpdate = true
//                         // predictionTexture.needsUpdate = true // saugat
//                     } else {
//                         alert("Wrong dimensions for forestMap image, check that the region is correct")
//                     }
//                     ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
//                     ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
//                 }
//             }
//             fr.readAsDataURL(file)
//         } else {
//             alert('Invalid file type, must be .png or .json!')
//             ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
//             ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
//         }

//         ;(document.getElementById('upload') as HTMLInputElement).files = null
//     }
// }

// fetch(`${host}img/elevation${metaState.region}.tiff`).then((res) =>
// fetch(`${host}img/test0.1.tiff`).then((res) =>
//     res.arrayBuffer().then(function (arr) {
//         var tif = tiff.decode(arr)
//         data = tif[0].data as Float32Array
//     })
// )
window.onload = init

var segsToPixels2: {
    [key: number]: {
        [key: number]: Array<number>
    }
} = {}
var persDatas: {
    [key: number]: Int16Array
} = {}

var persTextures: { [key: number]: THREE.Texture } = {}
var dataTextures: { [key: number]: THREE.Texture } = {}
var segsMax: { [key: number]: number } = {}

;(document.getElementById('loader') as HTMLElement).style.display = 'none'
;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
// ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
persLoader.load(
    './img/rainbow.png',
    function (texture) {
        uniforms.colormap.value = texture
    },
    undefined,
    function (err) {
        console.error('An error happened.')
    }
)
// const light = new THREE.SpotLight()
// light.position.set(4000, 4000, 20)
// scene.add(light)
// const ambient = new THREE.AmbientLight( 0x404040 ); // soft white light
// scene.add( ambient );

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 5000)
camera.position.set(0, 0, 2000)

const renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true })
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.shadowMap.enabled = true

document.body.appendChild(renderer.domElement)

let controls = new OrbitControls(camera, renderer.domElement)
controls.dampingFactor = 1.25
controls.enableDamping = true
controls.maxPolarAngle = Math.PI / 1.5
controls.minPolarAngle = 1.2
controls.minDistance = 0
controls.maxAzimuthAngle = 0.8
controls.minAzimuthAngle = -0.65

var canvas = document.createElement('canvas')
var annCanvas = document.createElement('canvas')
var predCanvas = document.createElement('canvas')
var labelsCanvas = document.createElement('canvas')
var superpixelCanvas = document.createElement('canvas')
var confidenceCanvas = document.createElement('canvas')

var context : CanvasRenderingContext2D
var predContext : CanvasRenderingContext2D
var superpixelContext: CanvasRenderingContext2D
var confidenceContext: CanvasRenderingContext2D
var labelsContext: CanvasRenderingContext2D
var forestMapTexture : THREE.Texture
var labelsTexture : THREE.Texture

var predictionTexture: THREE.Texture // saugat
var superpixelTexture: THREE.Texture // saugat
var confidenceTexture: THREE.Texture // saugat

const gui = new GUI({ width: window.innerWidth / 5 })
var params = {
    blur: 0,
    dimension: metaState.flat == 0,
    forestMap: false,
    brushSize: 8,
    pers: 6,
    persShow: false,
    data: false,
    guide: 0,
    forest: true,
    not_forest: false,
    clear: false,
    prediction: true, // saugat
    superpixel: false, // saugat
    labels: false, // saugat
    confidence: false, // saugat
    forestMapKey: 0, // saugat
    superpixelKey: 0, // saugat
}
let persIndex: { [key: number]: number } = {
    1: 0.32,
    2: 0.16,
    3: 0.08,
    4: 0.04,
    5: 0.02,
    6: 0,
}
var persVal = persIndex[params.pers]
// var persIndex = persToIndex[params.pers];

var uniforms = {
    z: { value: metaState.flat == 0 ? 500 : 0 },
    diffuseTexture: { type: 't', value: new THREE.Texture() },
    forestMapTexture: { type: 't', value: new THREE.Texture() },
    predictionTexture: { type: 't', value: new THREE.Texture() },
    superpixelTexture: { type: 't', value: new THREE.Texture() },
    confidenceTexture: { type: 't', value: new THREE.Texture() },
    labelsTexture: { type: 't', value: new THREE.Texture() },
    dataTexture: { type: 't', value: new THREE.Texture() },
    persTexture: { type: 't', value: new THREE.Texture() },
    colormap: { type: 't', value: new THREE.Texture() },
    forestMap: { value: 1 },
    prediction: {value: 0},
    superpixel: {value: 0},
    confidence: {value: 0},
    labels: {value: 0},
    forestMapKey: {value: 0},
    superpixelKey: {value: 0},
    data: { value: 0 },
    segsMax: { type: 'f', value: 0 },
    persShow: { value: 0 },
    hoverValue: { type: 'f', value: 0 },
    guide: { value: params.guide },
    dimensions: { type: 'vec2', value: [100, 100] },
    not_forest: { type: 'bool', value: params.not_forest },
    forest: { type: 'bool', value: params.forest },
    quadrant: { value: metaState.quadrant },
}
const viewFolder = gui.addFolder('Settings')

// viewFolder
//     .add(params, 'flood')
//     .onChange(() => {
//         params.dry = !params.flood
//         viewFolder.updateDisplay()
//     })
//     .name('Annotate Flood')

// viewFolder
//     .add(params, 'dry')
//     .onChange(() => {
//         params.flood = !params.dry
//         viewFolder.updateDisplay()
//     })
//     .name('Annotate Dry Area')

// if (metaState.flat == 0) {
//     viewFolder
//         .add(params, 'dimension')
//         .onChange(() => {
//             scene.remove(scene.children[0])
//             if (params.dimension) {
//                 uniforms.z.value = 500
//                 scene.add(meshes[3])
//             } else {
//                 uniforms.z.value = 0
//                 scene.add(meshes[2])
//             }
//         })
//         .name('3D View')
// }


// saugat
viewFolder
.add(params, 'prediction')
.onChange(() => {
    if (params.prediction) {
        uniforms.prediction.value = 1
    } else {
        uniforms.prediction.value = 0
    }
})
.name('Show Prediction')
//saugat

viewFolder
    .add(params, 'forestMap')
    .onChange(() => {
        if (params.forestMap) {
            uniforms.forestMap.value = 1
        } else {
            uniforms.forestMap.value = 0
        }
    })
    .name('Show Burn Scars Map')

viewFolder
    .add(params, 'labels')
    .onChange(() => {
        if (params.labels) {
            uniforms.labels.value = 1
        } else {
            uniforms.labels.value = 0
        }
    })
    .name('Show Labels')


let sizeMap = {
    brushSize: {
        '4x4': 4,
        '8x8': 8,
        '16x16': 16,
        '32x32': 32,
    },
}

viewFolder
    .add(sizeMap, 'brushSize', sizeMap.brushSize)
    .setValue(8)
    .onChange((value) => {
        params.brushSize = value
    })
    .name('Brush Size')

viewFolder
    .add(
        {
            x: () => {
                new TWEEN.Tween(controls.target)
                    .to(
                        {
                            x: (regionBounds[1] + regionBounds[0]) / 2,
                            y: (regionBounds[2] + regionBounds[3]) / 2,
                            z: 0,
                        },
                        1000
                    )
                    .easing(TWEEN.Easing.Cubic.Out)
                    .onUpdate(() => {
                        controls.update()
                    })
                    .start()
                new TWEEN.Tween(camera.position)
                    .to(
                        {
                            x: (regionBounds[1] + regionBounds[0]) / 2,
                            y: (regionBounds[2] + regionBounds[3]) / 2,
                            z: 2000,
                        },
                        1000
                    )
                    .easing(TWEEN.Easing.Cubic.Out)
                    .onUpdate(() => {
                        camera.updateProjectionMatrix()
                    })
                    .start()
            },
        },
        'x'
    )
    .name('Reset Camera View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(-500, regionDimensions[1] / 2, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Left View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(regionDimensions[0] + 500, regionDimensions[1] / 2, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Right View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(regionDimensions[0] / 2, regionDimensions[1] + 500, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Top View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(regionDimensions[0] / 2, -500, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Bottom View')

viewFolder.open()
// meshFolder.open()

function segSelect(x: number, y: number, color: string) {
    context!.fillStyle = color
    var value = persDatas[persVal][x + y * regionDimensions[0]]
    var pixels = segsToPixels2[persVal][value]
    for (var i = 0; i < pixels.length; i++) {
        var x = pixels[i] % regionDimensions[0]
        var y = regionDimensions[1] - 1 - Math.floor(pixels[i] / regionDimensions[0])
        if (color == 'clear') {
            context!.clearRect(x, y, 1, 1)
            sessionData.annotatedPixelCount--
        } else {
            context!.fillRect(x, y, 1, 1)
            sessionData.annotatedPixelCount++
        }
    }
    forestMapTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // saugat
}

function connectedSegSelect(x: number, y: number, flood: boolean, clear: boolean) {
    var color = 'blue'
    if (flood) {
        color = 'red'
    }
    if (clear) {
        color = 'clear'
    }
    visited = new Map()
    BFS(x, y, 'BFS_Segment', color)
}

const searchFunction = {
    BFS_Down: {
        E: (x: number, y: number, value: number) => data[persVal][x + 1 + y * regionDimensions[0]] <= value,
        W: (x: number, y: number, value: number) => data[persVal][x - 1 + y * regionDimensions[0]] <= value,
        N: (x: number, y: number, value: number) =>
            data[persVal][x + (y + 1) * regionDimensions[0]] <= value,
        S: (x: number, y: number, value: number) =>
            data[persVal][x + (y - 1) * regionDimensions[0]] <= value,
        EN: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y + 1) * regionDimensions[0]] <= value,
        WN: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y + 1) * regionDimensions[0]] <= value,
        SW: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y - 1) * regionDimensions[0]] <= value,
        SE: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y - 1) * regionDimensions[0]] <= value,
    },
    BFS_Hill: {
        E: (x: number, y: number, value: number) => data[persVal][x + 1 + y * regionDimensions[0]] >= value,
        W: (x: number, y: number, value: number) => data[persVal][x - 1 + y * regionDimensions[0]] >= value,
        N: (x: number, y: number, value: number) =>
            data[persVal][x + (y + 1) * regionDimensions[0]] >= value,
        S: (x: number, y: number, value: number) =>
            data[persVal][x + (y - 1) * regionDimensions[0]] >= value,
        EN: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y + 1) * regionDimensions[0]] >= value,
        WN: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y + 1) * regionDimensions[0]] >= value,
        SW: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y - 1) * regionDimensions[0]] >= value,
        SE: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y - 1) * regionDimensions[0]] >= value,
    },
    BFS_Segment: {
        E: (x: number, y: number, value: number) =>
            persDatas[persVal][x + 1 + y * regionDimensions[0]] == value,
        W: (x: number, y: number, value: number) =>
            persDatas[persVal][x - 1 + y * regionDimensions[0]] == value,
        N: (x: number, y: number, value: number) =>
            persDatas[persVal][x + (y + 1) * regionDimensions[0]] == value,
        S: (x: number, y: number, value: number) =>
            persDatas[persVal][x + (y - 1) * regionDimensions[0]] == value,
        EN: (x: number, y: number, value: number) =>
            persDatas[persVal][x + 1 + (y + 1) * regionDimensions[0]] == value,
        WN: (x: number, y: number, value: number) =>
            persDatas[persVal][x - 1 + (y + 1) * regionDimensions[0]] == value,
        SW: (x: number, y: number, value: number) =>
            persDatas[persVal][x - 1 + (y - 1) * regionDimensions[0]] == value,
        SE: (x: number, y: number, value: number) =>
            persDatas[persVal][x + 1 + (y - 1) * regionDimensions[0]] == value,
    },
}

const valueFunction = {
    BFS_Down: (x: number, y: number) => data[persVal][x + y * regionDimensions[0]],
    BFS_Hill: (x: number, y: number) => data[persVal][x + y * regionDimensions[0]],
    BFS_Segment: (x: number, y: number) =>
        persDatas[persVal][x + y * regionDimensions[0]],
}

const fillFunction = {
    BFS_Down: (x: number, y: number) => [x, y],
    BFS_Hill: (x: number, y: number) => [x, y],
    BFS_Segment: (x: number, y: number) => [x, y],
}

var visited = new Map()
function BFS(x: number, y: number, direction: string, color: string) {
    context!.fillStyle = color
    var stack = []
    visited.set(`${x}, ${y}`, 1)
    stack.push(x, y)
    type ObjectKey = keyof typeof searchFunction
    let _direction = direction as ObjectKey
    while (stack.length > 0) {
        y = stack.pop()!
        x = stack.pop()!
        if (
            x < regionBounds[0] ||
            x > regionBounds[1] ||
            y < regionBounds[2] ||
            y > regionBounds[3]
        ) {
            continue
        }
        let [fillX, fillY] = fillFunction[_direction](x, y)
        if (color == 'clear') {
            sessionData.annotatedPixelCount--
            context!.clearRect(fillX, fillY, 1, 1)
        } else {
            sessionData.annotatedPixelCount++
            context!.fillRect(fillX, fillY, 1, 1)
        }
        var value = valueFunction[_direction](x, y)
        if (searchFunction[_direction].E(x, y, value)) {
            if (!visited.get(`${x + 1}, ${y}`)) {
                visited.set(`${x + 1}, ${y}`, 1)
                stack.push(x + 1, y)
            }
        }
        if (searchFunction[_direction].W(x, y, value)) {
            if (!visited.get(`${x - 1}, ${y}`)) {
                visited.set(`${x - 1}, ${y}`, 1)
                stack.push(x - 1, y)
            }
        }
        if (searchFunction[_direction].N(x, y, value)) {
            if (!visited.get(`${x}, ${y + 1}`)) {
                visited.set(`${x}, ${y + 1}`, 1)
                stack.push(x, y + 1)
            }
        }
        if (searchFunction[_direction].S(x, y, value)) {
            if (!visited.get(`${x}, ${y - 1}`)) {
                visited.set(`${x}, ${y - 1}`, 1)
                stack.push(x, y - 1)
            }
        }
        if (searchFunction[_direction].EN(x, y, value)) {
            if (!visited.get(`${x + 1}, ${y + 1}`)) {
                visited.set(`${x + 1}, ${y + 1}`, 1)
                stack.push(x + 1, y + 1)
            }
        }
        if (searchFunction[_direction].WN(x, y, value)) {
            if (!visited.get(`${x - 1}, ${y + 1}`)) {
                visited.set(`${x - 1}, ${y + 1}`, 1)
                stack.push(x - 1, y + 1)
            }
        }
        if (searchFunction[_direction].SW(x, y, value)) {
            if (!visited.get(`${x - 1}, ${y - 1}`)) {
                visited.set(`${x - 1}, ${y - 1}`, 1)
                stack.push(x - 1, y - 1)
            }
        }
        if (searchFunction[_direction].SE(x, y, value)) {
            if (!visited.get(`${x + 1}, ${y - 1}`)) {
                visited.set(`${x + 1}, ${y - 1}`, 1)
                stack.push(x + 1, y - 1)
            }
        }
    }
    forestMapTexture.needsUpdate = true
    // uniforms.forestMapTexture.value = forestMapTexture;
}

function fpart(x: number) {
    return x - Math.floor(x)
}
function rfpart(x: number) {
    return 1 - fpart(x)
}

const pointer = new THREE.Vector2()
const raycaster = new THREE.Raycaster()
var skip = true
var skipCounter = 0
const onMouseMove = (event: MouseEvent) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1
    if (skipCounter == 4) {
        skip = false
        skipCounter = 0
    } else {
        skipCounter++
    }
}
var polyPoints: Array<number> = []

function performRayCasting() {
    raycaster.setFromCamera(pointer, camera)
    const intersects = raycaster.intersectObjects(scene.children)
    var point = intersects[0].point
    var x = Math.trunc(point.x)
    var y = Math.ceil(point.y)
    return [x, y]
}

function hoverHandler() {
    let [x, y] = performRayCasting()
    y = regionDimensions[1] - 1 - y
    let localId = persDatas[persVal][x + y * regionDimensions[0]]
    uniforms.hoverValue.value = localId
    params.guide = 1
    uniforms.guide.value = params.guide
}

function buttonPressHandlerSuperpixel() {
    params.superpixelKey = 1
    uniforms.superpixelKey.value = params.superpixelKey
}

function buttonPressHandlerPrediction() {
    params.forestMapKey = 1
    uniforms.forestMapKey.value = params.forestMapKey
}

function BFSHandler(x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    visited = new Map()
    var type = 'BFS_Hill'
    var color = 'blue'
    if (flood) {
        type = 'BFS_Down'
        color = 'red'
    }
    if (clear) {
        color = 'clear'
    }
    BFS(x, y, type, color)
    logMyState('f', 'BFS', flood, clear, camera, pointer, x, y, undefined, undefined, time)
}

function brushHandler(key: string, x: number, y: number, forest: boolean, clear: boolean) { // flood means forest, dry means not forest
    sessionData.numberofClick++
    // context!.fillStyle = 'blue'
    // if (forest) {
    //     context!.fillStyle = 'green'
    // }
    context!.fillStyle = 'green'
    labelsContext!.fillStyle = 'green'
    if (clear){
        labelsContext!.fillStyle = 'blue'
    }

    if (clear) {
        context!.clearRect(
            x - Math.floor(params.brushSize / 2),
            y - Math.floor(params.brushSize / 2),
            params.brushSize,
            params.brushSize
        )

        // translate to labels canvas
        // labelsContext!.fillStyle = 'blue'
        labelsContext!.fillRect(
            x - Math.floor(params.brushSize / 2),
            y - Math.floor(params.brushSize / 2),
            params.brushSize,
            params.brushSize
        )
        sessionData.annotatedPixelCount -= params.brushSize * params.brushSize
    } else {
        context!.fillRect(
            x - Math.floor(params.brushSize / 2),
            y - Math.floor(params.brushSize / 2),
            params.brushSize,
            params.brushSize
        )

        // translate to labels canvas
        // labelsContext!.fillStyle = 'green'
        labelsContext!.fillRect(
            x - Math.floor(params.brushSize / 2),
            y - Math.floor(params.brushSize / 2),
            params.brushSize,
            params.brushSize
        )
        sessionData.annotatedPixelCount += params.brushSize * params.brushSize
    }
    forestMapTexture.needsUpdate = true
    labelsTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // saugat
    // uniforms.forestMapTexture.value = forestMapTexture
    logMyState(key, 'brush', forest, clear, camera, pointer, x, y, params.brushSize, undefined, time)
}

function brushLineHandler(linePixels: Array<number>, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    // context!.fillStyle = 'blue'
    // if (forest) {
    //     context!.fillStyle = 'green'
    // }
    context!.fillStyle = 'green'
    labelsContext!.fillStyle = 'green'
    if (clear){
        labelsContext!.fillStyle = 'blue'
    }

    for (var i = 0; i < linePixels.length; i += 2) {
        if (clear) {
            context!.clearRect(
                linePixels[i] - Math.floor(params.brushSize / 2),
                regionDimensions[1] - 1 - linePixels[i + 1] - Math.floor(params.brushSize / 2),
                params.brushSize,
                params.brushSize
            )

            // // translate to labels canvas
            // labelsContext!.fillStyle = 'blue'
            labelsContext!.fillRect(
                linePixels[i] - Math.floor(params.brushSize / 2),
                regionDimensions[1] - 1 - linePixels[i + 1] - Math.floor(params.brushSize / 2),
                params.brushSize,
                params.brushSize
            )
            
            sessionData.annotatedPixelCount -= params.brushSize * params.brushSize
        } else {
            context!.fillRect(
                linePixels[i] - Math.floor(params.brushSize / 2),
                regionDimensions[1] - 1 - linePixels[i + 1] - Math.floor(params.brushSize / 2),
                params.brushSize,
                params.brushSize
            )

            // // translate to labels canvas
            // labelsContext!.fillStyle = 'green'
            labelsContext!.fillRect(
                linePixels[i] - Math.floor(params.brushSize / 2),
                regionDimensions[1] - 1 - linePixels[i + 1] - Math.floor(params.brushSize / 2),
                params.brushSize,
                params.brushSize
            )
            sessionData.annotatedPixelCount += params.brushSize * params.brushSize
        }
    }
    forestMapTexture.needsUpdate = true
    labelsTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // saugat
    logMyState(
        't',
        'brushLine',
        flood,
        clear,
        camera,
        undefined,
        undefined,
        undefined,
        params.brushSize,
        linePixels,
        time
    )
}

function polygonSelectionHandler(x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    context!.fillStyle = 'blue'
    if (flood) {
        context!.fillStyle = 'red'
    }
    if (clear) {
        var cy = polyPoints.pop()!
        var cx = polyPoints.pop()!
        context!.clearRect(cx - 2, cy - 2, 4, 4)
        sessionData.annotatedPixelCount -= 16 //follow this with the line selection to minimize the double counting
    } else {
        polyPoints.push(x, y)
        context!.fillRect(x - 2, y - 2, 4, 4)
        sessionData.annotatedPixelCount += 16 //follow this with the line selection to minimize the double counting
    }
    logMyState(
        'p',
        'polygonSelector',
        flood,
        clear,
        camera,
        pointer,
        x,
        y,
        params.brushSize,
        undefined,
        time
    )
    forestMapTexture.needsUpdate = true
}

function polygonFillHandler(flood: boolean, clear: boolean, linePoints?: Array<number>) {
    sessionData.numberofClick++
    if (linePoints) {
        polyPoints = linePoints
    }
    var type = 'BFS_Hill'
    var color = 'blue'
    if (flood) {
        color = 'red'
        type = 'BFS_Down'
    }
    context!.fillStyle = color
    context!.beginPath()
    context!.moveTo(polyPoints[0], polyPoints[1])
    for (var i = 2; i < polyPoints.length; i += 2) {
        context!.lineTo(polyPoints[i], polyPoints[i + 1])
        context!.clearRect(polyPoints[i] - 2, polyPoints[i + 1] - 2, 4, 4)
    }
    context!.closePath()
    if (clear) {
        color = 'clear'
        context!.globalCompositeOperation = 'destination-out'
        context!.fill()
        // second pass, the actual painting, with the desired color
        context!.globalCompositeOperation = 'source-over'
        context!.fillStyle = 'rgba(0,0,0,0)'
    }
    context!.fill()
    var linePixels: Array<number> = []
    for (var i = 0; i < polyPoints.length; i += 2) {
        var x0 = polyPoints[i]
        var y0 = polyPoints[i + 1]
        var x1, y1
        if (i + 2 == polyPoints.length) {
            x1 = polyPoints[0]
            y1 = polyPoints[1]
        } else {
            x1 = polyPoints[i + 2]
            y1 = polyPoints[i + 3]
        }
        var steep: boolean = Math.abs(y1 - y0) > Math.abs(x1 - x0)
        if (steep) {
            ;[x0, y0] = [y0, x0]
            ;[x1, y1] = [y1, x1]
        }
        if (x0 > x1) {
            ;[x0, x1] = [x1, x0]
            ;[y0, y1] = [y1, y0]
        }
        var dx = x1 - x0
        var dy = y1 - y0
        var gradient
        if (dx == 0) {
            gradient = 1
        } else {
            gradient = dy / dx
        }
        var xend = x0
        var yend = y0
        var xpxl1 = xend
        var ypxl1 = yend
        if (steep) {
            linePixels.push(ypxl1, xpxl1)
            linePixels.push(ypxl1 + 1, xpxl1)
        } else {
            linePixels.push(xpxl1, ypxl1)
            linePixels.push(xpxl1, ypxl1 + 1)
        }
        var intery = yend + gradient
        xend = x1
        yend = y1
        var xpxl2 = xend
        var ypxl2 = yend
        if (steep) {
            linePixels.push(ypxl2, xpxl2)
            linePixels.push(ypxl2 + 1, xpxl2)
        } else {
            linePixels.push(xpxl2, ypxl2)
            linePixels.push(xpxl2, ypxl2 + 1)
        }
        if (steep) {
            for (var x = xpxl1 + 1; x < xpxl2; x++) {
                linePixels.push(Math.floor(intery), x)
                linePixels.push(Math.floor(intery) + 1, x)
                intery = intery + gradient
            }
        } else {
            for (var x = xpxl1 + 1; x < xpxl2; x++) {
                linePixels.push(x, Math.floor(intery))
                linePixels.push(x, Math.floor(intery) + 1)
                intery = intery + gradient
            }
        }
    }
    visited = new Map()
    for (var i = 0; i < linePixels.length; i += 2) {
        BFS(linePixels[i], linePixels[i + 1], type, color)
    }
    logMyState(
        'o',
        'polygonFill',
        flood,
        clear,
        camera,
        undefined,
        undefined,
        undefined,
        undefined,
        polyPoints,
        time
    )
    polyPoints = []
    forestMapTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // saugat
}

function segAnnotationHandler(key: string, x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    var color = 'blue'
    if (flood) {
        color = 'red'
    }
    if (clear) {
        color = 'clear'
    }
    context!.fillStyle = color
    segSelect(x, y, color)
    logMyState(key, 'segmentation', flood, clear, camera, pointer, x, y, undefined, undefined, time)
}

function connectedSegAnnotationHandler(
    key: string,
    x: number,
    y: number,
    flood: boolean,
    clear: boolean
) {
    sessionData.numberofClick++
    connectedSegSelect(x, y, flood, clear)
    logMyState(
        key,
        'connectedSegmentation',
        flood,
        clear,
        camera,
        pointer,
        x,
        y,
        undefined,
        undefined,
        time
    )
}

let [lastX, lastY] = [0, 0]
const onKeyPress = (event: KeyboardEvent) => {
    if (event.key == 'z') {
        var eve
        for (var i = gameState.length - 1; i > 0; i--) {
            if (!gameState[i]['mouseEvent'].undone && !gameState[i]['mouseEvent'].clear) {
                sessionData.numberofUndo++
                gameState[i]['mouseEvent'].undone = true
                eve = gameState[i]['mouseEvent']
                break
            }
        }
        if (eve) {
            eventFunction[eve.label](eve.x, eve.y, eve.flood, !eve.clear, eve.linePoints)
        }
    } else if (event.key == 'r') {
        var eve
        for (var i = gameState.length - 1; i > 0; i--) {
            if (!gameState[i]['mouseEvent'].redone && gameState[i]['mouseEvent'].clear) {
                sessionData.numberofRedo++
                gameState[i]['mouseEvent'].redone = true
                eve = gameState[i]['mouseEvent']
                break
            }
        }
        if (eve) {
            eventFunction[eve.label](eve.x, eve.y, eve.flood, !eve.clear, eve.linePoints)
        }
    }

    if (event.repeat && skip) {
        return
    }
    skip = true

    if (event.key == 'm') {
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('exploration') as HTMLButtonElement).innerHTML = 'Continue ->'
    } 
    // else if (event.key == 'f' && metaState.BFS) {
    //     let [x, y] = performRayCasting()
    //     // console.log("x: ", x)
    //     // console.log("y1: ", y)

    //     y = regionDimensions[1] - 1 - y
    //     // console.log("y2: ", y)
    //     BFSHandler(x, y, params.flood, params.clear)

    //     // const pixelIndex = y * regionDimensions[1] + x
    //     // const pixelVal = confidenceJSON[pixelIndex]
        
    //     // console.log("pixelVal: ", pixelVal)f
    //     // if (pixelVal == 0){
    //     //     y = regionDimensions[1] - 1 - y
    //     //     BFSHandler(x, y, params.flood, params.clear)
    //     // }
    //     // y = regionDimensions[1] - 1 - y
    //     // BFSHandler(x, y, params.flood, params.clear)
    // } 
    else if (event.key == 't' && metaState.brushSelection) {
        let [x, y] = performRayCasting()
        if (
            !(
                x < regionBounds[0] ||
                x > regionBounds[1] ||
                y < regionBounds[2] ||
                y > regionBounds[3]
            )
        ) {
            if (event.repeat) {
                var linePixels = []
                var x0 = lastX
                var y0 = lastY
                var x1 = x
                var y1 = y
                var steep: boolean = Math.abs(y1 - y0) > Math.abs(x1 - x0)
                if (steep) {
                    ;[x0, y0] = [y0, x0]
                    ;[x1, y1] = [y1, x1]
                }
                if (x0 > x1) {
                    ;[x0, x1] = [x1, x0]
                    ;[y0, y1] = [y1, y0]
                }
                var dx = x1 - x0
                var dy = y1 - y0
                var gradient
                if (dx == 0) {
                    gradient = 1
                } else {
                    gradient = dy / dx
                }
                var xend = x0
                var yend = y0
                var xpxl1 = xend
                var ypxl1 = yend
                if (steep) {
                    linePixels.push(ypxl1, xpxl1)
                    linePixels.push(ypxl1 + 1, xpxl1)
                } else {
                    linePixels.push(xpxl1, ypxl1)
                    linePixels.push(xpxl1, ypxl1 + 1)
                }
                var intery = yend + gradient
                xend = x1
                yend = y1
                var xpxl2 = xend
                var ypxl2 = yend
                if (steep) {
                    linePixels.push(ypxl2, xpxl2)
                    linePixels.push(ypxl2 + 1, xpxl2)
                } else {
                    linePixels.push(xpxl2, ypxl2)
                    linePixels.push(xpxl2, ypxl2 + 1)
                }
                if (steep) {
                    for (var z = xpxl1 + 1; z < xpxl2; z++) {
                        linePixels.push(Math.floor(intery), z)
                        linePixels.push(Math.floor(intery) + 1, z)
                        intery = intery + gradient
                    }
                } else {
                    for (var z = xpxl1 + 1; z < xpxl2; z++) {
                        linePixels.push(z, Math.floor(intery))
                        linePixels.push(z, Math.floor(intery) + 1)
                        intery = intery + gradient
                    }
                }
                brushLineHandler(linePixels, params.forest, params.clear)
            }
            lastX = x
            lastY = y
            brushHandler('t', x, regionDimensions[1] - 1 - y, params.forest, params.clear)
        }
    } 
    // else if (event.key == 'p' && metaState.polygonSelection) {
    //     let [x, y] = performRayCasting()
    //     y = regionDimensions[1] - 1 - y
    //     if (
    //         !(
    //             x < regionBounds[0] ||
    //             x > regionBounds[1] ||
    //             y < regionBounds[2] ||
    //             y > regionBounds[3]
    //         )
    //     ) {
    //         // y = regionDimensions[1] - y
    //         polygonSelectionHandler(x, y, params.flood, params.clear)
    //     }
    // } else if (event.key == 'o' && metaState.polygonSelection) {
    //     polygonFillHandler(params.flood, params.clear)
    //     // } else if (event.key == 's' && metaState.segEnabled) {
    //     //     let [x, y] = performRayCasting()
    //     //     segAnnotationHandler('s', x, y, params.flood, params.clear)
    // } else if (event.key == 's' && metaState.segEnabled) {
    //     let [x, y] = performRayCasting()
    //     y = regionDimensions[1] - 1 - y
    //     connectedSegAnnotationHandler('s', x, y, params.flood, params.clear)
    // }
    else if (event.key == ' ') {
        buttonPressHandlerPrediction()
    }
}
const onKeyUp = (event: KeyboardEvent) => {
    // if (event.key == 'g') {
    //     params.guide = 0
    //     uniforms.guide.value = params.guide
    // }
    // else 
    if (event.key == ' '){
        params.forestMapKey = 0
        uniforms.forestMapKey.value = params.forestMapKey
    }
    else if (event.key == 'Enter'){
        params.superpixelKey = 0
        uniforms.superpixelKey.value = params.superpixelKey
    }
}

async function startUp() {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('keydown', onKeyPress)
    window.addEventListener('keyup', onKeyUp)
    document.getElementById('cancel')?.addEventListener('click', () => {
        ;(document.getElementById('uploadForm') as HTMLFormElement).style.display = 'none'
        ;(document.getElementById('download') as HTMLElement).style.display = 'block'
    })
    // // thresholdSlider.addEventListener('input', updateCanvas)
    // document.getElementById('thresholdSlider')?.addEventListener('input', updateCanvas)
}

var diffuseTexture : THREE.Texture
var texContext : CanvasRenderingContext2D
;document.getElementById('submit')!.addEventListener('click', function(e) {
    let student_id = (document.getElementById('studentId') as HTMLInputElement).value
    let testRegion = (document.getElementById('testRegion') as HTMLInputElement).value
    
    if (student_id == ""){
        alert("Please provide your correct student id!")
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'none'
        return
    }
    else if (testRegion == ""){
        alert("Please provide the correct Test Region ID!")
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'none'
        return
    }


    e.preventDefault()
    if ((document.getElementById('stl') as HTMLInputElement).files![0]) {
        let file = (document.getElementById('stl') as HTMLInputElement).files![0]
        if (file.type == "image/png") {
            ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
            let fr = new FileReader()
            fr.onload = async function (e) {
                let image = document.createElement('img')
                image.src = e.target!.result as string
                image.onload = function() { 
                    regionBounds = [0, image.width, 0, image.height]
                    regionDimensions = [image.width, image.height]
                    controls.target = new THREE.Vector3(regionDimensions[0] / 2, regionDimensions[1] / 2, -2000)

                    var texCanvas = document.createElement('canvas')
                    texCanvas.width = image.width
                    texCanvas.height = image.height
                    texContext = texCanvas.getContext('2d')!

                    diffuseTexture = new THREE.Texture(texCanvas)

                    // var annCanvas = document.createElement('canvas')
                    annCanvas.width = image.width
                    annCanvas.height = image.height

                    // var predCanvas = document.createElement('canvas')
                    predCanvas.width = image.width
                    predCanvas.height = image.height
                    predContext = predCanvas.getContext('2d', {willReadFrequently: true})!

                    // var superpixelCanvas = document.createElement('canvas')
                    superpixelCanvas.width = image.width
                    superpixelCanvas.height = image.height
                    superpixelContext = superpixelCanvas.getContext('2d')!

                    // var confidenceCanvas = document.createElement('canvas')
                    confidenceCanvas.width = image.width
                    confidenceCanvas.height = image.height
                    confidenceContext = confidenceCanvas.getContext('2d')!

                    context = annCanvas.getContext('2d')!
                    
                    labelsCanvas.width = image.width
                    labelsCanvas.height = image.height
                    labelsContext = labelsCanvas.getContext('2d')!

                    forestMapTexture = new THREE.Texture(annCanvas)
                    labelsTexture = new THREE.Texture(labelsCanvas)

                    predictionTexture = new THREE.Texture(predCanvas) // saugat
                    superpixelTexture = new THREE.Texture(superpixelCanvas) // saugat
                    confidenceTexture = new THREE.Texture(confidenceCanvas) // saugat

                    uniforms.diffuseTexture.value = diffuseTexture
                    uniforms.forestMapTexture.value = forestMapTexture
                    uniforms.predictionTexture.value = predictionTexture // saugat
                    uniforms.labelsTexture.value = labelsTexture
                    uniforms.superpixelTexture.value = superpixelTexture // saugat
                    uniforms.confidenceTexture.value = confidenceTexture // saugat
                    const meshMaterial = new THREE.RawShaderMaterial({
                        uniforms: uniforms,
                        vertexShader: terrainShader._VS,
                        fragmentShader: terrainShader._FS,
                    })
                    texContext.drawImage(image, 0, 0)

                    // if (!(document.getElementById("topology") as HTMLInputElement).checked) {
                    //     var imageData = texContext!.getImageData(0, 0, image.width, image.height).data
                    //     let temp = []
                    //     for (let i = 0; i < imageData.length; i+=4) {
                    //         temp.push(imageData[i])
                    //     }
                    //     data = {0: temp}
                    // }

                    diffuseTexture.needsUpdate = true
                    forestMapTexture.needsUpdate = true
                    // predictionTexture.needsUpdate = true // saugat
                    uniforms.dimensions.value = [image.width, image.height]
                    var formData = new FormData();
                    formData.append('file', file);
                    ajax({
                        url: `http://127.0.0.1:5005/stl?testRegion=${testRegion}`,
                        type: 'POST',
                        data: formData,
                        processData: false,
                        contentType: false,
                        xhr: function() {
                            var xhr = new XMLHttpRequest()
                            xhr.responseType = 'blob'
                            return xhr
                        },
                        success: async function(data) {
                            const terrainLoader = new STLLoader()
                            try {
                                var test = window.URL.createObjectURL(data)
                                let response: THREE.BufferGeometry = await terrainLoader.loadAsync(
                                    test
                                ) 
                                mesh = new THREE.Mesh(response, meshMaterial)
                                mesh.receiveShadow = true
                                mesh.castShadow = true
                                mesh.position.set(0, 0, -100)
                                scene.add(mesh)
                                
                                testRegion = encodeURIComponent(testRegion);
                                student_id = encodeURIComponent(student_id);

                                const predBuffer = await fetch(`http://127.0.0.1:5005/pred?predict=${1}&testRegion=${testRegion}&taskId=${student_id}&initial=${1}`).then(response => response.arrayBuffer());
                                console.log("arraybuffer: ", predBuffer)

                                // Convert ArrayBuffer to base64
                                const base64ImagePred = arrayBufferToBase64(predBuffer)

                                // Create an Image element
                                const imgPred = new Image();
                                const imgForest = new Image();

                                // Set the source of the Image to the base64-encoded PNG data
                                imgPred.src = 'data:image/png;base64,' + base64ImagePred;
                                imgForest.src = 'data:image/png;base64,' + base64ImagePred;

                                // Wait for the image to load
                                imgPred.onload = () => {

                                    // Set canvas dimensions to match the image dimensions
                                    predCanvas.width = imgPred.width;
                                    predCanvas.height = imgPred.height;

                                    console.log("height: ", predCanvas.height)
                                    console.log("width: ", predCanvas.width)

                                    // Draw the image on the canvas
                                    predContext!.drawImage(imgPred, 0, 0);
                                    predictionTexture.needsUpdate = true // saugat
                                    // console.log("predictionTexture: ", predictionTexture)

                                    predDataOrig = predContext.getImageData(0, 0, predCanvas.width, predCanvas.height);
                                    // console.log("predDataOrig: ", predDataOrig.data);
                                };

                                // Wait for the image to load
                                imgForest.onload = () => {

                                    // Set canvas dimensions to match the image dimensions
                                    annCanvas.width = imgForest.width;
                                    annCanvas.height = imgForest.height;

                                    console.log("height: ", annCanvas.height)
                                    console.log("width: ", annCanvas.width)

                                    context!.drawImage(imgForest, 0, 0);
                                    forestMapTexture.needsUpdate = true
                                };

                                
                            } catch (e) {
                                console.error(`error on reading STL file a.stl`)
                            }                
                            ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
                            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
                        },
                        error: function(xhr, status, error) {
                            ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
                            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
                            console.log('Error uploading file: ' + error)
                        }
                    });
                }
            }
            fr.readAsDataURL(file)
        } else {
            alert('Invalid file type, must be .png!')
        }
    } else {
        alert('No data uploaded!')
    }

    // For topology based segmentation
    // if ((document.getElementById('data') as HTMLInputElement).files![0] && (document.getElementById('topology') as HTMLInputElement).checked) {
    //     let file = (document.getElementById('data') as HTMLInputElement).files![0]
    //     if (file.type == "image/tiff") {
    //         ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
    //         ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
    //         var formData = new FormData();
    //         formData.append('file', file);
    //         ajax({
    //             url: `http://127.0.0.1:5005/topology?testRegion=${testRegion}`,
    //             type: 'POST',
    //             data: formData,
    //             processData: false,
    //             contentType: false,
    //             dataType: 'json',
    //             success: async function(d) {
    //                 console.log(d)
    //                 data = d['data']
    //                 for (var i = 0; i < pers.length; i++) {
    //                     var thresh = pers[i]
    //                     persDatas[thresh] = new Int16Array(d['segmentation'][thresh])
    //                     var max = 0
    //                     var imageData = new Uint8Array(4 * persDatas[thresh].length)
    //                     // segsToPixels2[thresh] = {}
    //                     var imageData2 = new Uint8Array(4 * data[thresh].length)
    //                     for (var x = 0; x < regionDimensions[0]; x++) {
    //                         for (var y = 0; y < regionDimensions[1]; y++) {
    //                             var segID = persDatas[thresh][x + y * regionDimensions[0]]
    //                             if (segID > max) {
    //                                 max = segID
    //                             }
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4] = Math.floor(segID / 1000)
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 1] = Math.floor((segID % 1000) / 100)
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 2] = Math.floor((segID % 100) / 10)
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 3] = segID % 10
    //                             // if (segsToPixels2[thresh][segID]) {
    //                             //     segsToPixels2[thresh][segID].push(x)
    //                             // } else {
    //                             //     segsToPixels2[thresh][segID] = [x]
    //                             // }
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4] = Math.floor(255 * data[thresh][y * regionDimensions[0] + x])
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 1] = Math.floor(255 * data[thresh][y * regionDimensions[0] + x])
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 2] = Math.floor(255 * data[thresh][y * regionDimensions[0] + x])
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 3] = 255
    //                         }
    //                     }
    //                     segsMax[thresh] = max
    //                     persTextures[thresh] = new THREE.DataTexture(
    //                         imageData,
    //                         regionDimensions[0],
    //                         regionDimensions[1]
    //                     )
    //                     persTextures[thresh].needsUpdate = true
    //                     dataTextures[thresh] = new THREE.DataTexture(
    //                         imageData2,
    //                         regionDimensions[0],
    //                         regionDimensions[1]
    //                     )
    //                     dataTextures[thresh].needsUpdate = true
    //                 } 
    //                 uniforms.dataTexture.value = dataTextures[persVal]
    //                 uniforms.persTexture.value = persTextures[persVal]
    //                 uniforms.segsMax.value = segsMax[persVal]
    //                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
    //                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
    //             },
    //             error: function(xhr, status, error) {
    //                 console.log(xhr)
    //                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
    //                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
    //                 console.log('Error uploading file: ' + error)
    //             }
    //         });
    //     } else {
    //         alert('Invalid file type, must be .tiff for data!')
    //     }
    // } 

    if ((document.getElementById('texture') as HTMLInputElement).files![0]) {
        let file = (document.getElementById('texture') as HTMLInputElement).files![0]
        if (file.type == "image/png") {
            ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
            let fr = new FileReader()
            fr.onload = async function (e) {
                let image = document.createElement('img')
                image.src = e.target!.result as string
                image.onload = function() { 
                    texContext!.drawImage(image, 0, 0)
                }
            }
            fr.readAsDataURL(file)
        } else {
            alert('Invalid file type, must be .png!')
        }
    }
})


// // Load static files (elevation and RGB png) from img folder based on regionId
// const elevationUrl = `${host}img/Region_${testRegion}_Elevation.png`;
// console.log("elevationUrl: ", elevationUrl)

// var diffuseTexture : THREE.Texture
// var texContext : CanvasRenderingContext2D

// fetch(elevationUrl)
//   .then(response => {
//     if (!response.ok) {
//       throw new Error(`Failed to fetch: ${elevationUrl}`);
//     }
//     return response.blob();
//   })
//   .then(blob => {
//     ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
//     ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'

//     // Use the blob data (e.g., display the image)
//     const image = document.createElement('img');
//     image.src = URL.createObjectURL(blob);

//     image.onload = function() { 
//         regionBounds = [0, image.width, 0, image.height]
//         regionDimensions = [image.width, image.height]
//         controls.target = new THREE.Vector3(regionDimensions[0] / 2, regionDimensions[1] / 2, -2000)

//         var texCanvas = document.createElement('canvas')
//         texCanvas.width = image.width
//         texCanvas.height = image.height
//         texContext = texCanvas.getContext('2d')!

//         console.log("width: ", image.width)

//         diffuseTexture = new THREE.Texture(texCanvas)

//         // var annCanvas = document.createElement('canvas')
//         annCanvas.width = image.width
//         annCanvas.height = image.height

//         // var predCanvas = document.createElement('canvas')
//         predCanvas.width = image.width
//         predCanvas.height = image.height
//         predContext = predCanvas.getContext('2d', {willReadFrequently: true})!

//         // var superpixelCanvas = document.createElement('canvas')
//         superpixelCanvas.width = image.width
//         superpixelCanvas.height = image.height
//         superpixelContext = superpixelCanvas.getContext('2d')!

//         // var confidenceCanvas = document.createElement('canvas')
//         confidenceCanvas.width = image.width
//         confidenceCanvas.height = image.height
//         confidenceContext = confidenceCanvas.getContext('2d')!

//         context = annCanvas.getContext('2d')!
        
//         labelsCanvas.width = image.width
//         labelsCanvas.height = image.height
//         labelsContext = labelsCanvas.getContext('2d')!

//         forestMapTexture = new THREE.Texture(annCanvas)
//         labelsTexture = new THREE.Texture(labelsCanvas)

//         predictionTexture = new THREE.Texture(predCanvas) // saugat
//         superpixelTexture = new THREE.Texture(superpixelCanvas) // saugat
//         confidenceTexture = new THREE.Texture(confidenceCanvas) // saugat

//         uniforms.diffuseTexture.value = diffuseTexture
//         uniforms.forestMapTexture.value = forestMapTexture
//         uniforms.predictionTexture.value = predictionTexture // saugat
//         uniforms.labelsTexture.value = labelsTexture
//         uniforms.superpixelTexture.value = superpixelTexture // saugat
//         uniforms.confidenceTexture.value = confidenceTexture // saugat
//         const meshMaterial = new THREE.RawShaderMaterial({
//             uniforms: uniforms,
//             vertexShader: terrainShader._VS,
//             fragmentShader: terrainShader._FS,
//         })
//         texContext.drawImage(image, 0, 0)

//         diffuseTexture.needsUpdate = true
//         forestMapTexture.needsUpdate = true
//         // predictionTexture.needsUpdate = true // saugat
//         uniforms.dimensions.value = [image.width, image.height]
//         var formData = new FormData();

//         // const file = new File([''], elevationUrl, { type: 'image/png' });
//         formData.append('file', blob);
//         ajax({
//             // url: `http://127.0.0.1:5005/stl?testRegion=${testRegion}`,
//             url: `http://127.0.0.1:5005/stl`,
//             type: 'POST',
//             data: formData,
//             processData: false,
//             contentType: false,
//             xhr: function() {
//                 var xhr = new XMLHttpRequest()
//                 xhr.responseType = 'blob'
//                 return xhr
//             },
//             success: async function(data) {
//                 const terrainLoader = new STLLoader()
//                 try {
//                     var test = window.URL.createObjectURL(data)
//                     let response: THREE.BufferGeometry = await terrainLoader.loadAsync(
//                         test
//                     ) 
//                     mesh = new THREE.Mesh(response, meshMaterial)
//                     mesh.receiveShadow = true
//                     mesh.castShadow = true
//                     mesh.position.set(0, 0, -100)
//                     scene.add(mesh)

//                     const predBuffer = await fetch(`http://127.0.0.1:5005/pred?predict=${1}&testRegion=${testRegion}&taskId=${student_id}`).then(response => response.arrayBuffer());
//                     console.log("arraybuffer: ", predBuffer)

//                     // Convert ArrayBuffer to base64
//                     const base64ImagePred = arrayBufferToBase64(predBuffer)

//                     // Create an Image element
//                     const imgPred = new Image();
//                     const imgForest = new Image();

//                     // Set the source of the Image to the base64-encoded PNG data
//                     imgPred.src = 'data:image/png;base64,' + base64ImagePred;
//                     imgForest.src = 'data:image/png;base64,' + base64ImagePred;

//                     // Wait for the image to load
//                     imgPred.onload = () => {

//                         // Set canvas dimensions to match the image dimensions
//                         predCanvas.width = imgPred.width;
//                         predCanvas.height = imgPred.height;

//                         console.log("height: ", predCanvas.height)
//                         console.log("width: ", predCanvas.width)

//                         // Draw the image on the canvas
//                         predContext!.drawImage(imgPred, 0, 0);
//                         predictionTexture.needsUpdate = true // saugat
//                         // console.log("predictionTexture: ", predictionTexture)

//                         predDataOrig = predContext.getImageData(0, 0, predCanvas.width, predCanvas.height);
//                         // console.log("predDataOrig: ", predDataOrig.data);
//                     };

//                     // Wait for the image to load
//                     imgForest.onload = () => {

//                         // Set canvas dimensions to match the image dimensions
//                         annCanvas.width = imgForest.width;
//                         annCanvas.height = imgForest.height;

//                         console.log("height: ", annCanvas.height)
//                         console.log("width: ", annCanvas.width)

//                         context!.drawImage(imgForest, 0, 0);
//                         forestMapTexture.needsUpdate = true
//                     };

                    
//                 } catch (e) {
//                     console.error(`error on reading STL file a.stl`)
//                 }                
//                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
//                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
//             },
//             error: function(xhr, status, error) {
//                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
//                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
//                 console.log('Error uploading file: ' + error)
//             }
//         });
//     }

//     // document.body.appendChild(img);
//   })
//   .catch(error => console.error(error));


// const RGBUrl = `${host}img/Region_${testRegion}_RGB.png`;
// console.log("RGBUrl: ", RGBUrl)


// fetch(RGBUrl)
//     .then(response => {
//     if (!response.ok) {
//         throw new Error(`Failed to fetch: ${RGBUrl}`);
//     }
//     return response.blob();
//     })
//     .then(blob => {
//         ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
//         ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'

//         // Use the blob data (e.g., display the image)
//         const rgb = document.createElement('img');
//         rgb.src = URL.createObjectURL(blob);

//         rgb.onload = function() { 
//             texContext!.drawImage(rgb, 0, 0)
//         }
//     })

//     .catch(error => console.error(error));



function arrayBufferToBase64(buffer: ArrayBuffer): string {
    // const binary = String.fromCharCode(...new Uint8Array(buffer));
    // return window.btoa(binary);

    // const binary = Buffer.from(buffer).toString('base64');
    // return binary;

    var base64 = btoa(
        new Uint8Array(buffer)
          .reduce((data, byte) => data + String.fromCharCode(byte), '')
      );
    return base64;
}


function disposeUniform() {
    type ObjectKeyUniforms = keyof typeof uniforms
    for (let key in uniforms) {
        if (uniforms[key as ObjectKeyUniforms]) {
            let x: any = uniforms[key as ObjectKeyUniforms]
            if (x['type'] !== undefined && x['type'] == 't') {
                x['value'].dispose()
                uniforms[key as ObjectKeyUniforms].value = new THREE.Texture()
            }
        }
    }
    for (let key in persTextures) {
        // persTextures[key].dispose()
        persTextures[key] = new THREE.Texture()
        dataTextures[key] = new THREE.Texture()
    }
}

window.addEventListener('resize', onWindowResize, false)
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
    gui.width = window.innerWidth / 5
    render()
}

function animate() {
    requestAnimationFrame(animate)
    stats_mb.update()
    if (camera.position.z <= 100) {
        camera.position.z = 100
        camera.updateProjectionMatrix()
    }
    if (!overRideControl) {
        controls.update()
    }
    TWEEN.update()
    // let position = new THREE.Vector3()
    // camera.getWorldPosition(position)
    render()
}

function render() {
    renderer.render(scene, camera)
}

function getCameraLastStage() {
    return {
        position: camera.position.clone(),
        lookAt: controls.target,
    }
}

animate()

export {
    canvas,
    startUp,
    controls,
    mesh,
    pointer,
    renderer,
    camera,
    TWEEN,
    raycaster,
    scene,
    params,
    uniforms,
    gui,
    disposeUniform,
    annCanvas,
    arrayBufferToBase64,
    superpixelContext,
    superpixelTexture,
    superpixelCanvas,
    predCanvas,
    labelsCanvas,
    predictionTexture,
    predContext,
    labelsContext,
    confidenceCanvas,
    confidenceTexture,
    confidenceContext,
    context, // for forestMap
    forestMapTexture,
    labelsTexture,
    retrainSession,
    // testRegion,
//     student_id
}
