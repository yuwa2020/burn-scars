from flask import jsonify, Flask, send_file, request, make_response
from werkzeug.utils import secure_filename
import os
from PIL import Image
import gzip
import numpy as np
from flask_cors import CORS

# from vtkmodules.all import (
#     vtkTIFFReader,
#     vtkImageExtractComponents
# )

import cv2
import gc
# from vtkmodules.util.numpy_support import vtk_to_numpy
import json
import subprocess, sys

# from topologytoolkit import (
#     ttkFTMTree,
#     ttkTopologicalSimplificationByPersistence,
#     ttkScalarFieldSmoother
# )

from al import train, run_prediction


app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "static/files"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, app.config["UPLOAD_FOLDER"])


# extractComponent = vtkImageExtractComponents()
# extractComponent.SetInputConnection(pread.GetOutputPort())
# extractComponent.SetComponents(0)
# extractComponent.Update()

# smoother = ttkScalarFieldSmoother()
# smoother.SetInputConnection(0, pread.GetOutputPort())
# smoother.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
# smoother.SetNumberOfIterations(5)
# smoother.Update()

# simplify = ttkTopologicalSimplificationByPersistence()
# simplify.SetInputConnection(0, smoother.GetOutputPort())
# simplify.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
# simplify.SetThresholdIsAbsolute(False)
# simplify.SetPersistenceThreshold(50)
# simplify.Update()

# tree = ttkFTMTree()
# tree.SetInputConnection(0, simplify.GetOutputPort())
# tree.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
# tree.SetTreeType(2)
# tree.SetWithSegmentation(1)
# tree.Update()

@app.route('/stl', methods=['POST'])
def stl():
    TEST_REGION = int(request.args.get('testRegion', 1))
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
#         subprocess.check_output(['./hmm', f.filename, 'a.stl', '-z', '500', '-t', '10000000'])
        payload = make_response(send_file(f'Region_{TEST_REGION}.stl'))
        payload.headers.add('Access-Control-Allow-Origin', '*')
        # os.remove('a.stl')
        # os.remove(f.filename)

        return payload


@app.route('/pred')
def pred():
    student_id = request.args.get('taskId')
    predict = int(request.args.get('predict', 0))
    TEST_REGION = int(request.args.get('testRegion', 1))

    if predict:
        run_prediction(TEST_REGION, student_id)
        payload = make_response(send_file(f'R{TEST_REGION}_pred_test.png'))
    else:
        payload = make_response(send_file(f'R{TEST_REGION}_pred_test.png'))
    
    payload.headers.add('Access-Control-Allow-Origin', '*')
    return payload


@app.route('/retrain', methods=['POST'])
def retrain():
    student_id = request.args.get('taskId')
    file = request.files.get('image')
    TEST_REGION = int(request.args.get('testRegion', 1))

    if file:
        print('image is here')
        # file = request.files['image']

        # Process the file as needed, for example, save it to the server
        file.save(f'./R{TEST_REGION}_labels.png')

        train(TEST_REGION, student_id)

        payload = make_response(jsonify({'status': 'success', 'taskId': student_id}), 200)
        payload.headers.add('Access-Control-Allow-Origin', '*')

        with open(f"./status_{student_id}.txt", 'w') as file:
            file.write("completed")

        return payload

    payload = make_response(jsonify({'status': 'error', 'taskId': student_id}), 400)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    with open(f"./status_{student_id}.txt", 'w') as file:
        file.write("error")

    return payload


@app.route('/check-status', methods=['GET'])
def check_status():
    student_id = request.args.get('taskId')
    TEST_REGION = int(request.args.get('testRegion', 1))
    print("student_id: ", student_id)

    # logic to check the status of the task
    with open(f"./status_{student_id}.txt", 'r') as file:
        status = file.read()
    
    print("status: ", status)

    payload = make_response(jsonify({'status': status}), 200)
    payload.headers.add('Access-Control-Allow-Origin', '*')

    return payload



# @app.route('/topology', methods=['POST'])
# def topology():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(f.filename)
#         pread = vtkTIFFReader()
#         pread.SetFileName(f.filename)
#         extractComponent = vtkImageExtractComponents()
#         extractComponent.SetInputConnection(pread.GetOutputPort())
#         extractComponent.SetComponents(0)
#         extractComponent.Update()
#         simplify = ttkTopologicalSimplificationByPersistence()
#         simplify.SetInputConnection(0, extractComponent.GetOutputPort())
#         simplify.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
#         simplify.SetThresholdIsAbsolute(False)
#         tree = ttkFTMTree()
#         tree.SetInputConnection(0, simplify.GetOutputPort())
#         tree.SetInputArrayToProcess(0, 0, 0, 0, "Tiff Scalars")
#         tree.SetTreeType(2)
#         tree.SetWithSegmentation(1)
#         response = {'data': {}, 'segmentation': {}}
#         dmax = 1
#         dmin = 0
#         for i in [0, 0.02, 0.04, 0.08, 0.16, 0.32]:
#             simplify.SetPersistenceThreshold(i)
#             simplify.Update()
#             tree.Update()
#             if i == 0:
#                 dmax = np.max(vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0)))
#                 dmin = np.min(vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0)))
#                 response['data'][i] = ((vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0)) - dmin) / (dmax - dmin)).tolist()
#             else:
#                 response['data'][i] = (vtk_to_numpy(simplify.GetOutput().GetPointData().GetArray(0))).tolist()
#             response['segmentation'][i] = vtk_to_numpy(tree.GetOutput(2).GetPointData().GetArray(2)).tolist()
#         content = gzip.compress(json.dumps(response).encode('utf8'), 9)
#         payload = make_response(content)
#         payload.headers.add('Access-Control-Allow-Origin', '*')
#         payload.headers['Content-length'] = len(content)
#         payload.headers['Content-Encoding'] = 'gzip'
#         del(response)
#         os.remove(f.filename)
#         return payload
    
# @app.route('/test', methods=['POST'])
# def test():
#     response = {"success": "success"}
#     # ranges = [0.02, 0.04, 0.06, 0.08, 0.1]
#     # ranges = [0.02]
#     # for i in ranges:
#     #     simplify.SetPersistenceThreshold(i)
#     #     simplify.Update()
#     #     tree.Update()
#     #     test = vtk_to_numpy(tree.GetOutput(2).GetPointData().GetArray(2))
#     #     response[i] = {'array': test.tolist(), "max": int(np.max(test))}
#     #     del test
#     payload = jsonify(response)
#     payload.headers.add('Access-Control-Allow-Origin', '*')
#     return payload
    
if __name__ == '__main__':
   app.run()