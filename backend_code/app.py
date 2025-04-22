from flask import jsonify, Flask, send_file, request, make_response
import os
import numpy as np
from flask_cors import CORS
import subprocess

from al import train, run_prediction


app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "static/files"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, app.config["UPLOAD_FOLDER"])


@app.route('/stl', methods=['POST'])
def stl():
    TEST_REGION = int(request.args.get('testRegion', 1))
    print("TEST_REGION: ", TEST_REGION)
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)

        base_path = "/Users/yudai/Documents/iu-coding/hmm/burn_scars_AL_code/backend_code/"
        hmm_path = os.path.join(base_path, "hmm")
        stl_path = os.path.join(base_path, f"stl/Region_{TEST_REGION}.stl")
        subprocess.check_output([hmm_path, f.filename, stl_path, '-z', '500', '-t', '10000000'])
        print("testRegion: ", TEST_REGION)
        payload = make_response(send_file(stl_path))
        payload.headers.add('Access-Control-Allow-Origin', '*')
        # os.remove('a.stl')
        # os.remove(f.filename)

        return payload


@app.route('/pred', methods=['GET'])
def pred():
    student_id = request.args.get('taskId')
    predict = int(request.args.get('predict', 0))
    TEST_REGION = int(request.args.get('testRegion', 1))
    initial = int(request.args.get('initial', 0))

    # to handle the situation where a user does AL for a while, terminated and starts again (all the process has to be completed in 1 go, there can be no break in between)
    if initial:
        try:
            with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'w') as file:
                file.write(str(0))
        except FileNotFoundError:
            pass

    if predict:
        run_prediction(TEST_REGION, student_id)
        payload = make_response(send_file(f'./users/{student_id}/output/R{TEST_REGION}_pred_test.png'))
    else:
        payload = make_response(send_file(f'./users/{student_id}/output/R{TEST_REGION}_pred_test.png'))
    
    payload.headers.add('Access-Control-Allow-Origin', '*')
    return payload


@app.route('/retrain', methods=['POST'])
def retrain():
    student_id = request.args.get('taskId')
    file = request.files.get('image')
    file_2 = request.files.get('image_2')
    TEST_REGION = int(request.args.get('testRegion', 1))

    if not os.path.exists(f"./users/{student_id}"):
        os.mkdir(f"./users/{student_id}")

    if not os.path.exists(f"./users/{student_id}/output"):
        os.mkdir(f"./users/{student_id}/output")
    
    # read cycle from txt file
    try:
        with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'r') as fp:
            content = fp.read()
            al_cycle = int(content) 
    except FileNotFoundError:
        al_cycle = 0

    if file and file_2:
        print('images are here')
        # file = request.files['image']

        # Process the file as needed, for example, save it to the server
        file.save(f'./users/{student_id}/output/R{TEST_REGION}_labels.png')

        file_2.save(f'./users/{student_id}/output/R{TEST_REGION}_pred_label.png')

        train(TEST_REGION, student_id, al_cycle)

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

    
if __name__ == '__main__':
   app.run()