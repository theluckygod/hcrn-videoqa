from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource

from init_api import load_model, inference_e2e, load_conf, load_data_type_conf
import numpy as np

import tempfile
import skvideo.io
import os
import base64
import time
import threading
from queue import Empty, Queue

BATCH_SIZE = 1
BATCH_TIMEOUT = 0.01
CHECK_INTERVAL = 0.01
requests_queue = Queue()

app = Flask(__name__)
api = Api(app)

app_conf_path = "./app_config.yml"
app_conf = load_conf(app_conf_path)
data_type_conf = load_data_type_conf(app_conf)

DATA_FORMAT_VALUES =  app_conf['data_format_values']

vocab_dict, model_dict = {}, {}
motion_model, appr_model, vocab_dict, model_dict = load_model(app_conf)


def abort_if_config_doesnt_exist(data_type):
    check_data_format = True
    if data_type not in DATA_FORMAT_VALUES:
        check_data_format = False
    if check_data_format == False:
        abort(404, message="Config not match {}".format(
            {"check_data_format": check_data_format}))

parser = reqparse.RequestParser()
parser.add_argument('video', help='Video (.mp4) input -- type=string base64_encoded')
parser.add_argument('data_type', help='Format of dataset -- type=string')
parser.add_argument('ques', help='Question input -- type=string')

def handle_requests():
    """
    This function handle requests.
    """
    while True:
        requests_batch = []
        while not (
            len(requests_batch) > BATCH_SIZE or
            (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        quess = [request['input']['ques'] for request in requests_batch]
        data_types = [request['input']['data_type'] for request in requests_batch]
        video_datas = [request['input']['video'] for request in requests_batch]

        try:
            ques = quess[0]
            data_type = data_types[0]
            video_data = video_datas[0]
            request = requests_batch[0]
            print("\n\n--------------------------")
            video_data = base64.b64decode(video_data)

            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_data)
            video_path = (tfile.name).replace('\\', '/')
            tfile.close()
            os.rename(video_path, video_path + ".mp4")
            video_path = video_path + ".mp4"
            print("Saved a video to", video_path)
        except:
            print("Cannot save video {video_path}!!!")
            continue

        video_data = None
        try:
            video_data = skvideo.io.vread(video_path)
        except:
            print('Cannot read {} as video'.format(video_path))
            continue

        print("Question:", ques)

        if video_data is not None:
            try:
                model = model_dict[data_type]
                vocab = vocab_dict[data_type]
                start_time = time.time()
                data = inference_e2e(data_type_conf[data_type], motion_model, appr_model, vocab, model, video_data, ques)
                print("--- inference time: %s seconds ---" % (time.time() - start_time))
                print("Answer:", data)
                request['output'] = data
            except:
                print("Fail to infer")
                request['output'] = "Fail"
                continue        

threading.Thread(target=handle_requests).start()

# Todo
class Inference(Resource):
    def post(self):
        args = parser.parse_args()
        ques, data_type, video_data = args['ques'], args['data_type'], args['video']
        abort_if_config_doesnt_exist(data_type)
        
        crequest = {'input': args, 'time': time.time()}
        requests_queue.put(crequest)

        
        while 'output' not in crequest:
            time.sleep(CHECK_INTERVAL)

        data = crequest['output']

        if not isinstance(data, str) or (isinstance(data, str) and data == "Fail") :
            return {"answer": None}, 403 # Fail to infer
            
        return {"answer": data}, 201 # infer successfully

##
## Actually setup the Api resource routing here
##
api.add_resource(Inference, '/inference')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)