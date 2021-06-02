from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource

from init_api import load_model, inference_e2e, load_conf, load_data_type_conf
import numpy as np

import tempfile
import skvideo.io
import os
import base64
import time

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


# Todo
class Inference(Resource):
    def post(self):
        print("\n\n--------------------------")

        args = parser.parse_args()
        ques, data_type, video_data = args['ques'], args['data_type'], args['video']
        video_data = base64.b64decode(video_data)

        abort_if_config_doesnt_exist(data_type)

        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_data)
        video_path = (tfile.name).replace('\\', '/')
        tfile.close()
        os.rename(video_path, video_path + ".mp4")
        video_path = video_path + ".mp4"
        print("Saved a video to", video_path)

        video_data = None
        try:
            video_data = skvideo.io.vread(video_path)
        except:
            print('Cannot read {} as video'.format(video_path))

        print("Question:", ques)

        if video_data is not None:
            try:
                model = model_dict[data_type]
                vocab = vocab_dict[data_type]
                start_time = time.time()
                data = inference_e2e(data_type_conf[data_type], motion_model, appr_model, vocab, model, video_data, ques)
                print("--- inference time: %s seconds ---" % (time.time() - start_time))
                print("Answer:", data)
                return {"answer": data}, 201
            except:
                print("Fail to infer")
                return None, 403

        print("Something goes wrong")
        return None, 403


##
## Actually setup the Api resource routing here
##
api.add_resource(Inference, '/inference')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)