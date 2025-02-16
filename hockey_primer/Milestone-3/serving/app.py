"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""

import os
import json
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import numpy as np

import comet_ml
from comet_ml.exceptions import CometRestApiException

# import ift6758

COMET_API_KEY = os.environ.get("COMET_API_KEY","YOpcMk2b4epnXXRcSzewfCSwg")
COMET_API_KEY = "YOpcMk2b4epnXXRcSzewfCSwg"
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
DEFAULT_WORKSPACE = os.environ.get("DEFAULT_WORKSPACE", 'ift6758-milestone2-team07')
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", 'simple_both')

app = Flask(__name__)

feats_catalogue = {
    "simple_dist": ['Distance_from_net'],
    "simple_angle": ['angle_from_net'],
    "simple_both": ['Distance_from_net', 'angle_from_net']
}

# setup basic logging configuration
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# any other initialization before the first request (e.g. load default model)
logging.info(f'Loading default model {DEFAULT_MODEL}')

# logging.info(f'Loading comet_ml api key {COMET_API_KEY}')
api = comet_ml.api.API(api_key=COMET_API_KEY)

model_path = f'models/{DEFAULT_MODEL}'
model_name = DEFAULT_MODEL

#model_file = os.listdir(model_path)[0]
#model = joblib.load(os.path.join(model_path, model_file))


def set_model(_model_name, workspace):

    global model_name, model

    model_path = f'models/{_model_name}'

    message = ''
    success = False
    if _model_name in feats_catalogue:
        if not os.path.exists(model_path):

            try:
                api.download_registry_model(
                    workspace=workspace,
                    registry_name=_model_name,
                    output_path=model_path
                )

            except CometRestApiException:
                app.logger.info(f'comet_ml api exception: couldnt load model {_model_name}')
                message = f'comet_ml api exception: couldnt load model {_model_name}'

        if os.path.exists(model_path):
            model_name = _model_name
            model_file = os.listdir(model_path)[0]
            model = joblib.load(os.path.join(model_path, model_file))

            success = True
            app.logger.info(f'loaded model {model_name}')
            message = f'loaded model {model_name}'
    else:
        app.logger.info(f'non available model {_model_name}')
        message = f'non available model {_model_name}'

    return success, message


set_model(DEFAULT_MODEL, DEFAULT_WORKSPACE)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """

    # DOESNT WORK IN NEWER VERSIONS OF FLASK

    pass


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    with open(LOG_FILE, 'r') as f:
        st = f.read().split('\n')

    st = '\n'.join(st[-100:])

    response = {'logs': st}
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data

    req = request.get_json()

    app.logger.info(req)

    # check to see if the model you are querying for is already downloaded

    #api.get_registry_model_names(workspace='ift6758-milestone2-team07')

    global model_name, model

    if 'model_name' not in req or 'workspace' not in req:
        return jsonify({'error_msg': 'expected input has model_name and workspace'})

    success, message = set_model(req['model_name'], req['workspace'])

    response = {
        'success': success,
        'current_model': model_name,
        'current_model_features': feats_catalogue[model_name],
        'message': message
    }

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    req = request.get_json()
    app.logger.info(req)

    global model

    usage = 'expected input: {"features": [{"<feat_1>": <val_1>, "<feat_2>": <val_2>, ...}, {"<feat_1>": <val_1>, "<feat_2>": <val_2>, ...}] }'

    if 'features' not in req or (not isinstance(req['features'], list)) or (len(req['features']) == 0) or (not isinstance(req['features'][0], dict)):
        app.logger.info(f'bad input {req}')
        return jsonify({'error_msg': usage})
    
    x = np.array([[sample[t] for t in feats_catalogue[model_name]] for sample in req['features']])

    pred = model.predict_proba(pd.DataFrame(x, columns=feats_catalogue[model_name]))

    response = {'predicted': pred.tolist()}

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
