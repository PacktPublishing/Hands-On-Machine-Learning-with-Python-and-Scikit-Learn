# -*- coding: utf-8 -*-

from __future__ import print_function

from flask import Flask, jsonify, request
from sklearn.externals import joblib

import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import logging

import sys
import os


# set up logger
def get_logger():
    logger = logging.getLogger("WineAPI")
    logger.setLevel(logging.INFO)
    lformat = "%(asctime)s - %(levelname)s - %(module)s - "\
              "%(funcName)s - %(message)s"

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(lformat))

    logger.addHandler(handler)
    return logger


# set top level
logger = get_logger()

# pickle location - load this only ONCE to avoid significant
# IO overhead for each call
model_loc = "model.pkl"
if not os.path.exists(model_loc):
    raise ValueError("Model does not exist yet!")
model = joblib.load(model_loc)
logger.info("Loaded model from %s" % model_loc)
    
# our app
app = Flask(__name__)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    """This method is routed when a user POSTs a JSON of data. The model
    will be scored and the predictions returned in JSON format.
    """
    # default msg if GET
    msg = "Send me a valid POST"
    if request.method == 'POST':
        request_json = request.get_json()

        try:
            data = request_json['data']
            logger.info("%i samples passed" % len(data))
        except Exception as ex:
            msg = "Bad data: %s (exception=%s, message=%s)" \
                  % (request_json, ex.__class__.__name__, str(ex))

        else:
            # form the json response from predictions
            try:
                preds = model.predict(data).tolist()
                return jsonify(dict(predictions=preds,
                                    message="Valid POST"))
            # if the predict fails
            except Exception as ex2:
                msg = "Predict failed (exception=%s, message=%s)" \
                      % (ex2.__class__.__name__, str(ex2))

    # if we get to this point, it was a GET or an invalid post
    output = dict(message=msg,
                  predictions=None)

    return jsonify(output)


@app.route("/test", methods=["GET"])
def test():
    """Test if the app is live"""
    if request.method == 'POST':
        msg = "I don't want your stinkin' data! I only GET!"
    else:
        msg = "App is live!"
    return jsonify(dict(message=msg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate predictions for wine quality. This depends on "
                    "the existence of a model\npickled into 'model.pkl'. If "
                    "not found, this will raise an error and exit.",
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('--host', dest='host', type=str,
                        help='The host on which to launch the Flask app. '
                             'Default is "localhost"')

    parser.add_argument('--port', dest='port', type=int,
                        help='The port for the Flask app. Default is 5000.')

    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='The port for the Flask app. Default is 5000.')

    parser.set_defaults(host='localhost', port=5000, debug=False)
        
    # parse
    args = parser.parse_args()
    host = args.host
    port = args.port
    debug = args.debug

    logger.info("Host: %s" % host)
    logger.info("Port: %i" % port)
    logger.info("Debug: %r" % debug)
    
    # set debug to False in prod
    app.run(host=host, port=port, debug=debug, 
            threaded=True)
