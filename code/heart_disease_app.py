from __future__ import print_function

from flask import Flask, jsonify, request
from sklearn.externals import joblib

import numpy as np
import warnings
import pandas as pd

import argparse
import sys
import os

# pickle location - load this only ONCE to avoid significant
# IO overhead for each call
model_loc = "heart_disease_model.pkl"
if not os.path.exists(model_loc):
    raise ValueError("Model does not exist yet! Expected model to "
                     "exist at %s" % model_loc)
    
# class to hold the model after we load it
class ModelWrapper(object):
    def __init__(self):
        self.model = None
        
# the singleton
wrapper = ModelWrapper()
    
# our app
app = Flask(__name__)
colnames = ['age', 'sex', 'cp', 'trestbps', 'chol', 'cigperday', 
            'fbs', 'famhist', 'restecg', 'thalach', 'exang', 
            'oldpeak', 'slope', 'ca', 'thal']


# Post-processing function
def is_certain_class(predictions, cls=3, proba=0.3):
    # find the row arg maxes (ones that are predicted 'cls')
    argmaxes = predictions.argmax(axis=1)
    
    # get the probas for the cls of interest
    probas = predictions[:, cls]
    
    # boolean mask that becomes our prediction vector
    return ((argmaxes == cls) & (probas >= proba)).astype(int)


# make sure to give it the predict endpoint!
@app.route("/predict", methods=["POST", "GET"])
def predict():
    """This method is routed when a user POSTs a JSON of data. The model
    will be scored and the predictions returned in JSON format.
    """
    # default return if GET
    output = dict(message="Send me a valid POST! I accept JSON " \
                          "data only:\n\n\t{data=[...]}",
                  predictions=None)

    if request.method == 'POST':
        request_json = request.get_json(silent=True)

        try:
            data = pd.DataFrame.from_records(request_json['data'], 
                                             columns=colnames)
        except Exception as ex:
            output['message'] = "Bad data: %s (exception=%s, message=%s)" \
                                % (request_json, ex.__class__.__name__, 
                                   str(ex))

        else:
            # form the json response from predictions
            try:
                preds = is_certain_class(wrapper.model.predict_proba(data)).tolist()
                output['message'] = 'Valid POST (n_samples=%i)' % len(data)
                output['predictions'] = preds
            
            # if the predict fails
            except Exception as ex2:
                output['message'] = "Predict failed (exception=%s, message=%s)" \
                                    % (ex2.__class__.__name__, str(ex2))
    else:
        warnings.warn("Received non-POST: %s" % request.method)

    return jsonify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Launch a Flask app that generates predictions for "
                    "heart_disease via REST")
    
    parser.add_argument('--host', dest='host', type=str,
                        help='The host to launch the app on.')
    
    parser.add_argument('--port', dest='port', type=int,
                        help='The port to launch the app on.')
    
    parser.add_argument('--debug', dest='debug', type=bool, default=True,
                        help='Whether or not to run in debug mode')
    
    parser.add_argument('--threaded', dest='threaded', type=bool, default=True,
                        help='Whether or not to run in threaded mode')
        
    # parse the args out
    args = parser.parse_args()
        
    print("Host:     %s" % args.host)
    print("Port:     %i" % args.port)
    print("Debug:    %r" % args.debug)
    print("Threaded: %r" % args.threaded)
    
    # NOW load the model up
    wrapper.model = joblib.load(model_loc)
    
    # set debug to False in prod
    app.run(host=args.host, port=args.port, 
            debug=args.debug, threaded=args.threaded)
