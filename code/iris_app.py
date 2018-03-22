from __future__ import print_function

from flask import Flask, jsonify, request
from sklearn.externals import joblib

import numpy as np

import sys
import os

# pickle location - load this only ONCE to avoid significant
# IO overhead for each call
model_loc = "iris.pkl"
if not os.path.exists(model_loc):
    raise ValueError("Model does not exist yet!")
model = joblib.load(model_loc)
    
# our app
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
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
        except Exception as ex:
            msg = "Bad data: %s (exception=%s, message=%s)" \
                  % (request_json, ex.__class__.__name__, str(ex))

        else:
            # form the json response from predictions
            try:
                preds = model.predict_proba(data).tolist()
                classes = [int(np.argmax(probas)) for probas in preds]
                
                return jsonify(dict(probabilities=preds,
                                    predicted_class=classes,
                                    message="Valid POST"))
            # if the predict fails
            except Exception as ex2:
                msg = "Predict failed (exception=%s, message=%s)" \
                      % (ex2.__class__.__name__, str(ex2))

    # if we get to this point, it was a GET or an invalid post
    output = dict(message=msg,
                  probabilities=None,
                  predicted_class=None)

    return jsonify(output)


if __name__ == '__main__':
    # this is a very rudimentary method for getting the args
    if len(sys.argv) != 3:
        raise OSError("Usage: python iris_app.py <host> <port>")
        
    host, port = sys.argv[1], sys.argv[2]
    try:
        port = int(port)
    except:
        raise ValueError("Port must be an int! Got %s" % port)
        
    print("Host: %s" % host)
    print("Port: %i" % port)
    
    # set debug to False in prod
    app.run(host=host, port=port, debug=True, threaded=True)
