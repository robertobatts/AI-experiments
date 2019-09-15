from flask import Flask, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask

app = Flask(__name__)
model = load_model('diabete_model')
graph = tf.get_default_graph()

# request model prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'GET':
    a = request.args['a']
    b = request.args['b']
    c = request.args['c']
    d = request.args['d']
    e = request.args['e']
    f = request.args['f']
    g = request.args['g']
    h = request.args['h']
  elif request.method == 'POST':
    a = request.args['a']
    b = request.args['b']
    c = request.args['c']
    d = request.args['d']
    e = request.args['e']
    f = request.args['f']
    g = request.args['g']
    h = request.args['h']

  with graph.as_default():
    result = model.predict_classes(np.array([[a, b, c, d, e, f, g, h]]))[0].tolist()
    data = {'result': result}
    return flask.jsonify(data)


# start Flask server
app.run(host='0.0.0.0', port=5000, debug=False)
