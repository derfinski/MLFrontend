from flask import Flask, redirect


import base64

import io

from flask import request

import trimesh
import os

import visualkeras as vk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt

from flask import send_file, render_template
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = set(["off"])
DATA_DIR = '/content/drive/MyDrive/raw'
PARSED_DIR = './parsed/'
MODEL_DIR = './3dModel/'
NUM_POINTS = 2048
NUM_CLASSES = 10
FRONTEND_DIR = 'app.html'
CLASS_DIR = 'class.html'
UPLOAD_DIR = './uploads/'

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))
values = ["sofa", "monitor", "desk", "bed", "bathtub", "nightstand", "chair", "toilet", "dresser", "table"]
current_prediction = []


@app.route("/")
def home():
    return redirect("/static/app.html", 302)


def get_model_weights_pointnet(path):
    model = os.path.join(MODEL_DIR, path)
    print("[+] model loaded")
    return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    # The decay rate is initialized as 0.5 and gradually increased to 0.99 in the paper
    # Somehow, decay rate of 0.0 works best here
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)


# Dense -> BatchNormalization -> ReLU block
def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        # this step performs orthogonal transformation followed by Frobenius norm
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="Identity",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    # Reshape feature transform
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def get_pointnet(weights):
    inputs = keras.Input(shape=(NUM_POINTS, 3))
    # T-net 1 with 3x3 output matrix
    x = tnet(inputs, 3)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    # T-net 2 with 64x64 output matrix
    x = tnet(x, 64)
    x = conv_bn(x, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()
    model.load_weights(weights)
    return model


models = [get_pointnet(get_model_weights_pointnet("model_raw_untrained.keras")),
          get_pointnet(get_model_weights_pointnet("model_raw_trained.keras"))]
models_names = ["pointnet untrained", "pointnet trained"]

model = models[0]

def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    mesh = trimesh.load(decoded)
    points = mesh.sample(2048)
    return points


@app.route('/plot.png')
def plot_png():
    plt.cla()
    mesh = trimesh.load(os.path.join(UPLOAD_DIR, 'file.off'))
    points = mesh.sample(2048)
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 'red')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/barchart.png')
def plot_bar():
    global current_prediction
    plt.cla()



    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.bar(values, list(current_prediction))
    ax1.set_xticklabels(values, rotation=65)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route("/static/app.html", methods=["GET", "POST"])
def frontend():
    upload = False
    predicted_object = ""
    if("upload" in request.args.keys()):
        upload = True if request.args["upload"] == "true" else False
        if upload:
            global current_prediction
            mesh = trimesh.load(os.path.join(UPLOAD_DIR, 'file.off'))
            points = mesh.sample(2048)
            points = np.expand_dims(points, axis=0)
            #print(points.shape)
            predict = model.predict(points)
            current_prediction = predict[0]

            prediction_index = np.argmax(current_prediction)

            predicted_object = values[prediction_index]

    return render_template(FRONTEND_DIR, models=models_names, upload = upload, name = predicted_object)


@app.route("/class")
def classify():
    global current_prediction

    mesh = trimesh.load(os.path.join(UPLOAD_DIR, 'file.off'))
    points = mesh.sample(2048)
    points = np.expand_dims(points, axis=0)
    #print(points.shape)
    predict = model.predict(points)
    current_prediction = predict[0]
    return render_template(CLASS_DIR, predict=predict)



@app.route("/model.png")
def showModel():
    global model
    #keras.utils.plot_model(model = model, to_file="model.png", dpi=400)
    vk.layered_view(model, to_file='model.png')
    return send_file("model.png", mimetype='image/png')

@app.route("/upload", methods=["POST"])
def upload_file():
    global model

    if request.method == "POST":
        #print(request.form)
        if not "model" in request.form.keys():
            return "No Model selected"
        model_index = models_names.index(request.form["model"])
        model = models[model_index]

        if not "file" in request.files:
            return "No file part in the form."
        f = request.files["file"]
        if f.filename == "":
            return "No file selected."
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(UPLOAD_DIR, "file.off"))
            return redirect("/static/app.html?upload=true", code=302)
        return "File not allowed."
    return "Upload file route"


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port = 80)
