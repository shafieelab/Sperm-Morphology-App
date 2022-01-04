import csv
import json
import os
from os.path import isdir

from flask import Flask, request, jsonify, send_file

from core.MultiTemplateMatching_smartphone import extract_slides
from core.test_md_nets import run_md_nets
from utils import generate_slides_txt_phone
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app)

root_dir = "/var/www/html/Sperm_Morphology_App/data/"
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif', 'tiff'])


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/run_sperm_morph', methods=['POST'])
def run_sm_model():  # e2e run
    input_data = request.json
    run_id = input_data['run_id']
    print("run_id", run_id)
    # run_id = "2"

    # folder_name = os.listdir(root_dir + run_id + "/")
    folder_name = sorted([fdr for fdr in os.listdir(root_dir + run_id + "/") if isdir(root_dir + run_id + "/" + fdr)],
                    key=lambda x: len(root_dir + run_id + "/"+x))[0]
    print("folder_name", folder_name)
    extract_slides(root_dir, run_id,folder_name)
    txt_img_files_path = generate_slides_txt_phone(root_dir,run_id)
    # output_csv_path = run_md_nets(project_root=root_dir, run_id=run_id, img_paths_file='data/2/2.txt')
    run_md_nets(project_root=root_dir,run_id=run_id,img_paths_file=txt_img_files_path)
    return jsonify({"status": "success"})


def read_csv(csvFilePath):
    # create a dictionary
    data = []

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:
            # Assuming a column named 'No' to
            # be the primary key
            # key = rows['Slide_name']
            data.append(rows)

    return data


@app.route("/get_all_run_ids", methods=['POST'])  # Just to check the server is running
def get_all_run_ids():
    # input_data = request.json
    # run_id = input_data['run_id']

    print("index", "=" * 50)
    # slides = os.listdir(root_dir + run_id + "/" + "slides_heads")

    run_ids = os.listdir(root_dir)

    run_ids = sorted(run_ids,key= lambda x: x.split("___")[0].replace("Run-",""))
    if len(run_ids) == 0:
        return jsonify({"run_ids": []})
    user_data = ""
    for user in run_ids:
        user_data += user + ";"

    return jsonify({"run_ids": user_data[:-1]})


@app.route("/get_run_results", methods=['POST'])  # Just to check the server is running
def get_run_results():
    input_data = request.json
    run_id = input_data['run_id']

    csv_file_path = root_dir + run_id + "/logs/sperm/Xception/1_test_a_sd4_to_a_sd1_f/slide_prediction_sperm.csv"
    data = read_csv(csv_file_path)
    print("index", "=" * 50)
    # slides = os.listdir(root_dir + run_id + "/" + "slides_heads")

    return jsonify({"run_id_results": data})


@app.route("/single_file_upload", methods=['POST'])  # Just to check the server is running
def single_file_upload():
    print(' * received form with', list(request.form.items()))
    print(' * received files with', list(request.files.items()))
    # print(request.files)
    run_id = request.form['run_id']

    # check if the post request has the file part
    for file_name,file in request.files.items():

        if file and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:

            # filename = secure_filename(file.filename)
            os.makedirs(os.path.dirname(os.path.join(root_dir,run_id, file_name)), exist_ok=True)
            # print(file_name,file.filename,filename)
            file.save(os.path.join(root_dir,run_id, file_name))
            # file.save(os.path.join(root_dir+"/"+run_id+], filename))
            # print(' * file uploaded', filename)
    return 'uploaded successfully'


@app.route("/download_run_results", methods=['POST'])  # Just to check the server is running
def download_run_results():
    input_data = request.json
    run_id = input_data['run_id']

    csv_file_path = root_dir + run_id + "/logs/sperm/Xception/1_test_a_sd4_to_a_sd1_f/slide_prediction_sperm.csv"
    print("index", "=" * 50)
    # slides = os.listdir(root_dir + run_id + "/" + "slides_heads")

    return send_file(csv_file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
    # run_sm_model()
