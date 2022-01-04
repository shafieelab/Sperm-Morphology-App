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