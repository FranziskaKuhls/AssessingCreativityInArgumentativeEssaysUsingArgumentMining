import json
from AclEvaluator import AclEvaluator
import os.path
from pydantic import BaseModel
from typing import List
from flask import Flask, request, Response
from flask_cors import cross_origin


class Source(BaseModel):
    text: str
    dataset: str = "climate_change"


class Answer(BaseModel):
    fluency: float
    flexibility: float
    originality: float
    topics: List[int]


app = Flask(__name__)

origins = [
    "*"
]


@app.route('/creativity', methods=['POST'])
@cross_origin()
def creativity():
    if request.method == 'POST':
        text = request.json['text']
        fluency, flexibility, originality, topics = acl_evaluator.evaluate_essay(text)

        response = {}
        response['fluency'] = fluency
        response['flexability'] = flexibility
        response['originality'] = originality
        response['topics'] = str(topics)
        response = json.dumps(response)
        return Response(response, status=200)


@app.route('/models', methods=['GET'])
@cross_origin()
def load_models():
    # check if file exists
    if os.path.isfile('/model/TopicModel/topic_model_argumentative_sentences_new_version'):
        response = "Files exists"
    else:
        response = "Topic Modell does not exist!"
    return response


if __name__ == '__main__':
    acl_evaluator = AclEvaluator()
    app.run(host='0.0.0.0', port=5001)
