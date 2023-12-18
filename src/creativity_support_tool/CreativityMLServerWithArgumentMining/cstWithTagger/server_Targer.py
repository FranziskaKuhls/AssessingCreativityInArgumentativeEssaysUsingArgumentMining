import json
from TargerEvaluator import TargerEvaluator
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
        essay = request.json['text']
        essay_tags = request.json['tags']
        fluency, flexibility, originality, topics = acl_evaluator.evaluate_essay(essay, essay_tags)

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
    if os.path.isfile('/model/TopicModel/topic_model_with_targer'):
        response = "Files exists"
    else:
        response = "Topic Modell does not exist!"
    return response


if __name__ == '__main__':
    acl_evaluator = TargerEvaluator()
    app.run(host='0.0.0.0', port=5002)
