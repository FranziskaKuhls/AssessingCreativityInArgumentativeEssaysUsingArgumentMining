from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from transformers import BertTokenizer
import torch
import numpy as np

from typing import List


class Classificator:
    def __init__(self, model_path):
        self.num_labels = 3
        self.model_path = model_path
        self.label_list = ["NoArgument", "Argument_against", "Argument_for"]
        self.max_seq_length = 64
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, do_lower_case=True, max_length=self.max_seq_length)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=self.num_labels)

    def classify(self, sentence: str) -> List[str]:
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.detach().cpu().numpy()
        predicted_labels = []
        for prediction in np.argmax(logits, axis=1):
            predicted_labels.append(self.label_list[prediction])

        return predicted_labels

    def get_argumentative_sentences_from_sentence_list(self, sentence_list: List[str]) -> List[str]:
        argumentative_sentences_list = []
        for sentence in sentence_list:
            label = self.classify(sentence)
            if "NoArgument" not in label:
                argumentative_sentences_list.append(sentence)
        return argumentative_sentences_list


