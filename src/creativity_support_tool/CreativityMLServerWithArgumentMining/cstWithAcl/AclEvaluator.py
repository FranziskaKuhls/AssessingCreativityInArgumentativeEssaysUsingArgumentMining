import re

from src.creativity_support_tool.CreativityMLServerWithArgumentMining.cstWithAcl.metrics import acl_metrics


class AclEvaluator:
    def __init__(self):
        self.metrics = acl_metrics()
        self.delimiters_pattern = r'[!?.;-]'

    def evaluate_essay(self, input_essay: str):
        sentence_list = self.__from_essay_to_sentences(input_essay)
        argumentative_sentence_ist = self.metrics.get_argumentative_sentences(sentence_list)
        fluency = self.metrics.get_fluency(argumentative_sentence_ist, len(sentence_list))
        flexibility = self.metrics.get_flexibility(argumentative_sentence_ist)
        originality, topics = self.metrics.get_originality(argumentative_sentence_ist)
        return fluency, flexibility, originality, topics

    def __from_essay_to_sentences(self, input_essay: str):
        lines = input_essay.splitlines()
        sentence_list = []
        for line in lines:
            sentence_in_line = re.split(self.delimiters_pattern, line)
            for sentence in sentence_in_line:
                if len(sentence) <= 8:
                    continue
                if sentence[0] == " ":
                    sentence_list.append(sentence[1:] + ".\n")
                else:
                    sentence_list.append(sentence + ".\n")
        return sentence_list
