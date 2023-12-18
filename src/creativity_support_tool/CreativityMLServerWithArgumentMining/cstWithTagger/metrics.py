from bertopic import BERTopic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import re


class TaggerMetrics:
    def __init__(self):
        self.bert_model = BERTopic.load(
            "model/TopicModel/topic_model_with_targer")
        self.topic_sizes = dict(self.bert_model.topic_sizes_)

        embeddings = np.array(self.bert_model.topic_embeddings_)
        # remove -1 (outliers)
        embeddings = embeddings[1:]
        self.similarity_matrix = cosine_similarity(embeddings)

    def get_topics_per_segment(self, essay_segments):
        topic_list = []
        for segment in essay_segments:
            topic_label = self.bert_model.transform(segment)[0][0]
            topic_list.append(topic_label)
        return topic_list

    def get_segments(self, essay: str, essay_tags) -> List[str]:
        return self.__get_tagged_strings_from_essay(essay, essay_tags)

    def get_fluency(self, essay, essay_segments, essay_topics):
        """
        number of segments
        """
        #fluency = len(essay_segments) / len(self.__get_sentences_from_essay(essay))
        #return min(fluency, 1)

        essay_words_count = self.__get_string_size(essay)
        fluency = 0
        for i in range(len(essay_topics)):
            segment = essay_segments[i]
            topic = essay_topics[i]
            if topic != -1:
                fluency += 1

        #for segment in essay_segments:
        #    segment_size = self.__get_string_size(segment)
        #    fluency += segment_size / essay_words_count

        return fluency / len(essay_segments)

    def get_flexibility(self, topic_list):
        """
        Similarity Segments
        """
        global_similarity = 0

        if len(topic_list) > 1:
            for i in topic_list:
                other_topics = [x for x in topic_list if x != i]
                topic_similarity = 0
                for j in other_topics:
                    topic_similarity += self.similarity_matrix[i, j]

                topic_similarity = topic_similarity / len(other_topics)
                global_similarity += topic_similarity / len(topic_list)
        else:
            global_similarity = 1

        return 1 - global_similarity

    def get_originality(self, topic_list):
        """
        originality of arguments
        schauen welcher String in welches Topic passt und wie groß das Topic ist. (wenn -1 spezialbehandlung)
        """
        if len(topic_list) == 0:
            return 0
        originality = 0
        for topic in topic_list:
            if topic == -1:
                originality += 1
            else:
                topic_size = self.topic_sizes[topic]
                originality += 1 / topic_size
        return originality / len(topic_list)

    def __get_string_size(self, string: str):
        return len(string.split(" "))

    def __get_sentences_from_essay(self, essay):
        delimiters_pattern = r'[!?.;-]'
        lines = essay.splitlines()
        sentence_list = []
        for line in lines:
            sentence_in_line = re.split(delimiters_pattern, line)
            for sentence in sentence_in_line:
                if sentence[0] == " ":
                    sentence_list.append(sentence[1:] + ".\n")
                else:
                    sentence_list.append(sentence + ".\n")
        return sentence_list

    def __get_tagged_strings_from_essay(self, essay: str, list_tagged_items):
        list_tagged_strings_in_essay = []
        list_tagged_items = self.__split_item_list_string(list_tagged_items[1:-1])
        for item in list_tagged_items:
            type, start, end = self.__get_type_start_and_end_from_String_item(item)
            if type == "CLAIM" or type == "PREMISE":
                if end is None:
                    list_tagged_strings_in_essay.append(essay[start:])
                else:
                    list_tagged_strings_in_essay.append(essay[start:end])
        return list_tagged_strings_in_essay

    def __split_item_list_string(self, item_list_string):
        pattern = r'{[^}]+}'
        tokens = re.findall(pattern, item_list_string)
        return tokens

    def __get_type_start_and_end_from_String_item(self, tagged_item):
        pattern_type = r'"type":\s*"(.*?)"'
        pattern_start = r'"start":\s*([0-9]+)'
        pattern_end = r'"end":\s*([0-9]+)'
        type_value_match = re.search(pattern_type, tagged_item)
        start_value_match = re.search(pattern_start, tagged_item)
        end_value_match = re.search(pattern_end, tagged_item)
        type_value = None
        start_value = None
        end_value = None

        if (type_value_match != None):
            type_value = type_value_match.group(1)
        if (start_value_match != None):
            start_value = int(start_value_match.group(1))
        if (end_value_match != None):
            end_value = int(end_value_match.group(1))

        return type_value, start_value, end_value
