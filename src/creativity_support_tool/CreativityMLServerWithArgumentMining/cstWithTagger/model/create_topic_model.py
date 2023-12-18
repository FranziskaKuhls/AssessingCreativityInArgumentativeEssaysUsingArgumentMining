from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

import re
import csv


class TopicModellCreatorWithTarger:
    def __init__(self, list_paths_to_essay_with_tagged_types, number_of_topic_minimum: int):
        self.csv_path = list_paths_to_essay_with_tagged_types
        self.topic_count_min = number_of_topic_minimum

    def show_model_info(self):
        bert_model = BERTopic.load(
            "creativity_support_tool/CreativityMLServerWithArgumentMining/cstWithTagger/model/TopicModel/topic_model_with_targer")
        topic_info = bert_model.get_topic_info()
        print(topic_info)
        print(bert_model.topic_sizes_)

    def create_model(self):
        list_training_sentences = []
        with open(self.csv_path, 'r', encoding="utf8") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                essay = row[1]
                tagged_essay = row[2]
                list_training_sentences.extend(self.__get_tagged_strings_from_essay(essay, tagged_essay))
        # Remove stop words from definition not from model
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

        # Seed the model
        umap_model = UMAP(n_neighbors=self.topic_count_min, n_components=5,
                          min_dist=0.0, metric='cosine', random_state=42)

        topic_model = BERTopic(language="english",
                               min_topic_size=6,
                               umap_model=umap_model,
                               vectorizer_model=vectorizer_model,
                               calculate_probabilities=True,
                               verbose=True)

        # Fit the model with our data

        docs_filtered = [x for x in list_training_sentences if len(x.split()) > 1]

        topics, probs = topic_model.fit_transform(docs_filtered)

        """### View results"""

        topic_model.visualize_barchart(top_n_topics=30)
        topic_model.visualize_distribution(probs[200], min_probability=0.015)
        topic_model.visualize_heatmap(n_clusters=None, width=1000, height=1000)

        # Save the model
        topic_model.save(
            "creativity_support_tool/CreativityMLServerWithArgumentMining/cstWithTagger/model/TopicModel/topic_model_with_targer")

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
