from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from typing import List


class TopicModellCreator:
    def __init__(self, list_paths_to_argument_sentences_file: List[str], number_of_topic_minimum: int):
        self.list_paths = list_paths_to_argument_sentences_file
        self.topic_count_min = number_of_topic_minimum

    def show_model_info(self):
        bert_model = BERTopic.load(
            "creativity_support_tool/CreativityMLServerWithArgumentMining/cstWithAcl/model/TopicModel/topic_model_argumentative_sentences_new_version")
        topic_info = bert_model.get_topic_info()
        print(topic_info)
        print(bert_model.topic_sizes_)

    def create_model(self):
        list_argumentative_sentences: List[str] = []


        for path in self.list_paths:
            with open(path, 'r', encoding="utf8") as arg_file:
                sentence = arg_file.readline()
                while sentence:
                    list_argumentative_sentences.append(sentence)
                    sentence = arg_file.readline()
        print("Training Sentences count: " + str(len(list_argumentative_sentences)))
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

        docs_filtered = [x for x in list_argumentative_sentences if len(x.split()) > 1]

        topics, probs = topic_model.fit_transform(docs_filtered)

        """### View results"""

        topic_model.visualize_barchart(top_n_topics=30)
        topic_model.visualize_distribution(probs[200], min_probability=0.015)
        topic_model.visualize_heatmap(n_clusters=None, width=1000, height=1000)

        # Save the model
        topic_model.save("creativity_support_tool/CreativityMLServerWithArgumentMining/cstWithAcl/model/TopicModel/topic_model_argumentative_sentences_new_version")
