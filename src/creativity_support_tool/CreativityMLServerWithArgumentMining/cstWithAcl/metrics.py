from typing import List
from bertopic import BERTopic

from src.ArgumentMining.acl2019.argumentSimilarity.similarity_determiner import SimilarityDeterminer
from src.ArgumentMining.acl2019.argumentClassification.classificator import Classificator


# with argument classification and argument similarity
class acl_metrics:
    def __init__(self):
        classificator_model_path = "model/bert_output/argument_classification_ukp_all_data_large_model/"
        similarity_model_path = "model//bert_output/ukp_aspects_all/"

        self.similarity_determiner = SimilarityDeterminer(similarity_model_path)
        self.classificator = Classificator(classificator_model_path)
        self.bert_model = BERTopic.load("model/TopicModel/topic_model_argumentative_sentences_new_version")

    def get_argumentative_sentences(self, sentence_list: List[str]):
        return self.classificator.get_argumentative_sentences_from_sentence_list(sentence_list)

    def get_fluency(self, argumentative_sentence_list: List[str], sentence_count):
        """
        number of arguments
        """
        return len(argumentative_sentence_list)/sentence_count

    def get_flexibility(self, argumentative_sentence_list: List[str]):
        """
        pairwise distance between arguments
        """
        similarity_num: float = 0
        count_tuples = 0
        if len(argumentative_sentence_list) == 0 or len(argumentative_sentence_list) == 1:
            return 0
        for i in range(len(argumentative_sentence_list)):
            sentence1 = argumentative_sentence_list[i]
            for j in range(i, len(argumentative_sentence_list)):
                sentence2 = argumentative_sentence_list[j]
                if sentence1 != sentence2:
                    sim = self.similarity_determiner.get_similarity(sentence1, sentence2)
                    similarity_num += sim[0]
                    count_tuples += 1
        return similarity_num/count_tuples

    def get_originality(self, argumentative_sentence_list: List[str]):
        return self.__originality_with_topic_model(argumentative_sentence_list)

    def __originality_with_topic_model(self, argumentative_sentence_list: List[str]):
        found_topics = []
        topic_sizes = dict(self.bert_model.topic_sizes_)
        originality_topic = 0
        if len(argumentative_sentence_list) == 0:
            return 0
        for sentence in argumentative_sentence_list:
            #vectorized_sentence = bert_model.vectorizer_model.transform([sentence])
            topic_label = self.bert_model.transform(sentence)[0][0]
            found_topics.append(topic_label)
            if topic_label == -1:
                originality_topic += 1
            else:
                # topic info contains index, size and words
                topic_size = topic_sizes[topic_label]
                originality_topic += 1/topic_size
        return originality_topic / len(argumentative_sentence_list), found_topics
