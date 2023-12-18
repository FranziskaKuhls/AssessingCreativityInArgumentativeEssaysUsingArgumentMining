from src.creativity_support_tool.CreativityMLServerWithArgumentMining.cstWithTagger.model.create_topic_model import \
    TopicModellCreatorWithTarger
from src.creativity_support_tool.CreativityMLServerWithArgumentMining.cstWithAcl.model.create_topic_model import \
    TopicModellCreator

def preprocess_make_acl_model():
    model_creator = TopicModellCreator(
        ["data/argumentative-creative-essays/Argumentative_Sentences-creative-essays-climate-change.txt"],
        number_of_topic_minimum=10)
    model_creator.create_model()
    model_creator.show_model_info()


def preprocess_make_targer_model():
    path = "data/argumentative-creative-essays/Corpus-creative-essays-climate-change-tagged.csv"
    model_creator = TopicModellCreatorWithTarger(path, number_of_topic_minimum=10)
    model_creator.create_model()
    model_creator.show_model_info()


if __name__ == "__main__":
    preprocess_make_acl_model()
    preprocess_make_targer_model()
