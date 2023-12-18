from metrics import TaggerMetrics
from TargerAPIAdapter import TargetAdapter


class TargerEvaluator:
    def __init__(self):
        self.metrics = TaggerMetrics()
        self.apiAdapter = TargetAdapter()

    def evaluate_essay(self, essay: str, tagged_item_list):
        segments = self.metrics.get_segments(essay, tagged_item_list)
        topic_list = self.metrics.get_topics_per_segment(segments)

        fluency = self.metrics.get_fluency(essay, segments, topic_list)
        flexibility = self.metrics.get_flexibility(topic_list)
        originality = self.metrics.get_originality(topic_list)
        return fluency, flexibility, originality, topic_list
