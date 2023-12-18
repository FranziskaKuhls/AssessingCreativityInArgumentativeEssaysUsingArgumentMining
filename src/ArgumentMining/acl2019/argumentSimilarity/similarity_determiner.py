from transformers import BertTokenizer
import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from typing import List

from src.ArgumentMining.acl2019.argumentSimilarity.SigmoidBERT import SigmoidBERT
from src.ArgumentMining.acl2019.argumentSimilarity.train import InputExample, convert_examples_to_features


class SimilarityDeterminer:
    def __init__(self, model_path: str):
        self.max_seq_length = 64
        self.eval_batch_size = 8
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        self.model = SigmoidBERT.from_pretrained(model_path,)

    def get_similarity(self, sentence1: str, sentence2: str):
        input_examples = [InputExample(text_a=sentence1, text_b=sentence2, label=-1)]
        eval_features = convert_examples_to_features(input_examples, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        predicted_logits = []
        with torch.no_grad():
            for input_ids, input_mask, segment_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.detach().cpu().numpy()
                predicted_logits.extend(logits[:, 0])

        return predicted_logits

    def __create_distance_matrix(self, argumentative_sentence_list: List[str]) -> List[List[str]]:
        distance_matrix = []
        for sentence in argumentative_sentence_list:
            distance_row = []
            for sentence2 in argumentative_sentence_list:
                if sentence == sentence2:
                    distance_row.append(0)
                    continue
                else:
                    sim = self.get_similarity(sentence, sentence2)
                    distance_row.append(1 - sim[0])
            distance_matrix.append(distance_row)
        return distance_matrix

    def get_best_cluster_count(self, argumentative_sentence_list: List[str]) -> int:
        distance_matrix = numpy.array(self.__create_distance_matrix(argumentative_sentence_list))
        best_score = 0
        cluster_count = 0
        for j in range(2, len(argumentative_sentence_list) - 1):
            agg_cluster = AgglomerativeClustering(n_clusters=j, linkage='average', affinity='euclidean')
            agg_cluster.fit(distance_matrix)
            labels = agg_cluster.labels_
            sil_score = silhouette_score(distance_matrix, labels)
            if sil_score > best_score:
                best_score = sil_score
                cluster_count = j
        return cluster_count