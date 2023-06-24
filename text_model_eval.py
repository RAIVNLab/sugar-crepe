import argparse
import json
import os

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import pipeline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Vera:
    def __init__(self, model, model_cache_dir=None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
        self.model = transformers.T5EncoderModel.from_pretrained(model, torch_dtype='auto', offload_folder='offload', cache_dir=model_cache_dir)
        self.model = self.model.to(device)
        self.model.D = self.model.shared.embedding_dim
        self.linear = torch.nn.Linear(self.model.D, 1, dtype=self.model.dtype).to(device)
        self.linear.weight = torch.nn.Parameter(self.model.shared.weight[32099, :].unsqueeze(0))  # (1, D)
        self.linear.bias = torch.nn.Parameter(self.model.shared.weight[32098, 0].unsqueeze(0))  # (1)
        self.model.eval()
        self.t = self.model.shared.weight[32097, 0].item()

    def run(self, statement):
        input_ids = self.tokenizer.batch_encode_plus([statement], return_tensors='pt', padding='longest', truncation='longest_first', max_length=128).input_ids.to(device)
        with torch.no_grad():
            output = self.model(input_ids)
            last_hidden_state = output.last_hidden_state.to(device)  # (B=1, L, D)
            hidden = last_hidden_state[0, -1, :]  # (D)
            logit = self.linear(hidden).squeeze(-1)  # ()
            logit_calibrated = logit / self.t
            score = logit.sigmoid()
            score_calibrated = logit_calibrated.sigmoid()
        return score_calibrated.item()

    def runs(self, statements):
        tok = self.tokenizer.batch_encode_plus(statements, return_tensors='pt', padding='longest')
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_indices = attention_mask.sum(dim=1, keepdim=True) - 1  # (B, 1)
            last_indices = last_indices.unsqueeze(-1).expand(-1, -1, self.model.D)  # (B, 1, D)
            last_hidden_state = output.last_hidden_state.to(device)  # (B, L, D)
            hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1)  # (B, D)
            logits = self.linear(hidden).squeeze(-1)  # (B)
            logits_calibrated = logits / self.t
            scores = logits.sigmoid()
            scores_calibrated = logits_calibrated.sigmoid()
        return np.array([i.item() for i in scores_calibrated.detach().cpu()])


class GrammarModel:
    def __init__(self, model_cache_dir=None):
        self.model = pipeline("text-classification", model="textattack/distilbert-base-uncased-CoLA")

    def run(self, statement):
        with torch.no_grad():
            output = self.model(statement)[0]
            score = output['score'] if output['label'] == 'LABEL_1' else 1 - output['score']
        return score

    def runs(self, statements):
        with torch.no_grad():
            scores = []
            for output in self.model(statements):
                score = output['score'] if output['label'] == 'LABEL_1' else 1 - output['score']
                scores.append(score)
        return np.array(scores)


@torch.no_grad()
def text_retrieval(pos_text, neg_text, model):
    pos_score = model.run(pos_text)
    neg_score = model.run(neg_text)
    return 1 if pos_score > neg_score else 0


def evaluate(dataset, model):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            correct = text_retrieval(data['caption'], data['negative_caption'], model)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cache_dir', default=None, type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--output', type=str, default=None, help="Directory to where results are saved")

    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_obj'    : f'{args.data_root}/add_obj.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_obj': f'{args.data_root}/replace_obj.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_obj'   : f'{args.data_root}/swap_obj.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    os.makedirs(args.output, exist_ok=True)

    model = Vera('liujch1998/vera', args.model_cache_dir)
    print(f"Evaluating Vera model")
    metrics = evaluate(dataset, model)
    print(metrics)
    print(f"Dump results to: {os.path.join(args.output, f'vera.json')}")
    json.dump(metrics, open(os.path.join(args.output, f'vera.json'), 'w'), indent=4)

    model = GrammarModel(args.model_cache_dir)
    print(f"Evaluating grammar model")
    metrics = evaluate(dataset, model)
    print(metrics)
    print(f"Dump results to: {os.path.join(args.output, f'grammar.json')}")
    json.dump(metrics, open(os.path.join(args.output, f'grammar.json'), 'w'), indent=4)


