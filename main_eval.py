import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

import open_clip

models = [
    ('RN50', 'openai'),
    ('RN101', 'openai'),
    ('RN50x4', 'openai'),
    ('ViT-B-32', 'openai'),
    ('RN50x16', 'openai'),
    ('RN50x64', 'openai'),
    ('ViT-L-14', 'openai'),
    ('ViT-B-32-quickgelu', 'datacomp_s_s13m_b4k'),
    ('ViT-B-32-quickgelu', 'datacomp_m_s128m_b4k'),
    ('ViT-B-16', 'datacomp_l_s1b_b8k'),
    ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-g-14', 'laion2b_s12b_b42k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k'),
    ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
    ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
    ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
]


def load_model(args, pretrained, device):
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=pretrained,
        cache_dir=args.model_cache_dir,
        device=device
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    return model, tokenizer, transform


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, transform, device):
    pos_text = tokenizer(pos_text).to(device)
    pos_text_embedding = model.encode_text(pos_text, normalize=True)
    neg_text = tokenizer(neg_text).to(device)
    neg_text_embedding = model.encode_text(neg_text, normalize=True)
    image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)
    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0


def evaluate(image_root, dataset, model, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path)
            correct = text_retrieval(data['caption'], data['negative_caption'], image, model, tokenizer, transform, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[f'{c}'] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="RN50", help="Model architecture to use from OpenCLIP")
    parser.add_argument('--pretrained', type=str, default="openai", help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--model_cache_dir', default=None, type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--output', type=str, default=None, help="Directory to where results are saved")

    parser.add_argument('--coco_image_root', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--all', action="store_true", default=False, help="Whether to test all the pretrained models in the paper")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_ent'    : f'{args.data_root}/add_ent.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_ent': f'{args.data_root}/replace_ent.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_ent'   : f'{args.data_root}/swap_ent.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    os.makedirs(args.output, exist_ok=True)

    if args.all:
        print("Evaluating all models")
        for modelname, pretrained in models:
            print(f"Evaluating {modelname}-{pretrained}")
            args.model = modelname
            args.pretrained = pretrained

            model, tokenizer, transform = load_model(args, pretrained, device)

            metrics = evaluate(args.coco_image_root, dataset, model, tokenizer, transform, device)
            print(metrics)
            print(f"Dump results to: {os.path.join(args.output, f'{args.model}-{args.pretrained}.json')}")
            json.dump(metrics, open(os.path.join(args.output, f'{args.model}-{args.pretrained}.json'), 'w'), indent=4)

    else:
        print(f"Evaluating {args.model}-{args.pretrained}")

        model, tokenizer, transform = load_model(args, args.pretrained, device)

        metrics = evaluate(args.coco_image_root, dataset, model, tokenizer, transform, device)
        print(metrics)
        print(f"Dump results to: {os.path.join(args.output, f'{args.model}-{args.pretrained}.json')}")
        json.dump(metrics, open(os.path.join(args.output, f'{args.model}-{args.pretrained}.json'), 'w'), indent=4)
