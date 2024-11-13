import argparse
import torch
import torch.nn as nn
import random

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np
import warnings
from argparse import ArgumentParser
from GenerateUtils_coa import generate_coa
from torch.nn.functional import relu, sigmoid
from torch.nn import Parameter
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from clip import clip
from ram.models import ram
from tqdm import tqdm
from typing import TextIO
from pathlib import Path

warnings.filterwarnings("ignore")


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


compositional_template = ["This image contains {}."]

single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]

def compute_clip_similarity(image, predict_labels, target_labels, clip_model, clip_preprocess):
    with torch.no_grad():
        predict_text = "This image contains " + ", ".join(predict_labels)
        predict_text = predict_text[:70]
        predict_text_input = clip.tokenize(predict_text).cuda()
        predict_text_features = clip_model.encode_text(predict_text_input)
        predict_text_features /= predict_text_features.norm(dim=-1, keepdim=True)

        target_text = "This image contains " + ", ".join(target_labels)
        target_text = target_text[:70]
        target_text_input = clip.tokenize(target_text).cuda()
        target_text_features = clip_model.encode_text(target_text_input)
        target_text_features /= target_text_features.norm(dim=-1, keepdim=True)

        image = clip_preprocess(image).unsqueeze(0).cuda()
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        predict_similarity = (image_features @ predict_text_features.T).squeeze().item()
        target_similarity = (image_features @ target_text_features.T).squeeze().item()
        # print(predict_similarity)
        # print(target_similarity)
    return 1 if predict_similarity > target_similarity else 0


def article(name):
    try:
        new_name = "an" if name[0] in "aeiou" else "a"
    except IndexError as e:
        new_name = name
    return new_name

def convert_to_rgb(image):
    return image.convert("RGB")

def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def compute_ram_similarity(image, predict_labels, clip_model, ram_model, transform):
    # templates = single_template
    # templates = multiple_templates
    templates = compositional_template
    predict_text = ", ".join(predict_labels)
    if len(predict_text) > 70:
        predict_text = predict_text[:70]
    predict_labels = [predict_text]
    with torch.no_grad():
        labels_embedding = []
        for label in predict_labels:
            texts = [
                template.format(
                    label, article=article(label)
                )
                for template in templates
            ]

            # texts = [
            #     "This is " + text if text.startswith("a") or text.startswith("the") else text
            #     for text in texts
            # ]
            # print(texts)

            texts = clip.tokenize(texts).cuda()
            text_embeddings = clip_model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            labels_embedding.append(text_embedding)
        labels_embedding = torch.stack(labels_embedding, dim=1)
        labels_embedding = labels_embedding.cuda()
        labels_embedding = labels_embedding.t()
        ram_model.label_embed = Parameter(labels_embedding.float())

        image = transform(image).unsqueeze(dim=0).cuda()
        image_embeds = ram_model.image_proj(ram_model.visual_encoder(image))
        image_embeds = image_embeds.cuda()
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).cuda()

        label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
            .repeat(image.shape[0], 1, 1)
        label_embed = label_embed.cuda()
        tagging_embed, _ = ram_model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )
        logits = ram_model.fc(tagging_embed).squeeze(-1)
        logits = torch.sigmoid(logits)
        # print(logits)

    return logits.mean()


def ram_filter(image, predict_labels, clip_model, ram_model, transform):
    class_threshold = torch.ones(len(predict_labels)) * 0.73
    # templates = single_template
    templates = multiple_templates
    # templates = compositional_template
    # predict_text = ", ".join(predict_labels)
    # predict_labels = [predict_text]
    with torch.no_grad():
        labels_embedding = []
        for label in predict_labels:
            # if len(label) > 70:
            #     label = label[:70]
            texts = [
                template.format(
                    label, article=article(label)
                )
                for template in templates
            ]

            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = [
                text[:70] if len(text) > 70 else text
                for text in texts
            ]
            # print(texts)
            texts = clip.tokenize(texts).cuda()
            text_embeddings = clip_model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            labels_embedding.append(text_embedding)
        labels_embedding = torch.stack(labels_embedding, dim=1)
        labels_embedding = labels_embedding.cuda()
        labels_embedding = labels_embedding.t()
        ram_model.label_embed = Parameter(labels_embedding.float())

        image = transform(image).unsqueeze(dim=0).cuda()
        image_embeds = ram_model.image_proj(ram_model.visual_encoder(image))
        image_embeds = image_embeds.cuda()
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).cuda()

        label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
            .repeat(image.shape[0], 1, 1)
        label_embed = label_embed.cuda()
        tagging_embed, _ = ram_model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )
        logits = ram_model.fc(tagging_embed).squeeze(-1)
        targets = torch.where(
            torch.sigmoid(logits) > class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(len(predict_labels)).to(image.device))
        # print(predict_labels)
        targets = targets.tolist()[0]
        # print(targets)
        new_labels = []
        for i in range(len(predict_labels)):
            if targets[i] == 1:
                new_labels.append(predict_labels[i])
        # print(new_labels)
        if len(new_labels) == 0:
            max_value = max(targets)
            for i in range(len(predict_labels)):
                if targets[i] >= max_value:
                    new_labels.append(predict_labels[i])
    return new_labels


def calculate_ram_score(image, predict_labels, clip_model, ram_model, transform):
    class_threshold = torch.ones(len(predict_labels)) * 0.73
    # templates = single_template
    templates = multiple_templates
    # templates = compositional_template
    # predict_text = ", ".join(predict_labels)
    # predict_labels = [predict_text]
    with torch.no_grad():
        labels_embedding = []
        for label in predict_labels:
            # if len(label) > 70:
            #     label = label[:70]
            texts = [
                template.format(
                    label, article=article(label)
                )
                for template in templates
            ]

            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = [
                text[:70] if len(text) > 70 else text
                for text in texts
            ]
            # print(texts)
            texts = clip.tokenize(texts).cuda()
            text_embeddings = clip_model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            labels_embedding.append(text_embedding)
        labels_embedding = torch.stack(labels_embedding, dim=1)
        labels_embedding = labels_embedding.cuda()
        labels_embedding = labels_embedding.t()
        ram_model.label_embed = Parameter(labels_embedding.float())

        image = transform(image).unsqueeze(dim=0).cuda()
        image_embeds = ram_model.image_proj(ram_model.visual_encoder(image))
        image_embeds = image_embeds.cuda()
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).cuda()

        label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
            .repeat(image.shape[0], 1, 1)
        label_embed = label_embed.cuda()
        tagging_embed, _ = ram_model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )
        logits = ram_model.fc(tagging_embed).squeeze(-1)
        targets = torch.where(
            torch.sigmoid(logits) > class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(len(predict_labels)).to(image.device))
        targets[targets == 0] = -1
        # targets = targets.tolist()[0]
        final_ram_score = torch.sigmoid(torch.sum(targets))
        # print(final_ram_score)

    return final_ram_score


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


def eval_model(args):
    # Model
    disable_torch_init()

    summary_file = args.summary_file

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # image_files = image_parser(args)
    image_files_s = args.image_file

    clip_model, clip_preprocess = clip.load('ViT-B/16')
    clip_model = clip_model.cuda()

    ram_model = ram(pretrained=args.ram_checkpoint, image_size=args.ram_input_size, vit=args.ram_backbone)
    ram_model = ram_model.cuda().eval()
    transform = get_transform(args.ram_input_size)

    # final_compositional_similarity = AverageMeter()
    total = len(image_files_s)

    clip_correct = 0
    ram_com_correct = 0
    ram_score = 0

    clip_correct_af = 0
    ram_com_correct_af = 0
    ram_score_af = 0

    for i in tqdm(range(len(image_files_s)), desc="process"):
        torch.cuda.empty_cache()
        image_files = [image_files_s[i]]
        # print(image_files)
        labels_onehot = args.labels[i]
        labels = np.take(args.taglist, np.where(labels_onehot == 1)[0])
        image = load_images(image_files)
        # generate initialized vocabularies
        vocabularies, outputs = generate_coa(image, image_processor, model, model_name, tokenizer, args)
        # print(outputs)
        # print("Predicts: {}, True labels: {}".format(vocabularies, labels))
        clip_correct += compute_clip_similarity(image[0], vocabularies, labels, clip_model, clip_preprocess)
        ram_com_correct += compute_ram_similarity(image[0], vocabularies, clip_model, ram_model, transform)
        ram_score += calculate_ram_score(image[0], vocabularies, clip_model, ram_model, transform)

        vocabularies = ram_filter(image[0], vocabularies, clip_model, ram_model, transform)
        # print("Filter Predicts: {}, True labels:{}".format(vocabularies, labels))
        clip_correct_af += compute_clip_similarity(image[0], vocabularies, labels, clip_model, clip_preprocess)
        ram_com_correct_af += compute_ram_similarity(image[0], vocabularies, clip_model, ram_model, transform)
        ram_score_af += calculate_ram_score(image[0], vocabularies, clip_model, ram_model, transform)

    clip_accuracy = clip_correct / total
    ram_com_accuracy = ram_com_correct / total
    ram_score_accuracy = ram_score / total

    clip_accuracy_af = clip_correct_af / total
    ram_com_accuracy_af = ram_com_correct_af / total
    ram_score_accuracy_af = ram_score_af / total

    print("CLIP Accuracy: {:.2f}".format(clip_accuracy * 100))
    print("RAM Compositional Accuracy: {:.2f}".format(ram_com_accuracy * 100))
    print("RAM Score Accuracy: {:.2f}".format(ram_score_accuracy * 100))

    print("After Filter CLIP Accuracy: {:.2f}".format(clip_accuracy_af * 100))
    print("After Filter RAM Compositional Accuracy: {:.2f}".format(ram_com_accuracy_af * 100))
    print("After Filter RAM Score Accuracy: {:.2f}".format(ram_score_accuracy_af * 100))

    with open(summary_file, "a", encoding="utf-8") as f:
        print_write(f, "CLIP Accuracy: {:.2f}".format(clip_accuracy * 100))
        print_write(f, "RAM Compositional Accuracy: {:.2f}".format(ram_com_accuracy * 100))
        print_write(f, "RAM Score Accuracy: {:.2f}".format(ram_score_accuracy * 100))
        print_write(f, "After Filter CLIP Accuracy: {:.2f}".format(clip_accuracy_af * 100))
        print_write(f, "After Filter RAM Compositional Accuracy: {:.2f}".format(ram_com_accuracy_af * 100))
        print_write(f, "After Filter RAM Score Accuracy: {:.2f}".format(ram_score_accuracy_af * 100))


def parse_extra_args():
    parser = ArgumentParser()

    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./SJML_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    # data
    parser.add_argument("--dataset", type=str, choices=("voc", "coco", "cub", "nus"), default="voc")
    parser.add_argument("--seed", type=int, default=100)

    extra_args = parser.parse_args()

    return extra_args


if __name__ == "__main__":
    parser = ArgumentParser()
    # miscellaneous
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    # data
    parser.add_argument("--dataset", type=str, choices=("voc", "coco", "nus"), default="voc")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./outputs")

    extra_args = parser.parse_args()

    # fix random seed
    seed = extra_args.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    split = extra_args.split

    output_dir = extra_args.output_dir + "/" + extra_args.dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_file = output_dir + "/" + "summary.txt"
    with open(summary_file, "a", encoding="utf-8") as f:
        print_write(f, f"**********split: {split} ****************")

    model_path = "put your model path"

    dataset = extra_args.dataset
    dataset_root_path = "put your dataset path"
    tag_file = os.path.join(dataset_root_path, f'{dataset}/{dataset}_ram_taglist.txt')
    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]


    X = np.load(os.path.join(dataset_root_path, f'{dataset}/formatted_val_images_split{split}.npy'))
    Y = np.load(os.path.join(dataset_root_path, f'{dataset}/formatted_val_labels_split{split}.npy'))

    image_path = ""
    if dataset == "voc":
        image_path = dataset_root_path + "voc/VOCdevkit/VOC2012/JPEGImages/"
    elif dataset == "coco":
        image_path = dataset_root_path + "coco/"

    image_file = [image_path + line for line in X]

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        # "query": prompt,
        "conv_mode": None,
        "dataset": "voc",
        "image_file": image_file,
        "labels": Y,
        "taglist": taglist,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 200,
        "ram_checkpoint": "./pretrained/ram_swin_large_14m.pth",
        "ram_backbone": "swin_l",
        "ram_input_size": 384,
        "summary_file": summary_file,
    })()

    eval_model(args)
