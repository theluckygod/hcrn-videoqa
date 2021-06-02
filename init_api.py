import yaml
import h5py
from scipy.misc import imresize
import skvideo.io
from PIL import Image
import json
import nltk

import torch
from torch import nn
import torchvision
import random
import numpy as np

import os
from os.path import join
import sys
sys.path.append('preprocess/')
from datautils import svqad_qa, utils
from utils import todevice

from preprocess.preprocess_features import build_resnext, run_batch

from config import cfg_from_file
import model.HCRN as HCRN

import time

def load_conf(conf_path):
    with open(conf_path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    return cfg

def load_data_type_conf(app_conf):
    data_type_conf = {}
    for data_type in app_conf['available_vqa_model']:
        data_type_conf[data_type] = load_conf(join(app_conf['conf_folder'], data_type + ".yml"))

    return data_type_conf

def extract_clips_with_consecutive_frames(cfg, video_data, feat_type):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
        feat_type: type of features ['motion', 'appearance']
    Returns:
        A list of raw features of clips.
    """
    num_clips = cfg["inference"]["num_clips"]
    num_frames_per_clip = cfg["inference"]["num_frames_per_clip"]

    valid = True
    clips = list()

    total_frames = video_data.shape[0]

    if feat_type == 'motion':
        motion_image_height, motion_image_width = cfg["inference"]["motion_image_height"], cfg["inference"]["motion_image_width"]
        img_size = (motion_image_height, motion_image_width)
    else:
        appr_image_height, appr_image_width = cfg["inference"]["appr_image_height"], cfg["inference"]["appr_image_width"]
        img_size = (appr_image_height, appr_image_width)
    
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            frame_data = np.array(img)
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        if feat_type == 'motion':
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    return clips, valid

def extract_motion_features(cfg, model, video_data):
    """
    Args:
        model: model of motion
        video_data: video data -- type=np.array shape=(,,,3)
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw motion features of clip.
    """    
    num_clips = cfg["inference"]["num_clips"]

    clips, valid = extract_clips_with_consecutive_frames(cfg,
                                                        video_data, 
                                                        feat_type='motion')
    clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
    if valid:
        clip_feat = model(clip_torch)  # (8, 2048)
        clip_feat = clip_feat.squeeze()
        clip_feat = clip_feat.detach().cpu().numpy()
    else:
        clip_feat = np.zeros(shape=(num_clips, 2048))
    return torch.from_numpy(clip_feat[None, :]).float()

def extract_appearance_features(cfg, model, video_data):
    """
    Args:
        model: model of appearance
        video_data: video data -- type=np.array shape=(,,,3)
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw appearance features of clip.
    """
    num_clips = cfg["inference"]["num_clips"]
    
    clips, valid = extract_clips_with_consecutive_frames(cfg, 
                                                        video_data, 
                                                        feat_type='appearance')
    clip_feat = []
    if valid:
        for clip_id, clip in enumerate(clips):
            feats = run_batch(clip, model)  # (16, 2048)
            feats = feats.squeeze()
            clip_feat.append(feats)
    else:
        clip_feat = np.zeros(shape=(num_clips, 16, 2048))
    clip_feat = np.asarray(clip_feat)  # (8, 16, 2048)
    return torch.from_numpy(clip_feat[None, :]).float()

def extract_question_features(vocab, ques):
    """
    Args:
        vocab:
        ques: string question
    Returns:
        A list of raw features of question.
    """
    question = ques.lower()[:-1]
    question_tokens = nltk.word_tokenize(question)
    question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
    question_len = len(question_encoded)

    question_encoded = np.asarray(question_encoded, dtype=np.int32)
    glove_matrix = None

    obj = {
        'questions': torch.from_numpy(question_encoded[None, :]).long(),
        'questions_len': torch.tensor([question_len]).long(),
        'glove': glove_matrix,
    }
    return obj

def invert_dict(d):
    return {v: k for k, v in d.items()}
def load_vocab(dataset_format, vocab_json):
    with open(vocab_json.format(dataset_format, dataset_format), 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab

def build_resnet():
    if not hasattr(torchvision.models, "resnet101"):
        raise ValueError('Invalid model "%s"' % "resnet101")
    if not 'resnet' in "resnet101":
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, "resnet101")(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model

def load_vqa_model(cfg, data_type, vocab):
    if data_type == 'svqad_qa':
        model = load_SVQAD_HCRN_model(cfg, vocab)

    return model

def load_SVQAD_HCRN_model(cfg, vocab):
    cfg_data = load_conf(join(cfg['conf_folder'], "svqad_qa.yml"))
    # cfg_from_file('configs/svqad_qa.yml')
    device = 'cuda'
    assert os.path.exists(cfg_data['inference']['ckpt_path'])
    
    ckpt_path=cfg_data['inference']['ckpt_path']
    # load pretrained model
    loaded = torch.load(ckpt_path, map_location='cpu')
    model_kwargs = loaded['model_kwargs']
    model_kwargs.update({'vocab': vocab})
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    return model

def inference(model, data, device='cuda'):
    model.eval()
    print('infering...')

    with torch.no_grad():
        batch_input = [todevice(x, device) for x in data]
        batch_size = 1

        logits = model(*batch_input).to(device)
        pred = logits.detach().argmax(1)

    return pred

def inference_e2e(cfg, motion_model, appr_model, vocab, model, video_data, question):
    # extract motion features
    print("extract motion features...")
    print("----extracting motion features...")
    motion_feat = extract_motion_features(cfg, motion_model, video_data)
    print("====finish extracting motion features")

    # extract appearance features
    print("extract appearance features...")
    print("----extracting appearance features...")
    appr_feat = extract_appearance_features(cfg, appr_model, video_data)
    print("====finish extracting appearance features")

    # extract question
    print("extract question features...")
    print("----extracting question features...")
    ques_feat = extract_question_features(vocab, question)
    print("====finish extracting question features")

    print("---------------------------------")
    print("model inference...")
    data = [None, None, appr_feat, motion_feat, ques_feat['questions'], ques_feat['questions_len']]
    preds = inference(model, data)
    print("====finish infering")

    answer_vocab = vocab['answer_idx_to_token']
    return answer_vocab[preds[0].item()]

def load_vocab_dict(cfg):
    vocab_dict = {}
    for data_type in cfg['available_vqa_model']:
        cfg_data = load_conf(join(cfg['conf_folder'], data_type + ".yml"))
        vocab_json = cfg_data['inference']['vocab_json']
        vocab_dict[data_type] = load_vocab(data_type + ".yml", vocab_json)

    return vocab_dict

def load_model_dict(cfg, vocab_dict):
    model_dict = {}
    for data_type in cfg['available_vqa_model']:
        model_dict[data_type] = load_vqa_model(cfg, data_type, vocab_dict[data_type])
    return model_dict

def load_model(cfg):
    # model == 'resnext101'
    print("\n---------------------------------")
    print("----buiding motion model...")
    motion_model = build_resnext()
    print("====finish building motion model")
    
    # model == 'resnet101'
    print("---------------------------------")
    print("----buiding appearance model...")
    if not cfg['multi_gpus']:
        torch.cuda.set_device(cfg['gpu_id'])
    appr_model = build_resnet()
    print("====finish building appearance model")

    # load question
    print("---------------------------------")
    print("----buiding question model...")
    vocab_dict = load_vocab_dict(cfg)
    print("====finish building question model")


    print("---------------------------------")
    print("----buiding vqa models...")
    model_dict = load_model_dict(cfg, vocab_dict)
    print("====finish building vqa models")

    return motion_model, appr_model, vocab_dict, model_dict
