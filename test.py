import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
#from timechat.models import *
#from timechat.processors import *
#from timechat.runners import *
#from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/timechat.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--text-query", default="What is he doing?", help="question the video")
    parser.add_argument("--video-path", default='examples/hotdog.mp4', help="path to video file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args(args=[])
    return args

if __name__ == "__main__":


    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    DIR = "ckpt/timechat"
    MODEL_DIR = f"{DIR}/timechat_7b.pth"

    model_config = cfg.model_cfg # eval_configs/timechat.yaml 下面的model部分
    model_config.device_8bit = args.gpu_id #default=0
    model_config.ckpt = MODEL_DIR # timechat_7b.pth
    model_cls = registry.get_model_class(model_config.arch)  #timechat.models.timechat.TimeChat
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id)) #load timechat
    model.eval() # 将模型设置为评估模式的方法

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train #eval_configs.timechat.yaml 33
    # vis_processor_cfg.name: alpro_video_eval class: timechat.processors.video_processor.AlproVideoEvalProcessor
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print(vis_processor_cfg.name)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    video, _ = load_video(
        video_path=args.video_path,
        n_frms=32,
        sampling="uniform", return_msg=True
    ) #C T H W
    # video = vis_processor.transform(video)
    print(video.size())
    C, T, H, W = video.shape #3,32,224,398
    ts.show(video.transpose(0, 1))

    img_list = []
    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    msg = chat.upload_video_without_audio(
        video_path=args.video_path,
        conv=chat_state,
        img_list=img_list,
        n_frms=96,
    )
    #print(msg)
    text_input = "You are given a cooking video from the YouCook2 dataset. Please watch the video and extract a maximum of 10 significant cooking steps. For each step, determine the starting and ending times and provide a concise description. The format should be: 'start time - end time, brief step description'. For example, ' 90 - 102 seconds, spread margarine on two slices of white bread'."
    print(text_input)

    chat.ask(text_input, chat_state)

    num_beams = args.num_beams
    temperature = args.temperature
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    print(llm_message)
