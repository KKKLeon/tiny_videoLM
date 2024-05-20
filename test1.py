import torch
from timechat.common.registry import registry
from decord import VideoReader
import decord
import numpy as np
from timechat.processors import transforms_video
from timechat.processors.base_processor import BaseProcessor
from timechat.processors.randaugment import VideoRandomAugment
from timechat.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
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

MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")
def upload_video_without_audio(self, video_path, conv, img_list, n_frms=8):
    msg = ""
    if isinstance(video_path, str):  # is a video path
        ext = os.path.splitext(video_path)[-1].lower()
        # print(video_path)
        # image = self.vis_processor(image).unsqueeze(0).to(self.device)
        video, msg = load_video(
            video_path=video_path,
            n_frms=n_frms,
            height=224,
            width=224,
            sampling="uniform", return_msg=True
        )
        print(video[0][0][0])
        video = self.vis_processor.transform(video)
        video = video.unsqueeze(0).to(self.device)
        # print(image)
        if self.model.qformer_text_input:
            # timestamp
            timestamps = msg.split('at')[1].replace('seconds.', '').strip().split(
                ',')  # extract timestamps from msg
            timestamps = [f'This frame is sampled at {t.strip()} second.' for t in timestamps]
            timestamps = self.model.tokenizer(
                timestamps,
                return_tensors="pt",
                padding="longest",
                max_length=32,
                truncation=True,
            )
    else:
        raise NotImplementedError
    # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
    if self.model.qformer_text_input:
        image_emb, _ = self.model.encode_videoQformer_visual(video, timestamp=timestamps)
    else:
        image_emb, _ = self.model.encode_videoQformer_visual(video)
    img_list.append(image_emb)
    conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg)
    return "Received."

if "__name__" == __main__:
    video_path = 'examples/hotdog.mp4'
    img_list = []
    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    msg = chat.upload_video_without_audio(
        video_path,
        conv=chat_state,
        img_list=img_list,
        n_frms=96,
    )