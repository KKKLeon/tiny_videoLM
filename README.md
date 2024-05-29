21.05 updated:
1. change visual encoder from eva-clip to UMT-L.
Visual encoder is utilized following the path: timechat.py line 93 -> blip.py line 73 -> uml.py line 424
2. support using Valley_instruct_65K to finetune the model.
To download the dataset, you need to go to https://huggingface.co/datasets/luoruipu1/Valley-Instruct-65k to acquire json file and pip install ffmpeg, yt-dlp.

26.05 update:
change LLM backbone to Phi2.

27.05 update:
1. update train yaml.
2. To adpat valley json to train, use utils/valley.py to process the annotation file.
