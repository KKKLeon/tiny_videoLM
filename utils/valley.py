import json
import pdb
import os
from pathlib import Path
from tqdm import tqdm

path = '/home/vault/b211dd/b211dd14/timechat/data/valley_instruct_65k.json'
with open(path, 'r') as fin:
    data = json.load(fin)

sources = {}
exist_video_data_num = {
    'VATEX': 0,
    'jukinmedia': 0
}

video_root = {
    'VATEX': '/home/vault/b211dd/b211dd14/timechat/data/',  # after cropping
    'jukinmedia': '/path/to/jukin/videos/'
}

instruct_data = []
for item in tqdm(data):
    vsource = item['source']
    if vsource == 'VATEX':
        vid = item['video']
    elif vsource == 'jukinmedia':
        vid = item['video']
    else:
        print("not existing resource!")

    if not os.path.exists(os.path.join(video_root[vsource], vid)):
        #print(f"{vsource} {vid} not found!!! Please check your video data directory!")
        continue

    exist_video_data_num[vsource] += 1
    jterm = {}
    jterm["video"] = os.path.join(video_root[vsource], vid)
    convs = []
    for ti in range(0, len(item["conversations"]), 2):
        turn_human = item["conversations"][ti]
        turn_gpt = item["conversations"][ti + 1]
        assert turn_human["from"] == "human"
        assert turn_gpt["from"] == "gpt"
        qa = {}
        qa["q"] = turn_human["value"]
        qa["q"] = qa["q"].replace("<video>\n", "")
        qa["q"] = qa["q"].replace("\n<video>", "")
        qa["a"] = turn_gpt["value"]
        convs.append(qa)

    jterm["QA"] = convs
    instruct_data.append(jterm)

print(exist_video_data_num)
valid_num = sum(list(exist_video_data_num.values()))
print("# samples have videos {}; # total samples {}".format(valid_num, len(data)))

with open("instruct_valley_{}k.json".format(round(valid_num/1000), 1), "w") as fout:
    json.dump(instruct_data, fout, indent=4)