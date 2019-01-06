import json

tmp = []
with open("/home/liqintong/code/ChatBot/DM/Tweets.txt", "r") as t:
    data = t.readlines()
    for d in data:
        tmp.append(d)

with open("/home/liqintong/code/ChatBot/DM/Tweets.json", "w") as fp:
    json.dump(tmp, fp, indent=4)  # indent参数是设置jsons缩进的
