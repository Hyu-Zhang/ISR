import json, random, time
import matplotlib.pyplot as plt
file_path = 'VideoDialog/data/text_data/train_set4DSTC7-AVSD.json'
content = json.load(open(file_path))["dialogs"]
x = []
for item in content:
    length = []
    for diag in item['dialog']:
        length.append(len(diag["answer"].split())) 
    num = random.choice(length)
    length = sum(length)/10
    x.append((length, num))
x.sort(key=lambda elem: elem[0])

x1 = [item[0] for item in x]
x2 = [item[1] for item in x]
plt.scatter(x1, x2)
plt.savefig('result_dot_{}.png'.format(time.time()))