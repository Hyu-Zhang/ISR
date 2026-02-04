import json

def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

paths = ['VideoDialog/data/text_data/train_set4DSTC7-AVSD.json','VideoDialog/data/text_data/valid_set4DSTC7-AVSD.json','VideoDialog/data/text_data/test_set4DSTC7-AVSD.json']
for file in paths:
    print(file)
    content = json.load(open(file))["dialogs"] # list

    q_num = 0
    a_num = 0
    count =  0
    max_q = 0
    max_a = 0
    caption_num = 0
    summary_num = 0
    q_list = []
    a_list = []
    for item in content:
        caption = item['caption'].split(" ")
        caption_num += len(caption)
        summary = item['summary'].split(" ")
        summary_num += len(summary)
        for qa in item["dialog"]:

            question = qa["question"].split(" ")
            answer = qa["answer"].split(" ")
            q_num += len(question)
            a_num += len(answer)
            q_list.append(len(question))
            a_list.append(len(answer))
            count += 1
            if len(question) > max_q:
                max_q = len(question)
            if len(answer) > max_a:
                max_a = len(answer)

    print("max question len:", max_q)
    print("avg question len:", q_num/count)
    print("max answer len:", max_a)
    print("avg answer len:", a_num/count)
    print("summary:", summary_num/len(content))
    print("caption:", caption_num/len(content))


