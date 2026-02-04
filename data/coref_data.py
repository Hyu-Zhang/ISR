from allennlp.predictors.predictor import Predictor
import json, os
import sys, math
import numpy as np
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost', port=9000)
print("stanfordnlp  finish")
predictor = Predictor.from_path("VideoDialog/coref-spanbert-large-2021.03.10.tar.gz")
print("allennlp  finish")

path = 'VideoDialog/data/coref_text_data/text_data/'
new_path = 'VideoDialog/data/coref_text_data/'
file_names = os.listdir(path)
for file_name in file_names:
    print(file_name)
    file_path = path + file_name
    content = json.load(open(file_path))
    new_dict = content
    new_dialogs = []
    for item in content["dialogs"]:
        print(item["image_id"])
        new_item = item
        dialog = item["dialog"]
        history = ''
        st_history = ''
        for index, qa in enumerate(dialog):
            question, answer = qa["question"], qa["answer"]
            print("before:", question)
            question = question.split()
            answer = answer.split()
            if index == 0:
                new_item["dialog"][index]["allennlp_question"] = " ".join(question)
                new_item["dialog"][index]["stanfordnlp_question"] = " ".join(question)
                if question[-1] == '?': question[-1]=','
                if answer[-1] != '.': answer.append('.')
                history = history + "{} {} ".format(" ".join(question), " ".join(answer))
                st_history = st_history + "{} {} ".format(" ".join(question), " ".join(answer))
                continue
            # allennlp
            result = predictor.predict(document=history + " ".join(question))
            len_q = len(question)
            len_doc = len(result["document"])
            start_index = len_doc - len_q
            end_index = len_doc 
            flag = 0
            for item_list in result["clusters"]:
                if flag == 1:
                    break
                if item_list[-1][1] < start_index:
                    continue
                for i in range(len(item_list)-1, -1, -1):
                    if item_list[i][0] >= start_index and item_list[i][1] < end_index and item_list[i][1] - item_list[i][0] == 0 and result["document"][item_list[i][0]] == 'it':
                        print("TRUE")
                        try:
                            phrase_list = result["document"][item_list[0][0]:item_list[0][1]+1]
                            q_local_index = item_list[i][0] - start_index
                            temp = question[:q_local_index]
                            while '<' in phrase_list or 'nt' in phrase_list:
                                if '<' in phrase_list:
                                    index = phrase_list.index('<')
                                    del phrase_list[index:index+3]
                                    phrase_list.insert(index, '<unk>')
                                if 'nt' in phrase_list:
                                    index = phrase_list.index('nt')
                                    if phrase_list[index-1] == 'do':
                                        del phrase_list[index-1:index+1]
                                        phrase_list.insert(index-1, 'dont')

                            temp.extend(phrase_list)
                            temp.extend(question[q_local_index + 1:])
                            question = temp
                        except:
                            print("error")
                            pass
                        flag = 1

            print("allennlp after:"," ".join(question))
            new_item["dialog"][index]["allennlp_question"] = " ".join(question)
            if question[-1] == '?': question[-1]=','
            if answer[-1] != '.': answer.append('.')
            history = history + "{} {} ".format(" ".join(question), " ".join(answer))
            
            # stanford nlp
            st_q, st_a = qa["question"], qa["answer"]
            st_q = st_q.split()
            st_a = st_a.split()
            if 'it' in st_q:
                question_index = index + 1
                try:
                    coreference = nlp.coref(st_history + " ".join(st_q))
                    for item_list in coreference:
                        for item in item_list:
                            if item[0] == question_index and item[3] == 'it':
                                phrase = item_list[0][3]
                                phrase = phrase.split(" ")
                                if st_q[item[1]-1] == 'it':
                                    tmp = st_q[0:item[1]-1]
                                    tmp.extend(phrase)
                                    tmp.extend(st_q[item[1]:])
                                    st_q = tmp
                                    break
                except:
                    print("Error")
                    pass
            print("stanfordnlp after:", " ".join(st_q))
            new_item["dialog"][index]["stanfordnlp_question"] = " ".join(st_q)
            if st_q[-1] == '?': st_q[-1]=','
            if st_a[-1] != '.': st_a.append('.')
            st_history = st_history + "{} {} ".format(" ".join(st_q), " ".join(st_a))

        new_dialogs.append(new_item)
    new_dict["dialogs"] = new_dialogs
    new_list_json = json.dumps(new_dict, indent=4)
    open(new_path + file_name, 'w', encoding='utf-8').write(new_list_json)
nlp.close()