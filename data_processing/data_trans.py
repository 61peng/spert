import json
import spacy
from data_val import custom_tokenizer
from transformers import BertTokenizer
import os

def JSON_reader(file):
    td = []
    with open(file,'r') as load_f:
        load_dict = json.load(load_f)
        for ids in load_dict.values():  # 遍历字典的值记得加values()
            mini_td = []
            mini_td.append(ids["text"])
            arr1 = []
            arr2 = []
            mini_dic = {}
            for ent in ids["entity"].values():
                mini_tup1 = (ent["start"],ent["end"],ent["label"],ent["text"])
                arr1.append(mini_tup1)
                arr2.append(ent["eid"])
            mini_dic["entities"] = arr1
            mini_td.append(mini_dic)

            rlt_list = []
            for rlt in ids["relation"].values():
                rlt_dict = {}
                rlt_dict["type"] = rlt['label']
                for arr in range(len(arr2)):
                    if rlt["arg0"] == arr2[arr]:
                        rlt_dict["head"] = arr
                    if rlt["arg1"] == arr2[arr]:
                        rlt_dict["tail"] = arr
                # rlt_dict["head"] = int(''.join(list(filter(str.isdigit, rlt["arg0"]))))-1
                # rlt_dict["tail"] = int(''.join(list(filter(str.isdigit, rlt["arg1"]))))-1
                rlt_list.append(rlt_dict)
            mini_td.append(rlt_list)

            mini_td.append(ids['id'])

            td.append(mini_td)

    return td


if __name__ == "__main__":
    path = 'data/datasets/data_public/origin_data'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    file_list = os.listdir(path)
    for f in file_list:
        file = os.path.join(path,f)
        jsonTD = JSON_reader(file)
        nlp = spacy.blank('en')
        nlp.tokenizer = custom_tokenizer(nlp)
        JSON_list = []

        for text, annotations,relation_anno,orig_id in jsonTD:
            sample_dict = {}

            # tokens
            docs = nlp.make_doc(text)
            sample_dict["tokens"] = []
            for token in docs:
                if tokenizer.tokenize(token.text) == []:
                    sample_dict["tokens"].append('[unused1]')
                else:
                    sample_dict["tokens"].append(token.text)
            
            # entities
            sample_dict["entities"] = []
            for i in range(len(annotations['entities'])):           
                ents = annotations['entities'][i]
                span = docs.char_span(ents[0],ents[1],alignment_mode = "expand")

                ent_dicts = {}
                ent_dicts["type"] = ents[2]
                ent_dicts["start"]  = span.start
                ent_dicts["end"] = span.end
                ent_dicts["spacy_span"] = sample_dict["tokens"][span.start: span.end]
                ent_dicts["brat_span"] = ents[3]
                sample_dict["entities"].append(ent_dicts)
                # else:
                    # print(ents[3])
                    # print(str(sample_dict['tokens'][span.start-2:span.end+2]) + "未被正常切分,实体span为"+ docs.text[ents[0]:ents[1]])

            # relations
            sample_dict["relations"] = []
            for j in range(len(relation_anno)):
                rls_dicts = {}
                rls_dicts["type"] = relation_anno[j]["type"]
                rls_dicts["head"] = relation_anno[j]["head"]
                rls_dicts["tail"] = relation_anno[j]["tail"]
                sample_dict["relations"].append(rls_dicts)

            # orig_id
            sample_dict["orig_id"] = orig_id

            JSON_list.append(sample_dict)
    

        with open(os.path.join("data/datasets/public_v3",f),"w") as target_file:
            json.dump(JSON_list,target_file)
            print(str(f) + "写入完成")
        




# 获得实体序号：int(''.join(list(filter(str.isdigit, '124rewrw342'))))-1