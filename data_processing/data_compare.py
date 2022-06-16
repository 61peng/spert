# data/log/public21_train/2022-06-05_20:38:30.340452/predictions_valid_epoch_80.json
# data/datasets/public_v2/public_test.json

import json

with open('data/log/public21_train/2022-06-05_20:38:30.340452/predictions_valid_epoch_80.json','r') as load_f1:
    load_dict1 = json.load(load_f1)
    
for dict1 in load_dict1:
    if not len(dict1['entities']) == 0:
        for ent in dict1['entities']:
            ent['span'] = dict1['tokens'][ent['start']:ent['end']]


with open('data/datasets/public_v2/public_test.json','r') as load_f2:
    load_dict2 = json.load(load_f2)

for dict2 in load_dict2:
    if not len(dict2['entities']) == 0:
        for ent in dict2['entities']:
            ent['span'] = dict2['tokens'][ent['start']:ent['end']]

load_dict3 = []

for i in range(len(load_dict1)):
    dicts = {}
    load_dict3.append(dicts)
    load_dict3[i]["tokens"] = load_dict1[i]['tokens']
    load_dict3[i]["predict_entities"] = load_dict1[i]['entities']
    load_dict3[i]["predict_relations"] = load_dict1[i]['relations']
    for dict2 in load_dict2:
        if dict2['tokens'] == load_dict1[i]['tokens']:
            load_dict3[i]["true_entities"] = dict2['entities']
            load_dict3[i]["true_realations"] = dict2['relations']

with open("data/datasets/data_compare.json","w") as f:
    json.dump(load_dict3,f)