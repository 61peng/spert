import os
import json

def dic_slice(dic, val_start, val_end, test_start, test_end, train_start, train_end):
    _dic = dic
    keys = list(_dic.keys())
    val_slice = {}
    test_slice = {}
    train_slice = {}
    for key in keys[val_start: val_end]:
        val_slice[key] = _dic[key]
    for key in keys[test_start: test_end]:
        test_slice[key] = _dic[key]
    for key in keys[train_start: train_end]:
        train_slice[key] = _dic[key]
    return train_slice, val_slice, test_slice

path = "data/datasets/data_public/data_dev_reference"
filenames = os.listdir(path)

# train_dict = {}
# valid_dict = {}
test_dict = {}
for filename in filenames:
    filepath = os.path.join(path, filename)
    dict = json.load(open(filepath))
    # val_start = 0
    # val_end = int(len(dict) / 10)
    # test_start = val_end
    # test_end = int(len(dict) / 5)
    # train_start = test_end
    # train_end = len(dict)
    # train_slice, valid_slice, test_slice = dic_slice(dict,val_start,val_end,test_start,test_end,train_start,train_end)
    # for key,value in train_slice.items():
    #     train_dict[key] = value
    # for key,value in valid_slice.items():
    #     valid_dict[key] = value
    for key,value in dict.items():
        test_dict[key] = value

# with open("data/datasets/data_public/data/train.json","w") as f1:
#     json.dump(train_dict,f1)
with open("data/datasets/data_public/data_dev_reference/test.json","w") as f2:
    json.dump(test_dict,f2)
# with open("data/datasets/data_public/data/valid.json","w") as f3:
#     json.dump(valid_dict,f3)