import json
import os
import operator

if __name__ == "__main__":
    path = 'data/datasets/data_public/'
    file_list = os.listdir(path)
    ent_dict = {}
    for f in file_list:
        file = os.path.join(path,f)
        load_dict = json.load(open(file,'r'))
        for value in load_dict.values():
            for ent in value['entity'].values():
                if ent["text"] not in ent_dict:
                    ent_dict[ent["text"]] = 1
                else:
                    ent_dict[ent["text"]] += 1

    ent_dict = dict(sorted(ent_dict.items(), key=operator.itemgetter(1),reverse=True))
    count = 0
    for ent in ent_dict.keys():
        if '$' in ent:
            count += 1

    print(count)
    
    json.dump(ent_dict,open('data/entities_dict.json','w'))
        




