import json
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

def _parse_tokens(jtokens, tokenizer):
    # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

    # parse tokens
    for _, token_phrase in enumerate(jtokens):
        
        # 将token phrase切分成token，并映射为词典对应id
        token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
        if not token_encoding:
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]        
        doc_encoding += token_encoding
    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]
    return doc_encoding

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 读数据
with open('data/datasets/public21/public_all.json','r') as load_f:
    load_dict = json.load(load_f)

    # print(len(load_dict))
    
token_dict = {}
for ld in load_dict[::-1]:
    for _, token_phrase in enumerate(ld["tokens"]):
        tokens = tokenizer.tokenize(token_phrase)
        if len(tokens) >= 5:
            if token_phrase not in token_dict:
                token_dict[token_phrase] = 1
            else:
                token_dict[token_phrase] += 1

# 按照次数排序
token_list = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)       
with open("data/datasets/public21/split_dict.txt","w") as f:
    
        # d_encoding = _parse_tokens(ld["tokens"], tokenizer)
        # if len(d_encoding) >= 512:
        #     load_dict.remove(ld)
        for token in token_list:
            f.write(token[0] + ": " + str(token[1]) + '\n')

    

    # print(len(load_dict))

# # 生成合并文件
# with open("data/datasets/public21/public_checked.json","w") as f:
#     json.dump(load_dict,f)
# # 生成测试集和训练集
# train_set, test_set = train_test_split(load_dict, test_size=0.1, random_state=42)
# with open("data/datasets/public21/public_train_checked.json","w") as f:
#     json.dump(train_set,f)
# with open("data/datasets/public21/public_test_checked.json","w") as f:
#     json.dump(test_set,f)