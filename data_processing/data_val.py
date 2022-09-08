import spacy
import re
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc

# 用户自定义分词规则
def custom_tokenizer(nlp):
    special_cases = {
        "sets":[{"ORTH": "set"}, {"ORTH": "s"}],
        "subsets":[{"ORTH": "subset"}, {"ORTH": "s"}],
        "vertices":[{"ORTH": "vertice"}, {"ORTH": "s"}],
        "polynomials":[{"ORTH": "polynomial"}, {"ORTH": "s"}],
        "bounds":[{"ORTH": "bound"}, {"ORTH": "s"}],
        "integers":[{"ORTH": "integer"}, {"ORTH": "s"}],
        "constants":[{"ORTH": "constant"}, {"ORTH": "s"}],
        "variables":[{"ORTH": "variable"}, {"ORTH": "s"}],
        "numbers":[{"ORTH": "number"}, {"ORTH": "s"}],
        "partitions":[{"ORTH": "partition"}, {"ORTH": "s"}],
        "queues":[{"ORTH": "queue"}, {"ORTH": "s"}],
        "intervals":[{"ORTH": "interval"}, {"ORTH": "s"}],
        "elements":[{"ORTH": "element"}, {"ORTH": "s"}],
        "parameters":[{"ORTH": "parameter"}, {"ORTH": "s"}],
        "graphs":[{"ORTH": "graph"}, {"ORTH": "s"}],
        "spaces":[{"ORTH": "space"}, {"ORTH": "s"}],
        "steps":[{"ORTH": "step"}, {"ORTH": "s"}],
        "sequences":[{"ORTH": "sequence"}, {"ORTH": "s"}],
        "edges":[{"ORTH": "edge"}, {"ORTH": "s"}],
        "trees":[{"ORTH": "tree"}, {"ORTH": "s"}],
        "variances":[{"ORTH": "variance"}, {"ORTH": "s"}],
        "polytopes":[{"ORTH": "polytope"}, {"ORTH": "s"}],
        "transformations":[{"ORTH": "transformation"}, {"ORTH": "s"}],
        "functions":[{"ORTH": "function"}, {"ORTH": "s"}],
        "\\mathcal{X}":[{"ORTH": "\\"}, {"ORTH": "mathcal{X}"}],
        "2d":[{"ORTH": "2"}, {"ORTH": "d"}],
        "2r":[{"ORTH": "2"}, {"ORTH": "r"}],
        "2R":[{"ORTH": "2"}, {"ORTH": "R"}],
        "2m":[{"ORTH": "2"}, {"ORTH": "m"}],
        "2q":[{"ORTH": "2"}, {"ORTH": "q"}],
        "3n":[{"ORTH": "3"}, {"ORTH": "n"}],
        "3N":[{"ORTH": "3"}, {"ORTH": "N"}],
        "4d":[{"ORTH": "4"}, {"ORTH": "d"}],
        "6D":[{"ORTH": "6"}, {"ORTH": "D"}],
        "al":[{"ORTH": "a"}, {"ORTH": "l"}],
        "aF":[{"ORTH": "a"}, {"ORTH": "F"}],
        "aP":[{"ORTH": "a"}, {"ORTH": "P"}],
        "aN":[{"ORTH": "a"}, {"ORTH": "N"}],
        "ds":[{"ORTH": "d"}, {"ORTH": "s"}],
        "dP":[{"ORTH": "d"}, {"ORTH": "P"}],
        "dX":[{"ORTH": "d"}, {"ORTH": "X"}],
        "dZ":[{"ORTH": "d"}, {"ORTH": "Z"}],
        "hu":[{"ORTH": "h"}, {"ORTH": "u"}],
        "ij":[{"ORTH": "i"}, {"ORTH": "j"}],
        "iE":[{"ORTH": "i"}, {"ORTH": "E"}],
        "kd":[{"ORTH": "k"}, {"ORTH": "d"}],
        "kY":[{"ORTH": "k"}, {"ORTH": "Y"}],
        "lD":[{"ORTH": "l"}, {"ORTH": "D"}],
        "mn":[{"ORTH": "m"}, {"ORTH": "n"}],
        "ml":[{"ORTH": "m"}, {"ORTH": "l"}],
        "md":[{"ORTH": "m"}, {"ORTH": "d"}],
        "mF":[{"ORTH": "m"}, {"ORTH": "F"}],
        "no":[{"ORTH": "n"}, {"ORTH": "o"}],
        "ns":[{"ORTH": "n"}, {"ORTH": "s"}],
        "nt":[{"ORTH": "n"}, {"ORTH": "t"}],
        "nF":[{"ORTH": "n"}, {"ORTH": "F"}],
        "pR":[{"ORTH": "p"}, {"ORTH": "R"}],
        "rr":[{"ORTH": "r"}, {"ORTH": "r"}],
        "sx":[{"ORTH": "s"}, {"ORTH": "x"}],
        "tD":[{"ORTH": "t"}, {"ORTH": "D"}],
        "tk":[{"ORTH": "t"}, {"ORTH": "k"}],
        "uv":[{"ORTH": "u"}, {"ORTH": "v"}],
        "xp":[{"ORTH": "x"}, {"ORTH": "p"}],
        "xt":[{"ORTH": "x"}, {"ORTH": "t"}],
        "zp":[{"ORTH": "z"}, {"ORTH": "p"}],
        "zP":[{"ORTH": "z"}, {"ORTH": "P"}],
        "zs":[{"ORTH": "z"}, {"ORTH": "s"}],
        "zS":[{"ORTH": "z"}, {"ORTH": "S"}],
        "AU":[{"ORTH": "A"}, {"ORTH": "U"}],
        "AB":[{"ORTH": "A"}, {"ORTH": "B"}],
        "AN":[{"ORTH": "A"}, {"ORTH": "N"}],
        "AG":[{"ORTH": "A"}, {"ORTH": "G"}],
        "Bp":[{"ORTH": "B"}, {"ORTH": "p"}],
        "BN":[{"ORTH": "B"}, {"ORTH": "N"}],
        "Bt":[{"ORTH": "B"}, {"ORTH": "t"}],
        "By":[{"ORTH": "B"}, {"ORTH": "y"}],
        "Bu":[{"ORTH": "B"}, {"ORTH": "u"}],
        "Dl":[{"ORTH": "D"}, {"ORTH": "l"}],
        "Dt":[{"ORTH": "D"}, {"ORTH": "t"}],
        "RD":[{"ORTH": "R"}, {"ORTH": "D"}],
        "RT":[{"ORTH": "R"}, {"ORTH": "T"}],
        "Sq":[{"ORTH": "S"}, {"ORTH": "q"}],
        "WU":[{"ORTH": "W"}, {"ORTH": "U"}],
        "WB":[{"ORTH": "W"}, {"ORTH": "B"}],
        "UG":[{"ORTH": "U"}, {"ORTH": "G"}],
        "UB":[{"ORTH": "U"}, {"ORTH": "B"}],
        "Ux":[{"ORTH": "U"}, {"ORTH": "x"}],
        "ZP":[{"ORTH": "Z"}, {"ORTH": "P"}],
        "aDl":[{"ORTH": "a"}, {"ORTH": "D"}, {"ORTH": "l"}],
        "lDP":[{"ORTH": "l"}, {"ORTH": "D"}, {"ORTH": "P"}],
        "AWG":[{"ORTH": "A"}, {"ORTH": "W"}, {"ORTH": "G"}],
        "AWB":[{"ORTH": "A"}, {"ORTH": "W"}, {"ORTH": "B"}]}  # 特例词典
    prefix_re = re.compile(r'''^[\[\(\{\}\$\|\.\^"'=@,\-\+_\\]''')  # 前缀标点
    suffix_re = re.compile(r'''[\]\)\}\$\^\|\.!"'-=@,_\\]$''')  # 后缀标点
    infix_re = re.compile(r'''[-~=\.\{\}^\(\)\/\[\]\$@\|\':;_\\,&<>\+]''')  # 中间标点
    simple_url_re = re.compile(r'''^https?://''')
    return Tokenizer(nlp.vocab, rules=special_cases,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                url_match=simple_url_re.match,
                                faster_heuristics=False)

def custom_tokenizerv2(nlp):
    special_cases = {}  # 特例词典
    prefix_re = re.compile(r'''^[_^,.@\[\(\{\]\)\}\$\+\/"'<>]''')  # 前缀标点
    suffix_re = re.compile(r'''[_^,.@!\[\(\{\]\)\}\$\+\/"'<>]$''')  # 后缀标点
    infix_re = re.compile(r'''[-~=_^,.&@\{\}\(\)\[\]\$\+\/]''')  # 中间标点
    simple_url_re = re.compile(r'''^https?://''')
    return Tokenizer(nlp.vocab, rules=special_cases,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                url_match=simple_url_re.match,
                                faster_heuristics=False)

def generate_tokens_v0(jtext,jentities):
    nlp = spacy.blank("en")
    nlp.tokenizer = custom_tokenizer(nlp)
    doc = nlp.make_doc(jtext)
    jtokens_v1 = [token.text for token in doc]
    jtokens_v2 = []
    for jentity in jentities.values():
        span = doc.char_span(jentity['start'],jentity['end'],alignment_mode = "expand")  # 映射到token
        entity_name = jtext[jentity['start']:jentity['end']]
        if not span.text == entity_name and not entity_name.strip() == '':
            if span.text in entity_name:
                jtokens_v1[span.start:span.end]
                pass
            elif entity_name in span.text:
                split_list = list(jtokens_v1[span.start:span.end][0].partition(entity_name))
                while '' in split_list:
                    split_list.remove('')
                jtokens_v1[span.start] = split_list

    for _token in jtokens_v1:
        if type(_token) == list:
            jtokens_v2.extend(_token)
        else:
            jtokens_v2.append(_token)
    
    return jtokens_v2

def generate_tokens(jtext,_jentities):
    nlp = spacy.blank("en")
    nlp.tokenizer = custom_tokenizer(nlp)
    doc = nlp.make_doc(jtext)
    jtokens = [token.text for token in doc]
    jentities = []
    for _jentity in _jentities.values():
        jentity = {}
        entity_name = jtext[_jentity['start']:_jentity['end']]
        span = doc.char_span(_jentity['start'],_jentity['end'],alignment_mode = "expand")  # 映射到token
        jentity['type'] = _jentity['label']
        jentity['start'] = span.start
        jentity['end'] = span.end
        jentity['name'] = entity_name
        jentity["ID"] = _jentity['eid']
        jentities.append(jentity)
    
    return jtokens,jentities

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
           spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)




if __name__ == "__main__":
    nlp = spacy.blank("en")
# 查看nlp.tokenizer的规则
# with open("data/datasets/public_v2/111.json","w") as f:
#     json.dump(nlp.tokenizer.rules,f)
    nlp.tokenizer = custom_tokenizerv2(nlp)
    text = "a one - hot vector $\\Xstvec \\in \\R^{2I}$ "
    doc = nlp(text)
    tok_exp = nlp.tokenizer.explain(text)
    for t in tok_exp:
        print(t[1], "\t", t[0])
    # print([token.text for token in doc])

# span = doc.char_span(190,198,alignment_mode = "expand")
# print(span)
# print(doc[span.start:span.end])
# print(doc.text[190:198])