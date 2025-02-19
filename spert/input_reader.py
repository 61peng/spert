import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
from parso import split_lines
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer

from spert import util
from spert.entities import Dataset, EntityType, RelationType, Entity, Relation, Document
from spert.opt import spacy
from data_processing.data_val import custom_tokenizer, custom_tokenizerv2, generate_tokens


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None, **kwargs):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size

    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label):
        return self._datasets[label]

    def get_entity_type(self, idx):
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx):
        relation = self._idx2relation_type[idx]
        return relation

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, logger)

    def read(self, dataset_path, dataset_label):
        # 创建dataset对象
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents.values(), desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset):
        jtext = doc["text"]
        jrelations = doc['relation']
        _jentities = doc['entity']

        # 解析text
        doc_tokens, doc_encoding, jentities = self._parse_tokens(jtext, _jentities, dataset, self._tokenizer)
        # 实体长度列表
        entities_len = [jentities[i]['end']-jentities[i]['start'] for i in range(len(jentities))]
        if len(entities_len) == 0:
            entities_len.append(0)
        # 过滤序列长度大于512和实体长度大于100的document
        if len(doc_encoding) <= 512 and max(entities_len) < 100:
            # 解析实体提及
            entities = self._parse_entities(jentities, doc_tokens, dataset)

            # 解析关系
            relations = self._parse_relations(jrelations, entities, dataset)

            # 创建document
            document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)

            return document

    def _parse_tokens(self, jtext, _jentities, dataset, tokenizer):
        doc_tokens = []
        
        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        
        jtokens, jentities = generate_tokens(jtext, _jentities)

        for i, token_phrase in enumerate(jtokens):
            token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)  # BPE切词（改）
            if not token_encoding:
                token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))  # 在doc_encoding中的start和end

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding, jentities
    
    def _parse_entities(self, jentities, doc_tokens, dataset):
        entities = []

        for _,jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = jentity['name']
            ID = jentity['ID']
            entity = dataset.create_entity(entity_type, tokens, phrase, ID)
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset):
        relations = []

        for jrelation in jrelations.values():
            relation_type = self._relation_types[jrelation['label']]

            head_idx = jrelation['arg0']
            tail_idx = jrelation['arg1']

            # 创建relation
            for entity in entities:
                if head_idx == entity.ID:
                    head = entity
                if tail_idx == entity.ID:
                    tail = entity

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations


class JsonPredictionInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, spacy_model: str = None,
                 max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, max_span_size=max_span_size, logger=logger)
        self._spacy_model = spacy_model

        self._nlp = spacy.blank(spacy_model) if spacy is not None and spacy_model is not None else None
        self._nlp.tokenizer = custom_tokenizerv2(self._nlp)
    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents.values(), desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset):

        jtext = document['text']
        doc = self._nlp.make_doc(jtext)
        jtokens = [t.text for t in doc]

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset, self._tokenizer)
        if len(doc_encoding) >= 512:
            for index in range(len(doc_tokens)):
                if doc_tokens[index].span_end >= 512:
                    doc_encoding = doc_encoding[:doc_tokens[index].span_start]
                    doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
                    doc_tokens = doc_tokens[:index]
                    break
        # create document
        document = dataset.create_document(doc_tokens, [], [], doc_encoding)

        return document

    def _parse_tokens(self, jtokens, dataset, tokenizer):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
            if not token_encoding:
                token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding
