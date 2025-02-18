import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
from torch import Tensor
import os
from itertools import groupby
from operator import itemgetter
import gc
import subprocess
import random


def load_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def get_q_embeddings(retrieve_model, dataset, split='test'):
    with open('embeddings/' + split + '/' + retrieve_model + '/q_' + dataset + '_' + retrieve_model + '_embeddings.pkl',
              'rb') as f:
        query_embedding_list = pickle.load(f)
    return query_embedding_list


def get_c_embeddings(retrieve_model, dataset, split='test'):
    with open(
            'embeddings/' + split + '/' + retrieve_model + '/c_' + dataset + '_' + retrieve_model + '_embeddings.pkl',
            'rb') as f:
        candidate_embedding_list = pickle.load(f)
    return candidate_embedding_list


class DATASTORE:
    def __init__(self, dataset, split):
        self.split_q_dir = '../../data/processed_data/' + split + '/' + dataset + '_' + split + '_qlist.json'
        self.split_c_dir = '../../data/processed_data/' + split + '/' + dataset + '_' + split + '_clist.json'
        self.query_sequence_list = load_json_lines(self.split_q_dir)
        self.candidate_sequence_list = load_json_lines(self.split_c_dir)
        self.dataset = dataset

    def get_query_amount(self):
        return len(self.query_sequence_list)

    def group_query_by_date(self):
        self.query_sequence_list.sort(key=itemgetter("query_date"))
        processed_qlist_by_date = {date: list(items) for date, items in
                                   groupby(self.query_sequence_list, key=itemgetter("query_date"))}
        return processed_qlist_by_date

    def group_candidate_by_date(self):
        self.candidate_sequence_list.sort(key=itemgetter("candidate_date"))
        processed_clist_by_date = {date: list(items) for date, items in
                                   groupby(self.candidate_sequence_list, key=itemgetter("candidate_date"))}
        return processed_clist_by_date

    def group_no_freeze_query_str_by_date(self):
        temp_query_sequence_list = []
        for query_sequence in self.query_sequence_list:
            if query_sequence['movement'] != 'freeze':
                query_str = get_embedding_sequence_str(query_sequence, query_or_candidate='query')
                query_sequence.update({'query_str': query_str})
                temp_query_sequence_list.append(query_sequence)
        temp_query_sequence_list.sort(key=itemgetter("query_date"))
        processed_qlist_by_date = {date: list(items) for date, items in
                                   groupby(temp_query_sequence_list, key=itemgetter("query_date"))}
        return processed_qlist_by_date

    def group_candidate_str_by_date(self):
        temp_candidate_sequence_list = []
        for candidate_sequence in self.candidate_sequence_list:
            candidate_str = get_embedding_sequence_str(candidate_sequence, query_or_candidate='candidate')
            candidate_sequence.update({'candidate_str': candidate_str})
            temp_candidate_sequence_list.append(candidate_sequence)
        temp_candidate_sequence_list.sort(key=itemgetter("candidate_date"))
        processed_clist_by_date = {date: list(items) for date, items in
                                   groupby(temp_candidate_sequence_list, key=itemgetter("candidate_date"))}
        return processed_clist_by_date

    def group_embeddings_by_date_gpu(self, retrieve_model, q_or_c):
        temp_embedding_list_by_date = []
        if q_or_c == 'query':
            embedding_list_by_date = get_q_embeddings(retrieve_model, self.dataset)
        elif q_or_c == 'candidate':
            embedding_list_by_date = get_c_embeddings(retrieve_model, self.dataset)
        flag = False
        for item in embedding_list_by_date:
            date1 = list(item.keys())[0]
            queries_on_date1 = item[date1]
            query_embeddings_on_date1 = []
            for query in queries_on_date1:
                raw_embedding = torch.tensor(query['embedding'])
                if not flag:
                    print('The shape of embedding: ', raw_embedding.shape)
                    print('The dimension: ', len(raw_embedding.shape))
                    flag = True
                if len(raw_embedding.shape) == 2:
                    q_embedding_gpu = raw_embedding[0].clone().detach().requires_grad_(True).to('cuda')
                elif len(raw_embedding.shape) == 1:
                    q_embedding_gpu = raw_embedding.clone().detach().requires_grad_(True).to('cuda')
                query['embedding'] = q_embedding_gpu
                query_embeddings_on_date1.append(query)
            temp_embedding_list_by_date.append({date1: query_embeddings_on_date1})
        return temp_embedding_list_by_date

    def get_candidate_str_by_index(self, candidate_index):
        candidate_sequence = [d for d in self.candidate_sequence_list if d['data_index'] == candidate_index][0]
        candidate_movement = candidate_sequence['movement']
        candidate_str = get_embedding_sequence_str(candidate_sequence, query_or_candidate='candidate')
        return candidate_str, candidate_movement

    def get_candidate_sequence_by_index(self, candidate_index):
        candidate_sequence = [d for d in self.candidate_sequence_list if d['data_index'] == candidate_index][0]
        return candidate_sequence

    def get_query_sequence_by_index(self, query_index):
        query_sequence = [d for d in self.query_sequence_list if d['data_index'] == query_index][0]
        # query_str = get_embedding_sequence_str(query_sequence, query_or_candidate='query')
        return query_sequence


def get_embedding_sequence_str(sequence, query_or_candidate):
    indicator_key = list(sequence.keys())[-1]
    if query_or_candidate == 'query':
        seq1 = {
            'query_stock': sequence['query_stock'],
            'query_date': sequence['query_date'],
            'recent_date_list': sequence['date_list'],
            'adjusted_close_list': sequence['adj_close_list'],
            # indicator_key: sequence[indicator_key]
        }
    elif query_or_candidate == 'candidate':
        seq1 = {
            'candidate_stock': sequence['candidate_stock'],
            'candidate_date': sequence['candidate_date'],
            'recent_date_list': sequence['date_list'],
            indicator_key: sequence[indicator_key]
        }
    return str(seq1)


def generate_candidate_prompt_for_prob(query_sequence, candidate_list, retrieve_number):
    query_date = query_sequence['query_date']
    query_stock = query_sequence['query_stock']
    # print('index: ', query_sequence_id)
    query_sequence_str = str(get_embedding_sequence_str(sequence=query_sequence, query_or_candidate='query'))
    instruction = (
        "Based on the following information, predict stock movement by filling in the [blank] with 'rise' or 'fall'. Just fill in the blank, do not explain.\n"
    )
    query_inst = ('\nQuery: On ' + query_date + ', the movement of $' + query_stock + ' is ' + '[blank].\n')
    retrieve_prompt = 'These are sequences that may affect this stock\'s price recently:\n'
    query_prompt = 'This is the query sequence:\n'

    # 随机抽取retrieve_number条
    retrieve_result = random.sample(candidate_list, retrieve_number)

    prompt_list = []
    candidate_index_list = []
    candidate_str_list = []
    for candidate_sequence in retrieve_result:
        candidate_index_list.append(candidate_sequence['data_index'])
        candidate_prompt = str({
            'candidate_sequence': str(
                get_embedding_sequence_str(sequence=candidate_sequence, query_or_candidate='candidate'))
        })
        candidate_str_list.append(candidate_prompt)
        prompt = instruction + retrieve_prompt + candidate_prompt + '\n' + query_prompt + query_sequence_str + '\n' + query_inst
        prompt_list.append(prompt)
    return candidate_index_list, candidate_str_list, prompt_list, query_sequence_str
