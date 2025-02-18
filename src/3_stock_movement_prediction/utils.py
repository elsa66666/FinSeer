import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
from torch import Tensor
import openai
import os
from itertools import groupby
from operator import itemgetter
import gc
import subprocess
import random


def toggle_llama_query(tokenizer, model, query):
    try:
        inputs = tokenizer(query, return_tensors="pt")
        # Generate
        generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=4096)
        answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer
    except Exception as exc:
        print(exc)
        return 'broken'


def cut_pattern(count_index, string, pattern):
    count = 0
    for m in re.finditer(pattern, string):
        count += 1
        if count == count_index:
            return string[m.end():]
    return string


def get_llama_response(query, tokenizer, model, cut_index=1):
    response_task = toggle_llama_query(tokenizer=tokenizer, model=model, query=query)
    while response_task == "broken":
        response_task = toggle_llama_query(tokenizer=tokenizer, model=model, query=query)
    response_task = cut_pattern(count_index=cut_index * 2,
                                string=response_task,
                                pattern='INST]').lstrip('\t').lstrip('')
    return response_task


def ask_llama3(prompt, tokenizer, model, device):
    messages = [
        {"role": "system", "content": "You are a stock analyst."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def check_answer(generated_answer, reference_label):
    if ('rise' or 'Rise' or 'A' or 'increase') in generated_answer:
        generated_label = 'rise'
    elif ('fall' or 'Fall' or 'B' or 'decline') in generated_answer:
        generated_label = 'fall'
    elif ('freeze' or 'Freeze') in generated_answer:
        generated_label = 'freeze'
    elif generated_answer == 'A':
        generated_label = 'rise'
    elif generated_answer == 'B':
        generated_label = 'fall'
    else:
        generated_label = 'manual_check'

    if reference_label == generated_label:
        flag = 1
    elif generated_label != 'manual_check':
        flag = 0
    else:
        flag = 'manual_check'
    return generated_label, flag


def get_prompt1(retrieve_number, query_sequence, example_sequence_list, is_similar=False):
    '''
    :param retrieve_number: retrieve条数
    :param query_sequence: 原始序列
    :param example_sequence_list: candidate原始序列
    :param is_similar: 是否是最相似的k条，默认是随机的k条
    :return:
    '''

    prompt00 = (
        "Given a stock context and a multiple choice question related to it, select the correct answer from the two options.\n            "
        "Question: ")
    prompt_task0 = "This is a JSON format stock price sequence."
    prompt_task1 = "This is a JSON format stock price sequence and a previous sequence of the same stock for reference."
    prompt_task_k = ("This is a JSON format stock price sequence and "
                     + str(retrieve_number)
                     + " previous sequences of the same stock for reference.")
    prompt000 = "\nWhat is the movement of the query adj_close price on the next trading day? \nThe query sequence:\n"
    prompt_parameter = (
        "\nOptions: A: rise, B: fall.\n            Please answer with A or B only.\n            Answer:\n            ")

    str_query = str(query_sequence)
    example_str = ''
    for sequence in example_sequence_list:
        if is_similar:
            sequence = {"date_list": sequence['candidate_data']['date_list'],
                        "open_list": sequence['candidate_data']['open_list'],
                        "high_list": sequence['candidate_data']['high_list'],
                        "low_list": sequence['candidate_data']['low_list'],
                        "close_list": sequence['candidate_data']['close_list'],
                        "adj_close_list": sequence['candidate_data']['adj_close_list'],
                        "volume_list": sequence['candidate_data']['volume_list'],
                        "movement": sequence['candidate_data']['movement'],
                        "similarity_to_query_sequence": sequence['score']}

        example_str += str(sequence) + '\n'
    # no retrieval
    if retrieve_number == 0:
        prompt_k = prompt00 + prompt_task0 + prompt000 + str_query + prompt_parameter
    else:
        if retrieve_number == 1:  # retrieve one example
            prompt_k = prompt00 + prompt_task1 + prompt000 + str_query + "\nReference sequences: \n" + example_str + prompt_parameter
        elif retrieve_number > 1:
            prompt_k = prompt00 + prompt_task_k + prompt000 + str_query + "\nReference sequences: \n" + example_str + prompt_parameter

    query_llama = ("<s>[INST] <<SYS>> \nYou are a stock analyst. \n<</SYS>>"
                   + prompt_k + '[/INST]')
    return query_llama


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def get_test_data(dataset, flag='test'):
    query_data = []
    directory = '../../data/processed_data/' + flag + '/' + dataset + '_' + flag + '_qlist.json'
    with open(directory, 'r') as f:
        for line in f:
            query_data.append(json.loads(line))

    candidate_data = []
    directory = '../../data/processed_data/' + flag + '/' + dataset + '_' + flag + '_clist.json'
    with open(directory, 'r') as f:
        for line in f:
            candidate_data.append(json.loads(line))

    return query_data, candidate_data


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


def calculate_embedding_similarity(retrieve_model, query_embedding, candidate_embedding):
    if retrieve_model == 'instructor':
        score = cosine_similarity(query_embedding, candidate_embedding)
        score = float(score[0][0])
    elif (retrieve_model == 'bge') or (retrieve_model == 'llm_embedder') or (retrieve_model == 'e5'):
        score = query_embedding @ candidate_embedding.T
        # score = float(score[0][0])
        score = float(score)
        # print(score)
    elif (retrieve_model == 'maven2') or (retrieve_model == 'maven3'):
        score = query_embedding @ candidate_embedding.T
        score = float(score)
        # print(score)
    elif retrieve_model == 'uae':
        # import torch.nn.functional as F
        # query_embedding = F.normalize(query_embedding, p=2, dim=1)
        # candidate_embedding = F.normalize(candidate_embedding, p=2, dim=1)
        score = query_embedding @ candidate_embedding.T
        score = float(score[0][0])
    return score


def ask_llama3_2(pipe, prompt):
    messages = [
        {"role": "system", "content": "You are a stock analyst."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    output = outputs[0]["generated_text"][-1]['content']
    return output


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
            'candidate_movement': sequence['movement'],
            'recent_date_list': sequence['date_list'],
            indicator_key: sequence[indicator_key]
        }
    return str(seq1)


def load_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


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


def load_similarity_result(retrieve_model, dataset, split):
    with open((
            'similar_candidates/' + split + '/' + retrieve_model + '/' + split + '_' + dataset + '_' + retrieve_model + '.pkl'),
            'rb') as f:
        all_retrieve_result = pickle.load(f)
    return all_retrieve_result


def generate_raw_prompt(query_sequence):
    query_date = query_sequence['query_date']
    query_stock = query_sequence['query_stock']
    # print('index: ', query_sequence_id)
    query_sequence_str = str(get_embedding_sequence_str(sequence=query_sequence, query_or_candidate='query')) + '\n'
    instruction = (
        "Based on the following information, predict stock movement by filling in the [blank] with 'rise' or 'fall'. Just fill in the blank, do not explain.\n"
    )
    query_inst = ('\nQuery: On ' + query_date + ', the movement of $' + query_stock + ' is ' + '[blank].\n')
    query_prompt = 'This is the query sequence:\n'

    prompt = instruction + query_prompt + query_sequence_str + query_inst
    return prompt


def generate_random_candidate_prompt(query_sequence, candidate_list, retrieve_number):
    query_date = query_sequence['query_date']
    query_stock = query_sequence['query_stock']
    # print('index: ', query_sequence_id)
    query_sequence_str = str(get_embedding_sequence_str(sequence=query_sequence, query_or_candidate='query')) + '\n'
    instruction = (
        "Based on the following information, predict stock movement by filling in the [blank] with 'rise' or 'fall'. Just fill in the blank, do not explain.\n"
    )
    query_inst = ('\nQuery: On ' + query_date + ', the movement of $' + query_stock + ' is ' + '[blank].\n')
    retrieve_prompt = 'These are sequences that may affect this stock\'s price recently:\n'
    query_prompt = 'This is the query sequence:\n'

    # 随机抽取retrieve_number条
    retrieve_result = random.sample(candidate_list, retrieve_number)

    candidate_prompt = ''
    for candidate_sequence in retrieve_result:
        candidate_prompt += str({
            'candidate_sequence': str(
                get_embedding_sequence_str(sequence=candidate_sequence, query_or_candidate='candidate'))
        }) + '\n'
    prompt = instruction + retrieve_prompt + candidate_prompt + query_prompt + query_sequence_str + query_inst
    return prompt


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
