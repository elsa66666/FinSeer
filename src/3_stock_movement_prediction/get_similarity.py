import os
import argparse

import torch
from sklearn.manifold import TSNE
from utils import *
from datetime import datetime
from dtaidistance import dtw
import numpy as np


def get_similarity(dataset, retrieve_model, split='test'):
    datastore = DATASTORE(dataset, split)
    query_embeddings_gpu = datastore.group_embeddings_by_date_gpu(retrieve_model, 'query')
    # print(query_embeddings_gpu[0][list(query_embeddings_gpu[0].keys())[0]])
    candidate_embeddings_gpu = datastore.group_embeddings_by_date_gpu(retrieve_model, 'candidate')
    all_similarity_list = []
    flag = 0
    print('Start calculating similarity.')
    for q_item in query_embeddings_gpu:
        q_date1 = list(q_item.keys())[0]
        queries_on_date1 = q_item[q_date1]
        query_date_dt = datetime.strptime(q_date1, format("%Y-%m-%d"))
        if query_date_dt > datetime.strptime('2000-01-01', format("%Y-%m-%d")):
            qualified_c_embedding_list = []
            qualified_c_index_list = []
            for c_item in candidate_embeddings_gpu:
                c_date = list(c_item.keys())[0]
                candidate_date_dt = datetime.strptime(c_date, format("%Y-%m-%d"))
                if query_date_dt > candidate_date_dt:
                    qualified_c_embedding_list += [i['embedding'] for i in c_item[c_date]]
                    qualified_c_index_list += [i['data']['data_index'] for i in c_item[c_date]]
            # ---------calculate all query similarities on date1--------- #
            similarity_on_date1 = []
            for query in queries_on_date1:
                query_embedding = query['embedding']
                temp_c_embedding_list = qualified_c_embedding_list
                temp_c_index_list = qualified_c_index_list
                candidate_embeddings = torch.stack(temp_c_embedding_list)
                temp_score_list = torch.matmul(candidate_embeddings, query_embedding.unsqueeze(1)).squeeze().tolist()
                result = [{'candidate_index': index, 'candidate_score': score} for index, score in
                          zip(temp_c_index_list, temp_score_list)]
                result.sort(key=lambda x: x['candidate_score'], reverse=True)
                result = result[0:10]   # 只存储得分最高的前十，避免内存溢出
                similarity_on_date1.append({
                    'query_index': query['data']['data_index'],
                    'similarity_list': result
                })
            print('finish toggling date ', q_date1)
            flag += 1
            all_similarity_list.append({q_date1: similarity_on_date1})

            save_dir = str(
                    'similar_candidates/' + split + '/' + retrieve_model + '/' + split + '_' + dataset + '_' + retrieve_model + '.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(all_similarity_list, f)
    return 0


def get_dtw_distance(dataset, split='test'):
    datastore = DATASTORE(dataset, split)
    all_query_sequences = datastore.group_query_by_date()
    all_candidate_sequences = datastore.group_candidate_by_date()
    all_similarity_list = []
    flag = 0
    print('Start calculating similarity.')
    for date1, queries_on_date1 in all_query_sequences.items():
        query_date_dt = datetime.strptime(date1, format("%Y-%m-%d"))
        # 获取所有qualified candidates
        qualified_c_sequence_list = []
        qualified_c_index_list = []
        for date2, candidates_on_date2 in all_candidate_sequences.items():
            candidate_date_dt = datetime.strptime(date2, format("%Y-%m-%d"))
            if query_date_dt > candidate_date_dt:
                for candidate in candidates_on_date2:
                    candidate_indicator = list(candidate.keys())[-1]
                    if candidate_indicator == 'adj_close_list':
                        qualified_c_sequence_list.append(np.array(candidate[candidate_indicator], dtype=np.double))
                        qualified_c_index_list.append(candidate['data_index'])
        # ---------calculate all query similarities on date1--------- #
        similarity_on_date1 = []
        for query in queries_on_date1:
            query_sequence = np.array(query['adj_close_list'], dtype=np.double)
            distance_list = [dtw.distance_fast(query_sequence, candidate_sequence, use_pruning=True)
                             for candidate_sequence in qualified_c_sequence_list]
            result = [{'candidate_index': index, 'dtw_distance': score} for index, score in
                      zip(qualified_c_index_list, distance_list)]
            result.sort(key=lambda x: x['dtw_distance'], reverse=False)  # 升序，距离越小越好
            result = result[0:10]  # 只存储得分最高的前十，避免内存溢出
            similarity_on_date1.append({
                'query_index': query['data_index'],
                'similarity_list': result
            })
        print('finish toggling date ', date1)
        flag += 1
        all_similarity_list.append({date1: similarity_on_date1})

    save_dir = str(
        'similar_candidates/' + split + '/dtw/' + split + '_' + dataset + '_dtw.pkl')
    with open(save_dir, 'wb') as f:
        pickle.dump(all_similarity_list, f)
    return 0

# def draw_embedding_example(model):
#     data, query_start_date = get_test_data(dataset='cikm18', flag='test')
#
#     # start_index用于判定起始index，因为起始行写入csv时加header
#     start_index = -1
#     retrieve_result_list = []
#
#     index_in_candidate = -1
#     index_in_query = -1
#     for i in range(len(data)):
#         query_sequence = data[i]
#         query_date = query_sequence['query_date']
#         query_date1_datetime_format = datetime.strptime(query_date, format("%Y-%m-%d"))
#         index_in_candidate += 1
#         if query_date1_datetime_format >= query_start_date:
#             if start_index == -1:
#                 start_index = i
#             # 序列的id
#             query_sequence_id = data[i]['data_index']
#             print('index: ', query_sequence_id)
#             index_in_query += 1
#             # 抽取最相似的k个
#             query_embedding, retrieve_result = get_similar_retrieval_list(query_date=query_date1_datetime_format,
#                                                                           query_sequence=query_sequence,
#                                                                           retrieve_model=model,
#                                                                           dataset=args.test_dataset,
#                                                                           index_in_query=index_in_query,
#                                                                           flag='test')
#             break
#     tnse = TSNE(n_components=2, perplexity=5, random_state=42)
#     # print(query_embedding.shape)
#     all_embeddings_list = []
#     if model == 'instructor':
#         all_embeddings_list.append(query_embedding[0])
#         all_embeddings_list.append(retrieve_result[0]["candidate_embedding"][0])
#         all_embeddings_list += [x["candidate_embedding"][0] for x in retrieve_result[-15:]]
#     elif model == 'e5':
#         all_embeddings_list.append(query_embedding.cpu()[0].numpy())
#         print(query_embedding.cpu()[0])
#         all_embeddings_list.append(retrieve_result[0]["candidate_embedding"].cpu()[0].numpy())
#         print(retrieve_result[0]["candidate_embedding"].cpu()[0].numpy().shape)
#         all_embeddings_list += [x["candidate_embedding"].cpu()[0].numpy() for x in retrieve_result[-15:]]
#     else:
#         all_embeddings_list.append(query_embedding)
#         all_embeddings_list.append(retrieve_result[0]["candidate_embedding"])
#         all_embeddings_list += [x["candidate_embedding"] for x in retrieve_result[-15:]]
#
#     embeddings_2d = tnse.fit_transform(np.array(all_embeddings_list))
#     plt.figure(figsize=(10, 8))
#     plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='r', label='Query Embedding', alpha=0.7)
#     plt.scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], color='g', label='Positive Candidate\'s Embedding', alpha=0.7)
#     plt.scatter(embeddings_2d[2:, 0], embeddings_2d[2:, 1], color='b', label='Negative Candidates\' Embedding',
#                 alpha=0.7)
#
#     # 添加图例和标题
#     # plt.title('FinSeer embedding example')
#     # plt.xlabel('PC 1')
#     # plt.ylabel('PC 2')
#     plt.legend(loc=0, prop={'size': 12})
#     plt.savefig((model + '_embedding.pdf'), dpi=120)
#     plt.show()
#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='cikm18', type=str)
    parser.add_argument('--retrieve_model', default='dtw', choices=['instructor', 'bge', 'llm_embedder',
                                                                        'uae', 'e5', 'FinSeer', 'dtw'])
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    path_to_write = ('similar_candidates/test/' + args.retrieve_model)
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)

    if args.retrieve_model != 'dtw':
        get_similarity(args.test_dataset, args.retrieve_model)
    else:
        get_dtw_distance(args.test_dataset)
