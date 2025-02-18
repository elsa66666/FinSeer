from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import argparse
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def count_movement_for_candidates(retrieve_result):
    movement_count = [0, 0]  # [rise, freeze, fall]
    score_list = []
    rise_multiple = 1.0
    fall_multiple = 1.0
    possibility_movement = ''
    for candidate_item in retrieve_result:
        candidate_movement = candidate_item['candidate_data']['movement']
        candidate_score = float(candidate_item['score'])
        score_list.append(score_list)
        if candidate_movement == 'rise':
            movement_count[0] += 1
            rise_multiple = rise_multiple * (1 - 0.01 * candidate_score)
        elif candidate_movement == 'fall':
            movement_count[1] += 1
            fall_multiple = fall_multiple * (1 - 0.01 * candidate_score)
    rise_possibility = 1 - rise_multiple
    fall_possibility = 1 - fall_multiple
    if rise_possibility > fall_possibility:
        possibility_movement = 'rise'
    elif fall_possibility > rise_possibility:
        possibility_movement = 'fall'
    print(movement_count)
    return movement_count, score_list, possibility_movement


def generate_prompt(query_sequence, all_retrieve_result, datastore):
    query_date = query_sequence['query_date']
    query_stock = query_sequence['query_stock']
    query_sequence_id = query_sequence['data_index']
    # print('index: ', query_sequence_id)
    query_sequence_str = str(get_embedding_sequence_str(sequence=query_sequence, query_or_candidate='query')) + '\n'
    instruction = (
        "Based on the following information, predict stock movement by filling in the [blank] with 'rise' or 'fall'. Just fill in the blank, do not explain.\n"
    )
    query_inst = ('\nQuery: On ' + query_date + ', the movement of $' + query_stock + ' is ' + '[blank].\n')
    query_prompt = 'This is the query sequence:\n'

    # [0:retrieve_number]: 定位到前k条
    # retrieve_result = all_retrieve_result[0:args.retrieve_number]
    # 统计candidate的涨跌分布
    # movement_count, score_list, possibility_movement = count_movement_for_candidates(retrieve_result)

    candidate_prompt = ''
    count_candidate_movement = [0, 0]  # rise, fall
    retrieve_prompt = ''
    for candidate in all_retrieve_result:
        # prompt01 += (candidate_prompt(candidate) + '\n')
        candidate_sequence, candidate_movement = datastore.get_candidate_str_by_index(candidate['candidate_index'])
        if candidate_movement == 'rise':
            count_candidate_movement[0] += 1
        elif candidate_movement == 'fall':
            count_candidate_movement[1] += 1
        if 'candidate_score' in list(candidate.keys()):
            if retrieve_prompt == '':
                retrieve_prompt = 'These are sequences that may affect this stock\'s price recently, where similarity score shows the similarity to the query sequence:\n'
            candidate_score = round(candidate['candidate_score'], 4)
            candidate_prompt += str({
                'candidate_sequence': candidate_sequence,
                'similarity_score': candidate_score
            }) + '\n'
        else:
            if retrieve_prompt == '':
                retrieve_prompt = 'These are sequences that may affect this stock\'s price recently, where dtw distancee shows the distance to the query sequence (smaller value means more similar):\n'
            candidate_score = round(candidate['dtw_distance'], 4)
            candidate_prompt += str({
                'candidate_sequence': candidate_sequence,
                'dtw_distance': candidate_score
            }) + '\n'

    prompt = instruction + retrieve_prompt + candidate_prompt + query_prompt + query_sequence_str + query_inst
    return prompt, count_candidate_movement


def predict_with_retrieval(llm_path, dataset, retrieve_model, retrieve_number, path_to_write):
    # 加载大模型
    pipe = pipeline(
            "text-generation",
            model=llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
    )
    datastore = DATASTORE(dataset, 'test')
    similarity_result = load_similarity_result(retrieve_model, dataset, 'test')
    flag = False
    query_number = datastore.get_query_amount()
    count = 0
    llm_output_directory = path_to_write

    for item in similarity_result:
        query_date = list(item.keys())[0]
        all_data_on_date1 = item[query_date]
        for stock_data in all_data_on_date1:
            query_index = stock_data['query_index']
            query_sequence = datastore.get_query_sequence_by_index(query_index)
            reference_answer = query_sequence['movement']
            qualified_candidate_list = stock_data['similarity_list'][0:args.retrieve_number]
            count += 1
            print('toggling ', count, ' in ', query_number, '\n')
            if reference_answer != 'freeze':
                prompt, count_candidate_movement = generate_prompt(query_sequence, qualified_candidate_list, datastore)
                response = ask_llama3_2(pipe, prompt)
                print('response: ', response)

                # check llm是否回答正确
                generated_label, check = check_answer(generated_answer=response, reference_label=reference_answer)

                print('generated label: ', generated_label)
                print('reference label: ', reference_answer)
                print('check: ', check)
                print('\n')
                # 结果保存到csv中
                df1 = pd.DataFrame({'index': [str(query_sequence['data_index'])],
                                    'query': [str(prompt)],
                                    'generated_answer': [str(response)],
                                    'generated_label': [str(generated_label)],
                                    'reference_label': [str(reference_answer)],
                                    'candidate_movement': [str(count_candidate_movement)],
                                    'check': [str(check)]})

                if not flag:
                    df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
                    flag = True
                else:
                    df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
    print("Output directory: ", os.path.abspath(llm_output_directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='acl18', type=str)
    parser.add_argument('--llm', default='stock_llm')  # our code support llama3.2 and variants as LLMs
    parser.add_argument('--retrieve_number', default=5)
    parser.add_argument('--retrieve_model', default='dtw')
    parser.add_argument('--flag', default='test')
    args = parser.parse_args()

    folder_to_write = '../../retrieve_result/3_similarity_retrieval/'
    if not os.path.exists(folder_to_write):
        os.makedirs(folder_to_write)

    path_to_write = folder_to_write+ + args.retrieve_model+'.csv'
    predict_with_retrieval(args.llm, args.test_dataset, args.retrieve_model, args.retrieve_number, path_to_write)
