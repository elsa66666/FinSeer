import pandas as pd
from utils import *
import random
from datetime import datetime
from transformers import pipeline
import argparse


def predict_random(llm_path, dataset, retrieve_number, path_to_write):
    pipe = pipeline(
        "text-generation",
        model=llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    datastore = DATASTORE(dataset, 'test')
    qlist_by_date = datastore.group_query_by_date()
    clist_by_date = datastore.group_candidate_by_date()
    flag = False
    query_number = datastore.get_query_amount()
    count = 0

    for query_date, query_sequence_list in qlist_by_date.items():
        query_date_dt = datetime.strptime(query_date, format("%Y-%m-%d"))
        qualified_candidate_list = []
        for candidate_date, candidate_sequence_list in clist_by_date.items():
            candidate_date_dt = datetime.strptime(candidate_date, format("%Y-%m-%d"))
            if query_date_dt > candidate_date_dt:
                qualified_candidate_list += candidate_sequence_list
        for i in range(len(query_sequence_list)):
            query_sequence = query_sequence_list[i]
            reference_answer = query_sequence['movement']
            count += 1
            print('toggling ', count, ' in ', query_number)
            if reference_answer != 'freeze':
                prompt = generate_random_candidate_prompt(query_sequence, qualified_candidate_list, retrieve_number)
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
                                    'check': [str(check)]})

                if not flag:
                    df1.to_csv(path_to_write, mode='a', index=False, header=True)
                    flag = True
                else:
                    df1.to_csv(path_to_write, mode='a', index=False, header=False)
    print("Output directory: ", os.path.abspath(path_to_write))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='bigdata22', type=str)
    parser.add_argument('--retrieve_number', default=5)
    args = parser.parse_args()
    folder_to_write = '../../retrieve_result/2_random_retrieval/'
    if not os.path.exists(folder_to_write):
        os.makedirs(folder_to_write)
    path_to_write = folder_to_write+'random.csv'
    predict_random('stock_llm', args.test_dataset, args.retrieve_number, path_to_write)
