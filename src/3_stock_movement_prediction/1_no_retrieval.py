from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from utils import *
from datetime import datetime
from transformers import pipeline
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_test_data(dataset):
    data = []
    directory = '../../data/processed_data/test/' + dataset + '_test_list.json'
    with open(directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # 在整个数据集中，query序列为起始日期的一年后，以确保每一条query都能检索近一年的序列
    query_start_date = datetime.strptime(data[0]['date_list'][0], format("%Y-%m-%d")) + relativedelta(years=1)
    return data, query_start_date


def get_context_json(sequence):
    context_json = {
        "query_stock": sequence['query_stock'],
        "recent_date_list": sequence['date_list'],
        "recent_open_list": sequence['open_list'],
        "recent_high_list": sequence['high_list'],
        "recent_low_list": sequence['low_list'],
        "recent_adjusted_close_list": sequence['adj_close_list'],
        "recent_volume_list": sequence['volume_list'],
        # "movement_list": sequence['movement_list']
    }
    return str(context_json)


def get_query_for_llama3_blank(query_sequence):
    query_date = query_sequence['query_date']
    query_stock = query_sequence['query_stock']
    query_sequence_id = query_sequence['data_index']
    print('index: ', query_sequence_id)
    context_json = str(get_context_json(sequence=query_sequence)) + '\n'
    end_prompt = (
            "Based on the context information, predict the movement by filling in the [blank] with 'rise' or 'fall'. Just fill in the blank, do not explain."
            + '\nQuery: On ' + query_date + ', the movement of $' + query_stock + ' is ' + '[blank].\n')

    query = context_json + end_prompt
    return query


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


def check_answer_and_get_df(query_sequence, prompt, response):
    reference_answer = query_sequence['movement']
    query_sequence_id = query_sequence['data_index']
    print('response: ', response)
    # check llm是否回答正确
    generated_label, check = check_answer(generated_answer=response, reference_label=reference_answer)
    print('generated label: ', generated_label)
    print('reference label: ', reference_answer)
    print('check: ', check)
    print('\n')
    # 结果保存到csv中
    df1 = pd.DataFrame({'index': [str(query_sequence_id)],
                        'query': [str(prompt)],
                        'generated_answer': [str(response)],
                        'generated_label': [str(generated_label)],
                        'reference_label': [str(reference_answer)],
                        'check': [str(check)]})
    return df1


def predict_with_llama3_blank(llama3_dir, test_dataset1):
    device = "cuda"  # the device to load the model onto
    model = AutoModelForCausalLM.from_pretrained(llama3_dir, torch_dtype=torch.bfloat16,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llama3_dir)

    data, query_start_date = get_test_data(dataset=test_dataset1)

    start_index = -1
    for i in range(len(data)):
        query_sequence = data[i]
        query_date = query_sequence['query_date']
        query_date_datetime = datetime.strptime(query_date, format("%Y-%m-%d"))
        reference_answer = query_sequence['movement']
        # 如果date在测试范围内，开始测试
        if (query_date_datetime >= query_start_date) and (reference_answer != 'freeze'):
            if start_index == -1:
                start_index = i

            prompt = get_query_for_llama3_blank(query_sequence)
            if i == start_index:
                print("---------------------LLM input------------------------\n", prompt)
            response = ask_llama3(prompt, tokenizer, model, device)
            df1 = check_answer_and_get_df(query_sequence, prompt, response)

            llm_output_directory = (
                    '../../retrieve_result/1_no_retrieval/[llama3]' + test_dataset1
                    + '.csv')
            if i == start_index:  # 首行写入时加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
            else:  # 后面写入时不用加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
    print("Output directory: ", os.path.abspath(llm_output_directory))


def ask_llama2(tokenizer, model, query):
    try:
        inputs = tokenizer(query, return_tensors="pt")
        # Generate
        generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=2048)
        answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer
    except Exception as exc:
        print(exc)
        return 'broken'


def response_llama2(query, tokenizer, model):
    query = ("<s>[INST] <<SYS>> \nYou are a stock analyst. \n<</SYS>>"
             + query + '[/INST]')
    response_task = ask_llama2(tokenizer=tokenizer, model=model, query=query)
    while response_task == "broken":
        response_task = ask_llama2(tokenizer=tokenizer, model=model, query=query)
    response_task = cut_pattern(count_index=2,
                                string=response_task,
                                pattern='INST]')
    return response_task


def cut_pattern(count_index, string, pattern):
    count = 0
    for m in re.finditer(pattern, string):
        count += 1
        if count == count_index:
            return string[m.end():]
    return string


def predict_with_llama2(llama2_path, test_dataset):
    tokenizer = LlamaTokenizer.from_pretrained(llama2_path)
    model = LlamaForCausalLM.from_pretrained(llama2_path, device_map='auto')
    data, query_start_date = get_test_data(dataset=test_dataset)

    start_index = -1
    for i in range(len(data)):
        query_sequence = data[i]
        query_date = query_sequence['query_date']
        query_date_datetime = datetime.strptime(query_date, format("%Y-%m-%d"))
        reference_answer = query_sequence['movement']
        # 如果date在测试范围内，开始测试
        if (query_date_datetime >= query_start_date) and (reference_answer != 'freeze'):
            if start_index == -1:
                start_index = i

            prompt = get_query_for_llama3_blank(query_sequence)
            if i == start_index:
                print("---------------------LLM input------------------------\n", prompt)
            response = response_llama2(prompt, tokenizer, model)
            print('response: ', response)
            df1 = check_answer_and_get_df(query_sequence, prompt, response)
            llm_output_directory = (
                    '../../retrieve_result/1_no_retrieval/[llama2]' + test_dataset
                    + '.csv')
            if i == start_index:  # 首行写入时加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
            else:  # 后面写入时不用加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
    print("Output directory: ", os.path.abspath(llm_output_directory))


def predict_with_finma(finma_path, test_dataset):
    tokenizer = LlamaTokenizer.from_pretrained(finma_path)
    model = LlamaForCausalLM.from_pretrained(finma_path, device_map='auto')
    data, query_start_date = get_test_data(dataset=test_dataset)

    start_index = -1
    for i in range(len(data)):
        query_sequence = data[i]
        query_date = query_sequence['query_date']
        query_date_datetime = datetime.strptime(query_date, format("%Y-%m-%d"))
        reference_answer = query_sequence['movement']
        query_sequence_id = query_sequence['data_index']
        response = ''
        # 如果date在测试范围内，开始测试
        if (query_date_datetime >= query_start_date) and (reference_answer != 'freeze') and (query_sequence_id > 4021312):
            if start_index == -1:
                start_index = i

            prompt = get_query_for_llama3_blank(query_sequence)
            if i == start_index:
                print("---------------------LLM input------------------------\n", prompt)
            while (response == "broken") or (response == ''):
                response = ask_llama2(tokenizer, model, ("Human: \n{"+prompt+"}\n\nAssistant: \n"))
                response = cut_pattern(1, response, 'Assistant:')
                response = response.lstrip('\t').lstrip(' ').lstrip('\n')
                response = response.lstrip()
                if response == 'Fall':
                    response = 'fall'
                elif response == 'Rise':
                    response = 'rise'
            # check llm是否回答正确
            df1 = check_answer_and_get_df(query_sequence, prompt, response)
            llm_output_directory = (
                    '../../retrieve_result/1_no_retrieval/[finma]' + test_dataset
                    + '.csv')
            if i == start_index:  # 首行写入时加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
            else:  # 后面写入时不用加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
        # print("Output directory: ", os.path.abspath(llm_output_directory))


def predict_raw(llm_path, dataset,output_dir):
    # 加载大模型
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
                prompt = generate_raw_prompt(query_sequence)
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
                    df1.to_csv(output_dir, mode='a', index=False, header=True)
                    flag = True
                else:
                    df1.to_csv(output_dir, mode='a', index=False, header=False)
    print("Output directory: ", os.path.abspath(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='bigdata22', type=str)
    parser.add_argument('--llm', default='stock_llm', choices=['llama3.2', 'stock_llm'])
    args = parser.parse_args()
    folder_to_write = '../../retrieve_result/1_no_retrieval/'
    if not os.path.exists(folder_to_write):
        os.makedirs(folder_to_write)
    output_dir = folder_to_write+args.llm+'.csv'
    predict_raw(args.llm, args.test_dataset, output_dir)

'''
    if args.llm == 'llama3':
        predict_with_llama3_blank(test_dataset1=args.test_dataset)
    elif args.llm == 'gpt':
        # predict_with_gpt(test_dataset=args.test_dataset)
        print()
    elif args.llm == 'finma':
        predict_with_finma(test_dataset=args.test_dataset)
    elif args.llm == 'llama2':
        predict_with_llama2(test_dataset=args.test_dataset)
'''

