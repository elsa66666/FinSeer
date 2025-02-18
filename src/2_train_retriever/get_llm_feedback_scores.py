import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from torch.nn.functional import softmax
import json
import argparse
from utils import *
import pandas as pd


def get_predict_index(prompt, model, tokenizer):
    device = "cuda"  # the device to load the model onto
    messages = [
        {"role": "system", "content": "You are a stock analyst."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    # Generate token IDs
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_length=1024,
                                       num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95,
                                       pad_token_id=tokenizer.eos_token_id)

    # Obtain logits for the generated token IDs
    with torch.no_grad():
        logits = model(input_ids).logits

    # Apply softmax to logits to get probabilities
    probabilities = softmax(logits, dim=-1)

    # Extract token probabilities for each generated sequence
    generated_probabilities = probabilities[0, -1, :]

    # Get the whole generated response
    # generated_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # get the predict_index. At this index, it will predict rise or fall.
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    flag = 0
    predict_index = -1
    for token, prob in zip(generated_tokens, generated_probabilities):
        predict_index += 1
        if token == '<|end_header_id|>':
            flag += 1
        if flag == 3:
            if (token == 'rise') or (token == 'fall'):
                break
    return predict_index


def get_probability_one_sequence(prompt, model, tokenizer, answer):
    device = "cuda"  # the device to load the model onto
    messages = [
        {"role": "system", "content": "You are a stock analyst."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=1024,
                                 num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95,
                                 pad_token_id=tokenizer.eos_token_id)
    # 解码生成的内容
    generated_tokens = tokenizer.convert_ids_to_tokens(outputs[0])

    # 算概率
    outputs_tensor = torch.tensor(outputs).to(input_ids.device)
    with torch.no_grad():
        logits = model(outputs_tensor).logits

    last_logits = logits[:, -outputs_tensor.size(1):, :]  # 选择生成部分的 logits
    probabilities = torch.softmax(last_logits, dim=-1)  # 计算概率

    # 获取每个生成 token 的概率
    token_probabilities = probabilities[0, range(outputs_tensor.size(1) - 1), outputs_tensor[0][1:]]
    start_index = input_ids.shape[1]
    # print(start_index)

    # 输出置信度和生成 token 列表长度是否一致
    # print(f"Token confidences length: {len(token_probabilities)}")
    # print(f"Generated tokens length: {len(generated_tokens)}")

    prob = 0.0
    for i in range(start_index - 2, len(generated_tokens)):
        # i = flag_index_list[-1] + 2
        # generated_token_id = outputs_tensor[0, i]  # 假设只处理 batch_size = 1
        max_token_id = torch.argmax(probabilities[0, i, :])  # 获取最大概率的 token ID
        max_token_probability = probabilities[0, i, max_token_id]  # 获取对应的概率值
        token_string = tokenizer.decode(max_token_id.item())
        # print(f"在第 {i} 个位置上概率最大的 token: {token_string}")
        # print(f"对应的最大概率: {max_token_probability.item():.4f}")
        if (token_string == 'rise') or (token_string == 'fall'):
            if answer == token_string:
                prob = max_token_probability.item()
            else:
                target_token_id = tokenizer.convert_tokens_to_ids(answer)
                prob = probabilities[0, i, target_token_id].item()
    print("prob: ", prob)
    return prob


def get_json_data(directory):
    data = []
    with open(directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_rise_and_fall_amount(data):
    rise_count = 0
    fall_count = 0
    for sequence in data:
        answer = sequence['answers'][0]
        if answer == 'rise':
            rise_count += 1
        elif answer == 'fall':
            fall_count += 1
    print(rise_count, fall_count)


def get_all_scores(llm):
    model = AutoModelForCausalLM.from_pretrained(
        llm,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(llm)
    query_list = []

    for dataset in [args.dataset]:
        datastore = DATASTORE(dataset, 'test')
        qlist_by_date = datastore.group_query_by_date()
        clist_by_date = datastore.group_candidate_by_date()
        query_number = datastore.get_query_amount()
        count = 0
        flag = False
        for query_date, query_sequence_list in qlist_by_date.items():
            query_date_dt = datetime.strptime(query_date, format("%Y-%m-%d"))
            qualified_candidate_list = []
            for candidate_date, candidate_sequence_list in clist_by_date.items():
                candidate_date_dt = datetime.strptime(candidate_date, format("%Y-%m-%d"))
                if query_date_dt > candidate_date_dt:
                    qualified_candidate_list += candidate_sequence_list
            for i in range(len(query_sequence_list)):
                query_sequence = query_sequence_list[i]
                query_id = query_sequence['data_index']
                reference_answer = query_sequence['movement']
                count += 1
                print(query_id, ': toggling ', count, ' in ', query_number)
                if reference_answer != 'freeze':
                    teacher_score_list = []
                    candidate_index_list, candidate_str_list, prompt_list, query_sequence_str = generate_candidate_prompt_for_prob(query_sequence, qualified_candidate_list, retrieve_number=200)
                    for prompt in prompt_list:
                        score = get_probability_one_sequence(prompt, model, tokenizer, reference_answer)
                        teacher_score_list.append(score)
                    query_json = {
                        "query_id": query_id,
                        "query": query_sequence_str,
                        "pos": [],
                        "neg": candidate_str_list,
                        "pos_index": [],
                        "neg_index": candidate_index_list,
                        "answers": [reference_answer],
                        "task": "icl",
                        "teacher_scores": teacher_score_list
                    }
                    query_list.append(query_json)
                    if not flag:
                        print(query_json)
                        flag = True

    with open(args.target, "w") as file:
        for obj in query_list:
            # 将JSON对象转换为字符串
            json_str = json.dumps(obj)
            # 将字符串写入文件，并添加换行符
            file.write(json_str + "\n")
    return query_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', default='acl18', type=str)
    parser.add_argument('--target', default='w_movement_acl18.scored.json', type=str)
    args = parser.parse_args()
    get_all_scores(llm='StockLLM')  # llama family are all supported for this code

