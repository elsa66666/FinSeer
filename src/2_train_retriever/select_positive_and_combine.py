import os
import json


def load_json_files_from_folder(folder_path):
    json_data = []

    # 检查文件夹路径是否存在
    if not os.path.isdir(folder_path):
        print("Error: Folder path does not exist.")
        return json_data

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 遍历文件夹中的所有文件
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否为JSON文件
        if file_name.endswith('.json'):
            # 打开并读取JSON文件
            with open((folder_path + '/' + file_name), 'r') as f:
                for line in f:
                    json_data.append(json.loads(line))

    return json_data


def allocate_pos_and_neg(all_data):
    all_json_list = []
    for sequence in all_data:
        candidate_list = sequence["neg"]
        candidate_index_list = sequence["neg_index"]
        candidate_scores_list = sequence["teacher_scores"]
        sorted_score_list = sorted(candidate_scores_list)  # 升序

        teacher_score_list = []
        original_index_for_top_1 = candidate_scores_list.index(sorted_score_list[-1])
        teacher_score_list.append(sorted_score_list[-1])
        low_index_list = []
        low_candidate_list = []
        for low_score in sorted_score_list[0:15]:
            original_index_for_low = candidate_scores_list.index(low_score)
            low_index = candidate_index_list[original_index_for_low]
            low_candidate = candidate_list[original_index_for_low]
            low_index_list.append(low_index)
            low_candidate_list.append(low_candidate)
            teacher_score_list.append(low_score)
        sequence_with_score = {
            "query_id": sequence["query_id"],
            "query": sequence["query"],
            "pos": [candidate_list[original_index_for_top_1]],
            "pos_index": [candidate_index_list[original_index_for_top_1]],
            "neg": low_candidate_list,
            "neg_index": low_index_list,
            "teacher_scores": teacher_score_list,
            "answers": sequence["answers"],
            "task": sequence["task"]
        }
        all_json_list.append(sequence_with_score)
    return all_json_list


if __name__ == "__main__":
    dir1 = 'scored_training'
    # json_data = load_json_files_from_folder(dir1)
    # all_json_list = allocate_pos_and_neg(json_data)
    data = []
    for dataset_name in ['acl18', 'bigdata22', 'stock23']:
        with open(dataset_name+'.scored.json', 'r') as f:
            for line in f:
                data.append(json.loads(line))
    all_json_list = allocate_pos_and_neg(data)
    with open('train.scored.json', "w") as file:
        for obj in all_json_list:
            # 将JSON对象转换为字符串
            json_str = json.dumps(obj)
            # 将字符串写入文件，并添加换行符
            file.write(json_str + "\n")
