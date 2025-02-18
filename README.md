This directory is code for FinSeer. 
- The datasets used in our paper are uploaded in [FinSeer_data](https://huggingface.co/datasets/ElsaShaw/finseer_data/).
- Our retriever is uploaded in [FinSeer](https://huggingface.co/ElsaShaw/FinSeer).
- Our fine-tuned stock llm is uploaded in [StockLLM](https://huggingface.co/ElsaShaw/StockLLM)


### Environments

```shell
# for baseline RAG models and retriever training
pip install InstructorEmbedding
pip install -U FlagEmbedding
pip install sentence-transformers==2.2.2
pip install protobuf==3.20.0
pip install yahoo-finance
python -m pip install -U angle-emb
pip install transformers==4.33.2  # UAE
```

### train the retriever
step 1. get llm feedback scores (src/2_train_retriever/get_llm_feedback_scores.py)
- dataset: acl18, bigdata22, stock23
- target: the file to save, a json with llm probability scores
```python
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default='acl18', type=str)
parser.add_argument('--target', default='acl18.scored.json', type=str)
args = parser.parse_args()
get_all_scores(llm='StockLLM')  # llama family are all supported for this code
```

step 2. select positive and negative candidates (src/2_train_retriever/select_positive_and_combine.py)

Before the step, you should have generated acl18.scored.json, bigdata22.scored.json, stock23.scored.json

This step is to select candidates for all three datasets, and generate a combined train.scored.json file.

Then, follow the steps in [this link](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder), you can finetune your own 'FinSeer' using train.scored.json data.

### predict stock movement with RAG model
step 1. get embeddings of queries and candidates (src/3_stock_movement_prediction/get_embeddings.py)

- q_or_c: query or candidate, we generate the embeddings of query sequences and candidate sequences separately
```python
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--test_dataset', default='bigdata22', type=str)
parser.add_argument('--embedding_model', default='e5',
                    choices=['instructor', 'uae', 'bge', 'llm_embedder', 'e5', 'FinSeer'])
parser.add_argument('--q_or_c', default='candidate')
args = parser.parse_args()
```

step 2. calculate similarity of query and qualified candidates (and get top-5 related candidates)

step 3. predict stock movement with or without retrieval
That is what our three files do, 
- 1_no_retrieval.py
- 2_random_retrieval.py
- 3_similarity_retrieval.py
