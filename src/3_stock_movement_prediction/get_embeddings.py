from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
import argparse
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import multiprocessing
from angle_emb import AnglE, Prompts
from utils import *
import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EMBEDDER:
    def __init__(self, embedder_name):
        self.embedder_name = embedder_name
        self.root = '/data/your_name/embedding_models/'  # change this root into yours
        if self.embedder_name == 'instructor':
            if os.path.exists(self.root + 'instructor-large'):
                print('Loading local model ...')
                self.model = INSTRUCTOR(self.root + 'instructor-large')
            else:
                print('No local models, downloading ...')
                self.model = INSTRUCTOR('hkunlp/instructor-large')
        elif self.embedder_name == 'bge':
            if os.path.exists(self.root + 'bge-large-en-v1.5'):
                print('Loading local model ...')
                self.model = SentenceTransformer(self.root + 'bge-large-en-v1.5')
            else:
                print('No local models, downloading ...')
                self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        elif embedder_name == 'llm_embedder':
            if os.path.exists(self.root + 'llm-embedder'):
                print('Loading local model ...')
                self.model = SentenceTransformer(self.root + 'llm-embedder', device=args.device)
            else:
                print('No local models, downloading ...')
                self.model = SentenceTransformer('BAAI/llm-embedder')
        elif embedder_name == 'FinSeer':
            if os.path.exists(self.root + 'finseer'):
                print('Loading local model ...')
                self.model = SentenceTransformer(self.root + 'finseer', device=args.device)
            print('No local models, downloading ...')
            else:
                self.model = SentenceTransformer('')  # complete it
        elif embedder_name == 'e5':
            if os.path.exists(self.root + 'e5-mistral-7b-instruct'):
                print('Loading local model ...')
                self.model = SentenceTransformer(self.root + 'e5-mistral-7b-instruct')
            else:
                print('No local models, downloading ...')
                self.model = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
        elif self.embedder_name == 'uae':
            if os.path.exists(self.root + 'UAE-Large-V1'):
                print('Loading local model ...')
                self.model = AnglE.from_pretrained(self.root + 'UAE-Large-V1', pooling_strategy='cls').cuda()
            else:
                print('No local models, downloading ...')
                self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
            # self.model.set_prompt(prompt=None)

    def embed_query_and_return_json(self, query):
        query_str = query['query_str']
        # -----------toggling instructor------------ #
        # https://github.com/xlang-ai/instructor-embedding
        if self.embedder_name == 'instructor':
            sentences_a = [['Represent a stock sequence as query: ', query_str]]
            embedding = self.model.encode(sentences_a)

        # -----------toggling bge, llm_embedder------------ #
        # https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding
        # https://huggingface.co/BAAI/llm-embedder
        elif self.embedder_name in ['bge', 'llm_embedder', 'FinSeer']:
            embedding = self.model.encode(query_str, normalize_embeddings=True)

        # -----------toggling e5------------ #
        # https://huggingface.co/intfloat/e5-mistral-7b-instruct/tree/main
        elif self.embedder_name == 'e5':
            task = 'Represent a stock sequence as query: '
            queries = [get_detailed_instruct(task, query_str)]
            with torch.no_grad():
                self.model.max_seq_length = 4096
                embedding = self.model.encode(queries)

        # -----------toggling uae------------ #
        # https://huggingface.co/WhereIsAI/UAE-Large-V1
        elif self.embedder_name == 'uae':
            embedding = self.model.encode(query_str, to_numpy=True, prompt=None)
        return {'data': query, 'embedding': embedding}

    def embed_queries_in_parallel(self, queries_on_date1):
        with multiprocessing.Pool(processes=1) as pool:
            candidate_embedding_on_date1 = pool.starmap(self.embed_query_and_return_json,
                                                        [(q,) for q in queries_on_date1], chunksize=50)
        return candidate_embedding_on_date1

    def embed_candidate_and_return_json(self, candidate):
        candidate_str = candidate['candidate_str']
        if self.embedder_name == 'instructor':
            sentences_a = [['Represent a stock sequence as a retrieved candidate: ', candidate_str]]
            embedding = self.model.encode(sentences_a)
        elif self.embedder_name in ['bge', 'llm_embedder', 'FinSeer']:
            embedding = self.model.encode(candidate_str, normalize_embeddings=True)
        elif self.embedder_name == 'e5':
            with torch.no_grad():
                self.model.max_seq_length = 4096
                embedding = self.model.encode(candidate_str)
        elif self.embedder_name == 'uae':
            embedding = self.model.encode(candidate_str, to_numpy=True)
        return {'data': candidate, 'embedding': embedding}

    def embed_candidates_in_parallel(self, candidates_on_date1):
        with multiprocessing.Pool(processes=1) as pool:
            candidate_embedding_on_date1 = pool.starmap(self.embed_candidate_and_return_json,
                                                        [(c,) for c in candidates_on_date1], chunksize=50)
        return candidate_embedding_on_date1


def get_embeddings(test_dataset, embedder_name, q_or_c):
    datastore = DATASTORE(test_dataset, 'test')
    embedder = EMBEDDER(embedder_name)

    if q_or_c == 'query':
        qlist_by_date = datastore.group_no_freeze_query_str_by_date()
        query_embedding_list = []
        # -----------toggling queries by date------------ #
        for date1, queries_on_date1 in qlist_by_date.items():
            if embedder_name == 'e5':
                query_embedding_on_date1 = []
                for query in queries_on_date1:
                    embedding = embedder.embed_query_and_return_json(query)
                    query_embedding_on_date1.append(embedding)
                    print('finish embedding ', query['query_stock'], ' on ', date1)
            else:
                query_embedding_on_date1 = embedder.embed_queries_in_parallel(queries_on_date1)
            query_embedding_list.append({date1: query_embedding_on_date1})
            print(embedder_name, ', finish embedding date ', date1)
        # save query embeddings
        with open(('embeddings/test/' + embedder_name + '/q_' + test_dataset + '_' + embedder_name + '_embeddings.pkl'),
                  'wb') as f:
            pickle.dump(query_embedding_list, f)
        print('finish embedding queries.')

    elif q_or_c == 'candidate':
        clist_by_date = datastore.group_candidate_str_by_date()
        candidate_embedding_list = []
        # -----------toggling candidates by date------------ #
        count = 0
        group = 1
        for date1, candidates_on_date1 in clist_by_date.items():
            # print('start embedding date ', date1)
            '''
            if embedder_name == 'e5':
                candidate_embedding_on_date1 = []
                for candidate in candidates_on_date1:
                    embedding = embedder.embed_candidate_and_return_json(candidate)
                    candidate_embedding_on_date1.append(embedding)
                    print('finish embedding ', candidate['candidate_stock'], ' on ', date1)
            else:
            '''
            candidate_embedding_on_date1 = embedder.embed_candidates_in_parallel(candidates_on_date1)
            candidate_embedding_list.append({date1: candidate_embedding_on_date1})
            print(embedder_name, ', finish embedding date ', date1)
            count += 1
            # save candidate embeddings
            if count % 10 == 0:
                with open((
                        'embeddings/test/' + embedder_name + '/c_' + test_dataset + '_' + embedder_name + '_embeddings_' + str(
                    group) + '.pkl'),
                        'wb') as f:
                    pickle.dump(candidate_embedding_list, f)
                count = 0
                group += 1
                candidate_embedding_list = []
        # the last group
        if count % 10 != 0:
            with open((
                              'embeddings/test/' + embedder_name + '/c_' + test_dataset + '_' + embedder_name + '_embeddings_' + str(
                              group) + '.pkl'), 'wb') as f:
                pickle.dump(candidate_embedding_list, f)
        print('finish embedding candidates.')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='bigdata22', type=str)
    parser.add_argument('--embedding_model', default='e5',
                        choices=['instructor', 'uae', 'bge', 'llm_embedder', 'e5', 'FinSeer'])
    parser.add_argument('--q_or_c', default='candidate')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    path_to_write = ('embeddings/test/' + args.embedding_model)
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)

    multiprocessing.set_start_method('spawn', force=True)
    get_embeddings(args.test_dataset, args.embedding_model, args.q_or_c)
