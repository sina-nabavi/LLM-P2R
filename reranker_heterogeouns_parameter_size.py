import copy
from itertools import permutations
from tqdm import tqdm
import time
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import tempfile

def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        print(ranks[-1])
        return ranks[-1]

    for qid in topics:
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks

def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = []
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'content': f"{content}"})
    return messages

def sentence_similarity_batch(batch, query, model, pool=None, multiGPU=False):
    query_emb = model.encode(query, convert_to_tensor=True)
    if multiGPU:
        doc_emb = model.encode_multi_process(batch, pool, batch_size=len(batch))
        doc_emb = torch.from_numpy(doc_emb)
    else:
        doc_emb = model.encode(batch, convert_to_tensor=True)
    sim =  util.pytorch_cos_sim(query_emb,doc_emb)
    del query_emb
    del doc_emb
    gc.collect()
    torch.cuda.empty_cache()
    return sim

def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = permutation
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item

def get_sorted_indexes(input_list):
    sorted_indexes = [index for index, value in sorted(enumerate(input_list), key=lambda x: x[1], reverse=True)]
    return sorted_indexes

import torch.nn.functional as F
from torch import Tensor
import torch
import gc

def similarity_pipeline(item=None, rank_start=0, rank_end=100, clients_num=5, sentence_transformers_list=None):
    query = item['query']
    num_docs = len(item['hits'][rank_start:rank_end])
    client_docs_num = int((num_docs)/clients_num)
    messages = []
    if client_docs_num > 0:
        for i in range(clients_num-1):
            client_messages = create_permutation_instruction(item=item, rank_start=i*client_docs_num, rank_end=(i+1)*client_docs_num)  # chan
            messages.append(client_messages)

    client_messages = create_permutation_instruction(item=item, rank_start=(clients_num-1)*client_docs_num, rank_end=rank_end)  # chan
    messages.append(client_messages)

    doc_scores = []
    for i in range(len(messages)):
        pool = sentence_transformers_list[i].start_multi_process_pool()
        input_texts = []
        for message in messages[i]:
            input_texts.append(message['content'])
        doc_scores = doc_scores + sentence_similarity_batch(input_texts, query, sentence_transformers_list[i], pool, multiGPU=True)[0].tolist()
        sentence_transformers_list[i].stop_multi_process_pool(pool)
    permutation = get_sorted_indexes(doc_scores)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=100)         
    return item


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

import time
import time
import numpy as np

def split_by_delimiter(input_string, delimiter='*'):
    return input_string.split(delimiter)

def main():
    THE_INDEX = {
        'dl19': 'msmarco-v1-passage',
        'dl20': 'msmarco-v1-passage',
        'covid': 'beir-v1.0.0-trec-covid.flat',
        'arguana': 'beir-v1.0.0-arguana.flat',
        'touche': 'beir-v1.0.0-webis-touche2020.flat',
        'news': 'beir-v1.0.0-trec-news.flat',
        'scifact': 'beir-v1.0.0-scifact.flat',
        'fiqa': 'beir-v1.0.0-fiqa.flat',
        'scidocs': 'beir-v1.0.0-scidocs.flat',
        'nfc': 'beir-v1.0.0-nfcorpus.flat',
        'quora': 'beir-v1.0.0-quora.flat',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
        'fever': 'beir-v1.0.0-fever-flat',
        'robust04': 'beir-v1.0.0-robust04.flat',
        'signal': 'beir-v1.0.0-signal1m.flat',

    }

    THE_TOPICS = {
        'dl19': 'dl19-passage',
        'dl20': 'dl20-passage',
        'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test',
        'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test',
        'scidocs': 'beir-v1.0.0-scidocs-test',
        'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
        'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test',
        'signal': 'beir-v1.0.0-signal1m-test',

    }
    from pyserini.search import LuceneSearcher
    from pyserini.search import get_topics, get_qrels
    import tempfile
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llms", help="group of sentence transfomers to use", type=str, default='sentence-transformers/nli-mpnet-base-v2')
    args = parser.parse_args()
    sentence_transformers_list = split_by_delimiter(args.llms)
    sentence_transformers = [SentenceTransformer(sentence_transformer) for sentence_transformer in sentence_transformers_list]
    for data in ['dl19','dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']: 
        print()
        print('#' * 20)
        print(f'Now eval [{data}]')
        print('#' * 20)
        searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        rank_results = run_retriever(topics, searcher, qrels, k=100)
        reranked_data = []
        new_results = []
        for item in tqdm(rank_results):
            if len(item['hits']) == False:
                continue
            new_item = similarity_pipeline(item, rank_start=0, rank_end=100, clients_num=len(sentence_transformers), sentence_transformers_list=sentence_transformers)
            new_results.append(new_item)

        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        from trec_eval_new import EvalFunction
        EvalFunction.write_file(new_results, temp_file)
        EvalFunction.main(THE_TOPICS[data], temp_file)      


if __name__ == '__main__':
    main()