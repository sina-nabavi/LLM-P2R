#https://huggingface.co/models?library=sentence-transformers all the existing sentence transformers
#https://huggingface.co/sentence-transformers?sort_models=alphabetical#models for models by sentence transformers
#Number of clients 

# Google:
#     BERT: Various configurations include
    #     Base (110M parameters)
    #     Large (340M parameters)
#     RoBERTa, an optimized version of BERT, created by Facebook:
    #     Base (125M parameters)
    #     Large (355M parameters)

# T5:
#     Small (60M parameters)
#     Base (220M parameters)
#     Large (770M parameters)
#     XL (3B parameters)
#     XXL (11B parameters)

# GPT-Neo:
#     125M parameters
#     1.3B parameters
#     2.7B parameters

# OpenT5:
#     Small (300M parameters)
#     Base (1B parameters)
#     Large (3B parameters)
#     XL (13B parameters)

# GPT-2: It has several versions, including:
#     Small (124M parameters)
#     Medium (355M parameters)
#     Large (774M parameters)
#     XL (1.5B parameters)

# LLaMA-2:
#   7B, 13B and 70B

# LLaMA-3:
#   8B and 70B, and the larger not yet released!


## AutoLMs but I do not think I need them except for LLaMA-3
BERT = ['bert-base-uncased', 'bert-large-uncased']
RoBERTA = ['roberta-base', 'roberta-large']

# Sentence transformers
# T5_gtr = ['sentence-transformers/gtr-t5-base','sentence-transformers/gtr-t5-large','sentence-transformers/gtr-t5-xl','sentence-transformers/gtr-t5-xxl']
# T5_sentence = ['sentence-transformers/sentence-t5-base','sentence-transformers/sentence-t5-large','sentence-transformers/sentence-t5-xl','sentence-transformers/sentence-t5-xxl']
# Roberta_stsb = ['sentence-transformers/stsb-roberta-base', 'sentence-transformers/stsb-roberta-large'] #says depricated on HF
# LaBSE = ['sentence-transformers/LaBSE']
# MiniLM_L12 = ['sentence-transformers/all-MiniLM-L12-v1', 'sentence-transformers/all-MiniLM-L12-v2']
# MiniLM_L6 = ['sentence-transformers/all-MiniLM-L6-v1', 'sentence-transformers/all-MiniLM-L6-v2']
# DistilRoberta = ['sentence-transformers/distilroberta']
sentence_transformers = [
    'sentence-transformers/LaBSE',
    #
    'sentence-transformers/all-MiniLM-L12-v1',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-MiniLM-L6-v1',
    'sentence-transformers/all-MiniLM-L6-v2',
    #
    'sentence-transformers/msmarco-MiniLM-L-12-v3',
    'sentence-transformers/msmarco-MiniLM-L12-cos-v5',
    'sentence-transformers/msmarco-MiniLM-L6-cos-v5',
    'sentence-transformers/msmarco-MiniLM-L-6-v3',
    #
    'sentence-transformers/msmarco-bert-base-dot-v5',
    'sentence-transformers/msmarco-bert-co-condensor',
    #
    'sentence-transformers/all-distilroberta-v1',
    #
    'sentence-transformers/all-mpnet-base-v1',
    'sentence-transformers/all-mpnet-base-v2',
    #
    'sentence-transformers/all-roberta-large-v1',
    #
    'sentence-transformers/allenai-specter',
    #
    'sentence-transformers/average_word_embeddings_glove.6B.300d',
    'sentence-transformers/average_word_embeddings_glove.840B.300d',
    #
    'sentence-transformers/average_word_embeddings_komninos',
    #
    'sentence-transformers/average_word_embeddings_levy_dependency',
    #
    'sentence-transformers/bert-base-nli-cls-token',
    'sentence-transformers/bert-base-nli-max-tokens',
    'sentence-transformers/bert-base-nli-mean-tokens',
    'sentence-transformers/bert-large-nli-cls-token',
    'sentence-transformers/bert-large-nli-max-tokens',
    'sentence-transformers/bert-large-nli-mean-tokens',
    #
    'sentence-transformers/bert-base-nli-stsb-mean-tokens',
    'sentence-transformers/bert-large-nli-stsb-mean-tokens',
    #
    'sentence-transformers/bert-base-wikipedia-sections-mean-tokens',
    #
    'sentence-transformers/clip-ViT-B-16',
    'sentence-transformers/clip-ViT-B-32',
    'sentence-transformers/clip-ViT-L-14',
    #
    'sentence-transformers/clip-ViT-B-32-multilingual-v1',
    #
    'sentence-transformers/distilbert-base-nli-max-tokens',
    'sentence-transformers/distilbert-base-nli-mean-tokens',
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
    'sentence-transformers/distilbert-base-nli-stsb-quora-ranking',
    #
    'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
    #
    'sentence-transformers/distilroberta-base-msmarco-v1',
    'sentence-transformers/distilroberta-base-msmarco-v2',
    'sentence-transformers/msmarco-distilroberta-base-v2',
    #
    'sentence-transformers/distiluse-base-multilingual-cased',
    'sentence-transformers/distiluse-base-multilingual-cased-v1',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    #
    'sentence-transformers/facebook-dpr-ctx_encoder-multiset-base',
    'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
    #
    'sentence-transformers/facebook-dpr-question_encoder-multiset-base',
    'sentence-transformers/facebook-dpr-question_encoder-single-nq-base',
    #
    'sentence-transformers/gtr-t5-base',
    'sentence-transformers/gtr-t5-large',
    'sentence-transformers/gtr-t5-xl',
    'sentence-transformers/gtr-t5-xxl',
    #
    'sentence-transformers/msmarco-distilbert-base-dot-prod-v3',
    'sentence-transformers/msmarco-distilbert-base-tas-b',
    'sentence-transformers/msmarco-distilbert-base-v2',
    'sentence-transformers/msmarco-distilbert-base-v3',
    'sentence-transformers/msmarco-distilbert-base-v4',
    'sentence-transformers/msmarco-distilbert-cos-v5',
    'sentence-transformers/msmarco-distilbert-dot-v5',
    #
    'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned',
    'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch',
    #
    'sentence-transformers/msmarco-roberta-base-ance-firstp',
    'sentence-transformers/msmarco-roberta-base-v2',
    'sentence-transformers/msmarco-roberta-base-v3',
    #
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'sentence-transformers/multi-qa-MiniLM-L6-dot-v1',
    'sentence-transformers/multi-qa-distilbert-cos-v1',
    'sentence-transformers/multi-qa-distilbert-dot-v1',
    'sentence-transformers/multi-qa-mpnet-base-cos-v1',
    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    #
    'sentence-transformers/nli-bert-base',
    'sentence-transformers/nli-bert-base-cls-pooling',
    'sentence-transformers/nli-bert-base-max-pooling',
    'sentence-transformers/nli-bert-large',
    'sentence-transformers/nli-bert-large-cls-pooling',
    'sentence-transformers/nli-bert-large-max-pooling',
    #
    'sentence-transformers/nli-distilbert-base',
    'sentence-transformers/nli-distilbert-base-max-pooling',
    #
    'sentence-transformers/nli-distilroberta-base-v2',
    #
    'sentence-transformers/nli-mpnet-base-v2',
    #
    'sentence-transformers/nli-roberta-base',
    'sentence-transformers/nli-roberta-base-v2',
    'sentence-transformers/nli-roberta-large',
    #
    'sentence-transformers/nq-distilbert-base-v1',
    #
    'sentence-transformers/paraphrase-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-MiniLM-L3-v2',
    'sentence-transformers/paraphrase-MiniLM-L6-v2',
    #
    'sentence-transformers/paraphrase-TinyBERT-L6-v2',
    #
    'sentence-transformers/paraphrase-albert-base-v2',
    'sentence-transformers/paraphrase-albert-small-v2',
    #
    'sentence-transformers/paraphrase-distilroberta-base-v1',
    'sentence-transformers/paraphrase-distilroberta-base-v2',
    'sentence-transformers/distilroberta-base-paraphrase-v1',
    #
    'sentence-transformers/paraphrase-mpnet-base-v2',
    #
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    #
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    #
    'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
    #
    'sentence-transformers/quora-distilbert-base',
    #
    'sentence-transformers/quora-distilbert-multilingual',
    #
    'sentence-transformers/roberta-base-nli-mean-tokens',
    'sentence-transformers/roberta-base-nli-stsb-mean-tokens',
    'sentence-transformers/roberta-large-nli-mean-tokens',
    'sentence-transformers/roberta-large-nli-stsb-mean-tokens',
    #
    'sentence-transformers/sentence-t5-base',
    'sentence-transformers/sentence-t5-large',
    'sentence-transformers/sentence-t5-xl',
    'sentence-transformers/sentence-t5-xxl',
    #
    'sentence-transformers/stsb-bert-base',
    'sentence-transformers/stsb-bert-large',
    #
    'sentence-transformers/stsb-distilbert-base',
    #
    'sentence-transformers/stsb-distilroberta-base-v2',
    #
    'sentence-transformers/stsb-mpnet-base-v2',
    #
    'sentence-transformers/stsb-roberta-base',
    'sentence-transformers/stsb-roberta-base-v2',
    #
    'sentence-transformers/stsb-roberta-large',
    #
    'sentence-transformers/stsb-xlm-r-multilingual',
    #
    'sentence-transformers/use-cmlm-multilingual',
    #
    'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens',
    'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
    #
    'sentence-transformers/xlm-r-base-en-ko-nli-ststb',
    #
    'sentence-transformers/xlm-r-bert-base-nli-mean-tokens',
    'sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens',
    #
    'sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1',
    #
    'sentence-transformers/xlm-r-large-en-ko-nli-ststb',
]

datasets = [
    'sentence-transformers/parallel-sentences',
    'sentence-transformers/msmarco-hard-negatives',
    'sentence-transformers/NQ-retrieval',
    'sentence-transformers/reddit-title-body',
    'sentence-transformers/embedding-training-data']

hetero_group_different_sizes_versions_same_dataset_architecture_no_distill = [
    'sentence-transformers/LaBSE',
    'sentence-transformers/all-MiniLM-L12-v1_sentence-transformers/all-MiniLM-L12-v2_sentence-transformers/all-MiniLM-L6-v1_sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/msmarco-MiniLM-L-12-v3_sentence-transformers/msmarco-MiniLM-L12-cos-v5_sentence-transformers/msmarco-MiniLM-L6-cos-v5_sentence-transformers/msmarco-MiniLM-L-6-v3',
    'sentence-transformers/msmarco-bert-base-dot-v5_sentence-transformers/msmarco-bert-co-condensor',
    'sentence-transformers/all-distilroberta-v1',
    'sentence-transformers/all-mpnet-base-v1_sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-roberta-large-v1',
    'sentence-transformers/allenai-specter',
    'sentence-transformers/average_word_embeddings_glove.6B.300d_sentence-transformers/average_word_embeddings_glove.840B.300d',
    'sentence-transformers/average_word_embeddings_komninos',
    'sentence-transformers/average_word_embeddings_levy_dependency',
    'sentence-transformers/bert-base-nli-cls-token_sentence-transformers/bert-base-nli-max-tokens_sentence-transformers/bert-base-nli-mean-tokens_sentence-transformers/bert-large-nli-cls-token_sentence-transformers/bert-large-nli-max-tokens_sentence-transformers/bert-large-nli-mean-tokens',
    'sentence-transformers/bert-base-nli-stsb-mean-tokens_sentence-transformers/bert-large-nli-stsb-mean-tokens',
    'sentence-transformers/bert-base-wikipedia-sections-mean-tokens',
    'sentence-transformers/clip-ViT-B-16_sentence-transformers/clip-ViT-B-32_sentence-transformers/clip-ViT-L-14',
    'sentence-transformers/clip-ViT-B-32-multilingual-v1',
    'sentence-transformers/distilbert-base-nli-max-tokens_sentence-transformers/distilbert-base-nli-mean-tokens_sentence-transformers/distilbert-base-nli-stsb-mean-tokens_sentence-transformers/distilbert-base-nli-stsb-quora-ranking',
    'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
    'sentence-transformers/distilroberta-base-msmarco-v1_sentence-transformers/distilroberta-base-msmarco-v2_sentence-transformers/msmarco-distilroberta-base-v2',
    'sentence-transformers/distiluse-base-multilingual-cased_sentence-transformers/distiluse-base-multilingual-cased-v1_sentence-transformers/distiluse-base-multilingual-cased-v2',
    'sentence-transformers/facebook-dpr-ctx_encoder-multiset-base_sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
    'sentence-transformers/facebook-dpr-question_encoder-multiset-base_sentence-transformers/facebook-dpr-question_encoder-single-nq-base',
    'sentence-transformers/gtr-t5-base_sentence-transformers/gtr-t5-large_sentence-transformers/gtr-t5-xl_sentence-transformers/gtr-t5-xxl',
    'sentence-transformers/msmarco-distilbert-base-dot-prod-v3_sentence-transformers/msmarco-distilbert-base-tas-b_sentence-transformers/msmarco-distilbert-base-v2_sentence-transformers/msmarco-distilbert-base-v3_sentence-transformers/msmarco-distilbert-base-v4_sentence-transformers/msmarco-distilbert-cos-v5_sentence-transformers/msmarco-distilbert-dot-v5',
    'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned_sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch',
    'sentence-transformers/msmarco-roberta-base-ance-firstp_sentence-transformers/msmarco-roberta-base-v2_sentence-transformers/msmarco-roberta-base-v3',
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1_sentence-transformers/multi-qa-MiniLM-L6-dot-v1_sentence-transformers/multi-qa-distilbert-cos-v1_sentence-transformers/multi-qa-distilbert-dot-v1_sentence-transformers/multi-qa-mpnet-base-cos-v1_sentence-transformers/multi-qa-mpnet-base-dot-v1',
    'sentence-transformers/nli-bert-base_sentence-transformers/nli-bert-base-cls-pooling_sentence-transformers/nli-bert-base-max-pooling_sentence-transformers/nli-bert-large_sentence-transformers/nli-bert-large-cls-pooling_sentence-transformers/nli-bert-large-max-pooling',
    'sentence-transformers/nli-distilbert-base_sentence-transformers/nli-distilbert-base-max-pooling',
    'sentence-transformers/nli-distilroberta-base-v2',
    'sentence-transformers/nli-mpnet-base-v2',
    'sentence-transformers/nli-roberta-base_sentence-transformers/nli-roberta-base-v2_sentence-transformers/nli-roberta-large',
    'sentence-transformers/nq-distilbert-base-v1',
    'sentence-transformers/paraphrase-MiniLM-L12-v2_sentence-transformers/paraphrase-MiniLM-L3-v2_sentence-transformers/paraphrase-MiniLM-L6-v2',
    'sentence-transformers/paraphrase-TinyBERT-L6-v2',
    'sentence-transformers/paraphrase-albert-base-v2_sentence-transformers/paraphrase-albert-small-v2',
    'sentence-transformers/paraphrase-distilroberta-base-v1_sentence-transformers/paraphrase-distilroberta-base-v2_sentence-transformers/distilroberta-base-paraphrase-v1',
    'sentence-transformers/paraphrase-mpnet-base-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
    'sentence-transformers/quora-distilbert-base',
    'sentence-transformers/quora-distilbert-multilingual',
    'sentence-transformers/roberta-base-nli-mean-tokens_sentence-transformers/roberta-base-nli-stsb-mean-tokens_sentence-transformers/roberta-large-nli-mean-tokens_sentence-transformers/roberta-large-nli-stsb-mean-tokens',
    'sentence-transformers/sentence-t5-base_sentence-transformers/sentence-t5-large_sentence-transformers/sentence-t5-xl_sentence-transformers/sentence-t5-xxl',
    'sentence-transformers/stsb-bert-base_sentence-transformers/stsb-bert-large',
    'sentence-transformers/stsb-distilbert-base',
    'sentence-transformers/stsb-distilroberta-base-v2',
    'sentence-transformers/stsb-mpnet-base-v2',
    'sentence-transformers/stsb-roberta-base_sentence-transformers/stsb-roberta-base-v2_sentence-transformers/stsb-roberta-large',
    'sentence-transformers/stsb-xlm-r-multilingual',
    'sentence-transformers/use-cmlm-multilingual',
    'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens_sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
    'sentence-transformers/xlm-r-base-en-ko-nli-ststb',
    'sentence-transformers/xlm-r-bert-base-nli-mean-tokens_sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens',
    'sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1',
    'sentence-transformers/xlm-r-large-en-ko-nli-ststb'
]

remaining_exps = [
    # 'sentence-transformers/average_word_embeddings_glove.6B.300d*sentence-transformers/average_word_embeddings_glove.840B.300d',
    # 'sentence-transformers/average_word_embeddings_komninos',
    # 'sentence-transformers/average_word_embeddings_levy_dependency',
    # 'sentence-transformers/clip-ViT-B-16*sentence-transformers/clip-ViT-B-32*sentence-transformers/clip-ViT-L-14',
    'sentence-transformers/facebook-dpr-ctx_encoder-multiset-base*sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
    'sentence-transformers/facebook-dpr-question_encoder-multiset-base*sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
]

import os
for group in remaining_exps:
    command = "python query_generator_heterogeouns_size_stonly.py" + " --llms=" + group
    output = group.replace("sentence-transformers/", "").replace("*","_")
    output_file = "./size/"+ output + ".txt"
    command += " > " + output_file
    os.system(command)
