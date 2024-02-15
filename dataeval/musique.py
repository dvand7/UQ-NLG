import argparse
import functools
import os
import pathlib
import pickle

# import config
import datasets
import ipdb
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import _settings


@functools.lru_cache()
def get_fs_samples_prompt():
    data = datasets.load_dataset("dgslibisey/MuSiQue", split='train')
    indices = np.random.RandomState(42).choice(len(data), 5)
    ret = ''
    for i in indices:
        i = int(i)
        ret += '\nQ: ' + data[i]['question'] + '\nA: ' + data[i]['answer']
    return ret

def sample_to_prompt(sample, **kwargs):
    if isinstance(sample['question'], list):
        return [sample_to_prompt({'question': _}, **kwargs) for _ in sample['question']]
    # TODO: use unused split to sample random example questions
    return f"""Follow given examples and solve the Test Question at end in similar manner by decomposing the original questions
Examples: [Original Question]:What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?,
[Question1]: which woman portrayed Corliss Archer in the film Kiss and Tell?,
[Answer1]: Shirley Temple,
[Question2]: What government position was held by Shirley Temple?, [Answer2]: Chief of Protocol
[Final Answer]: Chief of Protocol
[Original Question]: When was the American lawyer, lobbyist and political consultant who was a senior member of the presidential campaign of Donald Trump born?,
[Question1]: Who is the American lawyer lobbyist and political consultant who was a senior member of the presidential campaign of Donald Trump?
[Answer1]: Paul Manafort,
[Question2]: When was Paul Manafort born?
[Answer2]: April 1 1949,
[Final Answer]: April 1 1949
[Original Question]: Does Andrew Johnson's presidential number exceed Elagabalus's Emperor number?,
[Question1]: What number president was Andrew Johnson?
[Answer1]: 17,
[Question2]: What number emperor  was Elagabalus?
[Answer2]: 25,
[Question3]: Is 17 greater than 25?
[Answer3]: No,
[Final Answer]: No
[Original Question]: What other political position did the person who introduced the DISCLOSE Act hold?,
[Question1]: Who introduced the DISCLOSE Act?
[Answer1]:Chris Van Hollen
[Question2]: What other political position did Chris Van Hollen hold?
[Answer2]: United States Senator
[Final Answer]: United States Senator
[Original Question]: Are any animals in Chinese calendar Chordata?
[Question1]: What animals are on the Chinese calendar?
[Answer1]:The chinese zodiac based on the Chinese calendar has a number of animals including dogs and pigs.
[Question2]: What is chordata?
[Answer2]: Chordata is a scientific classification of an animals phylum.
[Question3]: Which animals in zodiac calendar have a notochord and dorsal neural tube? Are they chordata ?
[Answer3]: phylum of pigs is chordata
[Final Answer]: pigs
[Original Question]: The youngest daughter of Lady Mary-Gaye Curzon stars with Douglas Smith and Lucien Laviscount  in what 2017 film?,
[Question1]: Who is the youngest daughter of Lady Mary-Gaye Curzon?
[Answer1]: Cressida Bonas,
[Question2]: Cressida Bonas stars with Douglas Smith and Lucien Laviscount in what 2017 film?
[Answer2]: The Bye Bye Man
[Final Answer]:The Bye Bye Man
Following the given examples generate step by step sub questions [Question1] [Answer1], [Question2] [Answer2] and generate [Final Answer] for question  by aggregating answers for sub questions [Question1], [Question2], Test Question:
[Original Question]: {sample['question']}"""

# def sample_to_prompt(sample, **kwargs):
#     if isinstance(sample['question'], list):
#         return [sample_to_prompt({'question': _}, **kwargs) for _ in sample['question']]
#     return f"""Answer these questions:{get_fs_samples_prompt()}
# Q: {sample['question']}
# A:"""

def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        pass
        # eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'LlamaTokenizerFast':
        pass
        # eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['[Original Question]:']] # only original question
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)


def get_dataset(tokenizer):
    df = pd.read_pickle('./data/musique.pkl')
    data = datasets.from_pandas(df)
    
    def process_instance(example):
        example['additional_answers'] = example['answer_aliases']
        example['prompt'] = sample_to_prompt({k:example[k] for k in ['question']})
        inputs = tokenizer(example['prompt'], padding=False, truncation=False)
        outputs = tokenizer(example['answer'], padding=False, truncation=False)
        example['input_ids'] = inputs['input_ids']
        example["attention_mask"] = inputs.attention_mask
        example["labels"] = outputs.input_ids.copy()
        example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
        return example
    
    data = data.map(process_instance, load_from_cache_file=False)
    data = data.remove_columns(['paragraphs', 'question_decomposition', 'answer_aliases'])

    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True)

    return data
if __name__ == '__main__':
    import models

    tokenizer = models.load_tokenizer('mistral-7b')
    #data = datasets.load_dataset("natural_questions", 'dev', beam_runner='DirectRunner')
    data = get_dataset(tokenizer)