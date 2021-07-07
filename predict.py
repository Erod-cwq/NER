from src.utils.model_utils import CRFModel
import torch
from transformers import BertTokenizer
import json
import re

model = CRFModel(bert_dir='../bert/usr', num_tags=77)

device = torch.device('cuda:0')

model.load_state_dict(torch.load('./output/checkpoint-26568/model.pt', map_location=torch.device('cpu')), strict=True)

model.to(device)
model.eval()
results = []
tokenizer = BertTokenizer('../bert/usr/vocab.txt')
with open('./data/final_test.txt', 'r', encoding='utf8') as f:
    index = 1
    for line in f.readlines():
        print(index)
        text = line.strip().split('')[1]

        tokens = []
        # text = re.sub('[（）]', '', text)
        for _ch in text:
            if _ch in [' ', '\t', '\n']:
                tokens.append('[BLANK]')
            else:
                if not len(tokenizer.tokenize(_ch)):
                    tokens.append('[INV]')
                else:
                    tokens.append(_ch)

        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=512,
                                            is_pretokenized=True,
                                            return_tensors='pt',
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        model_inputs = {'token_ids': encode_dict['input_ids'],
                        'attention_masks': encode_dict['attention_mask'],
                        'token_type_ids': encode_dict['token_type_ids']}

        for key in model_inputs:
            model_inputs[key] = model_inputs[key].to(device)

        pred_tokens = model(**model_inputs)[0][0]
        with open('ent2ids.json', 'r') as f:
            ent2ids = json.load(f)

        ids2ent = {}
        for key, value in ent2ids.items():
            ids2ent[value] = key

        result = list(map(lambda x: ids2ent[x], pred_tokens))
        results.append(result)
        index += 1

print(len(results))
with open('result.json', 'w', encoding='utf8') as res:
    json.dump(results, res)
