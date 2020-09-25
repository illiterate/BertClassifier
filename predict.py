import torch
from model import BertClassifier
from transformers import BertTokenizer, BertConfig
from train import get_bert_input

labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
bert_config = BertConfig.from_pretrained('../chinese-bert_chinese_wwm_pytorch')
bert_config.num_labels = len(labels)
model = BertClassifier(bert_config)
model.load_state_dict(torch.load('models/best_model.pkl', map_location=torch.device('cpu')))

tokenizer = BertTokenizer(vocab_file='../chinese-bert_chinese_wwm_pytorch/vocab.txt')

print('新闻类别分类')
while True:
    text = input('Input: ')
    input_id, attention_mask, token_type_id = get_bert_input(text, tokenizer)

    input_id = torch.tensor([input_id], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_id = torch.tensor([token_type_id], dtype=torch.long)

    predicted = model(
        input_id,
        attention_mask,
        token_type_id,
    )
    pred_label = torch.argmax(predicted, dim=1)

    print('Label:', labels[pred_label])
