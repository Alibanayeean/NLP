from torchvision.models import resnet152, ResNet152_Weights, ViT_H_14_Weights, vit_h_14
import numpy as np
from cleantext import clean
import re
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import os

class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = BertModel(BertConfig.from_pretrained('HooshvareLab/bert-fa-zwnj-base'))
        self.layer = nn.Sequential(
          nn.Dropout(),
          nn.LazyLinear(64),
          nn.Dropout(),
          nn.ReLU(),
          nn.LazyLinear(22),
        )


    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        pooled_output = pooled_output.pooler_output
        pooled_output = self.layer(pooled_output)
        return pooled_output
class TaaghcheDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Taaghche. """

    def __init__(self, tokenizer, comments, targets=None, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.tokenizer = tokenizer
        self.max_len = max_len


        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        if self.has_target:
            target = self.label_map[self.targets[item]]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')

        inputs = {
            'comment': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        if self.has_target:
            inputs['targets'] = torch.tensor(target, dtype=torch.long)

        return inputs

class ClassificationModel():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = SentimentModel()
        os.system('gdown --id 1-EJaP-GUcAwuwZCVWZyGFBBpROG8BO37')
        self.net.load_state_dict(torch.load("checkpoint.pth", map_location=torch.device('cpu')))
        self.tokenizer = BertTokenizer.from_pretrained('HooshvareLab/bert-fa-zwnj-base')
        self.net = self.net.to(self.device)
        self.net.eval()

    def classify_text(self, test_dataframe):
        dataloader = self.create_data_loader(test_dataframe.iloc[:, 0].to_numpy(), None, self.tokenizer, 256, 16, None)
        r = []
        for dl in dataloader:
            input_ids = dl['input_ids'].to(self.device)
            attention_mask = dl['attention_mask'].to(self.device)
            token_type_ids = dl['token_type_ids'].to(self.device)
            with torch.no_grad():
                outputs = self.net(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
                label_pred = torch.argmax(outputs, dim=1)
                for i in range(len(label_pred)):
                    val = label_pred[i].item() + 6
                    r.append(val)

        return r
    def create_data_loader(self, x, y, tokenizer, max_len, batch_size, label_list):
      dataset = TaaghcheDataset(
          comments=x,
          targets=y,
          tokenizer=tokenizer,
          max_len=max_len,
          label_list=label_list)

      return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last = False)

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext
    def cleaning(self, text):
        text = text.strip()

        # regular cleaning
        text = clean(text,
            fix_unicode=True,
            to_ascii=False,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="0",
            replace_with_currency_symbol="",
        )

        # cleaning htmls
        text = self.cleanhtml(text)

        # removing wierd patterns
        wierd_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u'\U00010000-\U0010ffff'
            u"\u200d"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\u3030"
            u"\ufe0f"
            u"\u2069"
            u"\u2066"
            # u"\u200c"
            u"\u2068"
            u"\u2067"
            "]+", flags=re.UNICODE)

        text = wierd_pattern.sub(r'', text)

        # removing extra spaces, hashtags
        text = re.sub("#", "", text)
        text = re.sub("\s+", " ", text)

        return text
    def final_clean(self, data):

        data['cleaned_comment'] = data.iloc[:, 0].apply(self.cleaning)
        data = data[['cleaned_comment']]
        data.columns = ['comment']

        return data
c = SentimentModel()