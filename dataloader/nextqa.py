import torch
from .base_dataset import BaseDataset
import pandas as pd
import json
import re
import random

class NextQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/nextqa/{split}.csv')
        self.cap = json.load(open('./data/nextqa/llava_v15_13b_n6.json', 'r'))
        self.features = torch.load(f'./data/{args.dataset}/clipvitl14.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.tokenizer = tokenizer
        self.split = split
        self.qtype_mapping = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx, vid):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        caption = self.cap[str(vid)]
        caption = self.find_caption_random(caption)

        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "

        # c_text = f"Description: {self.cap[str(vid)]}\n"
        c_text = f"Description: {caption}\n"

        text = {'c_text': c_text, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}

        return text

    def find_caption_random(self, text):
        # 使用正则表达式找到所有句号的位置
        periods_positions = [match.start() + 2 for match in re.finditer(r'\.', text)]
        periods_positions[-2:] = []
        periods_positions = [0] + periods_positions

        # 如果没有找到句号，则返回原字符串
        if not periods_positions:
            return text

        # 随机选择一个句号的位置
        selected_period_position = random.choice(periods_positions)
        extracted_text = text[selected_period_position:]

        return extracted_text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        qtype = self.qtype_mapping[self.data['type'].values[idx]]
        answer = self.data['answer'].values[idx]
        text = self._get_text(idx, vid)
        text_id, label, video_start, prefix_index, label_mask = self._get_text_token(text, answer)
        video, video_len = self._get_video(f'{vid}')
        return {"vid": vid,"video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "prefix_index": prefix_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}

    def __len__(self):
        return len(self.data)