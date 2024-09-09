import torch
from .base_dataset import BaseDataset
import json
import copy
import pysrt

class TVQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        json_path = f'./data/tvqa/tvqa_{split}.jsonl'
        feature_path = f'./data/tvqa/clipvitl14.pth'

        with open(json_path, "r") as f: 
            data_list = list(f)
        self.data = [json.loads(x) for x in data_list]
        self.features = torch.load(feature_path)
        self.subtitle_path = f'./data/tvqa/tvqa_subtitles/' # provided as castle_s01e01_seg02_clip_00.srt
        self.cap = json.load(open('./data/star/blip2_n6.json', 'r'))
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3 : '(D)', 4: '(E)'}
        self.num_options = 5
        self.sub = args.sub
        print(f"Num {split} data: {len(self.data)}") 

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

    def _get_text(self, idx, choices, vid, start, end):
        question = self.data[idx]["q"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        
        if self.sub:
            dialogue = ''
            
            for t in pysrt.open(self.subtitle_path+f'{vid}'+'.srt'):
                txt = t.text.replace('\n', ' ')
                st = t.start.minutes * 60 + t.start.seconds
                et = t.end.minutes * 60 + t.end.seconds
                if (st >= start and et <= end) or (st <= start and et  <= end and start <= et):
                    dialogue += ' ' + txt

            if dialogue != '': d_text = f"Dialogue: {dialogue}\n"
            else: d_text =  ''
            
        else: 
            d_text = ""
        caption = self.cap[str(vid)]
        caption = self.find_caption_random(caption)
        q_text = f"Question: {question}\n"
        o_text = f"Choices: \n"
        c_text = f"Description: {caption}\n"
        assert len(choices) == self.num_options, "Double check number of choices"
        for i, option in enumerate(choices):
            o_text += f"{self.answer_mapping[i]} {option}\n"

        a_text = f"Answer: The answer is "
        text = {'c_text': c_text, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'd_text': d_text}
        return text

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id][start * 3: (end + 1) * 3, :].float() # 3fps
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def _get_padding_id(self, text_id, prefix_index, prefix_i, prefix_main, type):
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        
        prefix = prefix_index
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, :len(tid)] = tid
                prefix = prefix_index
            else:
                if self.sub and prefix_i != prefix_main:
                    pad = self.max_seq_len - ((prefix_i) + (len(tid) - prefix_main))
                    padding_text_id[i, :prefix_i] = tid[:prefix_i]
                    padding_text_id[i, prefix_i: prefix_i + pad] = tid[prefix_i: prefix_i + pad]
                    padding_text_id[i, prefix_i + pad :] = tid[prefix_main:]

                    if type == "vqa":
                        prefix = len(padding_text_id[i]) - 4
                    elif type == "vaq":
                        if self.split == "train":
                            try:
                                prefix = (padding_text_id == self.tokenizer.q_token_id).nonzero(as_tuple=True)[1].item() + 2
                            except:
                                prefix = (padding_text_id == self.tokenizer.q_token_id).nonzero(as_tuple=True)[1][0].item() + 2
                        else:
                            prefix = (padding_text_id == self.tokenizer.q_token_id).nonzero(as_tuple=True)[1][0].item() + 2
                    else:
                        prefix = len(padding_text_id[i]) - self.max_feats - 1
                else:
                    padding_text_id[i] = tid[:self.max_seq_len]
                    prefix = prefix_index
                print('max sequence length overflow')

        return padding_text_id, prefix

    def _get_text_token(self, text, answer):
        vqa_id, vqa_prefix_index, vqa_video_start, vqa_prefix_i, vqa_prefix_q = self.tokenizer.encode_dvqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vaq_id, vaq_prefix_index, vaq_video_start, vaq_prefix_i, vaq_prefix_q = self.tokenizer.encode_dvaq(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        cap_id, cap_prefix_index, cap_video_start, cap_prefix_i, cap_prefix_q= self.tokenizer.encode_cap(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)

        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vaq_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vaq_id]
        cap_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in cap_id]
        
        vqa_padding_text_id, vqa_prefix_index = self._get_padding_id(vqa_id, vqa_prefix_index, vqa_prefix_i, vqa_prefix_q, "vqa")
        vaq_padding_text_id, vaq_prefix_index = self._get_padding_id(vaq_id, vaq_prefix_index, vaq_prefix_i, vaq_prefix_q, "vaq")
        cap_padding_text_id, cap_padding_text_id = self._get_padding_id(cap_id, cap_prefix_index, cap_prefix_i, cap_prefix_q, "cap")

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float()
        
        vaq_label = copy.deepcopy(vaq_padding_text_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label_mask = vaq_label.ge(0)
        vaq_label[~vaq_label_mask] = 0
        vaq_label_mask = vaq_label_mask.float()
        
        cap_label = copy.deepcopy(cap_padding_text_id)
        cap_label[:, :cap_prefix_index] = -1
        cap_label_mask = cap_label.ge(0)
        cap_label[~cap_label_mask] = 0
        cap_label_mask = cap_label_mask.float()
                
        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        vaq_text_mask = vaq_padding_text_id.ge(0)
        vaq_padding_text_id[~vaq_text_mask] = 0
        cap_text_mask = cap_padding_text_id.ge(0)
        cap_padding_text_id[~cap_text_mask] = 0
        
        # video index
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats)
        vaq_video_index = torch.arange(vaq_prefix_index, vaq_prefix_index + self.max_feats)
        cap_video_index = torch.arange(cap_prefix_index, cap_prefix_index + self.max_feats)
        
        text_id = {'vqa': vqa_padding_text_id, 'vaq': vaq_padding_text_id, 'cap': cap_padding_text_id}
        label = {'vqa': vqa_label, 'vaq': vaq_label, 'cap': cap_label}
        video_start = {'vqa': vqa_video_start, 'vaq': vaq_video_start, 'cap': cap_prefix_index}
        video_index = {'vqa': vqa_video_index, 'vaq': vaq_video_index, 'cap': cap_video_index}
        label_mask = {'vqa': vqa_label_mask, 'vaq': vaq_label_mask, 'cap': cap_label_mask}
        return text_id, label, video_start, video_index, label_mask

    def __getitem__(self, idx):
        vid = self.data[idx]['vid_name']
        qtype = -1
        choices =  [ self.data[idx][f'a{i}'] for i in range(self.num_options)]
        answer =  self.data[idx]['answer_idx']

        start, end = map(float, self.data[idx]['ts'].split('-'))
        try: 
            start, end = round(start), round(end)
        except: 
            start, end = -1000, 1000
        
        video, video_len = self._get_video(f'{vid}', start, end)
        text = self._get_text(idx, choices, f'{vid}', start, end)
        text_id, label, video_start, prefix_index, label_mask = self._get_text_token(text, answer)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "prefix_index": prefix_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}


    def __len__(self):
        return len(self.data)
