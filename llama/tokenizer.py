# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os
import torch

logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        
        self.v_token_id = 15167
        self.q_token_id = 16492
        self.a_token_id = 22550
        self.nl_id = 13
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, max_feats=10, split='train') -> List[int]:
        # assert type(s) is str

        # i_text = "Instruction: describe the video. Video: "
        # video_start = len(i_text)
        # t1 = [self.bos_id] + self.sp_model.encode(i_text)
        # if split == 'train':
        #     s = self.sp_model.encode(s) + [self.eos_id]
        #     t = [t1 + [-2 for _ in range(max_feats)] + s]
        #
        # else:
        #     t = [t1 + [-2 for _ in range(max_feats)]]

        video_start=10
        t = self.sp_model.encode(s) + [self.eos_id]
        # print(t)
        return t

    def encode_cap(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: describe the what happens in the video.\n"
        c_text = text['c_text']

        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        s2 = c_text
        t2 = self.sp_model.encode(s2) + [self.eos_id]
        t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
        prefix_index = t[0].index(12953) + 5

        return t, prefix_index, video_start

    def encode_vqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, video_start

    def encode_vaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        s2 = o_text + a_text

        s2 = s2 + answer_mapping[answer] + "\n" + q_text
        t2 = self.sp_model.encode(s2) + [self.eos_id]
        t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
        prefix_index = t[0].index(self.q_token_id) + 2

        return t, prefix_index, video_start
    
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def encode_dvqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the dialogue, video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)

        prefix_index = len(t[0]) - 4
        
        return t, prefix_index, video_start, prefix_i, prefix_main

    def encode_dvaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the dialogue, video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = o_text + a_text
        
        if split == 'train':
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)
        
        prefix_index = t[0].index(self.q_token_id) + 2
        
        return t, prefix_index, video_start, prefix_i, prefix_main

        i_text = "Instruction: Predict the video based on the dialogue, question and answer.\n"
        d_text = text['d_text']
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        s1, s2, s3 = i_text, d_text, q_text + o_text + a_text

        t1 = [self.bos_id] + self.sp_model.encode(s1)
        t2 = self.sp_model.encode(s2)
        prefix_i, prefix_q = len(t1), len(t1) + len(t2)

        if split == 'train':
            t3 = self.sp_model.encode(s3 + answer_mapping[answer] + "\n" + "Video:")
            t = [t1 + t2 + t3 + [-2 for _ in range(max_feats)] + [self.eos_id]]
        else:
            t = []
            for k, v in answer_mapping.items():
                t3 = self.sp_model.encode(s3 + v + "\n" + "Video:") + [-2 for _ in range(max_feats)] + [self.eos_id]
                t.append(t1 + t2 + t3)
                
        prefix_index = len(t[0]) - max_feats - 1
        
        return t, prefix_index, prefix_i, prefix_q