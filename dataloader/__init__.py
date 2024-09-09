import torch
from util import misc
from .nextqa import NextQA
from .dramaqa import DramaQA
from .star import STAR
from .vlep import VLEP
from .tvqa import TVQA
from .intentqa import IntentQA


dataset_mapping = {'nextqa': NextQA, 'star': STAR, 'dramaqa': DramaQA, 'vlep': VLEP, 'tvqa': TVQA, 'intentqa':IntentQA}
num_options_mapping = {'nextqa': 5, 'intentqa': 5, 'star': 4, 'dramaqa': 5, 'vlep': 2, 'tvqa': 5}

def load_data(args, tokenizer, split='train'):
    args.num_options = num_options_mapping[args.dataset]
    dataset = dataset_mapping[args.dataset](args=args, tokenizer=tokenizer, split=split)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_collate,
                                              pin_memory=args.pin_mem, drop_last=False)

    return data_loader

def batch_collate(batch):
    task_list = [ 
        'vqa', 
        'vaq', 
        'cap', 
    ]

    bs = len(batch)
    vid = [batch[i]["vid"] for i in range(bs)]
    video = torch.stack([batch[i]["video"] for i in range(bs)])
    video_len = torch.tensor([batch[i]["video_len"] for i in range(bs)], dtype=torch.long)
    text = [batch[i]["text"] for i in range(bs)]
    qid = [batch[i]["qid"] for i in range(bs)]
    qtype = torch.tensor([batch[i]['qtype'] for i in range(bs)])
    
    text_id = {}
    label = {}
    video_start = {}
    label_mask = {}
    prefix_index = {}

    # Loop through each task type and process accordingly
    for task in task_list:
        text_id[task] = torch.stack([batch[i]['text_id'][task] for i in range(bs)])
        label[task] = torch.stack([batch[i]['label'][task] for i in range(bs)])
        video_start[task] = [batch[i]["video_start"][task] for i in range(bs)]
        label_mask[task] = torch.stack([batch[i]["label_mask"][task] for i in range(bs)])
        prefix_index[task] = [batch[i]["prefix_index"][task] for i in range(bs)]

    answer = torch.tensor([batch[i]["answer"] for i in range(bs)])

    return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
            "prefix_index": prefix_index, "label_mask": label_mask, "qid": qid, "answer": answer, "qtype": qtype}