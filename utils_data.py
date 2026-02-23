import os
import pickle
from torch_geometric.data import Data
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *
from tqdm import tqdm
img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids,

def load_data_img(args):
    # problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))#21208(文本数据)
    # problems = json.load(open(os.path.join(args.data_root, 'scienceqa/subtriple_problems.json')))#21208(文本数据)
    # problems = json.load(open(os.path.join(args.data_root, 'scienceqa/context_subtriple_with_relations.json')))#21208(文本数据)
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))#21208(文本数据)
    subgraphs = pickle.load(open(os.path.join(args.data_root, 'scienceqa/final_single_process_optimized_cache_clean_subgraphs.pkl'),"rb"))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))#数据集划分 train:12726、val:4241、test:4241、trainval:16967、minitrain:1272、minval:424、minitest:424
    captions = json.load(open(args.caption_file))["captions"]   #为图像生成的包含视觉语义的文本标题 ， 11208
    name_maps = json.load(open('data/name_map.json'))#包含图像的数据中图像对应的索引，11208

    # check #加载图像特征
    if args.img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('vision_features/clip.npy')
    elif args.img_type == "detr":
        image_features = np.load('vision_features/detr.npy')
    elif args.img_type == "vit":
        image_features = torch.load("vision_features/vit.pth")  #torch.Size([11208, 145, 1024])
        # image_features = torch.load(os.path.join(args.data_root, 'vision_features/vit.pth'))
    else:
        image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", image_features.shape)
    # 遍历problems数据，对于每一个问题，如果数据中包含图像，则将图像对应的包含数据语义的文本标题加入到对应的problem数据中
    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""
    #分别获取训练集、验证集、测试集的索引qid
    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n") #12726
    print(f"number of val problems: {len(val_qids)}\n") #4241
    print(f"number of test problems: {len(test_qids)}\n")   #4241

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}

    with open(os.path.join(args.data_root, 'scienceqa/concept.txt'), "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]  # len(id2concept) = 799273

    # 假设subgraphs和id2concept已加载
    for key, value in tqdm(subgraphs.items(), desc="Processing subgraphs"):
        nodes = value["nodes"]
        edge_index = value["edge_index"]
        # id转文本
        node_concepts = [id2concept[int(i)] for i in nodes]
        # 现在node_concepts就是文本形式的节点
        subgraphs[key]["node_concepts"] = [c.replace('_', ' ') for c in node_concepts]

        # 创建重映射
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}
        subgraphs[key]["node_mapping"] = node_to_idx
        # 重映射边索引
        remapped_edge_index = []
        for src, dst in edge_index.t().tolist():
            remapped_edge_index.append([node_to_idx[src], node_to_idx[dst]])
        subgraphs[key]["edge_index"] = torch.tensor(remapped_edge_index).t().contiguous()

    # 返回问题数据，qids，图像索引，图像特征
    return problems, subgraphs, qids, name_maps, image_features

class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }


class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, subgraphs, qids, name_maps, tokenizer, source_len, target_len, triple_len, args, image_features, test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len    # 512
        self.summ_len = target_len  # 512

        self.triple_len = triple_len


        self.target_text = []
        self.source_text = []
        self.image_ids = []
        self.subgs = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            if qid in subgraphs:
                prompt, subgraph, target = build_train_pair(problems, subgraphs, qid, args, curr_le_data)
                self.subgs.append(subgraph)
            else:
                prompt, target = build_train_pair(problems, None, qid, args, curr_le_data)
                # Create empty subgraph for questions without subgraph data
                empty_subgraph = {
                    "nodes": [],
                    "edge_index": torch.tensor([[], []], dtype=torch.long),
                    "edge_type": [],
                    "node_concepts": []
                }
                self.subgs.append(empty_subgraph)
            
            self.target_text.append(target)
            self.source_text.append(prompt)

            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """返回input_ids, attention_masks, image_ids 和 target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        extra_triple_sub = self.subgs[index]

        # 对source_text和target_text进行规范化处理，去除多余的空格
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())
        # 将输入文本转化为token_ids
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        '''
        # 处理三元组数据
        if extra_triple_sub and len(extra_triple_sub) > 0:
            # 处理每个三元组
            triple_elements_ids = []
            triple_elements_masks = []

            # 处理所有三元组
            for triple in extra_triple_sub:
                # 对三元组的每个元素进行编码
                elements_ids = []
                elements_masks = []

                for element in triple:
                    element_encoded = self.tokenizer.encode_plus(
                        element,
                        max_length=self.triple_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    elements_ids.append(element_encoded["input_ids"].squeeze())
                    elements_masks.append(element_encoded["attention_mask"].squeeze())

                # 将三个元素的编码存储为一个三元组
                triple_elements_ids.append(elements_ids)
                triple_elements_masks.append(elements_masks)

            # 将列表转换为张量
            # 这里我们创建一个形状为 [num_triples, 3, triple_len] 的张量
            # 其中3表示三元组的三个元素(主体、关系、客体)
            triple_ids = torch.zeros((len(triple_elements_ids), 3, self.triple_len), dtype=torch.long)
            triple_attention_masks = torch.zeros((len(triple_elements_masks), 3, self.triple_len), dtype=torch.long)

            for i, (ids, masks) in enumerate(zip(triple_elements_ids, triple_elements_masks)):
                for j in range(3):
                    triple_ids[i, j] = ids[j]
                    triple_attention_masks[i, j] = masks[j]
        else:
            # 如果没有三元组，创建一个空的三元组张量
            triple_ids = torch.zeros((1, 3, self.triple_len), dtype=torch.long)
            triple_attention_masks = torch.zeros((1, 3, self.triple_len), dtype=torch.long)
        '''
        source_ids = source["input_ids"].squeeze()  # torch.Size([512])
        source_mask = source["attention_mask"].squeeze()    # torch.Size([512])
        target_ids = target["input_ids"].squeeze().tolist() # 长度为512的list

        # 修改这里：确保image_ids是float32类型
        image_ids = torch.tensor(image_ids, dtype=torch.float32).squeeze()   # torch.Size([145, 1024])
        
        # 处理子图数据，检查是否为空
        if extra_triple_sub["node_concepts"]:
            node_concepts = extra_triple_sub["node_concepts"]
            node_tokens = self.tokenizer.batch_encode_plus(
                node_concepts,
                max_length=self.triple_len,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            # 为空子图创建空的token
            node_tokens = self.tokenizer.batch_encode_plus(
                [""],  # 空字符串
                max_length=self.triple_len,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        if isinstance(extra_triple_sub, dict):
            # 处理边索引，确保即使为空也有正确的形状
            edge_index = extra_triple_sub["edge_index"]
            if edge_index.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            extra_triple_sub = Data(
                x=torch.tensor(extra_triple_sub["nodes"], dtype=torch.long) if len(extra_triple_sub["nodes"]) > 0 else torch.empty((0,), dtype=torch.long),
                edge_index=edge_index,
                edge_type=torch.tensor(extra_triple_sub["edge_type"], dtype=torch.long) if len(extra_triple_sub["edge_type"]) > 0 else torch.empty((0,), dtype=torch.long),
                node_tokens=node_tokens
            )
        # return {
        #     "input_ids": source_ids,
        #     "attention_mask": source_mask,
        #     "image_ids": image_ids,
        #     "triple_ids": triple_ids,  # 形状为 [num_triples, 3, triple_len]
        #     "triple_attention_masks": triple_attention_masks,  # 形状为 [num_triples, 3, triple_len]
        #     "labels": target_ids,
        # }
        return {
                    "input_ids": source_ids,
                    "attention_mask": source_mask,
                    "image_ids": image_ids,
                    "subgs": extra_triple_sub,
                    "labels": target_ids,
                }
from torch_geometric.data import Batch

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    image_ids = torch.stack([item["image_ids"] for item in batch])
    labels = torch.stack([torch.tensor(item["labels"], dtype=torch.long) for item in batch])
    
    # 处理子图数据，过滤掉可能的空子图
    subgs_list = []
    for item in batch:
        subg = item["subgs"]
        # 确保子图有有效的数据结构
        if hasattr(subg, 'x') and hasattr(subg, 'edge_index'):
            subgs_list.append(subg)
        else:
            # 获取第一个有效子图的node_tokens形状作为参考
            reference_tokens = None
            for ref_item in batch:
                ref_subg = ref_item["subgs"]
                if hasattr(ref_subg, 'node_tokens'):
                    reference_tokens = ref_subg.node_tokens
                    break
            
            # 如果找到参考tokens，使用相同的形状创建空tokens
            if reference_tokens is not None:
                empty_tokens = {
                    'input_ids': torch.zeros_like(reference_tokens['input_ids'][:1]),
                    'attention_mask': torch.zeros_like(reference_tokens['attention_mask'][:1])
                }
            else:
                # 默认形状
                empty_tokens = {
                    'input_ids': torch.zeros((1, 32), dtype=torch.long),
                    'attention_mask': torch.zeros((1, 32), dtype=torch.long)
                }
            
            # 创建一个空的Data对象
            empty_data = Data(
                x=torch.empty((0,), dtype=torch.long),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_type=torch.empty((0,), dtype=torch.long),
                node_tokens=empty_tokens
            )
            subgs_list.append(empty_data)
    
    # 关键：将PyG的Data对象批量化
    subgs_batch = Batch.from_data_list(subgs_list)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_ids": image_ids,
        "subgs": subgs_batch,
        "labels": labels,
    }
'''
def custom_collate_fn(batch):
    """
    自定义的collate函数，用于在每个批次中动态确定最大三元组数量
    """
    # 提取批次中的各个字段
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    # 确保image_ids是float32类型
    image_ids = torch.stack([item["image_ids"].to(torch.float32) for item in batch])
    
    # 将labels从列表转换为张量
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    labels = torch.stack(labels)
    
    # 获取批次中每个样本的三元组数量
    triple_counts = [item["triple_ids"].size(0) for item in batch]
    
    # 确定批次中的最大三元组数量
    max_triples_in_batch = max(triple_counts)
    
    # 为每个样本填充或截断三元组，使其数量达到批次中的最大值
    batch_triple_ids = []
    batch_triple_masks = []
    
    for i, item in enumerate(batch):
        sample_triples = item["triple_ids"]  # [num_triples, 3, triple_len]
        sample_masks = item["triple_attention_masks"]  # [num_triples, 3, triple_len]

        curr_triples = sample_triples.size(0)

        # 如果当前样本的三元组数量小于批次中的最大值，则填充
        if curr_triples < max_triples_in_batch:
            # 获取三元组的维度
            triple_element_count = sample_triples.size(1)  # 3
            triple_len = sample_triples.size(2)  # triple_len

            # 创建填充张量，注意维度为 [max_triples_in_batch - curr_triples, 3, triple_len]
            padding_triples = torch.zeros((max_triples_in_batch - curr_triples, triple_element_count, triple_len),
                                         dtype=torch.long, device=sample_triples.device)
            padding_masks = torch.zeros((max_triples_in_batch - curr_triples, triple_element_count, triple_len),
                                       dtype=torch.long, device=sample_masks.device)

            # 拼接原始张量和填充张量
            sample_triples = torch.cat([sample_triples, padding_triples], dim=0)
            sample_masks = torch.cat([sample_masks, padding_masks], dim=0)

        # 如果当前样本的三元组数量大于批次中的最大值，则截断
        elif curr_triples > max_triples_in_batch:
            sample_triples = sample_triples[:max_triples_in_batch]
            sample_masks = sample_masks[:max_triples_in_batch]

        batch_triple_ids.append(sample_triples)
        batch_triple_masks.append(sample_masks)

    # 将所有样本的三元组堆叠成一个批次
    triple_ids = torch.stack(batch_triple_ids)  # [batch_size, max_triples_in_batch, 3, triple_len]
    triple_attention_masks = torch.stack(batch_triple_masks)  # [batch_size, max_triples_in_batch, 3, triple_len]
    
    # 返回批次数据
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_ids": image_ids,
        "triple_ids": triple_ids,
        "triple_attention_masks": triple_attention_masks,
        "labels": labels,  # 现在是张量而不是列表
    }
'''