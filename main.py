import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
import logging
from datetime import datetime
import transformers
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    T5ForConditionalGeneration
# 修改导入语句，添加custom_collate_fn
from model import T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg, custom_collate_fn
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)
import nltk
import evaluate

# 添加自定义训练回调类，用于记录训练日志
class LoggingCallback(transformers.TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 设置Python标准日志库
        self.logger = logging.getLogger("training_logger")
        self.logger.setLevel(logging.INFO)

        # 创建文件处理器
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('[%(asctime)s] - INFO: step=%(step)s, loss=%(loss)s, grad_norm=%(grad_norm)s, lr=%(lr)s, epoch=%(epoch)s')
        file_handler.setFormatter(formatter)

        # 添加处理器到日志器
        self.logger.addHandler(file_handler)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """在每步结束时记录梯度信息"""
        if state.is_local_process_zero and hasattr(model, "get_gradients_for_logging"):
            # 获取梯度范数
            grad_norm = model.get_gradients_for_logging()
            # 将梯度范数保存到模型的状态中，以便在on_log中使用
            if not hasattr(model, "grad_norm_for_logging"):
                model.grad_norm_for_logging = {}
            model.grad_norm_for_logging[state.global_step] = grad_norm
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if state.is_local_process_zero and logs:
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 提取需要记录的信息
            step = state.global_step if hasattr(state, 'global_step') else 0
            loss = logs.get('loss', 'N/A')
            
            # 尝试从模型状态获取梯度范数
            grad_norm = 'N/A'
            if model is not None and hasattr(model, "grad_norm_for_logging") and step in model.grad_norm_for_logging:
                grad_norm = model.grad_norm_for_logging[step]
            
            learning_rate = logs.get('learning_rate', 'N/A')
            epoch = logs.get('epoch', 'N/A')
            
            # 写入日志文件 - 使用新的格式
            with open(self.log_path, 'a') as f:
                f.write(f"[{current_time}] - INFO: step={step}, loss={loss}, grad_norm={grad_norm}, lr={learning_rate}, epoch={epoch}\n")

                # 检查是否有评估结果
                eval_prefix = "eval_"
                eval_metrics = {k: v for k, v in logs.items() if k.startswith(eval_prefix)}
                if eval_metrics:
                    # 格式化评估结果
                    eval_str = ", ".join([f"{k.replace(eval_prefix, '')}={v}" for k, v in eval_metrics.items()])
                    f.write(f"[{current_time}] - EVAL: {eval_str}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--triple_len', type=int, default=16)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])

    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet', 'vit'],
                        help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--resume_from_checkpoint', type=str, default=True,
    #                     help='从特定checkpoint恢复训练，设置为"True"自动寻找最新checkpoint，或提供具体路径')
    # 添加数据加载线程数参数
    parser.add_argument('--num_workers', type=int, default=1,
                        help='数据加载的工作线程数量')
    
    args = parser.parse_args()
    return args


def T5Trainer(
        dataframe, args,
):
    # 设置随机种子，保证实验可复现
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 准备数据
    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    subgraphs  = dataframe['subgraphs']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")  # 'flan-alpaca-large'
        gpu_count = torch.cuda.device_count()
        # save_dir = "experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs2_op512_ep50"
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # 创建日志目录
    logs_dir = os.path.join(save_dir, "logs")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    # 创建日志文件路径
    log_file = os.path.join(logs_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    # 创建日志回调
    logging_callback = LoggingCallback(log_file)

    print(save_dir)
    if args.img_type is not None:  # args.img_type='vit'，多模态模型初始化
        patch_size = img_shape[args.img_type]  # (145, 1024)
        console.log(f"[model]: Loading modeling...\n")
        model = T5ForMultimodalGeneration.from_pretrained(args.model,
                                                          patch_size=patch_size)  # 加载多模态生成模型，args.model指定了预训练模型的路径，patch_size定义了图像特征的尺寸，用于多模态任务中的图像处理
        console.log(f"[model]: Model loading completed...\n")
        name_maps = dataframe['name_maps']  # 包含图像的数据中图像对应的索引，11208
        image_features = dataframe['image_features']  # 图像特征 torch.Size([11208, 145, 1024])

        # 获取训练集，验证集，测试集数据，
        # 包含data(原始数据), image_ids(初始化的图像特征), source_text(Question:...,Context:...,Options:...,Solution:), target_text(Solution:...)
        train_set = ScienceQADatasetImg(
            problems,
            subgraphs,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args.triple_len,
            args,
            image_features,
        )
        eval_set = ScienceQADatasetImg(
            problems,
            subgraphs,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args.triple_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            problems,
            subgraphs,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args.triple_len,
            args,
            image_features,
            args.test_le,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )

        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)  # 自动执行padding和attention mask等操作
    print("model parameters: ", model.num_parameters())  # 输出模型参数量 790496256

    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)

        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"
        return answer

        # accuracy for answer inference，评估答案预测的准确率

    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct += 1
        return {'accuracy': 1.0 * correct / len(targets)}

    # rougel for rationale generation
    metric = evaluate.load("rouge")  # 用于计算生成文本和参考文本之间的相关度，计算 n-gram 重叠率 和 最长公共子序列匹配
    # metric = evaluate.load("rouge", hf_endpoint="https://hf-mirror.com")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    # 评估生成推理的质量
    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)

        # if data_args.ignore_pad_token_for_loss:  # Replace -100 in the labels as we can't decode them.

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)
        # 计算ROUGE-L得分
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    # 只在训练结束后进行一次最终评估，主要目的是减少训练时间
    console.log("配置模型参数。。。")
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            logging_steps=1,  # 每10步记录一次日志
            logging_first_step=True,  # 确保第一步的损失也被记录
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="none",
            logging_dir=logs_dir,  # 添加日志目录
            logging_nan_inf_filter=True,  # 过滤NaN和Inf值
            gradient_accumulation_steps=1,  # 梯度累积步数
            max_grad_norm=1.0,  # 梯度裁剪阈值
            # resume_from_checkpoint=True
            # 添加数据加载器参数
            # dataloader_num_workers=args.num_workers,  # 设置数据加载器的工作线程数
            # dataloader_pin_memory=True,  # 启用内存锁定，加速CPU到GPU的数据传输
        )
    else:  # 在每个epoch结束后都进行一次评估，同时启用load_best_model_at_end，保存性能最好的模型
        training_args = Seq2SeqTrainingArguments(  # 配置模型的参数
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            logging_steps=1,  # 每10步记录一次日志
            logging_first_step=True,  # 确保第一步的损失也被记录
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,  # 在训练结束时加载最佳模型
            report_to="none",
            logging_dir=logs_dir,  # 添加日志目录
            logging_nan_inf_filter=True,  # 过滤NaN和Inf值
            gradient_accumulation_steps=1,  # 梯度累积步数
            max_grad_norm=1.0,  # 梯度裁剪阈值
            # resume_from_checkpoint=True
            # 添加数据加载器参数
            # dataloader_num_workers=args.num_workers,  # 设置数据加载器的工作线程数
            # dataloader_pin_memory=True,  # 启用内存锁定，加速CPU到GPU的数据传输
        )
    # Seq2SeqTrainer进行训模型的训练
    # 在T5Trainer函数中添加以下代码，位于创建trainer之前

    # 创建自定义训练器以记录梯度信息
    class GradientTracker(Seq2SeqTrainer):
        def training_step(self, model, inputs):
            """重写训练步骤以计算梯度范数"""
            # 调用父类的training_step
            loss = super().training_step(model, inputs)
    
            # 计算梯度范数
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                if hasattr(self.optimizer, "clip_grad_norm"):
                    # 一些优化器（如Adafactor）有自己的梯度裁剪方法
                    grad_norm = self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                else:
                    # 否则使用torch.nn.utils.clip_grad_norm_
                    parameters = model.parameters()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, self.args.max_grad_norm
                    ).item()
    
                # 添加一个方法让回调可以访问梯度范数
                def get_gradients_for_logging():
                    return grad_norm
    
                model.get_gradients_for_logging = get_gradients_for_logging
    
            return loss


        def get_train_dataloader(self):
           """重写获取训练数据加载器的方法，使用自定义的collate_fn"""
           if self.train_dataset is None:
               raise ValueError("Trainer: training requires a train_dataset.")

           train_dataset = self.train_dataset

           # 创建自定义的DataLoader
           from torch.utils.data import DataLoader

           return DataLoader(
               train_dataset,
               batch_size=self.args.train_batch_size,
               sampler=self._get_train_sampler(),
               collate_fn=custom_collate_fn,  # 使用自定义的collate函数
               drop_last=self.args.dataloader_drop_last,
               num_workers=self.args.dataloader_num_workers,
               pin_memory=self.args.dataloader_pin_memory,
           )

        def get_eval_dataloader(self, eval_dataset=None):
           """重写获取评估数据加载器的方法，使用自定义的collate_fn"""
           if eval_dataset is None and self.eval_dataset is None:
               raise ValueError("Trainer: evaluation requires an eval_dataset.")

           eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

           # 创建自定义的DataLoader
           from torch.utils.data import DataLoader

           return DataLoader(
               eval_dataset,
               batch_size=self.args.eval_batch_size,
               sampler=self._get_eval_sampler(eval_dataset),
               collate_fn=custom_collate_fn,  # 使用自定义的collate函数
               drop_last=self.args.dataloader_drop_last,
               num_workers=self.args.dataloader_num_workers,
               pin_memory=self.args.dataloader_pin_memory,
           )

        def get_test_dataloader(self, test_dataset):
           """重写获取测试数据加载器的方法，使用自定义的collate_fn"""
           # 创建自定义的DataLoader
           from torch.utils.data import DataLoader

           return DataLoader(
               test_dataset,
               batch_size=self.args.eval_batch_size,
               sampler=self._get_eval_sampler(test_dataset),
               collate_fn=custom_collate_fn,  # 使用自定义的collate函数
               drop_last=self.args.dataloader_drop_last,
               num_workers=self.args.dataloader_num_workers,
               pin_memory=self.args.dataloader_pin_memory,
           )


    # 使用自定义训练器替代原来的Seq2SeqTrainer
    trainer = GradientTracker(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel,
        callbacks=[logging_callback]  # 添加日志记录回调
    )
    # 模型训练
    if args.evaluate_dir is None:
        # console.log("[Training]: 模型训练开始...")

        # 检查是否存在checkpoint
        # checkpoint_path = None
        # if args.resume_from_checkpoint:
        #     # 检查保存目录中是否存在checkpoint文件夹
        #     checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint-")]
        #     if checkpoints:
        #         # 按照checkpoint编号排序，选择最新的checkpoint
        #         checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        #         checkpoint_path = os.path.join(save_dir, checkpoints[-1])
        #         console.log(f"[Training]: 发现检查点，从最新检查点恢复训练: {checkpoint_path}")
        #     else:
        #         console.log("[Training]: 未发现检查点，从头开始训练")

        # # 根据是否找到checkpoint决定是否从checkpoint恢复
        # trainer.train(resume_from_checkpoint=checkpoint_path)
        # console.log("[Training]: 模型训练完成，模型保存中...")
        # trainer.save_model(save_dir)
        # console.log("[Training]: 模型保存完成...")


        console.log("[Training]: 模型训练开始...")
        trainer.train()
        console.log("[Training]: 模型训练完成，模型保存中...")
        trainer.save_model(save_dir)
        console.log("[Training]: 模型保存完成...")
    # 在测试集上评估
    console.log("[Evaluation]: 在测试集上开始验证...")
    metrics = trainer.evaluate(eval_dataset=test_set, max_length=args.output_len)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    console.log("[Evaluation]: 测试集验证完成")
    # 在测试集上预测
    console.log("[Prediction]: 在测试集上开始预测...")
    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}

        num_fail = 0
        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)
            if extract_pred != "FAILED":
                if extract_pred in args.options:
                    extract_pred = args.options.index(extract_pred)
                else:
                    extract_pred = random.choice(range(0, len(args.options)))
            else:
                num_fail += 1
                extract_pred = random.choice(range(len(args.options)))  # random choose one option
            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref

        scores = get_scores(results_ans, results_rationale, results_reference,
                            os.path.join(args.data_root, "scienceqa/problems.json"))
        preds = [pred.strip() for pred in preds]
        output_data = {
            "num_fail": num_fail,
            "scores": scores,
            "preds": preds,
            "labels": targets}
        output_prediction_file = os.path.join(save_dir, "predictions_ans_test.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))

    # generate the rationale for the eval set
    if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
        torch.cuda.empty_cache()
        del predict_results, preds, targets
        predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len)
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds, targets = predict_results.predictions, predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                targets = predict_results.label_ids
                preds = preds.argmax(axis=2)

            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            targets = tokenizer.batch_decode(
                targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            output_data = {"preds": preds,
                           "labels": targets}
            output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))


if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()
    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # 数据加载
    if args.img_type is not None:
        problems, subgraphs, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'subgraphs': subgraphs,'qids': qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids}

    T5Trainer(
        dataframe=dataframe,
        args=args
    )
