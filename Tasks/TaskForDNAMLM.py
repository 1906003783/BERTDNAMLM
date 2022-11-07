from ast import arg
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging
import os
from utils import logger_init
from model import BertConfig
from model import BertForMaskedLM
from utils import LoadDNADataset,roc_auc,accuracy,evaluate
import torch
import time
import argparse

class ModelConfig:
    def __init__(self, masked_rate=0.4, train_set="train.csv", test_set="test.csv", inital_site=0, number_of_group=1, learning_rate=5E-5,batch_size=24,do_logging=True):
        self.project_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'dnabert')
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "bert_dna")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_set = train_set
        self.test_set = test_set
        self.train_file_path = os.path.join(
            self.dataset_dir, train_set)
        self.val_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.test_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        # self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.logs_save_dir = os.path.join(self.project_dir, '../records/bertmlm/raw')
        self.data_name = 'dnabert'
        self.model_save_path = os.path.join(
            self.model_save_dir, f'model_{self.data_name}.pt')
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = batch_size
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = learning_rate
        self.masked_rate = masked_rate
        self.masked_token_rate = 1
        self.masked_token_unchanged_rate = 1
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 100
        self.model_val_per_epoch = 1
        self.inital_site = inital_site
        self.number_of_group = number_of_group
        self.do_logging = do_logging
        self.attention = ''

        if do_logging:
            logger_init(log_file_name='dnabert', log_level=self.log_level,
                        log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(
            self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        if do_logging:
            logging.info(" ### 将当前配置打印到日志文件中 ")
            for key, value in self.__dict__.items():
                logging.info(f"### {key} = {value}")

def train(config):
    model = BertForMaskedLM(config)
    if os.path.exists(config.model_save_path):
        # loaded_paras = torch.load(config.model_save_path)
        # model.load_state_dict(loaded_paras)
        model = torch.load(config.model_save_path, map_location={'cuda:1':'cuda:0'})
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    data_loader = LoadDNADataset(vocab_path=config.vocab_path,
                                  batch_size=config.batch_size,
                                  max_sen_len=config.max_sen_len,
                                  max_position_embeddings=config.max_position_embeddings,
                                  pad_index=config.pad_index,
                                  is_sample_shuffle=config.is_sample_shuffle,
                                  random_state=config.random_state,
                                  data_name=config.data_name,
                                  masked_rate=config.masked_rate,
                                  masked_token_rate=config.masked_token_rate,
                                  masked_token_unchanged_rate=config.masked_token_unchanged_rate,
                                  inital_site=config.inital_site,
                                  number_of_group=config.number_of_group)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(
                config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            loss, mlm_logits = model(input_ids=b_token_ids,
                                                  attention_mask=b_mask,
                                                  token_type_ids=None,
                                                  masked_lm_labels=b_mlm_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            mlm_acc, _, _ = accuracy(
                mlm_logits, b_mlm_label, data_loader.PAD_IDX)
            auc = roc_auc(mlm_logits, b_mlm_label,
                          config.vocab_size, data_loader.PAD_IDX)
            if idx % 20 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f}, roc_auc: {auc:.3f}")
                # print(attn_weight.shape)

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        # attn_to_figure(attn_weight, epoch)
        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc, auc = evaluate(
                config, val_iter, model, data_loader.PAD_IDX)
            logging.info(
                f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, roc_auc: {auc:.3f}")
            if mlm_acc > max_acc:
                max_acc = mlm_acc
                # torch.save(model.state_dict(), config.model_save_path)
                torch.save(model, config.model_save_path)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DNA MLM Task.")
    parser.add_argument('-mr', '--masked_rate', type=float,
                        help='Masking rate of SNPs.', default=0.4)
    parser.add_argument('-is', '--inital_site', type=int,
                        help='Initial position of SNPs in dataset.', default=0)
    parser.add_argument('-ng', '--number_of_group', type=int,
                        help='Number of 512 SNPs groups in dataset.', default=1)
    parser.add_argument('-trs', '--train_set', type=str,
                        help='train_set', default='train.csv')
    parser.add_argument('-tes', '--test_set', type=str,
                        help='test_set', default='test.csv')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='learning_rate', default=5E-5)
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='batch_size', default=24)

    args = parser.parse_args()
    if args.number_of_group < 1:
        args.number_of_group = 1

    config = ModelConfig(masked_rate=args.masked_rate,
                         inital_site=args.inital_site,
                         number_of_group=args.number_of_group,
                         train_set=args.train_set,
                         test_set=args.test_set,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size
                         )
    train(config)

