import logging
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score



def roc_auc(mlm_logits, mlm_labels, vocab_size, PAD_IDX):
    with torch.no_grad():
        mlm_pred = mlm_logits.transpose(0, 1).reshape(-1, mlm_logits.shape[2])
        mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
        mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 获取mask位置的行索引
        mlm_pred = mlm_pred[mask, 5:]  # 去除预测为特殊标记可能性
        mlm_pred_sm = torch.softmax(mlm_pred, dim=1).cpu()
        mlm_true = mlm_true[mask]
        mlm_true = mlm_true.reshape(-1, 1).cpu()
        mlm_true = torch.zeros(mlm_true.shape[0], vocab_size).scatter_(
            dim=1, index=mlm_true, value=1)
        mlm_true = mlm_true[:, 5:]
        return roc_auc_score(y_true=mlm_true, y_score=mlm_pred_sm, average='macro', multi_class='ovr')


def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
    # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    return (mlm_acc, mlm_correct, mlm_total)


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, auc, cnt = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            mlm_logits = model(input_ids=b_token_ids,
                                          attention_mask=b_mask,
                                          token_type_ids=None)
            result = accuracy(mlm_logits, b_mlm_label, PAD_IDX)
            _, mlm_cor, mlm_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            auc += roc_auc(mlm_logits, b_mlm_label, config.vocab_size, PAD_IDX)
            cnt += 1
    model.train()
    return (float(mlm_corrects) / mlm_totals, auc / cnt)


class Vocab:

    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors

def vec2str(vec, start, max_len=512):
    v = vec[start: start + max_len]
    s = ''.join([str(i) for i in v])
    # logging.info(s)
    return s


def read_dnaseq(filepath=None, inital_site=0, number_of_group=1, max_len=512):
    df = pd.read_csv(filepath, header=None)
    df = df.to_numpy()
    paragraphs = []
    start = inital_site
    for i in range(number_of_group):
        for vec in df:
            paragraphs.append(vec2str(vec, start, max_len=max_len))
        start+=max_len
    random.shuffle(paragraphs)
    return paragraphs


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadDNADataset(object):
    def __init__(self,
                 vocab_path='./vocab.txt',
                 batch_size=24,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2022,
                 data_name='dnabert',
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 istraining=True,
                 inital_site=0,
                 number_of_group=1
                 ):
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        self.inital_site = inital_site
        self.number_of_group = number_of_group
        self.istraining=istraining
        random.seed(random_state)
        


    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = token_ids[mlm_pred_position]  # 10%的时间：保持词不变
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            rand_t=random.random()
            if rand_t < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            elif random.random() > self.masked_token_unchanged_rate:  # 0.5
                # 10%的时间：用随机词替换该词
                    masked_token_id = random.randint(0, len(self.vocab.stoi) - 1)
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    @cache
    def data_process(self, filepath, istraining=True, postfix='cache'):
        """
        本函数的作用是是根据格式化后的数据制作MLM任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        inital_site = self.inital_site
        number_of_group = self.number_of_group
        paragraphs = read_dnaseq(filepath, inital_site=inital_site, number_of_group=number_of_group, max_len=self.max_position_embeddings)
        # 返回的是一个二维列表，每个列表可以看做是一个段落（其中每个元素为一句话）
        data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        desc = f" ## 正在构造MLM样本({filepath.split('.')[1]})"
        for paragraph in tqdm(paragraphs, ncols=80, desc=desc):  # 遍历每个
            token_ids = [self.vocab[token] for token in list(paragraph)]
            if len(token_ids) > self.max_position_embeddings:
                # BERT预训练模型只取前512个字符
                token_ids = token_ids[:self.max_position_embeddings]
            #logging.debug(f" ## Mask之前token ids:{token_ids}")
            mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
            token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
            mlm_label = torch.tensor(mlm_label, dtype=torch.long)
            max_len = max(max_len, token_ids.size(0))
            data.append([token_ids, mlm_label])

        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_mlm_label = [], []
        for (token_ids, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        return b_token_ids, b_mask, b_mlm_label

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"
        test_data = self.data_process(filepath=test_file_path, istraining=False,
                                      postfix='test' + postfix)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            logging.info(f"## 成功返回测试集，一共包含样本{len(test_iter.dataset)}个")
            return test_iter
        data = self.data_process(filepath=train_file_path, istraining=True, postfix='train' + postfix)
        train_data, max_len = data['data'], data['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_data = self.data_process(
            filepath=val_file_path, istraining=False, postfix='val' + postfix)['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch)
        logging.info(f"## 成功返回训练集样本（{len(train_iter.dataset)}）个、开发集样本（{len(val_iter.dataset)}）个"
                     f"测试集样本（{len(test_iter.dataset)}）个.")
        return train_iter, test_iter, val_iter
