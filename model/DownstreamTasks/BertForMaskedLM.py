import logging
from ..BasicBert.Bert import BertModel
import torch.nn as nn
import torch



class BertForLMTransformHead(nn.Module):
    """
    用于BertForMaskedLM中的一次变换。 因为在单独的MLM任务中
    和最后NSP与MLM的整体任务中均要用到，所以这里单独抽象为一个类便于复用

    ref: https://github.com/google-research/bert/blob/master/run_pretraining.py
        第248-262行
    """

    def __init__(self, config, bert_model_embedding_weights=None):
        """
        :param config:
        :param bert_model_embedding_weights:
        the output-weights are the same as the input embeddings, but there is
        an output-only bias for each token. 即TokenEmbedding层中的词表矩阵
        """
        super(BertForLMTransformHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        if bert_model_embedding_weights is not None:
            self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
        # [hidden_size, vocab_size]
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size] Bert最后一层的输出
        :return:
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        # hidden_states:  [src_len, batch_size, vocab_size]
        return hidden_states


class BertForMaskedLM(nn.Module):
    """
    掩码语言预测模型
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMaskedLM, self).__init__()
        self.bert = BertModel(config)
        weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
        self.classifier = BertForLMTransformHead(config, weights)
        self.config = config

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None,  # [src_len,batch_size]
                ):
        all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]
        prediction_scores = self.classifier(sequence_output)
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size),
                                      masked_lm_labels.reshape(-1))
            return masked_lm_loss, prediction_scores
        else:
            return prediction_scores
