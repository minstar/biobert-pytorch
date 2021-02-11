# coding=utf-8

import os
import pdb
import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers import (
        BertConfig,
        BertModel,
        BertForTokenClassification,
        BertTokenizer,
        RobertaConfig,
        RobertaForTokenClassification,
        RobertaTokenizer
)

from torchcrf import CRF

class BioMultiNER(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(BioMultiNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_1 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_2 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_3 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_4 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_5 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_6 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_7 = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.classifier_8 = torch.nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, \
                ent_ids_1=None, ent_ids_2=None, ent_ids_3=None, ent_ids_4=None, \
                ent_ids_5=None, ent_ids_6=None, ent_ids_7=None, ent_ids_8=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)

        ### NCBI-disease ###
        d_logits = self.classifier_1(sequence_output)
        ent_ids_1 = torch.unsqueeze(ent_ids_1, 2)
        d_logits = ent_ids_1 * d_logits
        # ### BC5CDR-disease ###
        bd_logits = self.classifier_2(sequence_output)
        ent_ids_2 = torch.unsqueeze(ent_ids_2, 2)
        bd_logits = ent_ids_2 * bd_logits
        ### BC5CDR-chem ###
        bc_logits = self.classifier_3(sequence_output)
        ent_ids_3 = torch.unsqueeze(ent_ids_3, 2)
        bc_logits = ent_ids_3 * bc_logits
        ### BC4CHEMD ###
        c_logits = self.classifier_4(sequence_output)
        ent_ids_4 = torch.unsqueeze(ent_ids_4, 2)
        c_logits = ent_ids_4 * c_logits
        # ### BC2GM ###
        g_logits = self.classifier_5(sequence_output)
        ent_ids_5 = torch.unsqueeze(ent_ids_5, 2)
        g_logits = ent_ids_5 * g_logits
        ### JNLPBA 2 ###
        jn_logits = self.classifier_6(sequence_output)
        ent_ids_6 = torch.unsqueeze(ent_ids_6, 2)
        jn_logits = ent_ids_6 * jn_logits
        ### linnaeus ###
        li_logits = self.classifier_7(sequence_output)
        ent_ids_7 = torch.unsqueeze(ent_ids_7, 2)
        li_logits = ent_ids_7 * li_logits
        ### s800 ###
        s8_logits = self.classifier_8(sequence_output)
        ent_ids_8 = torch.unsqueeze(ent_ids_8, 2)
        s8_logits = ent_ids_8 * s8_logits

        logits = d_logits + g_logits + c_logits + bd_logits + bc_logits + jn_logits + li_logits + s8_logits
        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits


class BioUniNER(BertForTokenClassification):
    def __init__(self, config, num_labels=9):
        super(BioUniNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)

        ### NCBI-disease ###
        logits = self.classifier(sequence_output)

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits


class CRFNER(BertForTokenClassification):
    def __init__(self, config, num_labels=4):
        super(CRFNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        outputs = (logits, sequence_output)
        if labels is not None:
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)
                # active_labels = torch.where(
                #     active_loss, labels.view(-1), torch.tensor(3).type_as(labels)
                # )
                # active_logits = active_logits.view(batch_size, max_len, self.num_labels)
                # active_labels = active_labels.view(batch_size, max_len)
                attention_mask = attention_mask.type(torch.uint8)
                log_likelihood, sequence_of_tags = self.crf(logits, labels, mask=attention_mask, reduction='mean'), self.crf.decode(logits)
                return ((-1 * log_likelihood,) + outputs)
            else:
                log_likelihood = self.crf(logits, labels, reduction='mean')
                return -1 * log_likelihood
        else:
            return logits


class BiLSTMCRFNER(BertForTokenClassification):
    def __init__(self, config, num_labels=4):
        super(BiLSTMCRFNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.bilstm = torch.nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        sequence_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits, sequence_output)
        if labels is not None:
            if attention_mask is not None:
                attention_mask = attention_mask.type(torch.uint8)
                log_likelihood, sequence_of_tags = self.crf(logits, labels, mask=attention_mask, reduction='mean'), self.crf.decode(logits)
                return ((-1 * log_likelihood, ) + outputs)
            else:
                log_likelihood = self.crf(logits, labels, reduction='mean')
        
                return -1 * log_likelihood
        else:
            return logits


class CRFNER_MASKSIM(BertForTokenClassification):
    def __init__(self, config, num_labels=4, alpha=0):
        super(CRFNER_MASKSIM, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.alpha = alpha

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        outputs = (logits, sequence_output)

        sep_index = attention_mask.sum(dim=1) - token_type_ids.sum(dim=1)
        final_index = attention_mask.sum(dim=1)

        get_bef_embeds, get_aft_embeds = [], []
        for batch_idx in range(batch_size):
            bef_out = torch.mean(sequence_output[batch_idx][1:sep_index[batch_idx]], dim=0)
            aft_out = torch.mean(sequence_output[batch_idx][sep_index[batch_idx]+1:final_index[batch_idx]], dim=0)

            bef_out = bef_out.unsqueeze(dim=0)
            aft_out = aft_out.unsqueeze(dim=0)

            get_bef_embeds.append(bef_out)
            get_aft_embeds.append(aft_out)
        
        bef_embeddings = torch.cat(get_bef_embeds) # (batch, feat_dim)
        aft_embeddings = torch.cat(get_aft_embeds) # (batch, feat_dim)

        if labels is not None:
            if attention_mask is not None:
                attention_mask = attention_mask.type(torch.uint8)
                log_likelihood, sequence_of_tags = self.crf(logits, labels, mask=attention_mask, reduction='mean'), self.crf.decode(logits)
                cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                cossim_loss = torch.mean(1. - cossim(bef_embeddings, aft_embeddings))
                
                return (((1-self.alpha) * -1 * log_likelihood + self.alpha * cossim_loss,) + outputs)
            else:
                log_likelihood = self.crf(logits, labels, reduction='mean')
                cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                cossim_loss = torch.mean(1. - cossim(bef_embeddings, aft_embeddings))

                return (1-self.alpha) * -1 * log_likelihood + self.alpha * cossim_loss
        else:
            return logits

class BioNER(BertForTokenClassification):
    def __init__(self, config, num_labels=3, random_bias=False, freq_bias=False, pmi_bias=True):
        super(BioNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.random_bias = random_bias
        self.freq_bias = freq_bias
        self.pmi_bias = pmi_bias
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, bias_tensor=None, data_type=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        
        if data_type[0][0].item() == 1:
            if self.random_bias:
                rand_logits = torch.rand(batch_size, max_len, self.num_labels).cuda()
                logits = logits + rand_logits
            elif self.freq_bias or self.pmi_bias:
                logits = logits + bias_tensor
        
        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits

class GeneralNER(BertForTokenClassification):
    def __init__(self, config, num_labels=9, random_bias=False, freq_bias=False, pmi_bias=True):
        super(GeneralNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.random_bias = random_bias
        self.freq_bias = freq_bias
        self.pmi_bias = pmi_bias
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, bias_tensor=None, data_type=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        
        if data_type[0][0].item() == 1:
            if self.random_bias:
                rand_logits = torch.rand(batch_size, max_len, self.num_labels).cuda()
                logits = logits + rand_logits
            elif self.freq_bias or self.pmi_bias:
                logits = logits + bias_tensor
        
        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels)
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits