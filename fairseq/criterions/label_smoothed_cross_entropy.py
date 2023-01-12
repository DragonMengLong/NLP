# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
###########
import torch.nn as nn
import torch
###########


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        ##############
        if args.learn_multiloss_w:
            self.classifier_weight = Linear(args.decoder_layers, 1)
        else:
            self.classifier_weight = None
        self.depth_target = args.depth_select_target
        self.depth_lambda = args.depth_lambda
        if self.depth_target != "None":
            self.dep_cri = nn.CrossEntropyLoss()
        ##############

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parse nr."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        ############
        #这里的sample['net_input']是有3个键的，其中最后一个为prev_output_tokens
        #这是因为这里调用的是model，fairseq_model接受的是3个参数，真正传给encoder的是前两个
        #看\fairseq\models\fairseq_model.py中157行
        # print(sample['net_input']['src_tokens'].size())
        net_output, _, dec_layers_out, dict_n = model(**sample['net_input'])
        # print(net_output.size())
        loss_list = []
        nll_loss_list = []
        seq_logP_list = []
        seq_loss = {}
        loss = 0
        nll_loss = 0

        # loss, nll_loss, _ = self.compute_loss(model, (net_output, ), sample, reduce=reduce) # 重复计算了
        
        depth_num = 0
        if len(dec_layers_out) != 0:
            for item in dec_layers_out:
                depth_num += 1
                temp_loss, temp_nll_loss, logP = self.compute_loss(model, (item, ), sample, reduce=reduce)
                loss_list.append(temp_loss)
                nll_loss_list.append(temp_nll_loss)
                if self.depth_target != "None":
                    if self.depth_target == "seq_LL":
                        seq_logP_list.append(logP + self.depth_lambda * depth_num)
                    elif self.depth_target == "seq_cor":
                        seq_logP_list.append(logP + -self.depth_lambda * depth_num)
        
        multi_depth = None
        seq_loss["func_q"] = dict_n["pred"] # 62 * 6
        seq_loss["func_q_star"] = dict_n["pred"][:, 0].long()
        depth_loss = None

        if self.depth_target != "None": 
            for i in range(len(seq_logP_list)): 
                if i == 0:
                    multi_depth = seq_logP_list[i]
                else:
                    multi_depth = torch.cat((multi_depth, seq_logP_list[i]), 1) # seq_logP_list[i] 62*1 multi_depth 62*6
            
            if self.depth_target == "seq_LL":
                _, depth_index = multi_depth.min(-1, keepdim=True) # min 因为加了负号
                seq_loss["func_q_star"] = depth_index.squeeze(-1)  
            
            elif self.depth_target == "seq_cor":
                _, depth_index = multi_depth.max(-1, keepdim=True)
                seq_loss["func_q_star"] = depth_index.squeeze(-1)

            depth_loss = self.dep_cri(seq_loss["func_q"], seq_loss["func_q_star"]) # 有问题 直接用multi_depth 或者是 01矩阵 现在的相当于是分布和label计算

        # for i in range(len(loss_list)):
        #     print("loss " + str(i) + ":" + str(loss_list[i]))
        # for i in range(len(nll_loss_list)):
        #     print("loss " + str(i) + ":" + str(nll_loss_list[i]))

        for item in loss_list:
            loss += item
        for item in nll_loss_list:
            nll_loss += item
        loss = loss / 6.0 
        nll_loss = nll_loss / 6.0
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        # multi_loss
        ############

        # net_output = model(**sample['net_input']) #net_output中有个字典，'encoder_states': None}
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # logging_output = {
        #     'loss': utils.item(loss.data) if reduce else loss.data,
        #     'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
        #     'ntokens': sample['ntokens'],
        #     'nsentences': sample['target'].size(0),
        #     'sample_size': sample_size,
        # }
        # return loss, sample_size, logging_output

        ############
        return loss, sample_size, logging_output, depth_loss
        ############

    def compute_loss(self, model, net_output, sample, reduce=True):
        # print(net_output[0].size())
        lprobs = model.get_normalized_probs(net_output, log_probs=True) #输出log后的概率
        # print("compute_loss is :")
        # print(lprobs)

        ###########
        beam_batch_size, lenght, word_V = lprobs.size() # 62, 128, 10152
        # print(lprobs.size())
        ###########

        lprobs = lprobs.view(-1, lprobs.size(-1)) # 变为(batch*beam*len)*V
        target = model.get_targets(sample, net_output).view(-1, 1) # 将target变为 (batch*beam*len)*1 是目标在字典中的位置
        non_pad_mask = target.ne(self.padding_idx) # padding_idx=1，和之前的维度一样 去除其中padding的部分
        # 取lprobs在target位置上的值
        # nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]

        ############
        nll_loss_fake = -lprobs.gather(dim=-1, index=target)
        nll_loss = nll_loss_fake[non_pad_mask]

        # print(target)
        # print(target.size())
        # print(lprobs)
        # print(lprobs.size())
        # print(nll_loss_fake)
        # print(nll_loss_fake.size())
        # print(non_pad_mask)
        # print(non_pad_mask.size())
        # print(nll_loss)
        # print(nll_loss.size())
        # exit()

        if self.depth_target == "seq_LL" :
            lp_cp = nll_loss_fake.detach()
            non_pad_bb = non_pad_mask.view(beam_batch_size, lenght, -1).squeeze(-1)
            lp_bb = lp_cp.view(beam_batch_size, lenght, -1).squeeze(-1)
            lp_bb_nopad = lp_bb * non_pad_bb.float()
            lp_final_bb = lp_bb_nopad.sum(-1, keepdim=True) # 想法是将length加起来 得到 LLn 注意是不是lenght beam_batch_size 62, 1
        elif self.depth_target == "seq_cor":
            model_pred, model_pred_index = lprobs.max(-1, keepdim=True)
            # correct_num = nll_loss_fake == -model_pred # 用model_pred_index 和 target 可能会好一点 用那个分数的话要加负号
            correct_num = model_pred_index == target
            correct_num_bb = correct_num.view(beam_batch_size, lenght, -1).squeeze(-1)
            lp_final_bb = correct_num_bb.sum(-1, keepdim=True) # 得到一层的正确预处个数

        else:
            lp_final_bb = None
        ############

        # -1表示在最后一维做，keepdim为真时得到的结果保留了原tensor的第0维
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss, lp_final_bb

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

#############
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
#############
