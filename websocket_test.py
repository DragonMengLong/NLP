#-*- encoding:utf-8 -*-
import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import json
import logging
import time
import os
from collections import namedtuple
import fileinput
import sys

import torch
from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.utils import import_user_module

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')
model_list = ['single', 'multiple', 'single_ll' ,'single_c', 'multiple_ll', 'multiple_c']

def logger_init():
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    log_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    datetime = time.strftime("%Y-%m-%d", time.localtime())
    logger = logging.getLogger()
    log_filename = "./logs/{}.txt".format(datetime)
    if not os.path.exists(log_filename):
        os.mknod(log_filename)
    fh = logging.FileHandler(log_filename, mode='a')
    ch = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer

def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

def args_init(mode):
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    args.source_lang='de'
    args.target_lang='en'
    args.remove_bpe='@@ '
    if mode == 'single':
        args.path = 'checkpoints/iwslt14_de_en/align_train_single/checkpoint_last.pt'
        args.classifier = 'single'
        args.layer = 6
    if mode == 'multiple':
        args.path = 'checkpoints/iwslt14_de_en/align_train_multiple/checkpoint_last.pt'
        args.classifier = 'multiple'
        args.layer = 6
    if mode == 'single_ll':
        args.path = 'checkpoints/iwslt14_de_en/align_train_single_ll_both/checkpoint_last.pt'
        args.classifier = 'single'
    if mode == 'single_c':
        args.path = 'checkpoints/iwslt14_de_en/align_train_single_c_both/checkpoint_last.pt'
        args.classifier = 'single'
    if mode == 'multiple_ll':
        args.path = 'checkpoints/iwslt14_de_en/align_train_multiple_ll_both/checkpoint_last.pt'
        args.classifier = 'multiple'
    if mode == 'multiple_c':
        args.path = 'checkpoints/iwslt14_de_en/align_train_multiple_c_both/checkpoint_last.pt'
        args.classifier = 'multiple'
    return args

def load_model(args):
    import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    
    return {'task':task, 'args':args, 'max_positions':max_positions, 'use_cuda':use_cuda, 'generator':generator, 'models':models, 'tgt_dict':tgt_dict, 'src_dict':src_dict, 'align_dict':align_dict}

def translate(inputs, args, task, max_positions, use_cuda, generator, models, tgt_dict, src_dict, align_dict):
    start_id = 0
    results = []
    hypo_strs = []

    if inputs[0] == ("#"):
            exit()

    for batch in make_batches(inputs, args, task, max_positions):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
        }
        translations, avg_token = task.inference_step(generator, models, sample)

        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            results.append((start_id + id, src_tokens_i, hypos))

    # sort output to match input order
    for id, ssrc_token, hypos in sorted(results, key=lambda x: x[0]):
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            print('S-{}\t{}'.format(id, src_str))

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]: 
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
            hypo_strs.append(hypo_str)
            print('P-{}\t{}'.format(
                id,
                ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
            ))
            if args.print_alignment:
                print('A-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                ))

    # update running id counter
    start_id += len(results)  
    return hypo_strs, avg_token["avg_l"]

class MainHandler(tornado.websocket.WebSocketHandler):

    mode = None
    model_dict = None

    def check_origin(self, origin):
        return True
 
    def open(self):
        pass

    def on_message(self, msg):
        print("receive: " + msg)

        try:
            jm = json.loads(msg)
            source = jm['source']
            mode = jm['mode']
        except:
            self.write_message({
                'source':None,
                'mode':None,
                'error_code':3,
                'error_message':'json format error'
            })
            return
        
        time_start = time.perf_counter()

        if not mode in model_list:
            self.write_message({
                'source':None,
                'mode':None,
                'error_code':2,
                'error_message':'model not exist'
            })
            return

        if mode != self.mode:
            remode = True
            self.mode = mode
            self.model_dict = load_model(args_init(mode))
        else:
            remode = False

        inputs=[source]
        output, avg_l = translate(inputs, self.model_dict['args'] ,self.model_dict['task'], self.model_dict['max_positions'], self.model_dict['use_cuda'], self.model_dict['generator'], 
            self.model_dict['models'], self.model_dict['tgt_dict'], self.model_dict['src_dict'], self.model_dict['align_dict'],)
        time_end = time.perf_counter()

        self.write_message({
                'source':source,
                'mode':mode,
                'remode':remode,
                'time':time_end - time_start,
                'translation':output[0],
                'avg_l':6 if mode == 'single' or mode == 'multiple' else avg_l + 1  ,
                'error_code':0,
                'error_message':None
            })
        return

    def on_close(self):
        pass

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [('/main', MainHandler)]
        tornado.web.Application.__init__(self, handlers)
        
if __name__ == '__main__':
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(6006)
    tornado.ioloop.IOLoop.instance().start()

