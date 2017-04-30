from __future__ import print_function
import tensorflow as tf
import argparse
import cPickle as pickle
from collections import namedtuple
import numpy as np
import gzip
import timeit
import json
import csv
import sys
import os
import re


Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])

def encode_word(word, word2idx, tokens):
    code = None
    try:
        code = word2idx[word]
    except KeyError:
        code = word2idx[tokens.UNK]
    return code

def encode_chars(word, char2idx, tokens):
    char_code = [char2idx[tokens.START]] * (2 + len(word))
    for n, char in enumerate(word):
        char_code[n + 1] = char2idx[char]
    char_code[-1] = char2idx[tokens.END]
    word_len = len(char_code)
    return char_code, word_len

def encode_utterance(utterance, do_word, do_chars, vocab, tokens):
    e_words = None
    e_chars = None
    max_word_len = None
    seq_len = 0
    if do_word:
        e_words = [None] * len(utterance) # encoded words
        word2idx = vocab['word2idx']
    if do_chars:
        e_chars = [None] * len(utterance)
        char2idx = vocab['char2idx']
        max_word_len = 0
    returns = []
    for n_w, word in enumerate(utterance):
        if do_word:
            e_words[n_w] = encode_word(word, word2idx, tokens)
        if do_chars:
            # retrieve list of codes here
            cchars, word_len = encode_chars(word, char2idx, tokens)
            if max_word_len < word_len:
                max_word_len = word_len
            e_chars[n_w] = cchars
    if do_word:
        seq_len = len(e_words)
    else:
        seq_len = len(e_chars)
    return e_words, e_chars, max_word_len, seq_len

def main(opts):
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    spaces = re.compile('\s+')
    exclude_idx = opts.exclude_csv_fields
    if exclude_idx is not None:
        print('Excluding CSV indices: ', exclude_idx)
    label_idx = opts.label_idx
    if label_idx is not None:
        print('Label CSV index specified: ', label_idx)
        # add label idx to exclude idx for non reading it as txt
        exclude_idx.append(label_idx)
    tokens = opts.tokens
    with open(opts.csv_file, 'rb') as tr_f:
        # Read the whole text to get the vocab
        tr_reader = csv.reader(tr_f)
        csv_txt_records = {}
        csv_lab_records = []
        if opts.do_words:
            # store all appearing words w/ counts
            tr_words = {}
        if opts.do_chars:
            # store all appearing chars w/ counts
            tr_chars = {}
        def add_counts(dict_, elements):
            for el in elements:
                if el not in dict_:
                    dict_[el] = 1
                else:
                    dict_[el] += 1
            return dict_
        for i, line in enumerate(tr_reader):
            if i == 0:
                # skip header
                header = line
                txt_idxes = []
                for n in range(len(header)):
                    if n in exclude_idx:
                        # skip excluded field
                        continue
                    csv_txt_records[header[n]] = []
                    csv_lab_records = []
                    txt_idxes.append(n)
                print('Found txt indxes in header: ', txt_idxes)
                continue
            total_txt_line = ""
            for idx in txt_idxes:
                csv_txt_records[header[idx]].append(line[idx])
                total_txt_line += line[idx] + " "
            # merge strings to compute words and/or chars of both
            total_txt_line = total_txt_line[:-1] # get rid of last space
            csv_lab_records.append(line[label_idx])
            if opts.do_words:
                words = re.split(spaces, total_txt_line)
                tr_words = add_counts(tr_words, words)
            if opts.do_chars:
                chars = [ch for ch in total_txt_line if ch != ' ']
                tr_chars = add_counts(tr_chars, chars)

        print('Number of samples: ', len(csv_txt_records.values()[0]))
        # build the vocabulary dictionary
        vocab = {}
        if opts.do_words:
            vocab['word2idx'] = dict((k, v + 2) for v, k in
                                     enumerate(tr_words.keys()))
            assert tokens.UNK not in tr_words.keys()
            assert tokens.EOS not in tr_words.keys()
            vocab['word2idx'][tokens.UNK] = 0
            vocab['word2idx'][tokens.EOS] = 1
            print('Num of unqiue words : ', len(vocab['word2idx']))
            vocab['idx2word'] = dict((v, k) for k, v in
                                     vocab['word2idx'].iteritems())
        if opts.do_chars:
            vocab['char2idx'] = dict((k, v + 4) for v, k in
                                     enumerate(tr_chars.keys()))
            assert tokens.EOS not in tr_chars.keys()
            assert tokens.START not in tr_chars.keys()
            assert tokens.END not in tr_chars.keys()
            assert tokens.ZEROPAD not in tr_chars.keys()
            vocab['char2idx'][tokens.ZEROPAD] = 0
            vocab['char2idx'][tokens.EOS] = 1
            vocab['char2idx'][tokens.START] = 2
            vocab['char2idx'][tokens.END] = 3
            print('Num of unique chars: ', len(tr_chars.keys()))
            vocab['idx2char'] = dict((v, k) for k, v in
                                     vocab['char2idx'].iteritems())
        # encode utterances and write them to TFRecords
        out_path = os.path.join(opts.save_path,
                                opts.out_prefix + '_data.tfrecords')
        out_tf = tf.python_io.TFRecordWriter(out_path)
        keys = csv_txt_records.keys()
        max_seq_len = 0
        max_word_len = 0
        print('Encoding utterances in keys ', keys)
        total_utterances = len(csv_txt_records.values()[0])
        if label_idx is not None:
            utterances = zip(csv_lab_records, *csv_txt_records.values())
        beg_t = timeit.default_timer()
        # store dictionary with description of context features
        context_features_desc = {}
        # store dictionary with description of sequence features
        seq_features_desc = {}

        def register_context_feature_desc(feature_name, feature_type):
            assert feature_type == 'int64_list' or feature_type == 'float_lsit' \
                or feature_type == 'bytes_list', feature_type
            if feature_name not in context_features_desc:
                context_features_desc[feature_name] = feature_type

        def register_seq_feature_desc(feature_name, feature_type):
            assert feature_type == 'int64_list' or feature_type == 'float_lsit' \
                or feature_type == 'bytes_list', feature_type
            if feature_name not in seq_features_desc:
                seq_features_desc[feature_name] = feature_type

        # keep track of timings to know how much it lasts per record
        utt_times = []
        for u_i, utts in enumerate(utterances):
            first_utt_idx = 0
            #ex = tf.train.SequenceExample()
            # initialize the dictionaries of TFRecords features
            context_features = {}
            seq_features = {}
            if label_idx is not None:
                # first utt is label
                first_utt_idx += 1
            utt_beg_t = timeit.default_timer()
            for i, utt in enumerate(utts[first_utt_idx:]):
                utt = spaces.split(utt)
                #print('utt {}:{}'.format(i, utt))
                e_words, e_chars, word_len, \
                seq_len = encode_utterance(utt,
                                           opts.do_words,
                                           opts.do_chars,
                                           vocab,
                                           tokens)
                if max_seq_len < seq_len:
                    max_seq_len = seq_len
                if opts.do_chars and max_word_len < word_len:
                    max_word_len = word_len
                register_context_feature_desc('{}_length'.format(keys[i]),
                                              'int64_list')
                register_context_feature_desc('do_chars'.format(keys[i]),
                                              'int64_list')
                register_context_feature_desc('do_words'.format(keys[i]),
                                              'int64_list')
                context_features['{}_length'.format(keys[i])] = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
                context_features['do_chars'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(opts.do_chars == 'true')]))
                context_features['do_words'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(opts.do_words == 'true')]))
                if opts.do_words:
                    register_seq_feature_desc('{}_words'.format(keys[i]),
                                              'int64_list')
                    # Feature list for the words sequence
                    seq_features['{}_words'.format(keys[i])] = tf.train.FeatureList(
                        feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=e_words))]
                    )
                if opts.do_chars:
                    # Feature list of feature list for the chars sequence
                    register_seq_feature_desc('{}_chars_len'.format(keys[i]),
                                              'int64_list')
                    register_seq_feature_desc('{}_chars'.format(keys[i]),
                                              'int64_list')
                    # Register each words len to decode
                    word_lens = [len(word) for word in e_chars]
                    seq_features['{}_chars_len'.format(keys[i])] = tf.train.FeatureList(
                        feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=word_lens))]
                    )
                    # Register each word chars decomposition
                    seq_features['{}_chars'.format(keys[i])] = tf.train.FeatureList(
                        feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=e_char)) for e_char in e_chars]
                    )
            # build the example
            ex = tf.train.SequenceExample(
                context=tf.train.Features(feature=context_features),
                feature_lists=tf.train.FeatureLists(feature_list=seq_features)
            )
            out_tf.write(ex.SerializeToString())
            utt_end_t = timeit.default_timer()
            utt_times.append(utt_end_t - utt_beg_t)
            print('Processed utterances {}/{}\t, mtime/utt {} s'
                  '...'.format(u_i + 1,
                               total_utterances,
                               np.mean(utt_times)),
                  end='\r')
            sys.stdout.flush()
        end_t = timeit.default_timer()
        print('--> Done writing {} in {} s <--'.format(out_path, end_t - beg_t))
        # write the TFRecords file format to pickle such that can be recomposed
        tfr_format = {
            'context_features':context_features_desc,
            'sequence_features':seq_features_desc
        }
        fmt_path = os.path.join(opts.save_path,
                                opts.out_prefix + '_fmt.pkl.gz')
        # store format file
        with gzip.open(fmt_path, 'wb') as fmt_f:
            pickle.dump(tfr_format, fmt_f)

        print('Max seq len found: ', max_seq_len)
        vocab['max_seq_len'] = max_seq_len
        if opts.do_chars:
            print('Max word len found: ', max_word_len)
            vocab['max_word_len'] = max_word_len
        out_tf.close()
        vocab_path = os.path.join(opts.save_path,
                                  opts.out_prefix + '_vocab.pkl.gz')
        # store vocabulary
        with gzip.open(vocab_path, 'wb') as vocab_f:
            pickle.dump(vocab, vocab_f)

if __name__ == '__main__':
    description='Convert a CSV file into TFRecords. The CSV format is required '\
                'because every CSV separation is a sentence/utterance. Within '\
                'the utterance, every word is split in spaces, and chars can '\
                'also be retrieved if desired and flag is specified.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--csv_file', type=str, default=None,
                        help='Filename to be converted.')
    parser.add_argument('--exclude_csv_fields', type=int,
                        nargs='+', default=None,
                        help='List of indexes to be excluded from processing '
                             'in the CSV file (Def: None).')
    parser.add_argument('--label_idx', type=int, default=None,
                        help='CSV index where the label is, if any '
                             '(Def: None).')
    parser.add_argument('--out_prefix', type=str, default=None,
                        help='Prefix of filename for saved TFRecords.')
    parser.add_argument('--save_path', type=str, default='data/',
                        help='Path where output files are written (Def: data).')
    parser.add_argument('--text_encoding', type=str, default='utf-8',
                        help='Text encoding (Def: utf-8).')
    parser.add_argument('--do-chars', dest='do_chars', action='store_true',
                        help='Include chars of every word into the TFRecords'
                             '(Def: False).')
    parser.add_argument('--no-words', dest='do_words', action='store_false',
                        help='Exclude words (Def: False).')
    parser.set_defaults(do_chars=False, do_words=True)
    opts = parser.parse_args()
    # global constants for certain tokens
    opts.tokens = Tokens(
        EOS='\x04',
        UNK='<UNK>',    # unk word token
        START='\x02',  # start-of-word token
        END='\x03',    # end-of-word token
        ZEROPAD=' ' # zero-pad token
    )

    if not opts.do_words and not opts.do_chars:
        raise ValueError('At least words or chars must be encoded in the TFR!')

    if opts.csv_file is None:
        raise ValueError('A CSV input file must be specified!')

    if opts.out_prefix is None:
        raise ValueError('Please specify an output prefix for your '
                         'output files to be names <prefix>_<content>.')
    print('Parsed opts from arguments:')
    print(json.dumps(vars(opts), indent=2))
    main(opts)
