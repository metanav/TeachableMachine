import logging as log
import string
import unicodedata
import numpy as np
from openvino.inference_engine import IECore

class BERT():
    def __init__(self, model_xml, model_bin, input_names, output_names, vocab_file, device='MYRIAD'):
        self.context      = None
        self.input_names  = input_names
        self.output_names = output_names

        log.info("Loading vocab file:\t{}".format(vocab_file))
        with open(vocab_file, "r", encoding="utf-8") as r:
            self.vocab = {t.rstrip("\n"): i for i, t in enumerate(r.readlines())}

        log.info("{} tokens loaded".format(len(self.vocab)))
    
        log.info("Initializing Inference Engine")
        ie = IECore()
        ie.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'NO'}, "MYRIAD")
        version = ie.get_versions(device)[device]
        version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
        log.info("Plugin version is {}".format(version_str))
    
        # read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        ie_encoder = ie.read_network(model=model_xml, weights=model_bin)
    
        # maximum number of tokens that can be processed by network at once
        self.max_length = ie_encoder.input_info[self.input_names[0]].input_data.shape[1]
    
        # load model to the device
        log.info("Loading model to the {}".format(device))
        self.ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=device)

    # split word by vocab items and get tok codes
    # iteratively return codes
    def encode_by_voc(self, w, vocab):
        # remove mark and control chars
        def clean_word(w):
            # extract marks as separate chars to remove them later
            wo = ""  # accumulator for output word
            for c in unicodedata.normalize("NFD", w):
                c_cat = unicodedata.category(c)
                # remove mark nonspacing code and controls
                if c_cat != "Mn" and c_cat[0] != "C":
                    wo += c
            return wo
    
        w = clean_word(w)
        w = w.lower()
        s, e = 0, len(w)
        while e > s:
            subword = w[s:e] if s == 0 else "##" + w[s:e]
            if subword in vocab:
                yield vocab[subword]
                s, e = e, len(w)
            else:
                e -= 1
        if s < len(w):
            yield vocab['[UNK]']
    
    
    # split big text into words by spaces
    # iteratively return words
    def split_to_words(self, text):
        prev_is_sep = True  # mark initial prev as space to start word from 0 char
        for i, c in enumerate(text + " "):
            is_punc = (c in string.punctuation or unicodedata.category(c)[0] == "P")
            cur_is_sep = (c.isspace() or is_punc)
            if prev_is_sep != cur_is_sep:
                if prev_is_sep:
                    start = i
                else:
                    yield start, i
                    del start
            if is_punc:
                yield i, i + 1
            prev_is_sep = cur_is_sep
    
    
    # get the text and return list of token ids and start-end position for each id (in the original text)
    def text_to_tokens(self, text, vocab):
        tokens_id = []
        tokens_se = []
        for s, e in self.split_to_words(text):
            for tok in self.encode_by_voc(text[s:e], vocab):
                tokens_id.append(tok)
                tokens_se.append((s, e))
        log.info("Size: {} tokens".format(len(tokens_id)))
        return tokens_id, tokens_se
    
    
    # return entire sentence as start-end positions for a given answer (within the sentence).
    def find_sentence_range(self, context, s, e):
        # find start of sentence
        for c_s in range(s, max(-1, s - 200), -1):
            if context[c_s] in "\n\.":
                c_s += 1
                break
    
        # find end of sentence
        for c_e in range(max(0, e - 1), min(len(context), e + 200), +1):
            if context[c_e] in "\n\.":
                break
    
        return c_s, c_e
    
    
    # set context as one big string by given input arguments
    def set_context(self, context):
        self.context = context.replace('\n', ' ')
        # encode context into token ids list
        self.c_tokens_id, self.c_tokens_se = self.text_to_tokens(self.context, self.vocab)
        log.info("Context: {}\nSize: {} characters".format(self.context, len(self.context)))


    # inference
    def inference(self, question):
        if len(list(self.split_to_words(question))) < 3:
            log.info("Question must contain 3 words.")
            return None

        q_tokens_id, _ = self.text_to_tokens(question, self.vocab)

        # calculate number of tokens for context in each inference request.
        # reserve 3 positions for special tokens
        # [CLS] q_tokens [SEP] c_tokens [SEP]
        c_wnd_len = self.max_length - (len(q_tokens_id) + 3)

        # token num between two neighbour context windows
        # 1/2 means that context windows are overlapped by half
        c_stride = c_wnd_len // 2

        # array of answers from each window
        answers = []

        # init a window to iterate over context
        c_s, c_e = 0, min(c_wnd_len, len(self.c_tokens_id))

        # iterate while context window is not empty
        while c_e > c_s:
            # form the request
            tok_cls = self.vocab['[CLS]']
            tok_sep = self.vocab['[SEP]']
            input_ids = [tok_cls] + q_tokens_id + [tok_sep] + self.c_tokens_id[c_s:c_e] + [tok_sep]
            token_type_ids = [0] + [0] * len(q_tokens_id) + [0] + [1] * (c_e - c_s) + [0]
            attention_mask = [1] * len(input_ids)

            # pad the rest of the request
            pad_len = self.max_length - len(input_ids)
            input_ids += [0] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len

            # create numpy inputs for IE
            inputs = {
                self.input_names[0]: np.array([input_ids], dtype=np.int32),
                self.input_names[1]: np.array([attention_mask], dtype=np.int32),
                self.input_names[2]: np.array([token_type_ids], dtype=np.int32),
            }
            # infer by IE
            res = self.ie_encoder_exec.infer(inputs=inputs)
            # get start-end scores for context
            def get_score(name):
                out = np.exp(res[name].reshape((self.max_length,)))
                return out / out.sum(axis=-1)

            score_s = get_score(self.output_names[0])
            score_e = get_score(self.output_names[1])

            score_na = 0

            # find product of all start-end combinations to find the best one
            c_s_idx = len(q_tokens_id) + 2  # index of first context token in tensor
            c_e_idx = self.max_length - (1 + pad_len)  # index of last+1 context token in tensor
            score_mat = np.matmul(
                score_s[c_s_idx:c_e_idx].reshape((c_e - c_s, 1)),
                score_e[c_s_idx:c_e_idx].reshape((1, c_e - c_s))
            )
            # reset candidates with end before start
            score_mat = np.triu(score_mat)
            # reset long candidates (>max_answer_token_num)
            max_answer_token_num = 15
            score_mat = np.tril(score_mat, max_answer_token_num - 1)
            # find the best start-end pair
            max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
            max_score = score_mat[max_s, max_e] * (1 - score_na)

            # convert to context text start-end index
            max_s = self.c_tokens_se[c_s + max_s][0]
            max_e = self.c_tokens_se[c_s + max_e][1]
            # check that answers list does not have duplicates (because of context windows overlapping)
            same = [i for i, a in enumerate(answers) if a[1] == max_s and a[2] == max_e]
            if same:
                assert len(same) == 1
                # update existing answer record
                a = answers[same[0]]
                answers[same[0]] = (max(max_score, a[0]), max_s, max_e)
            else:
                # add new record
                answers.append((max_score, max_s, max_e))

            # check that context window reached the end
            if c_e == len(self.c_tokens_id):
                break

            # move to next window position
            c_s = min(c_s + c_stride, len(self.c_tokens_id))
            c_e = min(c_s + c_wnd_len, len(self.c_tokens_id))

        # print top 3 results
        answers = sorted(answers, key=lambda x: -x[0])
        answer  = ''
        for score, s, e in answers[:1]:
            answer = self.context[s:e]
            log.info("Answer: {} [score: {:0.2f}]".format(answer, score))
            c_s, c_e = self.find_sentence_range(self.context, s, e)
            log.debug("   " + self.context[c_s:s] + answer +  self.context[e:c_e])

        return answer
 
