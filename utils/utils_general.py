import os
import re
import time
from typing import Optional

import torch
import torch.utils.data as data
import torch.nn as nn
from openai import OpenAI

from utils.config import *


def _simple_detokenize(text_input):
    """Turn token list or string into a single line of text for metrics / LLM polish."""
    if isinstance(text_input, list):
        filtered_list = [
            t
            for t in text_input
            if t not in ['<EOS>', '<PAD>', '<GO>', '<UNK>', '<SOS>', '<EOT>', '<GO_S>', '<GO_E>', '<PAD_S>', '<PAD_E>', 'PAD', 'EOS']
            and t is not None
            and t != ''
        ]
        text_str = " ".join(filtered_list)
    elif isinstance(text_input, str):
        text_str = text_input
        text_str = text_str.replace('<EOS>', ' ').replace('<PAD>', ' ').replace('<GO>', ' ').replace('<UNK>', ' ')
        text_str = text_str.replace('<SOS>', ' ').replace('<EOT>', ' ').replace('<GO_S>', ' ').replace('<GO_E>', ' ')
        text_str = text_str.replace('<PAD_S>', ' ').replace('<PAD_E>', ' ').replace('PAD', ' ').replace('EOS', ' ')
    else:
        return ""

    text_str = text_str.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    text_str = text_str.replace(" 's", "'s").replace(" 't", "'t").replace(" 've", "'ve").replace(" 'm", "'m")
    text_str = text_str.replace(" 're", "'re").replace(" 'll", "'ll")
    text_str = text_str.replace(" (", "(").replace(" )", ")")
    text_str = text_str.replace(" :", ":").replace(" ;", ";")
    text_str = text_str.replace(" - ", "-")
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    return text_str


def strip_polish_response(text: str) -> str:
    """Remove common LLM wrappers so evaluation sees plain text."""
    if not text:
        return ""
    s = text.strip()
    for p in (
        "润色后：",
        "润色后:",
        "润色结果：",
        "润色结果:",
        "输出：",
        "输出:",
        "改写后：",
        "改写后:",
    ):
        if s.startswith(p):
            s = s[len(p) :].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


deepseek_client: Optional[OpenAI] = None
_deepseek_client_init_attempted = False
_deepseek_client_init_succeeded = False


def _initialize_deepseek_client():
    global deepseek_client, _deepseek_client_init_attempted, _deepseek_client_init_succeeded

    if _deepseek_client_init_succeeded:
        return True
    if _deepseek_client_init_attempted and not _deepseek_client_init_succeeded:
        return False
    _deepseek_client_init_attempted = True

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set. DeepSeek refinement disabled.")
        return False
    try:
        deepseek_client = OpenAI(api_key=api_key, base_url=DeepSeekConfig.BASE_URL)
        print("DeepSeek OpenAI client initialized successfully.")
        _deepseek_client_init_succeeded = True
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize DeepSeek client: {e}")
        return False


def polish_with_deepseek(context_str, raw_response, gold_response=None, max_retries=3):
    """Call DeepSeek to polish English system reply; keeps entities (prompt-constrained)."""
    if not _initialize_deepseek_client():
        return None

    prompt = f"""你是对话系统回复润色助手。下面给出「对话上下文」和模型生成的「原始回复草稿」。

请只润色措辞与句式，使英文更自然、通顺；不要改变语义，不要增删事实信息。

【硬性要求】
- 保留原始草稿中出现的所有实体：人名、地名、餐馆/酒店/景点名、地址、电话号码、时间、日期、价格、编号等，逐字保留（仅可调整前后空格或标点以符合语法）。
- 不要根据上下文编造或补全新实体；不要替换同义词实体（例如不要把具体店名改成泛称）。
- 输出仍为一句完整的系统回复正文，不要解释，不要列出修改说明。

对话上下文:
{context_str}

原始回复草稿:
{raw_response}

只输出润色后的回复正文一行（或自然的一段），不要有标题或前缀。
"""
    if gold_response:
        prompt += f"\n（可选参考，勿照抄）参考回复: {gold_response}\n"

    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            response = deepseek_client.chat.completions.create(
                model=DeepSeekConfig.MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
                timeout=60,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(DeepSeekConfig.RETRY_DELAY)
    print(f"DeepSeek polishing failed after {max_retries} attempts.")
    return None


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


if args['dataset'] == 'kvr':
    domains = {'navigate': 0, 'weather': 1, 'schedule': 2}
elif args['dataset'] == 'woz':
    domains = {'restaurant': 0, 'attraction': 1, 'hotel': 2}


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, lang):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.lang = lang

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        selector_index = torch.Tensor(self.data_info['selector_index'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['gold_sketch_response'] = self.data_info['sketch_response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        selector_index, _ = merge_index(item_info['selector_index'])
        ptr_index, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)

        max_seq_len = conv_arr.size(1)
        label_arr = _cuda(torch.Tensor([domains[label] for label in item_info['domain']]).long().unsqueeze(-1))
        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        selector_index = _cuda(selector_index.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        if (len(list(kb_arr.size())) > 1): kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())
        item_info['label_arr'] = []

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths
        return data_info


def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if (type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=dataset.collate_fn)
    return data_loader
