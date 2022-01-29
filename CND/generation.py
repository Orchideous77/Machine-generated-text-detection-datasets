import torch
import torch.nn.functional as F
import os
import argparse
import random
from tqdm import trange
from transformers import GPT2LMHeadModel
from math import *


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0,
                    repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            # 一次生成一个，在末尾
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            # batch, seq_len, hidden_size 第二维为生成的，取最后一个
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            # 不取unk的词
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


# 通过命令行参数--fast_pattern，指定模式
def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0,
             device='cpu',
             is_fast_pattern=False):
    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k,
                               top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)


def check_length(line, max_len):
    if len(line) > max_len:
        for i in range(max_len):
            if line[max_len-i-1] in '.。!！?？':
                return line[:max_len-i]
        return line[:max_len]
    else:
        return line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=200, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=2, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=40, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.9, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='models/fine_tuned_general/config.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='models/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='models/fine_tuned_general', type=str, required=False, help='模型路径')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='使用past加快生成速度')
    parser.add_argument('--save_samples', default=True, action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.5, type=float, required=False, help="重复生成惩罚参数")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenization import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenization import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if length == -1:
        length = model.config.n_ctx
    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
    fin = open('human_written.txt', 'r', encoding='utf-8')
    f = open(args.save_samples_path + '/t1.txt', 'w', encoding='utf-8')
    rates = [0.3]
    for line in fin:
        r = random.random()
        if r < 0.5:
            continue
        line = line.strip()
        line = check_length(line, 450)
        l0 = len(line)
        print(l0)
        for r in rates:
            l1 = floor(l0 * r)
            title = line[:l1]
            if len(title.strip()) == 0:
                continue
            raw_text = title.strip()
            context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
            generated = 0
            for _ in range(nsamples // batch_size):
                out = generate(
                    n_ctx=n_ctx,
                    model=model,
                    context=context_tokens,
                    length=l0 - l1,
                    is_fast_pattern=args.fast_pattern, tokenizer=tokenizer,
                    temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty,
                    device=device
                )
                for i in range(batch_size):
                    generated += 1
                    text = tokenizer.convert_ids_to_tokens(out)
                    for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                        if is_word(item) and is_word(text[i + 1]):
                            text[i] = item + ' '
                    for i, item in enumerate(text):
                        if item == '[MASK]':
                            text[i] = ''
                        elif item == '[CLS]':
                            text[i] = '\n\n'
                        elif item == '[SEP]':
                            text[i] = '\n'
                    text = ''.join(text).replace('##', '').replace('\n', '').strip()
                    print(len(text))
                    f.write(text + '\n')
                    #f.write(str(round(l1 / l0, 10)))
                    #f.write('\n')


if __name__ == '__main__':
    main()
