import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertTokenizer


def first_process(l):
    t1 = l.find('(CNN)')
    t2 = l.find('--')
    if t1 < 0 and t2 < 0:
        return l.strip()
    elif t1 >= 0 and t2 < 0:
        res = l[t1+5:].strip()
    else:
        res = l[t2+2:].strip()
    while res and not res[0].isalpha():
        res = res[1:]
    return res


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    path = './CNNStories/'
    max_length = 500
    max_file = 6750
    with open('cnn_stories_500.txt', 'w', encoding='utf-8', buffering=1) as fout:
        text_count = 0
        for file in tqdm(os.listdir(path)):
            with open(path + file, 'r', encoding='utf-8') as fin:
                text = first_process(fin.readline())
                token_count = len(tokenizer.tokenize(text))
                for line in fin.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    elif line[0] == '@':
                        break
                    else:
                        leg = len(tokenizer.tokenize(line))
                        if token_count + leg <= max_length:
                            text += (' ' + line)
                            token_count += leg
                        else:
                            break
                if text:
                    text_count += 1
                    fout.write(text + '\n')
            if text_count >= max_file:
                break
