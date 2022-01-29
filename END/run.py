import torch
from math import ceil
from tqdm import tqdm
from transformers import AutoConfig
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split


def save_list(datas, labels, path):
    print(path + ' data size:{}, {}'.format(len(datas), len(labels)))
    with open(path, 'w', encoding='utf-8') as fd:
        for data, label in zip(datas, labels):
            fd.write(data + '\t' + label + '\n')


if __name__ == "__main__":
    print("Preparing...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained('config.json')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, config=config, device=0)
    texts = []
    max_num = 4500
    with open('cnn_stories_500.txt', 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()):
            if i >= max_num:
                break
            line = line.strip()
            texts.append(line)
    print('Texts: ' + str(len(texts)))
    not_to_generate, to_generate, _1, _2 = train_test_split(texts, [0] * len(texts), test_size=2/3, random_state=42)
    not_to_add, to_add, _1, _2 = train_test_split(to_generate, [0] * len(to_generate), test_size=0.5)
    print("Start generating...")
    generated_texts = []
    gratio = 0.1
    with open('gptneo_generated.txt', 'a', encoding='utf-8', buffering=1) as fout:
        for text in tqdm(texts):
            leg = len(text)
            inp = text[:ceil(leg * gratio)]
            while inp[-1] != ' ':
                inp = inp[:-1]
            res = generator(inp, do_sample=True, min_length=450, max_length=500)
            res_text = res[0]['generated_text'].replace('\n', '').strip()
            generated_texts.append(res_text)
            fout.write(res_text + '\n')
    all_data = generated_texts + not_to_generate + to_add
    all_label = ['1'] * len(generated_texts) + ['0'] * (len(not_to_generate) + len(to_add))
    last_x, train_x, last_y, train_y = train_test_split(all_data, all_label, test_size=0.4)
    dev_x, test_x, dev_y, test_y = train_test_split(last_x, last_y, test_size=0.5)
    save_list(train_x, train_y, 'train.txt')
    save_list(test_x, test_y, 'test.txt')
    save_list(dev_x, dev_y, 'dev.txt')
