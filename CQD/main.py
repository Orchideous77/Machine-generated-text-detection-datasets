import requests
import json
import time
from tqdm import tqdm
from datetime import datetime

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36 Edg/90.0.818.46"
}


def retry(data, logger):
    logger.write(data['poem'] + ' retried\n')
    time.sleep(5)
    return requests.post(url="https://jiuge.cs.tsinghua.edu.cn/jiugepoem/task/send_jueju", data=data, headers=headers)


if __name__ == '__main__':
    keywords = []
    with open('test_inps_1.txt', 'r', encoding='utf-8') as fin:
        for i in fin.readlines():
            i = i.strip()
            if i not in keywords and i != '':
                keywords.append(i)
    print("number of keywords: {0}".format(len(keywords)))
    datas = {"yan": "7", "user_id": "k6FYP7ayc3zTMHxJaGHcDbnaYf6D2x"}
    fout = open('result3.txt', 'a', encoding='utf-8', buffering=1)
    logger = open('log2.txt', 'a', encoding='utf-8', buffering=1)
    start_time = datetime.now()
    print("start time: " + start_time.strftime('%m-%d %H:%M:%S'))
    for word in tqdm(keywords[:250]):
        for y in ['5', '7']:
            datas['yan'] = y
            datas['poem'] = word
            while True:
                try:
                    res_0 = requests.post(url="https://jiuge.cs.tsinghua.edu.cn/jiugepoem/task/send_jueju", data=datas, headers=headers)
                    res_0 = json.loads(res_0.text)
                except json.decoder.JSONDecodeError or requests.exceptions.ConnectionError:
                    continue
                else:
                    break
            cnt = 0
            while res_0['code'] == '0':
                time.sleep(0.5)
                try:
                    res = requests.post(url="https://jiuge.cs.tsinghua.edu.cn/jiugepoem/task/get_jueju",
                                        data={'celery_id': res_0['celery_id']}, headers=headers)
                    result = json.loads(res.text)
                except requests.exceptions.ConnectionError:
                    res_0 = retry(datas, logger)
                    continue
                except json.decoder.JSONDecodeError:
                    print(word, ' ', res, res.text)
                    continue
                if result['status'] == 'PENDING':
                    cnt += 1
                    time.sleep(0.8)
                elif result['status'] == 'NOOUTPUT' or cnt > 30:
                    res_0 = retry(datas)
                    continue
                elif result['status'] == 'SUCCESS':
                    poem = '|'.join(result['output'])
                    fout.write(poem + '\n')
                    logger.write('{} {} {}\n'.format(word, y, cnt))
                    break
                else:
                    logger.write('{} {} failed\n'.format(word, y))
                    break
            time.sleep(1)
    fout.close()
    logger.close()
    end_time = datetime.now()
    time_dif = end_time - start_time
    print("end time: " + end_time.strftime('%m-%d %H:%M:%S'))
    print("time usage: " + time_dif.strftime('%m-%d %H:%M:%S'))
