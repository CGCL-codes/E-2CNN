# %%
from tokenization import BasicTokenizer
import json


# sentence = "2018年England，蔡志坚在南京艺术学院求学时受过系统、正规的艺术教育和专业训练，深得刘海粟、罗叔子、陈之佛、谢海燕、陈大羽等著名中国画大师的指授，基本功扎实，加上他坚持从生活中汲取创作源泉，用心捕捉生活中最美最感人的瞬间形象，因而他的作品，不论是山水、花鸟、飞禽、走兽，无不充满了生命的灵气，寄托着画家的情怀，颇得自然之真趣"
bt = BasicTokenizer()
# sentence_tokens = bt.tokenize(sentence)
# print(sentence_tokens)
max_sent_len = 450 # BERT最大输入长度
output_data_dir = '../pure_finance_crosslabel/test.json'
with open(output_data_dir, "w", encoding="utf-8") as f_out:
    with open("/home/newsgrid/lmf/PURE/data/finance_crosslabel/test.json", 'r') as f:
        doc_id = 0
        cnt = 0
        for line in f.readlines():
            pure_dic = {'doc_key': str(doc_id), 'sentences':[], 'ner':[], "relations":[]}
            doc_id += 1
            spo_text = json.loads(line) 
            sentence_tokens = bt.tokenize(spo_text['text'])
            pure_dic['sentences'].append(sentence_tokens[:max_sent_len if len(sentence_tokens) > max_sent_len else len(sentence_tokens)])
            ner_list = []
            relation_list = []
            for spo in spo_text['spo_list']:
                s_start, s_end = 0, 0
                sub_tokens = bt.tokenize(spo['subject'])
                for i in range(len(sentence_tokens)):  
                    if sentence_tokens[i: i+len(sub_tokens)] == sub_tokens:
                        s_start, s_end = i, i + len(sub_tokens) - 1
                if [s_start, s_end, spo['subject_type']] not in ner_list and s_end < max_sent_len:
                    ner_list.append([s_start, s_end, spo['subject_type']])

                o_start, o_end = 0, 0
                obj_tokens = bt.tokenize(spo['object'])
                for i in range(len(sentence_tokens)):  
                    if sentence_tokens[i: i+len(obj_tokens)] == obj_tokens:
                        o_start, o_end = i, i + len(obj_tokens) - 1
                if [o_start, o_end, spo['object_type']] not in ner_list and o_end < max_sent_len:
                    ner_list.append([o_start, o_end, spo['object_type']])

                if s_end >= max_sent_len or o_end >= max_sent_len:
                    cnt += 1
                    continue
                relation_list.append([s_start, s_end, o_start, o_end, spo['predicate']])  
            pure_dic['ner'].append(ner_list)
            pure_dic['relations'].append(relation_list)
            f_out.write(str(json.dumps(pure_dic, ensure_ascii=False)))
            f_out.write('\n')
