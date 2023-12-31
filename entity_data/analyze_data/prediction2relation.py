import copy
import json
import argparse
import os


parser = argparse.ArgumentParser(description='data_path')
parser.add_argument('--pred_dir', default='entity_data/FCAR/', type=str)
parser.add_argument('--output_dir', default='relation_data/FCAR/', type=str)
# entity_type = {'人物': 'peo', '学校': 'school', '影视作品': 'video', 
# '歌曲': 'song', '图书作品': 'book', '出版社': 'press', '企业': 'comp', 
# '日期': 'date', '音乐专辑': 'album', '网络小说': 'fict', '网站': 'web', 
# '地点': 'loc', '国家': 'country', '文本': 'text', '机构': 'indus',}

entity_type = {'公司企业': 'comp',
	'人员':'peo',
	'地域':'loc',
	'业务':'busi',
	'行业':'indus',
    '产品': 'product'}

'''显式地将实体标签加入文本中'''
def add_marker_to_text(output_data_dir, pred_dir):
    with open(output_data_dir, "w", encoding="utf-8") as f_out:
        with open(pred_dir, "r", encoding="utf-8") as f:
            for line in f.readlines():
                spo_text = json.loads(line) 
                sent = spo_text['sentences'][0]

                lower_sent = {}
                entity_set = set()
                lower_sent['text'] = ""
                lower_sent['spo_list'] = []
                for (idx, char) in enumerate(sent):
                    for ner in spo_text['predicted_ner'][0]:
                        if ner[0] == idx: # 与实体开始位置匹配上
                            lower_sent['text'] = lower_sent['text'] + "[{}]".format(entity_type[ner[2]])
                    lower_sent['text'] = lower_sent['text'] + char
                    for ner in spo_text['predicted_ner'][0]:
                        if ner[1] == idx: # 与实体结束位置匹配上
                            lower_sent['text'] = lower_sent['text'] + "[/{}]".format(entity_type[ner[2]])
                for rel in spo_text['relations'][0]:
                    predicate = rel[4]
                    subject = ""
                    object = ""
                    subject_type = ""
                    object_type = ""
                    for ner in spo_text['predicted_ner'][0]:
                        if ner[0] == rel[0] and ner[1] == rel[1]:
                            subject_type = ner[2]
                            subject = "[{}]".format(entity_type[ner[2]])
                            for i in range(ner[0], ner[1]+1):
                                subject += sent[i]
                            subject += "[/{}]".format(entity_type[ner[2]])
                        if ner[0] == rel[2] and ner[1] == rel[3]:
                            object_type = ner[2]
                            object = "[{}]".format(entity_type[ner[2]])
                            for i in range(ner[0], ner[1]+1):
                                object += sent[i]
                            object += "[/{}]".format(entity_type[ner[2]])
                    lower_sent['spo_list'].append({"predicate": predicate, "subject_type": subject_type, "object_type": object_type, "subject": subject, "object": object})
                json.dump(lower_sent, f_out, ensure_ascii=False)      
                f_out.write("\n")
            
'''
在json数据中加入predict_entity:[[start_idx, end_idx, entity_type], [], ...]这一信息
'''            
def add_predict_entity(output_data_dir, pred_dir, is_train):
    with open(output_data_dir, "w", encoding="utf-8") as f_out:
        with open(pred_dir, "r", encoding="utf-8") as f:
            for line in f.readlines():
                spo_text = json.loads(line) 
                sent = spo_text['sentences'][0]

                lower_sent = {}
                entity_set = set()
                lower_sent['text'] = ""
                lower_sent['spo_list'] = []
                lower_sent['predict_entity'] = []
                char_idx = [] # 每个token的字符级别的index
                ner_list = []
                if is_train:
                    ner_list = spo_text['ner'][0]
                else:
                    ner_list = spo_text['predicted_ner'][0]
                for (idx, char) in enumerate(sent):
                    if idx == 0:
                        char_idx.append(0)
                    else:
                        char_idx.append(char_idx[idx-1] + len(sent[idx-1]))
                for (idx, char) in enumerate(sent):
                    
                    for ner in ner_list:
                        if ner[0] == idx: # 与实体开始位置匹配上
                            lower_sent['predict_entity'].append([char_idx[ner[0]], char_idx[ner[1]], ner[2]])
                    lower_sent['text'] = lower_sent['text'] + " " + char
                for rel in spo_text['relations'][0]:
                    predicate = rel[4]
                    subject = ""
                    object = ""
                    subject_type = ""
                    object_type = ""
                    for ner in spo_text['ner'][0]:
                        if ner[0] == rel[0] and ner[1] == rel[1]:
                            subject_type = ner[2]
                            subject = ""
                            for i in range(ner[0], ner[1]+1):
                                subject += sent[i]
                        if ner[0] == rel[2] and ner[1] == rel[3]:
                            object_type = ner[2]
                            object = ""
                            for i in range(ner[0], ner[1]+1):
                                object += sent[i]
                    lower_sent['spo_list'].append({"predicate": predicate, "subject_type": subject_type, "object_type": object_type, "subject": subject, "object": object})
                json.dump(lower_sent, f_out, ensure_ascii=False)      
                f_out.write("\n")


'''
英文
'''            
def add_predict_entity_eng(output_data_dir, pred_dir, is_train):
    with open(output_data_dir, "w", encoding="utf-8") as f_out:
        with open(pred_dir, "r", encoding="utf-8") as f:
            for line in f.readlines():
                spo_text = json.loads(line) 
                sent = spo_text['sentences'][0]

                lower_sent = {}
                entity_set = set()
                lower_sent['text'] = ""
                lower_sent['spo_list'] = []
                lower_sent['predict_entity'] = []
                char_idx = [] # 每个token的字符级别的index
                ner_list = []
                if is_train:
                    ner_list = spo_text['ner'][0]
                else:
                    ner_list = spo_text['predicted_ner'][0]
                for (idx, char) in enumerate(sent):
                    if idx == 0:
                        char_idx.append(0)
                    else:
                        char_idx.append(char_idx[idx-1] + len(sent[idx-1]))
                for (idx, char) in enumerate(sent):
                    
                    for ner in ner_list:
                        if ner[0] == idx: # 与实体开始位置匹配上
                            lower_sent['predict_entity'].append([char_idx[ner[0]], char_idx[ner[1]], ner[2]])
                    lower_sent['text'] = lower_sent['text'] + " " + char
                for rel in spo_text['relations'][0]:
                    predicate = rel[4]
                    subject = ""
                    object = ""
                    subject_type = ""
                    object_type = ""
                    for ner in spo_text['ner'][0]:
                        if ner[0] == rel[0] and ner[1] == rel[1]:
                            subject_type = ner[2]
                            subject = ""
                            for i in range(ner[0], ner[1]+1):
                                subject += sent[i]
                        if ner[0] == rel[2] and ner[1] == rel[3]:
                            object_type = ner[2]
                            object = ""
                            for i in range(ner[0], ner[1]+1):
                                object += sent[i]
                    lower_sent['spo_list'].append({"predicate": predicate, "subject_type": subject_type, "object_type": object_type, "subject": subject, "object": object})
                json.dump(lower_sent, f_out, ensure_ascii=False)      
                f_out.write("\n")


args = parser.parse_args()
train_path = args.pred_dir + 'train.json'
output_train_path = args.output_dir + 'train.json'
add_predict_entity(output_train_path, train_path, True)

dev_path = args.pred_dir + 'ent_pred_dev.json'
output_dev_path = args.output_dir + 'dev.json'
add_predict_entity(output_dev_path, dev_path, True)
# if not os.path.exists(dev_path):
#     dev_path = args.pred_dir + 'dev.json'
#     add_predict_entity(output_dev_path, dev_path, True)
# else:
#     add_predict_entity(output_dev_path, dev_path, False)

test_path = args.pred_dir + 'ent_pred_test.json'
output_test_path = args.output_dir + 'test.json'
add_predict_entity(output_test_path, test_path, False)
# if not os.path.exists(test_path):
#     test_path = args.pred_dir + 'test.json'
#     add_predict_entity(output_test_path, test_path, True)
# else:
#     add_predict_entity(output_test_path, test_path, False)