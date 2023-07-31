import json

def unicodeFile_convertTo_correct_file(save_file, ent_pred_dev):
    with open(save_file, "w") as save_f:
        with open(ent_pred_dev, "r", encoding="utf-8") as f:
            for line in f.readlines():
                spo_text = json.loads(line)
                save_f.write((json.dumps(spo_text, ensure_ascii=False)))
                save_f.write("\n")

save_file = "ent_pred_dev.json"
ent_pred_dev = "../bert_base_chinese_models/baidu50epoch/ent_pred_dev.json"
# unicodeFile_convertTo_correct_file(save_file, ent_pred_dev)
with open("../pure_baidu_tokenize_data/dev.json", "r", encoding="utf-8") as f:
    max_ner_len = 0
    for line in f.readlines():
        spo_text = json.loads(line)
        tmp = max_ner_len
        max_ner_len = max(max_ner_len, max(i[1]-i[0]+1 for i in spo_text['ner'][0]))
        if max_ner_len > tmp:
            print(spo_text['sentences'], spo_text['ner'][0])
    print(max_ner_len)
        