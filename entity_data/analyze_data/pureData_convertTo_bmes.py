import json

pure_data_dir = '../pure_baidu_tokenize_data/train.json'
output_data_dir ='/home/newsgrid/wangweihao/BERT-NER-Pytorch-master/datasets/finance/train.char.bmes'
entity_type = {'人物': 'PEO', '学校': 'SCHOOL', '影视作品': 'VIDEO', 
'歌曲': 'SONG', '图书作品': 'BOOK', '出版社': 'PRESS', '企业': 'COMP', 
'日期': 'DATE', '音乐专辑': 'ALBUM', '网络小说': 'FICT', '网站': 'WEB', 
'地点': 'LOC', '国家': 'COUNTRY', '文本': 'TEXT', '机构': 'INDUS',}
with open(output_data_dir, "w", encoding="utf-8") as f_out:
    with open(pure_data_dir, "r", encoding="utf-8") as f:
        for line in f.readlines():
            spo_text = json.loads(line) 
            sent = spo_text['sentences'][0]
            bios_sent = [[token, 'O'] for token in sent]
            for entit in spo_text['ner'][0]:
                if entit[0] == entit[1]:
                    bios_sent[entit[0]][1] = 'S-' + entity_type[entit[2]]
                else:
                    bios_sent[entit[0]][1] = 'B-' + entity_type[entit[2]]
                    for i in range(entit[0]+1, entit[1]):
                        bios_sent[i][1] = 'M-' + entity_type[entit[2]]
                    bios_sent[entit[1]][1] = 'E-' + entity_type[entit[2]]    
            for enti_bios in bios_sent:
                f_out.write(enti_bios[0] + " " + enti_bios[1] + "\n")
            f_out.write("\n")
            

