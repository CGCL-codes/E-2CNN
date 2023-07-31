import json

entity_type = {'人物': 'peo', '学校': 'school', '影视作品': 'video', 
'歌曲': 'song', '图书作品': 'book', '出版社': 'press', '企业': 'comp', 
'日期': 'date', '音乐专辑': 'album', '网络小说': 'fict', '网站': 'web', 
'地点': 'loc', '国家': 'country', '文本': 'text', '机构': 'indus',}

class Config(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_name = args.bert_name
        self.bert_dim = args.bert_dim

        self.train_path = 'relation_data/' + self.dataset + '/train.json'
        self.test_path = 'relation_data/' + self.dataset + '/test.json'
        self.dev_path = 'relation_data/' + self.dataset + '/dev.json'
        self.rel_path = 'relation_data/' + self.dataset + '/rel.json'
        self.entity_path = 'relation_data/' + self.dataset + '/entity.json'
        self.num_relations = len(json.load(open(self.rel_path, 'r')))

        self.save_weights_dir = 'relation_output/saved_weights/' + self.dataset + '/'
        self.save_logs_dir = 'relation_output/saved_logs/' + self.dataset + '/'
        self.result_dir = 'results/' + self.dataset + '/'

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.result_save_name = 'result.json'
