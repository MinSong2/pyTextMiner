import json
import pickle
import torch
from gluonnlp.data import SentencepieceTokenizer

from py_ner.bert_crf_ner_prediction import DecoderFromNamedEntitySequence
from py_ner.model.net import KobertCRFViz
from py_ner.data_utils.utils import Config
from py_ner.data_utils.vocab_tokenizer import Tokenizer
from py_ner.data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

from py_ner.bertviz.head_view import show

class BertCrfNerVisualization:
    def __init__(self, model_dir=''):
        #'./experiments/base_model_with_crf'
        self.model_dir = model_dir
        self.model_config = Config(json_path=self.model_dir + '/config.json')
        self.tokenizer = None
        self.model = None
        self.decoder_from_res = None

    def load_model(self, tokenizer_model_name, ner_model_name):
        # load vocab & tokenizer
        #tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
        tok_path = tokenizer_model_name
        ptr_tokenizer = SentencepieceTokenizer(tok_path)

        with open(self.model_dir + "/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        self.tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=self.model_config.maxlen)

        # load ner_to_index.json
        with open(self.model_dir + "/ner_to_index.json", 'rb') as f:
            ner_to_index = json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}

        # model
        self.model = KobertCRFViz(config=self.model_config, num_classes=len(ner_to_index), vocab=vocab)

        #ner_model_name = "./experiments/base_model_with_crf/best-epoch-16-step-1500-acc-0.993.bin"
        # load
        model_dict = self.model.state_dict()
        checkpoint = torch.load(ner_model_name, map_location=torch.device('cpu'))
        convert_keys = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key_name = k.replace("module.", '')
            if new_key_name not in model_dict:
                print("{} is not int model_dict".format(new_key_name))
                continue
            convert_keys[new_key_name] = v

        self.model.load_state_dict(convert_keys)
        self.model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=self.tokenizer, index_to_ner=index_to_ner)

    def visualize(self):
        input_text = '김대중 대통령은 노벨평화상을 받으러 스웨덴으로 출국해서 5박6일 동안 스웨덴에 머물며 대한민국의 위상을 높였다.'
        list_of_input_ids = self.tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
        x_input = torch.tensor(list_of_input_ids).long()
        list_of_pred_ids, _ = self.model(x_input)

        list_of_ner_word, decoding_ner_sentence = self.decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
        print("output>", decoding_ner_sentence)
        model_type = 'bert'
        show(self.model, model_type, self.tokenizer, decoding_ner_sentence, input_text)
        print("")

if __name__ == '__main__':
    model_dir = '../examples/exper/base_model_with_crf'
    visualizer = BertCrfNerVisualization(model_dir)

    tokenizer_model_name = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ner_model_name = "../examples/exper/base_model_with_crf/best-epoch-6-step-500-acc-0.943.bin"

    visualizer.load_model(tokenizer_model_name, ner_model_name)

    visualizer.visualize()