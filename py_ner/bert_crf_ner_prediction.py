import json
import pickle
import argparse
import torch
from py_ner.kobert.pytorch_kobert import get_pytorch_kobert_model
from py_ner.kobert.utils import get_tokenizer
from py_ner.model.net import KobertSequenceFeatureExtractor, KobertCRF, KobertBiLSTMCRF, KobertBiGRUCRF
from gluonnlp.data import SentencepieceTokenizer
from py_ner.data_utils.utils import Config
from py_ner.data_utils.vocab_tokenizer import Tokenizer
from py_ner.data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

class BertCRFNERPredictor:
    def __init__(self, model_dir=''):
        self.model_dir = model_dir
        self.model_config = Config(json_path=self.model_dir + '/config.json')
        self.tokenizer = None
        self.model = None
        self.decoder_from_res = None
        self.model_name = ''

    def load_model(self, model_name='bert_crf',
                   tokenizer_path='./ptr_lm_model/tokenizer_78b3253a26.model',
                   checkpoint_file='./experiments/base_model_with_crf_val/best-epoch-9-step-750-acc-0.980.bin'):
        # Vocab & Tokenizer
        # tok_path = get_tokenizer() # ./tokenizer_78b3253a26.model
        #tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
        tok_path = tokenizer_path
        ptr_tokenizer = SentencepieceTokenizer(tok_path)

        # load vocab & tokenizer
        with open(self.model_dir + "/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)

        self.tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=self.model_config.maxlen)

        # load ner_to_index.json
        with open(self.model_dir + "/ner_to_index.json", 'rb') as f:
            ner_to_index = json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}

        # Model
        # model = KobertSequenceFeatureExtractor(config=self.model_config, num_classes=len(ner_to_index))
        if model_name == 'bert_lstm_crf':
            self.model = KobertBiLSTMCRF(config=self.model_config, num_classes=len(ner_to_index), vocab=vocab)
            checkpoint = torch.load(checkpoint_file,
                                    map_location=torch.device('cpu'))
        elif model_name == 'bert_gru_crf':
            self.model = KobertBiGRUCRF(config=self.model_config, num_classes=len(ner_to_index), vocab=vocab)
            checkpoint = torch.load(checkpoint_file,
                                    map_location=torch.device('cpu'))
        else:
            self.model = KobertCRF(config=self.model_config, num_classes=len(ner_to_index), vocab=vocab)
            checkpoint = torch.load(checkpoint_file,
                                    map_location=torch.device('cpu'))
        self.model_name = model_name

        # load
        model_dict = self.model.state_dict()
        # checkpoint = torch.load("./experiments/base_model/best-epoch-9-step-600-acc-0.845.bin", map_location=torch.device('cpu'))
        # checkpoint = torch.load("./experiments/base_model_with_crf/best-epoch-16-step-1500-acc-0.993.bin", map_location=torch.device('cpu'))
        #checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin", map_location=torch.device('cpu'))
        # checkpoint = torch.load("./experiments/base_model_with_bilstm_crf/best-epoch-15-step-2750-acc-0.992.bin", map_location=torch.device('cpu'))
        # checkpoint = torch.load("./experiments/base_model_with_bigru_crf/model-epoch-18-step-3250-acc-0.997.bin", map_location=torch.device('cpu'))

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

        # n_gpu = torch.cuda.device_count()
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        self.model.to(device)

        self.decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=self.tokenizer, index_to_ner=index_to_ner)

    def predict(self, input_text):
        print("문장을 입력하세요: ")
        list_of_input_ids = self.tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
        x_input = torch.tensor(list_of_input_ids).long()

        ## for bert alone
        # y_pred = model(x_input)
        # list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()

        if self.model_name == 'bert_lstm_crf' or self.model_name == 'bert_gru_crf':
            ## for bert bilstm crf & bert bigru crf
            list_of_pred_ids = self.model(x_input, using_pack_sequence=False)
        else:
            ## for bert crf
            list_of_pred_ids = self.model(x_input)

        list_of_ner_word, decoding_ner_sentence = self.decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
        print("list_of_ner_word:", list_of_ner_word)
        #print("decoding_ner_sentence:", decoding_ner_sentence)
        return decoding_ner_sentence

class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        print(input_token)
        print(self.index_to_ner[6])

        _list = list_of_pred_ids[0].tolist()
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in _list[0]]

        print("len: {}, input_token:{}".format(len(input_token), input_token))
        print("len: {}, pred_ner_tag:{}".format(len(pred_ner_tag), pred_ner_tag))

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag, "prob":None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""

        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for token_str, pred_ner_tag_str in zip(input_token, pred_ner_tag):
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence

if __name__ == '__main__':
    model_dir = './experiments/base_model_with_crf_val'
    predictor = BertCRFNERPredictor(model_dir)

    model_name = 'bert_lstm_crf'
    tokenizer_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    predictor.load_model(model_name, tokenizer_path)

    text = '김대중 대통령은 노벨평화상을 받으러 스웨덴으로 출국해서 5박6일 동안 스웨덴에 머물며 대한민국의 위상을 높였다.'
    predictor.predict(text)
