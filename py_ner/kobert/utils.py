# coding=utf-8
# Copyright 2019 SK T-Brain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import requests
import hashlib

kobert_models = {
    'onnx_kobert': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/onnx/onnx_kobert_44529811f0.onnx',
        'fname': 'onnx_kobert_44529811f0.onnx',
        'chksum': '44529811f0'
    },
    'tokenizer': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/tokenizer/tokenizer_78b3253a26.model',
        'fname': 'tokenizer_78b3253a26.model',
        'chksum': '78b3253a26'
    }
}


def download(url, filename, chksum, cachedir='./ptr_lm_model'):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path,
                            'rb').read()).hexdigest()[:10] == chksum:
            print('using cached model')
            return file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                   '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    assert chksum == hashlib.md5(open(
        file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
    return file_path


def get_onnx(cachedir='./ptr_lm_model'):
    """Get KoBERT ONNX file path after downloading
    """
    model_info = kobert_models['onnx_kobert']
    return download(model_info['url'],
                    model_info['fname'],
                    model_info['chksum'],
                    cachedir=cachedir)

def get_tokenizer(cachedir='./ptr_lm_model'):
    """Get KoBERT Tokenizer file path after downloading
    """
    model_info = kobert_models['tokenizer']
    return download(model_info['url'],
                        model_info['fname'],
                        model_info['chksum'],
                        cachedir=cachedir)    
