"""Module for postprocessing and displaying transformer attentions.

"""

import json
from py_ner.bertviz.attention import get_attention

import os

def show(model, model_type, tokenizer, sentence_a, sentence_b=None):

    if sentence_b:
        vis_html = """
          <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="all">All</option>
              <option value="aa">Sentence A -> Sentence A</option>
              <option value="ab">Sentence A -> Sentence B</option>
              <option value="ba">Sentence B -> Sentence A</option>
              <option value="bb">Sentence B -> Sentence B</option>
            </select>
          </span>
          <div id='vis'></div> 
        """
    else:
        vis_html = """
          <span style="user-select:none">
            Layer: <select id="layer"></select>
          </span>
          <div id='vis'></div> 
        """

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()
    attn_data = get_attention(model, model_type, tokenizer, sentence_a, sentence_b)
    params = {
        'attention': attn_data,
        'default_filter': "all"
    }

    with open('bert_visualization.html', 'w') as f:
        _head = "<head>" + "<script src = \"https://d3js.org/d3.v4.min.js\" > </script>" + "</head>"
        f.write(_head)
        f.write(vis_html + '\n')

        f.write('window.params = %s' % json.dumps(params) + '\n')
        f.write(vis_js + '\n')