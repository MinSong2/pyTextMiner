from pprint import pprint
from pycrfsuite_spacing import TemplateGenerator
from pycrfsuite_spacing import CharacterFeatureTransformer
from pycrfsuite_spacing import sent_to_chartags
from pycrfsuite_spacing import sent_to_xy
from pycrfsuite_spacing import PyCRFSuiteSpacing

with open('../../data/134963_norm.txt', encoding='utf-8') as f:
    docs = [doc.strip() for doc in f]

print('n docs = %d' % len(docs))
pprint(docs[:5])

to_feature = CharacterFeatureTransformer(
    TemplateGenerator(begin=-2,
    end=2,
    min_range_length=3,
    max_range_length=3)
)

x, y = sent_to_xy('이것도 너프해 보시지', to_feature)
pprint(x)
print(y)

correct = PyCRFSuiteSpacing(
    to_feature = to_feature,
    feature_minfreq=3, # default = 0
    max_iterations=100,
    l1_cost=1.0,
    l2_cost=1.0
)
correct.train(docs, '../../model/korean_segmentation_model.crfsuite')

