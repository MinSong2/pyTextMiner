
class POSFilter:
    IN_TYPE = [list, tuple]
    OUT_TYPE = [list, tuple]

    def __init__(self, *posWanted):
        import re
        self.wanted = set(p for p in posWanted if not p.endswith('*'))
        self.re = re.compile('(' + '|'.join(p[:-1] for p in posWanted if p.endswith('*')) + ').*')

    def test(self, pos):
        if pos in self.wanted: return True
        if self.re.match(pos): return True
        return False

    def __call__(self, *args, **kwargs):
        return [i for i in args[0] if self.test(i[1])]

class StopwordFilter:
    IN_TYPE = [list, str]
    OUT_TYPE = [list, str]

    def __init__(self, stopwords = [], file = None):
        if file:
            stopwords = stopwords + [line.strip() for line in open(file, encoding='utf-8')]
        self.stopwords = set(stopwords)
        self.stopwordsPrefix = ('http', 'https', 'ftp', 'git', 'thatt')

    def __call__(self, *args, **kwargs):
        #any(e for e in test_list if e.startswith('three') or e.endswith('four'))
        return [i for i in args[0] if i.lower() not in self.stopwords and (i.lower().startswith(tuple(p for p in self.stopwordsPrefix)) == False)]

class SelectWordOnly:
    IN_TYPE = [tuple]
    OUT_TYPE = [str]

    def __call__(self, *args, **kwargs):
        return args[0][0]

class ToLowerCase:
    IN_TYPE = [str]
    OUT_TYPE = [str]

    def __call__(self, *args, **kwargs):
        return args[0].lower()