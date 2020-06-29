class WordCounter:
    IN_TYPE = [list, str]
    OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        from collections import Counter
        return list(Counter(args[0]).most_common())

