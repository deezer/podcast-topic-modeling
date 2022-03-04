def read_top_words_file(fname, encoding='utf-8'):
    topic_words = []
    with open(fname, 'r', encoding=encoding) as topic_file:
        for line in topic_file:
            words = line.rstrip().split()
            topic_words.append(words)
    return topic_words