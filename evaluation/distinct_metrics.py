def distinct_ngrams(n, replies):
    ngrams = set()

    for reply in replies:
        tokens = reply.split()

        for i in range(len(tokens)-n+1):
            sub_string = ' '.join(tokens[i:i+n])
            ngrams.add(sub_string)

    return len(ngrams)

def distinct_sentences(replies):
    sentences = set(replies)

    return len(sentences)

def token_count(replies):
    total_len = 0

    for reply in replies:
        tokens = reply.split()
        total_len += len(tokens)

    return total_len