import sys
import json
from os import path
from gensim.corpora.wikicorpus import WikiCorpus


base_dir = path.join(path.dirname(path.realpath(__file__)), path.pardir)
wiki_filename = 'simplewiki-20171103-pages-articles-multistream.xml.bz2'
wiki_path = path.join(base_dir, 'corpora', wiki_filename)
outname = path.join(base_dir, 'corpora', 'simplewikiselect')

index = []  # Save information about articles as they've been processed.

wiki = WikiCorpus(wiki_path, dictionary=True)  # dict=True avoids making vocab
wiki.metadata = True  # Want article titles
print("Loading Wikipedia archive (this may take a few minutes)... ", end="")
articles = list(wiki.get_texts())
print("Done.")

num_articles = len(articles)

print("Total Number of Articles:", num_articles)

MAX_WC = 20_000_000
ARTICLE_MIN_WC = 200
ARTICLE_MAX_WC = 10000

ac = 0
wc = 0
selected=[]

with open(outname + ".txt", "w") as f:
    line = 0
    for i in range(num_articles):
        article, (id, name) = articles[i]
        art_len = len(article)
        if art_len >= ARTICLE_MIN_WC and art_len <= ARTICLE_MAX_WC:
            text = " ".join(article)
            wc += art_len
            ac += 1
            pos = f.tell()
            index.append({"id": id, "name": name, "wc": art_len, "line": line,
                          "byte": pos})
            f.write(text)
            f.write("\n")
            line += 1

        if wc >= MAX_WC:
            break

print("Selected", ac, "documents. (", wc, "words )")

metadata = {
    "source": wiki_filename,
    "document_min_wc": ARTICLE_MIN_WC,
    "document_max_wc": ARTICLE_MAX_WC,
    "num_documents": ac,
    "num_words": wc,
    "fields": list(index[0].keys()),
    "index": index}

with open(outname + ".meta.json", "w") as f:
    json.dump(metadata, f, indent=4)

with open(outname + ".meta.txt", "w") as f:
    del metadata["index"]
    for key, val in metadata.items():
        f.write(key)
        f.write(": ")
        f.write(str(val))
        f.write("\n")
