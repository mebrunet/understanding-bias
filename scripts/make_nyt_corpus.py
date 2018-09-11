import re
import os
import logging
import sys
import json
import tarfile
import numpy as np
from xml.etree import ElementTree
from gensim.utils import tokenize

ARTICLE_MIN_WC = 100
ARTICLE_MAX_WC = 30_000

# cd ..


def get_text_content(root):
    path = './body/body.content/block[@class="full_text"]'
    full_text = root.find(path)
    if (full_text is None):
        return (None, 0)
    text = ''.join(full_text.itertext()).strip()
    # Ditch lead paragraph
    lines = text.split("\n")
    if lines[0].startswith('LEAD:'):
        lines = lines[1:-1]
    text = ' '.join(lines)
    # with open("corpora/scratch.txt", "a") as f:
    #     f.write(text)
    #     f.write("\n---\n")
    tokens = [t.lower() for t in tokenize(text)]
    wc = len(tokens)
    return (' '.join(tokens), wc)


def get_attr(root, path, extract='content'):
    try:
        result = root.find(path).get(extract)
    except Exception as e:
        result = None

    return result


def get_text(root, path):
    try:
        node = root.find(path)
        result = ' '.join(node.itertext()).strip()
    except Exception as e:
        result = None
    return result


def get_metadata(root):
    metadata = {}
    publication_date = './head/pubdata'
    news_desk = './head/meta[@name="dsk"]'
    byline = './body/body.head/byline[@class="print_byline"]'
    normalized_byline = './body/body.head/byline[@class="normalized_byline"]'
    headline = './body[1]/body.head/hedline/hl1'
    guid = './head/docdata/doc-id'
    metadata['headline'] = get_text(root, headline)
    metadata['date'] = get_attr(root, publication_date, 'date.publication')
    metadata['news_desk'] = get_attr(root, news_desk)
    # metadata['byline'] = get_text(root, byline)
    metadata['nbyline'] = get_text(root, normalized_byline)
    metadata['guid'] = get_attr(root, guid, 'id-string')
    return metadata


def process_contents(base_dir, outfile, article_meta, limit=None, verbose=False,
                     progress=True):
    ac = 0  # Total article count
    twc = 0  # Total word count
    i = 1
    for content in os.listdir(base_dir):
        if content.startswith('.'):
            continue
        folder = os.path.join(base_dir, content)
        # print(content)
        for subcontent in os.listdir(folder):
            if subcontent.startswith('.'):
                continue
            archive_path = os.path.join(folder, subcontent)
            # print('-', archive_path)
            archive = tarfile.open(archive_path, mode='r')
            for member in archive.getmembers():
                f = archive.extractfile(member)  # f can be None
                if f is None:
                    continue
                # print(member)
                content_string = f.read()
                # print(content_string)
                root = ElementTree.fromstring(content_string)
                (text, wc) = get_text_content(root)
                if (wc < ARTICLE_MIN_WC or wc > ARTICLE_MAX_WC):
                    continue

                # Now we have an article!
                metadata = get_metadata(root)
                metadata["wc"] = wc  # Append word count
                metadata["line"] = ac  # The line in the outfile
                metadata["byte"] = outfile.tell()  # The position of the start of the line
                ac += 1
                twc += wc
                article_meta.append(metadata)
                outfile.write(text)
                outfile.write("\n")
                if (verbose):
                    print('Finished File:', i)
                    print(metadata)
                    print(text)
                    print()
                if (progress and i % 1000 == 0):
                    print('Completed', i, 'files')
                i += 1
                if (limit is not None and i > limit):
                    return (ac, twc)
    return (ac, twc)


# List archives
article_meta = []  # Save information about the articles
outname = 'corpora/nytselect'

with open(outname + ".txt", mode='w') as f:
    base_dir = 'corpora/nyt_corpus/data/'
    (ac, twc) = process_contents(base_dir, f, article_meta, limit=None)

metadata = {
    "source": "NYT Corpus",
    "document_min_wc": ARTICLE_MIN_WC,
    "document_max_wc": ARTICLE_MAX_WC,
    "num_documents": ac,
    "num_words": twc,
    "fields": list(article_meta[0].keys()),
    "index": article_meta}

with open(outname + ".meta.json", "w") as f:
    json.dump(metadata, f, indent=4)

with open(outname + ".meta.txt", "w") as f:
    del metadata["index"]
    for key, val in metadata.items():
        f.write(key)
        f.write(": ")
        f.write(str(val))
        f.write("\n")
