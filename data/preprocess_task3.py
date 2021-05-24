import numpy as np
import os
import csv

examples = []
tokens = []
relations = []
bio_tags = []
from_seqs = []
to_seqs = []
root = {}
dir_name = "subtask3/test"
for file in os.listdir(dir_name):
    root = {}
    with open(os.path.join(dir_name, file), 'r') as f:
        words = []
        lines = f.readlines()
        for i in range(len(lines) -1):
            if lines[i] == '\n' or lines[i] == '':  # empty line:
                if lines[i + 1] == '\n' or lines[i + 1] == '':  # two empty line, end of a
                    # examples.append((" ".join(tokens), " ".join(bio_tags), " ".join(relations),
                    #                  " ".join(from_seqs), " ".join(to_seqs)))
                    assert len(tokens) == len(bio_tags) == len(relations) == len(from_seqs) == len(to_seqs)
                    # find the root
                    for rt_tag, rf_tag in zip(to_seqs, from_seqs):
                        # if rf_tag == '0':  # root
                        if 'T' in rt_tag:
                            root[rt_tag] = (to_seqs.index(rt_tag), len(to_seqs) - to_seqs[::-1].index(rt_tag))
                    # print(root)
                    # find any definition that corresponding to the root
                    rels = []
                    unique_ft_tag = list(set(to_seqs))
                    root['0'] = (0, 0)
                    for to_tag in unique_ft_tag:
                        if 'T' in to_tag:
                            # print(to_tag)
                            head_span = (to_seqs.index(to_tag), len(to_seqs) - to_seqs[::-1].index(to_tag))
                            try:
                                root_span = root[from_seqs[head_span[0]]]  #
                            except KeyError:
                                continue
                            if root_span[0] == 0 and root_span[1] == 0:
                                continue
                            # print(head_span, root_span)
                            if root_span[0] < head_span[0] < root_span[1] or head_span[0] < root_span[0] < head_span[1]:
                                continue
                            # print(tokens[head_span[0]:head_span[1]], tokens[root_span[0]:root_span[1]])
                            tokens_insert = [t for t in tokens]
                            if head_span[0] > root_span[0]:
                                tokens_insert.insert(root_span[0], '[ER]')
                                tokens_insert.insert(root_span[1] + 1, '[ER]')
                                tokens_insert.insert(head_span[0] + 2, '[EH]')
                                tokens_insert.insert(head_span[1] + 3, '[EH]')
                            else:
                                tokens_insert.insert(head_span[0], '[EH]')
                                tokens_insert.insert(head_span[1] + 1, '[EH]')
                                tokens_insert.insert(root_span[0] + 2, '[ER]')
                                tokens_insert.insert(root_span[1] + 3, '[ER]')

                            input_sent = " ".join(tokens_insert)
                            relation_label = relations[head_span[0]]
                            if relation_label == 0 or relation_label == "0":
                                continue
                            examples.append([input_sent, relation_label])

                    tokens.clear()
                    bio_tags.clear()
                    relations.clear()
                    from_seqs.clear()
                    to_seqs.clear()
                    i += 1
                else:  # single empty line, separator between each sentence
                    tokens.append(" ")
                    relations.append("0")
                    to_seqs.append("-1")
                    from_seqs.append("-1")
                    bio_tags.append("O")
            else:
                line = lines[i]
                splits = line.split()
                token = splits[0].strip()
                relation = splits[-1].strip()
                relation_from = splits[-2].strip()
                relation_to = splits[-3].strip()
                bio_tag = splits[-4].strip()
                from_seqs.append(relation_from)
                to_seqs.append(relation_to)
                tokens.append(token)
                relations.append(relation)
                bio_tags.append(bio_tag)
print(len(examples))

with open('subtask3/test.tsv', 'w') as fw:
    for e in examples:
        fw.write("\t".join(e).strip() + '\n')

