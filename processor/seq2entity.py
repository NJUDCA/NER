import numpy as np
import logging

LOG_SETTINGS = {
    'format': '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'filemode': 'a',
}

class Seq2Entity:
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        self.length = len(tokens)

    def get_per_entity(self):
        PER = []
        for i, (token, label) in enumerate(zip(self.tokens, self.labels)):
            try:
                if label == 'B-PER':
                    if 'per' in locals().keys():
                        PER.append(per)
                        del per
                    per = token
                    if i+1 == self.length:
                        PER.append(per)
                if label == 'I-PER':
                    per += token
                    if i + 1 == self.length:
                        PER.append(per)
                if label not in ['I-PER', 'B-PER']:
                    if 'per' in locals().keys():
                        PER.append(per)
                        del per
                    continue
            except Exception as e:
                logging.info('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        return PER

    def get_loc_entity(self):
        LOC = []
        for i, (token, label) in enumerate(zip(self.tokens, self.labels)):
            try:
                if label == 'B-LOC':
                    if 'loc' in locals().keys():
                        LOC.append(loc)
                        del loc
                    loc = token
                    if i+1 == self.length:
                        LOC.append(loc)
                if label == 'I-LOC':
                    loc += token
                    if i + 1 == self.length:
                        LOC.append(loc)
                if label not in ['I-LOC', 'B-LOC']:
                    if 'loc' in locals().keys():
                        LOC.append(loc)
                        del loc
                    continue
            except Exception as e:
                logging.info('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        return LOC

    def get_org_entity(self):
        ORG = []
        for i, (token, label) in enumerate(zip(self.tokens, self.labels)):
            try:
                if label == 'B-ORG':
                    if 'org' in locals().keys():
                        ORG.append(org)
                        del org
                    org = token
                    if i+1 == self.length:
                        ORG.append(org)
                if label == 'I-ORG':
                    org += token
                    if i + 1 == self.length:
                        ORG.append(org)
                if label not in ['I-ORG', 'B-ORG']:
                    if 'org' in locals().keys():
                        ORG.append(org)
                        del org
                    continue
            except Exception as e:
                logging.info('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        return ORG


output_dir = '../output/ChinaDaily/BERT/'

logging.basicConfig(level=logging.INFO, **LOG_SETTINGS, filename=output_dir + 'seq2entity.log')
logging.info('output_dir: {}'.format(output_dir))

input_file = output_dir + 'label_test.txt'
output_per = output_dir + 'per.txt'
output_loc = output_dir + 'loc.txt'
output_org = output_dir + 'org.txt'

seq = np.loadtxt(input_file, dtype=str)
seq2entity = Seq2Entity(seq[:, 0], seq[:, 2])
per = seq2entity.get_per_entity()
loc = seq2entity.get_loc_entity()
org = seq2entity.get_org_entity()
logging.info('PER: {}'.format(len(per)))
logging.info('LOC: {}'.format(len(loc)))
logging.info('ORG: {}'.format(len(org)))
np.savetxt(output_per, np.array(per), fmt="%s")
np.savetxt(output_loc, np.array(loc), fmt="%s")
np.savetxt(output_org, np.array(org), fmt="%s")


