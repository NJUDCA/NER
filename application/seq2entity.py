import numpy as np
import logging


class Seq2Entity:
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        self.length = len(tokens)
        logging.info('tokens: {}'.format(''.join(tokens)))
    
    def get_entity(self):
        per = self.get_per_entity()
        loc = self.get_loc_entity()
        org = self.get_org_entity()
        return {
            'PER': per,
            'LOC': loc,
            'ORG': org
        }

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
                logging.error('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        if len(PER) > 0:
            logging.info('{} PER: {}'.format(len(PER), ' '.join([str(p) for p in PER])))
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
                logging.error('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        if len(LOC) > 0:
            logging.info('{} LOC: {}'.format(len(LOC), ' '.join([str(l) for l in LOC])))
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
                logging.error('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        if len(ORG) > 0:
            logging.info('{} ORG: {}'.format(len(ORG), ' '.join([str(o) for o in ORG])))
        return ORG
