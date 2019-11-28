import numpy as np
import logging
import re


class Seq2Entity:
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        if len(tokens) != len(labels):
            raise IndexError('length of tokens {} and labels {} are not equal'.format(len(tokens), len(labels)))
        self.length = len(tokens)
        self.sent = ''.join(tokens)
    
    def get_entity(self):
        per = self.get_per_entity()
        loc = self.get_loc_entity()
        org = self.get_org_entity()
        book = self.get_book_entity()
        return {
            'PER': per,
            'LOC': loc,
            'ORG': org,
            'BOOK': book,
        }

    def get_per_entity(self):
        PER = []
        for i, (token, label) in enumerate(zip(self.tokens, self.labels)):
            try:
                if label == 'B-PER':
                    if 'per' in locals().keys():
                        end = i
                        PER.append((per, start, end))
                        del per
                    per = token
                    start = i
                    if i+1 == self.length:
                        end = i + 1
                        PER.append((per, start, end))
                if label == 'I-PER':
                    per += token
                    if i + 1 == self.length:
                        end = i + 1
                        PER.append((per, start, end))
                if label not in ['I-PER', 'B-PER']:
                    if 'per' in locals().keys():
                        end = i
                        PER.append((per, start, end))
                        del per
                    continue
            except Exception as e:
                logging.error('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        if len(PER) > 0:
            logging.info('{} PER: {}'.format(len(PER), ' '.join([str(p[0]) for p in PER])))
        return PER

    def get_loc_entity(self):
        LOC = []
        for i, (token, label) in enumerate(zip(self.tokens, self.labels)):
            try:
                if label == 'B-LOC':
                    if 'loc' in locals().keys():
                        end = i
                        LOC.append((loc, start, end))
                        del loc
                    loc = token
                    start = i
                    if i+1 == self.length:
                        end = i + 1
                        LOC.append((loc, start, end))
                if label == 'I-LOC':
                    loc += token
                    if i + 1 == self.length:
                        end = i + 1
                        LOC.append((loc, start, end))
                if label not in ['I-LOC', 'B-LOC']:
                    if 'loc' in locals().keys():
                        end = i
                        LOC.append((loc, start, end))
                        del loc
                    continue
            except Exception as e:
                logging.error('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        if len(LOC) > 0:
            logging.info('{} LOC: {}'.format(len(LOC), ' '.join([str(l[0]) for l in LOC])))
        return LOC

    def get_org_entity(self):
        ORG = []
        for i, (token, label) in enumerate(zip(self.tokens, self.labels)):
            try:
                if label == 'B-ORG':
                    if 'org' in locals().keys():
                        end = i
                        ORG.append((org, start, end))
                        del org
                    org = token
                    start = i
                    if i+1 == self.length:
                        end = i + 1
                        ORG.append((org, start, end))
                if label == 'I-ORG':
                    org += token
                    if i + 1 == self.length:
                        end = i + 1
                        ORG.append((org, start, end))
                if label not in ['I-ORG', 'B-ORG']:
                    if 'org' in locals().keys():
                        end = i
                        ORG.append((org, start, end))
                        del org
                    continue
            except Exception as e:
                logging.error('i: {}, token: {}, label: {}, error: {}'.format(i, token, label, str(e)))
        if len(ORG) > 0:
            logging.info('{} ORG: {}'.format(len(ORG), ' '.join([str(o[0]) for o in ORG])))
        return ORG
    
    def get_book_entity(self):
        BOOK =[]
        bookmark_pattern = r"《([^》]*)》"
        matches = re.finditer(bookmark_pattern, self.sent, re.MULTILINE)
        for match in matches:
            item = match.group()
            start = match.start()
            end = match.end()
            if self.tokens[start] in ['I-PER', 'I-LOC', 'I-ORG']:
                logging.info('bookmark {} embedded in entity'.format(item))
                continue
            BOOK.append((item, start, end))
        if len(BOOK) > 0:
            logging.info('{} BOOK: {}'.format(len(BOOK), ' '.join([str(b[0]) for b in BOOK])))
        return BOOK

