import re
import os
import logging


class Txt2Seq:
    def __init__(self, data_dir, txt_file, lang=None):
        self.lang = lang
        self.sentences = []
        input_file = '{}{}'.format(data_dir, txt_file)
        output_file = '{}predict.txt'.format(data_dir)
        if os.path.exists(input_file):
            text = open(input_file, 'r', encoding='UTF-8').read()
            sentences = text.replace('\n', '')
            self.split_sentence(sentences)
            self.sen2seq(output_file)

    def split_sentence(self, sentences):
        split_pattern = r'([.。!！?？])'
        if self.lang == 'en':
            split_pattern = r'([\'|.?!])'
        elif self.lang == 'zn':
            split_pattern = r'([。？！])'
        sentences = re.split(split_pattern, sentences)
        self.sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        logging.info('sentences after split: {}'.format(self.sentences))

    def sen2seq(self, output):
        with open(output, 'w+', encoding='UTF-8') as f:
            for sent in self.sentences:
                for char in sent:
                    if len(char.strip(' ')) > 0:
                        f.write('{} {}\n'.format(char, 'O'))
                    else:
                        continue
                f.write('\n')
