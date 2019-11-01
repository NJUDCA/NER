import numpy as np
import os
import time

class Txt2Sql:
    def __init__(self, node_file, link_file):
        self.type2category = {
            'EVENT': 1,
            'TIME': 2,
            'PER': 3,
            'LOC': 4,
            'ORG': 5,
            'BOOK': 6,
            'OTHER': 6,
        }
        self.node_file = node_file
        self.link_file = link_file
        self.node_list = []
        self.node_length = 0
        self.link_list = []
        self.link_length = 0
    
    def select_nodes_all(self):
        if os.path.exists(self.node_file):
            with open(self.node_file, encoding='UTF-8') as fr_node:
                self.node_list = np.loadtxt(fr_node, dtype=np.str, delimiter='\t')
            self.node_length = len(self.node_list)

    def select_links_all(self):
        if os.path.exists(self.link_file):
            with open(self.link_file, encoding='UTF-8') as fr_link:
                self.link_list = np.loadtxt(fr_link, dtype=np.str, delimiter='\t')
            self.link_length = len(self.link_list)

    def get_node_id(self, value, category):
        if self.node_length:
            try:
                for idx in range(self.node_length):
                    node = self.node_list[idx]
                    if node[1] == value and node[2] == str(category):
                        target = node[0]
                        return target
            except Exception as e:
                raise IndexError('get node id error: {}'.format(e))
        self.insert_node(value, category)
        return self.node_length
    
    def insert_node(self, value, category):
        id = self.node_length + 1
        try:
            with open(self.node_file, 'a+', encoding='UTF-8') as fw_node:
                fw_node.write('{}\t{}\t{}\n'.format(id, value, category))
            if self.node_length:
                self.node_list = np.concatenate((self.node_list, [[id, value, category]]), axis=0)
            else:
                self.node_list = np.array([[id, value, category]])
            self.node_length += 1
        except Exception as e:
            raise OSError('insert node error: {}'.format(e))
    
    def insert_link(self, source_id, target_id):
        for idx in range(self.link_length):
            link = self.link_list[idx]
            # opposition may cause different relationship
            if link[0] == source_id and link[1] == target_id:
                return 0
        try:
            with open(self.link_file, 'a+', encoding='UTF-8') as fw_link:
                fw_link.write('{}\t{}\n'.format(source_id, target_id))
            if self.link_length:
                self.link_list = np.concatenate((self.link_list, [[source_id, target_id]]), axis=0)
            else:
                self.link_list = np.array([[source_id, target_id]])
            self.link_length += 1
        except Exception as e:
            raise OSError('insert link error: {}'.format(e))

    def generate_nodes_links(self, root_input, leaf_input, root_type=None):
        start = time.time()
        print('Start generating ...')

        self.select_nodes_all()
        self.select_links_all()
        # root nodes first
        leaf_flag = 0
        for root in root_input:
            value = root[1]
            node_type = root_type
            category = self.type2category[node_type]
            source_id = self.get_node_id(value, category)
            # get related leaf nodes
            event_id = root[0]
            start_point = leaf_flag
            for leaf in leaf_input[start_point:]:
                if leaf[0] != event_id:
                    break
                value = leaf[-1]
                node_type = leaf[1]
                category = self.type2category[node_type]
                target_id = self.get_node_id(value, category)
                self.insert_link(source_id, target_id)
                leaf_flag += 1
        print('Finished in {:.2f} secs.'.format(time.time() - start))


# extract year from time
'''
time_file = '../data/ywsz/events/time.txt'
times = np.loadtxt(time_file, dtype=np.str, delimiter='\t', usecols=[0, 4, 5], encoding='UTF-8')
year_file = '../data/ywsz/events/time_year.txt'
time_year = []
for time_item in times:
    time_id = time_item[0]
    if len(time_item[1]) == 0:
        print(time_id)
        continue
    value = time_item[1]
    if len(time_item[2]) > 0:
        value += time_item[2]
    time_year.append([time_id, value])
with open(year_file, 'w+', encoding='UTF-8') as fw_year:
    np.savetxt(fw_year, np.array(time_year), fmt="%s", delimiter='\t')
'''

# class instance
year_file = '../data/ywsz/events/time_year.txt'
time_year = np.loadtxt(year_file, dtype=np.str, delimiter='\t', encoding='UTF-8')

entity_file = '../data/ywsz/events/entity_bert_crf.txt'
entities = np.loadtxt(entity_file, dtype=np.str, delimiter='\t', usecols=[0, 1, 2], encoding='UTF-8')

file_node = '../data/ywsz/events/nodes.txt'
file_link = '../data/ywsz/events/links.txt'

txt = Txt2Sql(file_node, file_link)
txt.generate_nodes_links(time_year, entities, root_type='TIME')