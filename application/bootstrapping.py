import numpy as np
from txt2seq import cut_max_seq


def bootstrapping():
    event_file = '../data/ywsz/events/events.txt'

    with open(event_file, encoding='UTF-8') as fr_event:
        events = np.loadtxt(fr_event, dtype=np.str)
    total_events = events.shape[0]
    print('total events: ', total_events)
    ids = np.arange(1, total_events+1)
    test_size = 456

    # _, events_test, _, ids_test = train_test_split(events, ids, test_size=test_size, shuffle=False)
    events_test = events[: test_size]
    ids_test = ids[: test_size]
    print('{} test events, {} test ids'.format(len(events_test), len(ids_test)))
    reduce(events_test, ids_test, 'test')
    # reduce(events, ids, 'train')


def reduce(events, ids, mode='test'):
    entity_file = '../data/ywsz/events/entity_bert_crf.txt'
    with open(entity_file, encoding='UTF-8') as fr_entity:
        entity = np.loadtxt(fr_entity, dtype=np.str)
        event_ids = entity[:, 0]

    fw_output = '../data/ywsz/events/bootstraping_{}.txt'.format(mode)
    fr_entity = open(entity_file, 'r', encoding='UTF-8')
    with open(fw_output, 'w+', encoding='UTF-8') as fw:
        for i, event in zip(ids, events):
            event_id = str('{:05d}').format(i)
            event = event.strip()
            labels = ['O'] * len(event)

            idx = [index for index in range(len(entity)) if event_ids[index] == event_id]
            for id in idx:
                elements = entity[id]
                if mode == 'test' and elements[1] == 'BOOK':
                    continue
                item_type = elements[1]
                item_value = elements[2]
                start = int(elements[3])
                end = int(elements[4])
                if item_value == event[start: end]:
                    labels[start] = 'B-{}'.format(item_type)
                    for flag in range(start+1, end):
                        labels[flag] = 'I-{}'.format(item_type)
                else:
                    raise ValueError('cannot matching entity {} to event {}: {}'.format(
                        item_value, i, ''.join(event[start: end]))
                    )
            seqs = cut_max_seq()
            seqs.cut(event, labels)
            token_seq, label_seq = seqs.get_cut_seqs()
            for i, tokens in enumerate(token_seq):
                labels = label_seq[i]
                for token, label in zip(tokens, labels):
                    fw.write('{} {}\n'.format(token, label))
                fw.write('\n')

    fr_entity.close()


if __name__ == "__main__":
    bootstrapping()
