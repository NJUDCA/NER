from sklearn.metrics import classification_report
from metrics import conlleval


# label_test_file = 'output/MSRA/crf/result.txt'
# eval_file = 'output/MSRA/crf/eval_crf.txt'

# label_test_file = 'output/ywevents/crf/result.txt'
# eval_file = 'output/ywevents/crf/eval_crf.txt'


label_test_file = 'output/ywevents/char2vec_blstm/results/test_label.txt'
eval_file = 'output/ywevents/char2vec_blstm/eval.txt'


def main():

    targets = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']
    y_test = []
    y_pred = []
    with open(label_test_file, 'r', encoding='UTF-8') as fr:

        line = fr.readline()
        while line:
            elements = line.strip().split('\t')
            if len(elements) == 3:
                y_test.append(elements[1])
                y_pred.append(elements[2])
            else:
                print(line)
            line = fr.readline()

    print('Test: {}\nPred: {}'.format(len(y_test), len(y_pred)))

    report = classification_report(y_test, y_pred, digits=4, target_names=targets)

    with open(eval_file, 'w+', encoding='UTF-8') as fw:
        fw.write('Classification report: \n')
        fw.write(report)


    test_report = conlleval.return_report(label_test_file)
    with open(eval_file, 'a+', encoding='UTF-8') as wf:
        wf.write(''.join(test_report))


if __name__ == '__main__':
    main()
