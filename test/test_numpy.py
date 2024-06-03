import numpy as np
import utils
from utils import InputExample, LogitsList


def main():
    logits = np.array([[[1, 3, 5, 7]], [[2, 4, 6, 8]], [[10, 12, 14, 16]]])
    logits = np.mean(logits, axis=0)
    print(f'logits: {logits}') # logits: [[ 4.33333333  6.33333333  8.33333333 10.33333333]]

    logits = utils.softmax(logits, axis=1).tolist()
    print(f'logits: {logits}') # logits: [[0.002144008783584632, 0.01584220117850691, 0.11705891323853293, 0.8649548767993754]]

    labels = ['1', '2', '3', '4', '5']
    logits = [6.534807506134894, 4.660593815113103, 6.546420306545415, 0.5433331549705217, 0.4566661856615058]
    max_prob = np.argmax(logits)
    print(f'max_prob: {max_prob}')
    logits = labels[np.argmax(logits).item()]
    print(f'logits: {logits}') 



if __name__ == "__main__":
    main()