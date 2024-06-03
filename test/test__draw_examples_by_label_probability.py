import numpy as np
import utils
from utils import InputExample, LogitsList





from typing import List

class InputExample:
    def __init__(self, guid, text_a, label=None, logits=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.logits = logits



def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples]
    print(f'label_probabilities: {label_probabilities }') # label_probabilities: [0.4, 0.65, 0.8, 0.9]
    print(f'len(label_probabilities): {len(label_probabilities)}') # len(label_probabilities): 4
    sum_label_probabilities = sum(label_probabilities)
    print(f'sum_label_probabilities: {sum_label_probabilities}') # sum_label_probabilities: 2.75
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    print(f'label_probabilities: {label_probabilities}') # label_probabilities: [0.14545454545454548, 0.23636363636363636, 0.29090909090909095, 0.32727272727272727]
    return np.random.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()




def main():
    # Example input data
    examples = [
        InputExample(guid='1', text_a='Example 1', logits=[0.3, 0.4, 0.25, 0.05]),
        InputExample(guid='2', text_a='Example 2', logits=[0.2, 0.65, 0.07, 0.08]),
        InputExample(guid='3', text_a='Example 3', logits=[0.8, 0.05, 0.05, 0.1]),
        InputExample(guid='4', text_a='Example 4', logits=[0.01, 0.03, 0.9, 0.06])
    ]

    num_examples = 2  # Number of examples to draw  

    # Running the function with our example data
    selected_examples = _draw_examples_by_label_probability(examples, num_examples)

    # Printing the selected examples
    for example in selected_examples:
        print(f'GUID: {example.guid}, Text: {example.text_a}, Logits: {example.logits}')
        """
            GUID: 1, Text: Example 1, Logits: [0.3, 0.4, 0.25, 0.05]
            GUID: 4, Text: Example 4, Logits: [0.01, 0.03, 0.9, 0.06]
        """



if __name__ == "__main__":
    main()