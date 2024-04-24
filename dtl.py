import pandas as pd
import math
from collections import Counter, defaultdict

df = {
    'Alt': ['T', 'T', 'F', 'T', 'T', 'F', 'F', 'F', 'F','T', 'F','T'],
    'Bar': ['F', 'F', 'T', 'F', 'F', 'T', 'T', 'F', 'T','T', 'F','T'],
    'Fry': ['F', 'F', 'F', 'T', 'T', 'F', 'F', 'F', 'T','T', 'F','T'],
    'Hun': ['T', 'T', 'F', 'T', 'F', 'T', 'F', 'T', 'F','T', 'F','T'],
    
    'Pat': ['Some', 'Full', 'Some', 'Full', 'Full', 'Some', 'None', 'Some', 'Full', 'Full', 'None','Full'],
    'Price': ['$$$', '$', '$', '$', '$$$', '$$', '$', '$$', '$','$$$', '$','$'],
    
    'Rain': ['F', 'F', 'F', 'F', 'F', 'T', 'T', 'T', 'T','F', 'F','F'],
    'Res': ['T', 'F', 'F', 'F', 'T', 'T', 'F', 'T', 'F','T', 'F','F'],
    'Type': ['French', 'Thai', 'Burger', 'Thai', 'French', 'Italian','Burger', 'Thai','Burger','Italian','Thai','Burger'],
    'Est': ['0-10', '30-60', '0-10', '10-30', '>60', '0-10','0-10', '0-10','>60','10-30','0-10','30-60'],
    'Wait': ['T', 'F', 'T', 'T', 'F', 'T', 'F', 'T', 'F', 'F', 'F', 'T']
}

df = pd.DataFrame(df)

# Convert the DataFrame to a list of tuples, each containing a dict of features and the label.
data_instances = list(df.apply(lambda row: ({col:row[col] for col in df.columns if col != 'Wait'}, row['Wait']), axis=1))

def calculate_entropy(class_probabilities):
    """Calculate entropy based on class probabilities."""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

def compute_class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def compute_data_entropy(data):
    labels = [label for _, label in data]
    probabilities = compute_class_probabilities(labels)
    return calculate_entropy(probabilities)

def compute_partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(compute_data_entropy(subset) * len(subset) / total_count for subset in subsets)

def partition_data(inputs, attribute):
    groups = defaultdict(list)
    for data_instance in inputs:
        key = data_instance[0][attribute]
        groups[key].append(data_instance)
    return groups

def calculate_information_gain(inputs, attribute):
    partitions = partition_data(inputs, attribute)
    return compute_data_entropy(inputs) - compute_partition_entropy(partitions.values())

class TreeNode:
    def __init__(self, attribute=None, attribute_value=None, is_leaf=False, prediction=None):
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.children = {}

    def add_child(self, attribute_value, node):
        self.children[attribute_value] = node

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def train(self, data):
        self.root = self._grow_tree(data, depth=0)

    def _grow_tree(self, data, depth):
        labels = [label for _, label in data]
        most_common_label = Counter(labels).most_common(1)[0][0]

        if len(set(labels)) == 1 or (self.max_depth and depth == self.max_depth):
            return TreeNode(is_leaf=True, prediction=most_common_label)

        gains = [
            (calculate_information_gain(data, attribute), attribute)
            for attribute, _ in data[0][0].items()
        ]

        max_gain, best_attribute = max(gains, key=lambda x: x[0])

        if max_gain == 0:
            return TreeNode(is_leaf=True, prediction=most_common_label)

        partitions = partition_data(data, best_attribute)
        new_node = TreeNode(attribute=best_attribute, attribute_value=most_common_label)
        for attribute_value, subset in partitions.items():
            child = self._grow_tree(subset, depth + 1)
            new_node.add_child(attribute_value, child)

        return new_node

    def predict(self, data_instance):
        node = self.root
        while not node.is_leaf:
            node = node.children.get(data_instance[node.attribute], node)
        return node.prediction

# Instantiate the decision tree classifier
tree_classifier = DecisionTree(max_depth=3)

# Train the classifier using the data instances
tree_classifier.train(data_instances)

# Predict using the trained classifier
predictions = [tree_classifier.predict(x[0]) for x in data_instances]

# Show the predictions
predictions
