from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf # works with tensorflow 1.6
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#------------------Functions------------------
def get_iris_df_tens():
    iris = load_iris()
    col_names = [x[:-5].replace(" ", "_") for x in iris.feature_names]
    iris_df = pd.DataFrame(iris.data, columns=col_names)
    iris_df = iris_df.join(pd.Series(iris.target, name='Type'))
    categories = dict(zip([0, 1, 2], iris.target_names))
    iris_df['Type_str'] = iris_df['Type'].apply(lambda x: categories[x])
    train, test = train_test_split(iris_df, test_size=0.3)
    return {"all":iris_df, "train": train, "test": test}, col_names
def get_tensor_features_and_labels(feture_names):
    my_feature_columns = []
    for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        my_feature_columns.append(tf.feature_column.numeric_column(key=col.replace(" ", "_")))
    labels = [tf.feature_column.categorical_column_with_vocabulary_list(key="Type",vocabulary_list=['setosa', 'versicolor', 'virginica'])]
    return my_feature_columns, labels
def input_fn(X, y, batch_size):  # An input function for training
    dataset = tf.data.Dataset.from_tensor_slices((dict(X),y)) # Convert the inputs to a Dataset.
    return dataset.shuffle(1000).repeat().batch(batch_size)
def eval_input_fn(X,y,batch_size):
    # Convert the inputs to a Dataset.
    X = dict(X)
    if y is None:
        inputs = X
    else:
        inputs = (X, y)

    dataset = tf.data.Dataset.from_tensor_slices(inputs) # Convert the inputs to a Dataset.
    assert batch_size is not None, "batch_size must not be None"
    return dataset.batch(batch_size)


#------------------Main------------------
iris, featue_names = get_iris_df_tens()
feat_names_tens, labels_tens = get_tensor_features_and_labels(featue_names)
batch_size = 5
classifier = tf.estimator.DNNClassifier (feature_columns=feat_names_tens,
                                         hidden_units=[10, 10],
                                         n_classes=3,
                                         model_dir='check_points/iris'
                                         )

classifier.train(input_fn=lambda: input_fn(iris["train"].ix[:, :4], iris["train"].Type, batch_size), steps= 10000)
print(classifier.model_dir)

eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(iris["test"].ix[:, :4], iris["test"].Type, batch_size))
print(eval_result)


# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'sepal_length': [5.1, 5.9, 6.9],
    'sepal_width': [3.3, 3.0, 3.1],
    'petal_length': [1.7, 4.2, 5.4],
    'petal_width': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(input_fn=lambda:eval_input_fn(predict_x, None, 1))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print("acc for {} is {:.3f}".format(expec, probability))

