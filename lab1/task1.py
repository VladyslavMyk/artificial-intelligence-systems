import numpy as np
from sklearn import preprocessing
Input_labels = ['red', 'blасk', 'red', 'green', 'Ьlack', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(Input_labels)
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)
test_labels = ['green', 'red', 'blасk']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))