import numpy as np
import neurolab as nl

target =  [[1, 0, 0, 0, 1,
            1, 1, 0, 1, 1,
            1, 0, 1, 0, 1, #M
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1],

           [1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1, #В
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1],

           [1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1, # O
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 0]

           ]
chars = ['M', 'V', 'O']
target = np.asfarray(target)
target[target == 0] = -1
net = nl.net.newhop(target)
output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())
