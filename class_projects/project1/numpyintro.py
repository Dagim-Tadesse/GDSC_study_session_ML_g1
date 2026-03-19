
import numpy as np

numbers = [10, 25, 7, 18, 30]

array = np.array(numbers)

mean_value = np.mean(array)
max_value = np.max(array)
sum_value = np.sum(array)

print("Numbers:", array)
print("Mean:", mean_value)
print("Maximum Value:", max_value)
print("Sum:", sum_value)
