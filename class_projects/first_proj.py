# Import NumPy
import numpy as np

# Step 1: Create a list of 5 numbers
numbers = [10, 25, 7, 18, 30]

# Step 2: Convert the list into a NumPy array
array = np.array(numbers)

# Step 3: Calculate mean, maximum, and sum
mean_value = np.mean(array)
max_value = np.max(array)
sum_value = np.sum(array)

# Step 4: Print the results clearly
print("Numbers:", array)
print("Mean:", mean_value)
print("Maximum Value:", max_value)
print("Sum:", sum_value)
