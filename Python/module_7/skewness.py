import numpy as np
import scipy.stats as stats

# Data: Numbers from 1 to 10
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 1. Mean
mean_value = np.mean(data)

# 2. Median
median_value = np.median(data)

# 3. Mode
mode_value = stats.mode(data)[0][0]  # mode() returns a tuple, so we take the first element

# 4. Skewness
skewness_value = stats.skew(data)

# Print the results
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")
print(f"Skewness: {skewness_value}")
