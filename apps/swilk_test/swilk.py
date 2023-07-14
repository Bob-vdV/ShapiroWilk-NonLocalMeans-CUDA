import math
import scipy

# Example data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

t3 = [-1.7, -1, -1, -.73, -.61, -.5, -.24, .45, .62, .81, 1]

print(scipy.stats.shapiro(data))

print(scipy.stats.shapiro(t3))