import math
import scipy

# WRONG CODE GIVEN BY CHATGPT
def shapiro_wilk_test(data):
    n = len(data)
    sorted_data = sorted(data)
    mean = sum(sorted_data) / n
    deviations = [x - mean for x in sorted_data]
    squared_deviations = [x ** 2 for x in deviations]
    variance = sum(squared_deviations) / (n - 1)
    std_dev = math.sqrt(variance)

    a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    m = n // 2
    for i in range(m):
        a[i] = sorted_data[i]
        a[n - i - 1] = sorted_data[n - i - 1]

    for i in range(n):
        w[i] = (a[i] - mean) / std_dev

    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    b[0] = 1 / math.sqrt(n)
    b[m] = -b[0]

    y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(n):
        y[i] = w[i] * b[i]

    s = sum(y)

    u = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(n):
        u[i] = y[i] - s / n

    numerator = sum([u[i] ** 2 for i in range(n)])
    denominator = sum([deviations[i] ** 2 for i in range(n)])

    w_statistic = numerator / denominator

    return w_statistic

# Example data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Perform Shapiro-Wilk test
w_statistic = shapiro_wilk_test(data)

# Print the test statistic
print("Test Statistic:", w_statistic)

print(scipy.stats.shapiro(data))