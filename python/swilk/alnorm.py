from math import exp, log, asin, sqrt
from matplotlib import pyplot as plt
import numpy as np


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def alnorm(x, upper=True):
    ltone = 7.0
    #  utzero = 18.66
    utzero = 38.0
    con = 1.28

    A1 = 0.398942280444
    A2 = 0.399903438504
    A3 = 5.75885480458
    A4 = 29.8213557808
    A5 = 2.62433121679
    A6 = 48.6959930692
    A7 = 5.92885724438
    B1 = 0.398942280385
    B2 = 3.8052e-8
    B3 = 1.00000615302
    B4 = 3.98064794e-4
    B5 = 1.98615381364
    B6 = 0.151679116635
    B7 = 5.29330324926
    B8 = 4.8385912808
    B9 = 15.1508972451
    B10 = 0.742380924027
    B11 = 30.789933034
    B12 = 3.99019417011
    z = x
    if not (z > 0):
        # negative of the condition to catch NaNs
        upper = False
        z = -z

    if not ((z <= ltone) or (upper and z <= utzero)):
        if upper:
            return 0
        else:
            return 1

    y = 0.5 * z * z
    if z <= con:
        temp = 0.5 - z * (A1 - A2 * y / (y + A3 - A4 / (y + A5 + A6 / (y + A7))))

    else:
        temp = (
            B1
            * exp(-y)
            / (
                z
                - B2
                + B3
                / (
                    z
                    + B4
                    + B5 / (z - B6 + B7 / (z + B8 - B9 / (z + B10 + B11 / (z + B12))))
                )
            )
        )

    if upper:
        return temp

    else:
        return 1 - temp


def poly(cc, nord, x):
    # /* Algorithm AS 181.2	Appl. Statist.	(1982) Vol. 31, No. 2
    # Calculates the algebraic polynomial of order nord-1 with array of coefficients cc.
    # Zero order coefficient is cc(1) = cc[0] */

    ret_val = cc[0]
    if nord > 1:
        p = x * cc[nord - 1]
        for j in range(nord - 2, 0, -1):
            p = (p + cc[j]) * x
        ret_val += p

    return ret_val


def calcpw(w, n):
    g = [-2.273, 0.459]
    c3 = [0.544, -0.39978, 0.025054, -6.714e-4]
    c4 = [1.3822, -0.77857, 0.062767, -0.0020322]
    c5 = [-1.5861, -0.31082, -0.083751, 0.0038915]
    c6 = [-0.4803, -0.082676, 0.0030302]

    w1 = 1 - w
    # /*	Calculate significance level for W */
    m = 0
    s = 0
    if n == 3:
        # /* exact P value : */
        pi6 = 1.90985931710274  # /* = 6/pi */
        stqr = 1.04719755119660  # /* = asin(sqrt(3/4)) */
        pw = pi6 * (asin(sqrt(w)) - stqr)
        if pw < 0.0:
            pw = 0
        return pw

    y = log(w1)
    logn = log(n)
    if n <= 11:
        gamma = poly(g, 2, n)
        if y >= gamma:
            pw = 1e-99  # /* an "obvious" value, was 'small' which was 1e-19f */
            return pw

        y = -log(gamma - y)
        m = poly(c3, 4, n)
        s = exp(poly(c4, 4, n))

    else:
        # /* n >= 12 */
        m = poly(c5, 4, logn)
        s = exp(poly(c6, 3, logn))

    pw = alnorm((y - m) / s, True)
    return pw


# Find the minimum of a strictly increasing function
def findMin(start, end, alpha, func):
    i = 0

    precision = 1e-15

    while abs(start - end) > precision:
        searchRange = end - start

        first = func(start)
        last = func(end)

        #print(f"i: {i} start: {start} end: {end}")
        i += 1

        if abs(first - alpha) < abs(last - alpha):
            # Minimum is left of start
            if (first - alpha) > 0:
                start = start - searchRange
                end = end - searchRange
                continue

            # Minimum is right of start
            else:
                end = end - 0.5 * searchRange
                continue
        else:
            # Minimum is right of end
            if last - alpha < 0:
                start = start + searchRange
                end = end + searchRange

            # Minimum is left of end
            else:
                start = start + 0.5 * searchRange
                continue

    return start


def main():
    start = 0
    end = 1 - 1e-16
    alpha = 0.05
    n = 9

    def wrapper(x):
        return calcpw(x, n)

    x = np.linspace(start, end, 10000000, dtype=np.longdouble)

    test = np.vectorize(wrapper)

    y = test(x)

    nearest = x[find_nearest_idx(y, alpha)]
    print(nearest)

    plt.title(f"Threshold value for Shapiro-Wilk test (n={n})")
    plt.xlabel("w")
    plt.ylabel("pw")

    plt.plot(
        x,
        y,
    )
    plt.hlines(alpha, start, end, colors="orange")
    plt.plot(nearest, alpha, marker="o", color="black")

    offset = 0.03
    plt.text(start, alpha + offset, "α = 0.05", color="C1", ha="left")

    plt.text(
        nearest, alpha + offset, f"({nearest:.2f}, {alpha})", color="black", ha="right"
    )

    plt.savefig("pw.png")
    plt.savefig("pw.svg", bbox_inches='tight',pad_inches = 0)
    plt.show()

    print(findMin(start, end, alpha, wrapper))


if __name__ == "__main__":
    main()
