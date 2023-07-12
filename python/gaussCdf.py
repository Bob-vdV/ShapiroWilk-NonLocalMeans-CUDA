from matplotlib import pyplot as plt
import numpy as np

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gaussCdf(z):
    """
        // input = z-value (-inf to +inf)
        // output = p under Normal curve from -inf to z
        // e.g., if z = 0.0, function returns 0.5000
        // ACM Algorithm #209
        double y; // 209 scratch variable
        double p; // result. called ‘z’ in 209
        double w; // 209 scratch variable
    """
    if (z == 0.0):
        p = 0.0
        
    else:
        y = abs(z) / 2.0
        if (y >= 3.0):
            p = 1.0
        elif (y < 1.0):
            w = y * y
            p = ((((((((0.000124818987 * w - 0.001075204047) * w + 0.005198775019) * w - 0.019198292004) * w + 0.059054035642) * w - 0.151968751364) * w + 0.319152932694) * w - 0.531923007300) * w + 0.797884560593) * y * 2.0
        
        else:
            y = y - 2.0
            p = (((((((((((((-0.000045255659 * y + 0.000152529290) * y - 0.000019538132) * y - 0.000676904986) * y + 0.001390604284) * y - 0.000794620820) * y - 0.002034254874) * y + 0.006549791214) * y - 0.010557625006) * y + 0.011630447319) * y - 0.009279453341) * y + 0.005353579108) * y - 0.002141268741) * y + 0.000535310849) * y + 0.999936657524
        

    if (z > 0.0):
    
        return (p + 1.0) / 2.0
    

    return (1.0 - p) / 2.0
    

def main():
    start = -3
    end = 3
    alpha = 0.05

    x = np.linspace(start, end, 10000000, dtype=np.longdouble)

    test = np.vectorize(gaussCdf)

    y = test(x)

    nearest = x[find_nearest_idx(y, alpha)]
    print(nearest)

    plt.title("Threshold value for Shapiro-Wilk test")
    plt.xlabel("w")
    plt.ylabel("t")

    plt.plot(x, y,)
    plt.hlines(alpha, start, end, colors="orange")
    plt.plot(nearest, alpha, marker='o', color="black")

    offset = 0.03
    plt.text(end, alpha + offset, "α = 0.05", color="C1", ha="right")

    plt.text(nearest, alpha + offset, f"({nearest:.2f}, {alpha})", color="black", ha="right")

    plt.savefig("pw.png")
    plt.savefig("pw.svg")
    plt.show()



if __name__ == "__main__":
    main()