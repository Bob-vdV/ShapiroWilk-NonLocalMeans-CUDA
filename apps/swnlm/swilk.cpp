/**
 * Shapiro-Wilk test for normality.
 *
 * The following code is ported from Javascript to PHP to JAVA to C++.
 * Original script: https://github.com/rniwa/js-shapiro-wilk/blob/master/shapiro-wilk.js
 * Java code:       https://github.com/elcronos/shapiro-wilk
 *
 */

#include "swilk.hpp"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <vector>

using namespace std;

/**
 * Used internally by shapiro wilk algorithm
 *
 * Compute the quantile function for the normal distribution. For small to moderate probabilities, algorithm referenced
 * below is used to obtain an initial approximation which is polished with a final Newton step. For very large arguments, an algorithm of Wichura is used.
 * Used by ShapiroWilk Test
 * Ported java function found at
 * Ported by Javascript implementation found at https://raw.github.com/rniwa/js-shapiro-wilk/master/shapiro-wilk.js
 * Originally ported from http://svn.r-project.org/R/trunk/src/nmath/qnorm.c
 *
 * @param p
 * @param mu
 * @param sigma
 * @return
 */
double normalQuantile(double p, double mu, double sigma)
{
    // The inverse of cdf.

    assert(sigma >= 0);

    if (sigma == 0)
    {
        return mu;
    }

    double r;
    double val;

    double q = p - 0.5;

    if (0.075 <= p && p <= 0.925)
    {
        r = 0.180625 - q * q;
        val = q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608) / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1);
    }
    else
    { /* closer than 0.075 from {0,1} boundary */
        /* r = min(p, 1-p) < 0.075 */
        if (q > 0)
        {
            r = 1 - p;
        }
        else
        {
            r = p; /* = R_DT_Iv(p) ^=  p */
        }

        r = sqrt(-log(r)); /* r = sqrt(-log(r))  <==>  min(p, 1-p) = exp( - r^2 ) */

        if (r <= 5.0)
        { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
            r += -1.6;
            val = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r + 1.42343711074968357734) / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r + 2.05319162663775882187) * r + 1.0);
        }
        else
        { /* very close to  0 or 1 */
            r += -5.0;
            val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r + 5.4637849111641143699) * r + 6.6579046435011037772) / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r + 0.59983220655588793769) * r + 1.0);
        }

        if (q < 0.0)
        {
            val = -val;
        }
        /* return (q >= 0.)? r : -r ;*/
    }
    return mu + sigma * val;
}

/**
 * Used internally by Shapiro wilk algorithm
 *
 * Returns the p-value of a specific z score for Gaussian
 * Ported from Java to C++ code posted at https://github.com/datumbox/datumbox-framework/blob/develop/datumbox-framework-core/src/main/java/com/datumbox/framework/core/statistics/distributions/ContinuousDistributions.java
 * Ported from C# to java code posted at http://jamesmccaffrey.wordpress.com/2010/11/05/programmatically-computing-the-area-under-the-normal-curve/
 *
 * @param z
 * @return
 */
double gaussCdf(double z)
{
    // input = z-value (-inf to +inf)
    // output = p under Normal curve from -inf to z
    // e.g., if z = 0.0, function returns 0.5000
    // ACM Algorithm #209
    double y; // 209 scratch variable
    double p; // result. called ‘z’ in 209
    double w; // 209 scratch variable

    if (z == 0.0)
    {
        p = 0.0;
    }
    else
    {
        y = abs(z) / 2.0;
        if (y >= 3.0)
        {
            p = 1.0;
        }
        else if (y < 1.0)
        {
            w = y * y;
            p = ((((((((0.000124818987 * w - 0.001075204047) * w + 0.005198775019) * w - 0.019198292004) * w + 0.059054035642) * w - 0.151968751364) * w + 0.319152932694) * w - 0.531923007300) * w + 0.797884560593) * y * 2.0;
        }
        else
        {
            y = y - 2.0;
            p = (((((((((((((-0.000045255659 * y + 0.000152529290) * y - 0.000019538132) * y - 0.000676904986) * y + 0.001390604284) * y - 0.000794620820) * y - 0.002034254874) * y + 0.006549791214) * y - 0.010557625006) * y + 0.011630447319) * y - 0.009279453341) * y + 0.005353579108) * y - 0.002141268741) * y + 0.000535310849) * y + 0.999936657524;
        }
    }

    if (z > 0.0)
    {
        return (p + 1.0) / 2.0;
    }

    return (1.0 - p) / 2.0;
}

/**
 * Used internally by ShapiroWilkW().
 *
 * @param cc
 * @param nord
 * @param x
 * @return
 */
double poly(double cc[], int nord, double x)
{
    /* Algorithm AS 181.2    Appl. Statist.    (1982) Vol. 31, No. 2
    Calculates the algebraic polynomial of order nord-1 with array of coefficients cc.
    Zero order coefficient is cc(1) = cc[0] */

    double ret_val = cc[0];
    if (nord > 1)
    {
        double p = x * cc[nord - 1];
        for (int j = nord - 2; j > 0; --j)
        {
            p = (p + cc[j]) * x;
        }
        ret_val += p;
    }
    return ret_val;
}

/**
 * Used internally by ShapiroWilkW()
 *
 * @param x
 * @return
 */
int sign(double x)
{
    if (x == 0)
    {
        return 0;
    }
    return (x > 0) ? 1 : -1;
}

/**
 * Calculates P-value for ShapiroWilk Test
 *
 * @param x
 * @return
 */
void ShapiroWilkW(vector<double> &x, double &w, double &pw)
{
    sort(std::begin(x), std::end(x));

    int n = x.size();

    assert(n >= 3);
    assert(n <= 5000);

    int nn2 = n / 2;
    double a[nn2 + 1]; /* 1-based */

    /*
        ALGORITHM AS R94 APPL. STATIST. (1995) vol.44, no.4, 547-551.
        Calculates the Shapiro-Wilk W test and its significance level
    */
    double small = 1e-19;

    /* polynomial coefficients */
    double g[] = {-2.273, 0.459};
    double c1[] = {0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056};
    double c2[] = {0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633};
    double c3[] = {0.544, -0.39978, 0.025054, -6.714e-4};
    double c4[] = {1.3822, -0.77857, 0.062767, -0.0020322};
    double c5[] = {-1.5861, -0.31082, -0.083751, 0.0038915};
    double c6[] = {-0.4803, -0.082676, 0.0030302};

    /* Local variables */
    int i, j, i1;

    double ssassx, summ2, ssumm2, gamma, range;
    double a1, a2, an, m, s, sa, xi, sx, xx, y, w1;
    double fac, asa, an25, ssa, sax, rsn, ssx, xsx;

    pw = 1.0;
    an = (double)n;

    if (n == 3)
    {
        a[1] = 0.70710678; /* = sqrt(1/2) */
    }
    else
    {
        an25 = an + 0.25;
        summ2 = 0.0;
        for (i = 1; i <= nn2; i++)
        {
            a[i] = normalQuantile((i - 0.375) / an25, 0, 1); // p(X <= x),
            summ2 += a[i] * a[i];
        }
        summ2 *= 2.0;
        ssumm2 = sqrt(summ2);
        rsn = 1.0 / sqrt(an);
        a1 = poly(c1, 6, rsn) - a[1] / ssumm2;

        /* Normalize a[] */
        if (n > 5)
        {
            i1 = 3;
            a2 = -a[2] / ssumm2 + poly(c2, 6, rsn);
            fac = sqrt((summ2 - 2.0 * (a[1] * a[1]) - 2.0 * (a[2] * a[2])) / (1.0 - 2.0 * (a1 * a1) - 2.0 * (a2 * a2)));
            a[2] = a2;
        }
        else
        {
            i1 = 2;
            fac = sqrt((summ2 - 2.0 * (a[1] * a[1])) / (1.0 - 2.0 * (a1 * a1)));
        }
        a[1] = a1;
        for (i = i1; i <= nn2; i++)
        {
            a[i] /= -fac;
        }
    }

    /* Check for zero range */

    range = x[n - 1] - x[0];

    assert(range >= small);

    /* Check for correct sort order on range - scaled X */

    xx = x[0] / range;
    sx = xx;
    sa = -a[1];
    for (i = 1, j = n - 1; i < n; j--)
    {
        xi = x[i] / range;

        assert(xx - xi <= small);

        sx += xi;
        i++;
        if (i != j)
        {
            sa += sign(i - j) * a[min(i, j)];
        }
        xx = xi;
    }

    /* Calculate W statistic as squared correlation        between data and coefficients */

    sa /= n;
    sx /= n;
    ssa = ssx = sax = 0.;
    for (i = 0, j = n - 1; i < n; i++, j--)
    {
        if (i != j)
        {
            asa = sign(i - j) * a[1 + min(i, j)] - sa;
        }
        else
        {
            asa = -sa;
        }
        xsx = x[i] / range - sx;
        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
    }

    /* W1 equals (1-W) calculated to avoid excessive rounding error        for W very near 1 (a potential problem in very large samples) */

    ssassx = sqrt(ssa * ssx);
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    w = 1.0 - w1;
    // System.out.println("w =" + w);
    /* Calculate significance level for W */

    if (n == 3)
    {                                   /* exact P value : */
        double pi6 = 1.90985931710274;  /* = 6/pi */
        double stqr = 1.04719755119660; /* = asin(sqrt(3/4)) */
        pw = pi6 * (asin(sqrt(w)) - stqr);
        if (pw < 0.)
        {
            pw = 0;
        }
        // return w;
        return;
    }
    y = log(w1);
    xx = log(an);
    if (n <= 11)
    {
        gamma = poly(g, 2, an);
        if (y >= gamma)
        {
            pw = 1e-99; /* an "obvious" value, was 'small' which was 1e-19f */
            // return w;
            return;
        }
        y = -log(gamma - y);
        m = poly(c3, 4, an);
        s = exp(poly(c4, 4, an));
    }
    else
    { /* n >= 12 */
        m = poly(c5, 4, xx);
        s = exp(poly(c6, 3, xx));
    }

    // Oops, we don't have pnorm
    // pw = pnorm(y, m, s, 0/* upper tail */, 0);
    double z = (y - m) / s;
    // System.out.println("z =" + z);
    pw = gaussCdf(z);
    // return w;
    return;
}
