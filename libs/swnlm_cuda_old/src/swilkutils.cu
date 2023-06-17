#include "swilkutils.cuh"

#include <cmath>
#include <cassert>

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
__host__ __device__ double normalQuantile(const double p, const double mu, const double sigma)
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
__host__ __device__ double gaussCdf(const double z)
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
__host__ __device__ double poly(const double cc[], const int nord, const double x)
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
__host__ __device__ int sign(const double x)
{
    if (x == 0)
    {
        return 0;
    }
    return (x > 0) ? 1 : -1;
}

// Simple bubble sort for sorting small arrays
__host__ __device__ void sortArr(double *arr, const size_t size){
    for(size_t i=0;i<size; i++){
        for(size_t j = i + 1; j < size; j++){
            if(arr[i] > arr[j]){
                const double temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }

}
