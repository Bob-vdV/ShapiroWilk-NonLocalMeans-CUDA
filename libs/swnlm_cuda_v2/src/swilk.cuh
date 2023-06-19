#ifndef SWNLM_SWILK_HPP
#define SWNLM_SWILK_HPP

#include "swilkutils.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cassert>

namespace ShapiroWilk
{
    // IMPORTANT: a should have allocated size+1
    void setup(double *a, const int size)
    {
        const int n = size;

        assert(n >= 3);
        assert(n <= 5000);

        const int nn2 = n / 2;

        /*
            ALGORITHM AS R94 APPL. STATIST. (1995) vol.44, no.4, 547-551.
            Calculates the Shapiro-Wilk W test and its significance level
        */

        const double c1[] = {0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056};
        const double c2[] = {0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633};

        /* Local variables */
        int i, i1;

        double summ2, ssumm2;
        double a1, a2, an;
        double fac, an25, rsn;

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
    }

    /**
     * Calculates P-value for ShapiroWilk Test
     *
     * @param x
     * @return
     */
    __device__ void test(double *x, const double *a, const int size, double &w, double &pw)
    {
        const int n = size;

        sortArr(x, size);

        pw = 1.0;

        /* polynomial coefficients */
        const double g[] = {-2.273, 0.459};
        const double c3[] = {0.544, -0.39978, 0.025054, -6.714e-4};
        const double c4[] = {1.3822, -0.77857, 0.062767, -0.0020322};
        const double c5[] = {-1.5861, -0.31082, -0.083751, 0.0038915};
        const double c6[] = {-0.4803, -0.082676, 0.0030302};

        /* Local variables */
        int i, j;

        double ssassx, gamma;
        double m, s, sa, xi, sx, xx, y, w1;
        double asa, ssa, sax, ssx, xsx;

        const double an = n;

        /* Check for zero range */

        const double range = x[n - 1] - x[0];
        assert(range > 0);

        /* Check for correct sort order on range - scaled X */

        xx = x[0] / range;
        sx = xx;
        sa = -a[1];
        for (i = 1, j = n - 1; i < n; j--)
        {
            xi = x[i] / range;

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

        /*
         * W1 equals (1-W) calculated to avoid excessive rounding error
         * for W very near 1 (a potential problem in very large samples)
         */

        ssassx = sqrt(ssa * ssx);
        w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
        w = 1.0 - w1;
        /* Calculate significance level for W */

        if (n == 3)
        {                                         /* exact P value : */
            const double pi6 = 1.90985931710274;  /* = 6/pi */
            const double stqr = 1.04719755119660; /* = asin(sqrt(3/4)) */
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
        const double z = (y - m) / s;
        pw = gaussCdf(z);
        return;
    }
}

#endif