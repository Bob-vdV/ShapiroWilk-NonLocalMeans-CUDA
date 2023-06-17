#ifndef SWNLM_SWILKUTILS_HPP
#define SWNLM_SWILKUTILS_HPP

double normalQuantile(const double p, const double mu, const double sigma);

double gaussCdf(const double z);

double poly(const double cc[], const int nord, const double x);

int sign(const double x);



#endif