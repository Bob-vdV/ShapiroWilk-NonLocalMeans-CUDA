#ifndef SWNLM_SWILK_HPP
#define SWNLM_SWILK_HPP

#include <vector>
#include <cstddef>

class ShapiroWilk
{
private:
    const size_t size;
    std::vector<double> a;

public:
    ShapiroWilk(const size_t size);
    void test(double *x, double &w, double &pw) const;
};

#endif