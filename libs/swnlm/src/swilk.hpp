#ifndef SWNLM_SWILK_HPP
#define SWNLM_SWILK_HPP

#include <vector>

class ShapiroWilk
{
private:
    const size_t size;
    std::vector<double> a;

public:
    ShapiroWilk(const size_t size);
    void test(std::vector<double> &x, double &w, double &pw) const;
};

void test(std::vector<double> &x, double &w, double &pw);

#endif