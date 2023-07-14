#include "swilk.cuh"

#include <iostream>
#include <vector>

using namespace std;

int main()
{

    int maxSize = 11;

    vector<double> vec(maxSize);
    double *a = vec.data();

    ShapiroWilk::setup(a, 10);

    // Simple test
    double t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    double w, pw;
    ShapiroWilk::test(t1, a, 10, w, pw);

    cout << "w: " << w << ", pw: " << pw << '\n';

    // Test sorting
    double t2[] = {2, 1, 0, 3, 9, 5, 7, 6, 8, 4};
    ShapiroWilk::test(t2, a, 10, w, pw);
    cout << "w: " << w << ", pw: " << pw << '\n';

    // Example for R code that failed with ifault=7
    ShapiroWilk::setup(a, 11);
    double t3[] = {-1.7, -1, -1, -.73, -.61, -.5, -.24, .45, .62, .81, 1};

    ShapiroWilk::test(t3, a, 11, w, pw);
    cout << "w: " << w << ", pw: " << pw << '\n';
}