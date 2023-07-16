#include "swilk.cuh"

#include <iostream>
#include <vector>

using namespace std;

int main()
{
    using T = double;

    int maxSize = 11;

    vector<T> vec(maxSize);
    T *a = vec.data();

    ShapiroWilk::setup(a, 10);

    // Simple test
    T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    T w;
    ShapiroWilk::test(t1, a, 10, w);

    cout << "w: " << w << '\n';

    // Test sorting
    T t2[] = {2, 1, 0, 3, 9, 5, 7, 6, 8, 4};
    ShapiroWilk::test(t2, a, 10, w);
    cout << "w: " << w << '\n';

    // Example for R code that failed with ifault=7
    ShapiroWilk::setup(a, 11);
    T t3[] = {-1.7, -1, -1, -.73, -.61, -.5, -.24, .45, .62, .81, 1};

    ShapiroWilk::test(t3, a, 11, w);
    cout << "w: " << w << '\n';
}