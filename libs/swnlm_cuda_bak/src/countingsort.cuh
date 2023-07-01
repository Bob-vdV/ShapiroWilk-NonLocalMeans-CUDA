#ifndef COUNTINGSORT_CUH
#define COUNTINGSORT_CUH

// array with values [-255...255]
template <typename T>
__device__ void countingSort(T *array, size_t size)
{
    /*
    int *hist = NULL;
    while (*hist == NULL)
    {
        hist = new int[2 * range + 1];
    }*/
    const int range = 255;

    int hist[2 * range + 1];
    memset(hist, 0, (2 * range + 1) * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        //printf("arr: %d, addr:%d\n", (int)array[i], (int)array[i] + range);
        hist[(int)array[i] + range]++;
    }

    //printf("2");
    size_t arrIdx = 0;
    size_t histIdx = 0;
    while (arrIdx != size)
    {
        while (hist[histIdx] != 0)
        {
            array[arrIdx] = histIdx - range;
            hist[histIdx]--;
            arrIdx++;
        }
        histIdx++;
        //printf("3");
    }
    //printf("4");
}

#endif