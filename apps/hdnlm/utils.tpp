/**
 * Copies a slice of inputMat at given dimension and index to the output.
 *
 */
template <typename T>
void mat3toMat2(const cv::Mat &inputMat, cv::Mat &outputMat, const int dim, const int idx)
{
    const int sz0 = inputMat.size[0];
    const int sz1 = inputMat.size[1];
    const int sz2 = inputMat.size[2];
    const int type = inputMat.type();

    assert(inputMat.dims == 3);

    switch (dim)
    {
    case 0:
    {
        const int size[] = {sz1, sz2};
        outputMat.create(2, size, type);

        for (int row = 0; row < sz1; row++)
        {
            for (int col = 0; col < sz2; col++)
            {
                outputMat.at<T>(row, col) = inputMat.at<T>(idx, row, col);
            }
        }

        return;
    }

    case 1:
    {
        const int size[] = {sz0, sz2};
        outputMat.create(2, size, type);

        for (int row = 0; row < sz0; row++)
        {
            for (int col = 0; col < sz2; col++)
            {
                outputMat.at<T>(row, col) = inputMat.at<T>(row, idx, col);
            }
        }

        return;
    }

    case 2:
    {
        const int size[] = {sz0, sz1};
        outputMat.create(2, size, type);

        for (int row = 0; row < sz0; row++)
        {
            for (int col = 0; col < sz1; col++)
            {
                outputMat.at<T>(row, col) = inputMat.at<T>(row, col, idx);
            }
        }

        return;
    }
    default:
    {
        assert(inputMat.dims > dim);
    }
    }
}

/**
 * Copies the 2D input into a slice of the 3D output
 *
 * Prerequisites: outputMat must be initialized and allocated.
 *
 */
template <typename T>
void mat2toMat3(const cv::Mat &inputMat, cv::Mat &outputMat, const int dim, const int idx)
{
    const int sz0 = outputMat.size[0];
    const int sz1 = outputMat.size[1];
    const int sz2 = outputMat.size[2];

    assert(inputMat.dims == 2);
    assert(outputMat.dims == 3);

    switch (dim)
    {
    case 0:
    {
        for (int row = 0; row < sz1; row++)
        {
            for (int col = 0; col < sz2; col++)
            {
                outputMat.at<T>(idx, row, col) = inputMat.at<T>(row, col);
            }
        }
        return;
    }

    case 1:
    {
        for (int row = 0; row < sz0; row++)
        {
            for (int col = 0; col < sz2; col++)
            {
                outputMat.at<T>(row, idx, col) = inputMat.at<T>(row, col);
            }
        }
        return;
    }

    case 2:
    {
        for (int row = 0; row < sz0; row++)
        {
            for (int col = 0; col < sz1; col++)
            {
                outputMat.at<T>(row, col, idx) = inputMat.at<T>(row, col);
            }
        }
        return;
    }
    default:
    {
        assert(outputMat.dims > dim);
    }
    }
}
