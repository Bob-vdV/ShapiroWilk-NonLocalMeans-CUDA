// Non-local Means Algorithm for Medical image Denoising
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

#define NLM_DEBUG 1

int i, j, k1, l1, l2, k2, l3, k3, p, q;
const int R = 5;
const int S = 5;
const int P = 20;
const int Q = 20;
const int h = 10;

float W1[R][S], W2[R][S];
float NL, Z, sij;
float nlm_kernel[R][S];

void nlm(Mat &inputMat, Mat &outputMat)
{
    // calculation of non-liner filter
    // search windows: (P+R-1)×(Q+S-1) e.g P=Q=16 → 20×20
    // W1[R][S], W2[R][S], nlm_kernel[R][S], h=10
    // input[N][M], output[N][M]

    int N = inputMat.rows;
    int M = inputMat.cols;

    outputMat = Mat::zeros(Size(N, M), CV_8UC1);

    float sum;
    for (i = (P / 2) - 1; i < N - (P / 2) + 1; i++)
    {
        for (j = (Q / 2) - 1; j < M - (Q / 2) + 1; j++)
        {
            for (k1 = 0; k1 < R; k1++)
            {
                for (l1 = 0; l1 < S; l1++)
                {
                    W1[k1][l1] = inputMat.at<uchar>(i - (R / 2) + k1, j - (S / 2) + l1); //    [i - (R / 2) + k1][j - (S / 2) + l1];
                }
            }
            NL = 0;
            Z = 0;
            for (p = 0; p < P; p++)
            {
                for (q = 0; q < Q; q++)
                {
                    for (k2 = 0; k2 < R; k2++)
                    {
                        for (l2 = 0; l2 < S; l2++)
                        {
                            W2[k2][l2] = inputMat.at<uint8_t>(i - (P / 2) + 1 + p + k2, j - (Q / 2) + 1 + q + l2); //[i - (P / 2) + 1 + p + k2][j - (Q / 2) + 1 + q + l2];
                        }
                    }
                    sum = 0;
                    for (k3 = 0; k3 < R; k3++)
                    {
                        for (l3 = 0; l3 < S; l3++)
                        {
                            sum += nlm_kernel[k3][l3] * ((W1[k3][l3] - W2[k3][l3]) * (W1[k3][l3] - W2[k3][l3]));
                        }
                    }
                    sij = exp(-sum / (h * h));
                    // update Z and NL
                    Z += sij;
                    NL += sij * inputMat.at<uint8_t>(i - (P / 2) + 1 + p + (R / 2), j - (Q / 2) + 1 + q + (S / 2)); // [i - (P / 2) + 1 + p + (R / 2)][j - (Q / 2) + 1 + q + (S / 2)];
                }
            }
            outputMat.at<uint8_t>(i, j) = NL / Z; //  [i][j] = NL / Z;
        }
    }
}

//----------------------------------------------------------------------------------
void mk_gsn_krnl()
{
    float su = 1; // standard deviation of gaussian kernel
    float sm = 0; // sum of all kernel elements (for normalization)
    const int f = 2;
    for (int x = 0; x < R; x++)
    {
        for (int y = 0; y < S; y++)
        {
            int ab = x - f; // horizontal distance of pixel from center(f+1, f+1)
            int cd = y - f; // vertical distance of pixel from center (f+1, f+1)
            nlm_kernel[x][y] = 100 * exp(((ab * ab) + (cd * cd)) / (-2 * (su * su)));
            sm += nlm_kernel[x][y];
        }
    }
    // printf("\n\n\n");
    for (int x = 0; x < R; x++)
        for (int y = 0; y < S; y++)
            nlm_kernel[x][y] = nlm_kernel[x][y] / f;
    for (int x = 0; x < R; x++)
    {
        for (int y = 0; y < S; y++)
        {
            nlm_kernel[x][y] = (nlm_kernel[x][y] / sm); // normalization
                                                        // printf("%f  ", nlm_kernel[x][y]);
        }
        // printf("\n");
    }
}

void CopyMatToArray(vector<uchar> &array, Mat &mt)
{
    array.clear();
    if (mt.isContinuous())
    {
        array.assign(mt.data, mt.data + mt.total() * mt.channels());
    }
    else
    {
        for (int i = 0; i < mt.rows; ++i)
        {
            array.insert(array.end(), mt.ptr<uchar>(i), mt.ptr<uchar>(i) + mt.cols * mt.channels());
        }

        /*
            int col = 0;
            for (int i = 0; i < M; i++)
            {
                col = 0;
                for (int j = 0; j < N; j++)
                {
                    input[i][j] = mt.at<uchar>(i, col);
                    col += 3;
                }
            }*/
    }
}
/*
void CopyArrayToMat(vector<uchar> &array, Mat &mt)
{

    mt = Mat::zeros(Size(M, N), CV_8UC1);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            mt.at<uchar>(i, j) = output[i][j];
}*/

void test(string imagePath)
{
    Mat input_mat, gray_mat, output_mat;

    // read input image and store in input
    // input_mat = imread("F:/Programming_start2019_9/Visal_Studio_2019/Source/1.JPG");
    // input_mat = imread("E:/Papers_98/Moradifar__Maryam/images_Nsaltperper/NCT1.png");
    // input_mat = imread("E:/Papers_98/Moradifar__Maryam/images_speckle_1/NCT1.png");
    input_mat = imread(imagePath);

    if (input_mat.empty())
        std::cout << "failed to open img.jpg" << std::endl;
    else if (NLM_DEBUG)
        std::cout << "img.jpg loaded OK" << std::endl;

	// Convert to grayscale
	cv::cvtColor(input_mat, gray_mat, cv::COLOR_RGB2GRAY);

    // show  input image
    if (NLM_DEBUG)
    {
        imshow("Input", gray_mat);
        cout << "Please Wait..." << endl;
        waitKey(1);
    }

    chrono::system_clock::time_point begin = chrono::high_resolution_clock::now();

    mk_gsn_krnl();
    nlm(gray_mat, output_mat);

    chrono::system_clock::time_point end = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    cout << imagePath << '\t';
    cout << duration / 1000 << '\n';

    // show output image
    if (NLM_DEBUG)
    {
        cout << "Finished.";
        imshow("Output", output_mat);
        waitKey(0);
    }
}

//----------------------------------------------------------------------------------------
int main()
{
    vector<string> images = {
        "../images/64x64.png",
        "../images/128x128.png",
        "../images/256x256.png",
        "../images/512x512.png",
        "../images/1024x1024.png",
    };

    for (auto image : images)
    {
        test(image);
    }
}