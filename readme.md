#  ShapiroWilk NonLocalMeans CUDA
C++ and CUDA implementations of Conventional Non Local Means (CNLM) and Shapiro-Wilk NLM (SWNLM). This repository is part of my bachelor thesis, which is also included [here](./NLM_BobvdVuurst_Bachelor_Thesis-final.pdf). The summary of the results are discussed below.



## Denoising Quality
PSNR and MSSIM values for CNLM and SWNLM-CUDA for the [standard images](https://www.imageprocessingplace.com/root_files_V3/image_databases.htm) are in the table below. The CUDA implementations have almost the exact same denoising quality as their sequential implementations. SWNLM-CUDA has better denoising quality than CNLM-CUDA on average.

<table>
<thead>
  <tr>
    <th>Noisy</th>
    <th>CNLM-CUDA</th>
    <th>SWNLM-CUDA</th>
  </tr>
</thead>
<tbody>
<tr>
<td>
<img src="output\standard\house_sigma=40_noisy.png" alt="Noisy" width="100%"/>
</td>
<td>
<img src="output\standard\house_sigma=40_searchRadius=8_neighborRadius=3_denoiser=cnlm_cuda.png" alt="CNLM-CUDA" width="100%"/>
</td>
<td>
<img src="output\standard\house_sigma=40_searchRadius=8_neighborRadius=3_denoiser=swnlm_cuda.png" alt="SWNLM-CUDA" width="100%"/>
</td>
</tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th rowspan="2">Image</th>
    <th colspan="3">PSNR</th>
    <th colspan="3">MSSIM</th>
  </tr>
  <tr>
    <th>Noisy</th>
    <th>CNLM-CUDA</th>
    <th>SWNLM-CUDA</th>
    <th>Noisy</th>
    <th>CNLM-CUDA</th>
    <th>SWNLM-CUDA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Boat</td>
    <td>16.3804</td>
    <td>24.6654</td>
    <td>25.4696</td>
    <td>0.1529</td>
    <td>0.4342</td>
    <td>0.4911</td>
  </tr>
  <tr>
    <td>Cameraman</td>
    <td>16.6392</td>
    <td>25.9880</td>
    <td>25.1043</td>
    <td>0.1192</td>
    <td>0.3217</td>
    <td>0.4719</td>
  </tr>
  <tr>
    <td>House</td>
    <td>16.5133</td>
    <td>26.7664</td>
    <td>29.7184</td>
    <td>0.0778</td>
    <td>0.2648</td>
    <td>0.5006</td>
  </tr>
  <tr>
    <td>Lake</td>
    <td>16.6491</td>
    <td>24.8156</td>
    <td>25.0256</td>
    <td>0.1901</td>
    <td>0.4739</td>
    <td>0.5340</td>
  </tr>
  <tr>
    <td>Livingroom</td>
    <td>16.3487</td>
    <td>24.2110</td>
    <td>24.9051</td>
    <td>0.1637</td>
    <td>0.4348</td>
    <td>0.4741</td>
  </tr>
  <tr>
    <td>Mandril</td>
    <td>16.2563</td>
    <td>23.4073</td>
    <td>23.5753</td>
    <td>0.2120</td>
    <td>0.4962</td>
    <td>0.4405</td>
  </tr>
  <tr>
    <td>Peppers</td>
    <td>16.4532</td>
    <td>25.7017</td>
    <td>27.1050</td>
    <td>0.1105</td>
    <td>0.4091</td>
    <td>0.5667</td>
  </tr>
  <tr>
    <td>Plane</td>
    <td>16.7011</td>
    <td>25.9475</td>
    <td>26.7001</td>
    <td>0.1364</td>
    <td>0.3619</td>
    <td>0.5111</td>
  </tr>
</tbody>
</table>

## Execution Times
Execution times of CNLM and SWNLM are measured from denoising downscaled images of the [Poly U dataset](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset). The tests are performed with an AMD Ryzen 5 1600 and an Nvidia GTX 1080. A higher relative speedup is achieved with SWNLM-CUDA and execution times are low enough to make it practically usable. However, CNLM-CUDA is still roughly 8 times faster than SWNLM-CUDA. 

### CNLM
| resolution  | avg sequential time (s) | avg CUDA time (s) | achieved speedup |
|-------------|-------------------------|-------------------|------------------|
| 324 × 216   | 1.7873                  | 0.0400            | 44.6833          |
| 648 × 432   | 7.1577                  | 0.1580            | 45.3017          |
| 1296 × 864  | 28.6027                 | 0.6350            | 45.0436          |
| 2592 × 1728 | 114.4240                | 2.5407            | 45.0370          |

### SWNLM
| resolution  | avg sequential time (s) | avg CUDA time (s) | achieved speedup |
|-------------|-------------------------|-------------------|------------------|
| 324 × 216   | 45.5367                 | 0.3213            | 141.7116         |
| 648 × 432   | 182.1727                | 1.2627            | 144.2761         |
| 1296 × 864  | 729.9890                | 5.0773            | 143.7741         |
| 2592 × 1728 | 2921.9523               | 20.3050           | 143.9031         |
