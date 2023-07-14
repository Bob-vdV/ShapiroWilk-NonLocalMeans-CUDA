/*
 *  Ported from Javascript version: https://github.com/rniwa/js-shapiro-wilk
 *
 *  Ported from http://svn.r-project.org/R/trunk/src/nmath/qnorm.c
 *
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998       Ross Ihaka
 *  Copyright (C) 2000--2005 The R Core Team
 *  based on AS 111 (C) 1977 Royal Statistical Society
 *  and   on AS 241 (C) 1988 Royal Statistical Society
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 */

#ifndef SWILK_HPP
#define SWILK_HPP

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
// #include <cuda.h>
#include <iostream>
#include <vector>
#include <math.h>

namespace ShapiroWilk
{
    void setup(double *a, const int size);

    __host__ __device__ void test(double *x, const double *a, const int size, double &w, double &pw);

};

#endif