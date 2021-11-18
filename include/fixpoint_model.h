#ifndef CUER_FIXPOINTMODEL_H_
#define CUER_FIXPOINTMODEL_H_
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include "struct.h"
#include "tools.h"

namespace CUER
{
class FixpointModel
{
private:
public:
    static int factorial(size_t n);
    static int summation(size_t n);
    static int get_comparisons(size_t num_threads, size_t num_tuples);
    static float* get_ratio(size_t *allocation, size_t len_);
};

} // namespace CUER
#endif //CUER_FIXPOINTMODEL_H_