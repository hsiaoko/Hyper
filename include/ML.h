#ifndef CUER_ML_H_
#define CUER_ML_H_
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>

#include "../lib/REENet/include/core.h"
#include "../lib/REENet/lib/cfasttext/include/cfasttext.h"
using namespace std;
namespace CUER
{

    class ML
    {
    private:
        std::vector<reenet::REEModule *> *vec_reemodules;
        std::vector<std::vector<std::string> *> *tuples_;
        //std::vector<std::vector<float *>*> *vec_attr_embeding_2D;
        int embeding_size_;
        std::map<int, reenet::REEModule *> *map_reemodules;
        //std::vector<int> *vec_aid_;

    public:
        std::map<int, std::vector<float *> *> *map_attr_embeding_2D_;
        std::map<int, std::string> *bin_pth;
        std::map<int, std::string> *model_pth;

        ML(std::vector<int> *, std::vector<string> *, std::vector<string> *, std::vector<std::vector<std::string> *> *);
        ML(std::map<int, std::vector<float *> *> *, std::map<int, string> *, std::map<int, string> *);
        ML();
        std::vector<std::vector<float *> *> *attrs_embeding(std::vector<std::vector<std::string> *> *);
        float *attr_embeding(size_t aid, std::string str_);

        static void show_embeding(float *eb, size_t eb_size);
        static float euclidean_distance(float *eb_1, float *eb_r, int eb_size);
        static void relu(float *input, size_t length, float tau);
        static void sigmoid(float *input, size_t length);
        static float *mm(float *a, float *b, std::pair<size_t, size_t> shape_a, std::pair<size_t, size_t> shape_b);
        static float *mm_euclidean_distance(float *a, float *b, std::pair<size_t, size_t> shape);
        static float euclidean_distance(char *attr_l, char *attr_r, int length_);
    };

} //namespace CUER

#endif //CUER_ML_H_