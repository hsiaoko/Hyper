
#ifndef CUER_CORE_H_
#define CUER_CORE_H_
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include "matcher.h"
#include "tools.h"
#include "data_loader.h"
#include "ML.h"

using namespace std;
/*
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t = cal;                                               \
        if (error !\ cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s: %d", __FILE__, __LINE__);                       \
            printf("code: %d, reason: %s\n", eror, cudaGetErrorString(error)); \
            exit(1)                                                            \
        }                                                                      \
    }
*/
namespace CUER
{
    __device__ float euclidean_distance(float *eb_1, float *eb_r, int eb_size);
    __device__ void relu(float *input, size_t length, float tau);
    __device__ void sigmoid(float *input, size_t length);
    __device__ float *mm(float *a, float *b, size_t shape_a_first, size_t shape_b_second, size_t shape_b_first, size_t shape_b_seocond);
    __device__ float *mm_euclidean_distance(float *a, float *b, size_t shape_first, size_t shape_second);
    __device__ float euclidean_distance(char *attr_l, char *attr_r, int length_);
    __device__ void mm_euclidean_distance(float *a, float *b, float *c, size_t shape_first, size_t shape_second);
    __device__ void mm(float *a, float *b, float *c, size_t shape_a_first, size_t shape_a_second, size_t shape_b_first, size_t shape_b_second);

    __global__ void helloFromCpu(void);

    __device__ bool callMatch(
        char *tuple_l,
        char *tuple_r,
        size_t *rule_aid,
        float *rule_threshold,
        size_t *attr_len_attr,
        size_t *attr_start_point,
        size_t tuple_size,
        size_t rule_size);
    __device__ bool eq(char *attr_l, char *attr_r);
    /*
__device__ double lev_jaro_ratio(size_t *len1, const char *string1,
                                 size_t *len2, const char *string2);

__device__ double lev_jaro_ratio(size_t len1, const char *string1,
                                 size_t len2, const char *string2);
*/
    __device__ double lev_jaro_ratio(size_t len1, const char *string1,
                                     size_t len2, const char *string2, int *tag);

    __global__ void debug(char *d_arr);

    __global__ void resolutionSelfJoinV3(
        char *d_tuples,
        char *d_match,
        const size_t line_size,
        const size_t data_size,
        const size_t max_match_per_thread,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        size_t num_blocking,
        size_t *blocking_size_arr,
        size_t *blocking_start_point,
        size_t *d_int);

    __device__ int cuda_sprintf(char *buf, const char *fmt, ...);

    __device__ size_t cuda_len(char *string_);

    __host__ std::vector<std::vector<std::vector<std::string> *> *> *blocking(
        std::vector<std::vector<std::string> *> *tuples,
        size_t num_blocks,
        size_t aid);

    __host__ std::vector<std::vector<std::vector<std::string> *> *> *blocking(
        std::vector<std::vector<std::string> *> *tuples,
        size_t num_blocks,
        std::vector<pair<size_t, size_t>> *aid /*<aid, num_prefix>*/);

    __global__ void TEST(
        int *offset);
    __global__ void say_hello(size_t *d_attr_aid, int *offset);

    __global__ void dirtyER(
        char *d_tuples,
        char *d_match,
        const size_t line_size,
        const size_t data_size,
        //const size_t max_match,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        int *offset);

    __global__ void dirtyER(
        char *d_tuples,
        char *d_match,
        const size_t line_size,
        const size_t data_size,
        //const size_t max_match,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        int *offset,
        int max_match);

    __global__ void clearnClearnER(
        char *d_tuples_l,
        char *d_tuples_r,
        char *d_match,
        const size_t line_size,
        const size_t data_size_l,
        const size_t data_size_r,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        int *offset);

    __global__ void MLclearnClearnER(
        char *d_tuples_l,
        char *d_tuples_r,
        char *d_match,
        const size_t line_size,
        const size_t data_size_l,
        const size_t data_size_r,
        float *d_ML_mm,
        float bias,
        size_t *d_rule_aid,
        size_t *d_ML_aid,
        float *d_rule_threshold,
        //float *d_ML_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        pair<int, int> d_ML_mm_shape,
        const size_t tuple_size,
        float *d_sim,
        int *offset);

    // __device__ int cuda_vsprintf(char *buf, const char *fmt, va_list args);

    __host__ void clearnclearnHyper(
        char *d_tuples_l,
        char *d_tuples_r,
        int line_size,
        size_t data_size_l,
        size_t data_size_r,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        float *h_sim,
        char *h_match_vec,
        char *d_match_vec,
        int *d_offset,
        int *h_offset,
        std::pair<size_t, size_t> dim_grid,
        std::pair<size_t, size_t> dim_block,
        int size_tag,
        size_t max_match,
        cudaStream_t *stream,
        int* signal_vec,
        int iblock,
        int *signal);


    __host__ void dirtyHyper(
        char *d_tuples_l,
        int line_size,
        size_t data_size_l,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        float *h_sim,
        char *h_match_vec,
        char *d_match_vec,
        int *d_offset,
        int *h_offset,
        std::pair<size_t, size_t> dim_grid,
        std::pair<size_t, size_t> dim_block,
        int size_tag,
        size_t max_match,
        cudaStream_t *stream,
        int* signal_vec,
        int iblock,
        int *signal);

} //namespace CUER
#endif //CUER_CORE_H_