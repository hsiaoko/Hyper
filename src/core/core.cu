#include "../../include/core.h"

namespace CUER
{

    __device__ __host__ size_t cuda_len(char *string_)
    {
        size_t i = 0;
        while (string_[i] != '\0' && i < 2048)
        {
            i++;
        }
        return i;
    }

    __device__ float euclidean_distance(float *eb_1, float *eb_r, int eb_size)
    {
        float dis = 0;
        for (int i = 0; i < eb_size; i++)
        {
            dis += (eb_1[i] - eb_r[i]) * (eb_1[i] - eb_r[i]);
        }
        return dis;
    }

    __device__ void relu(float *input, size_t length, float tau)
    {
        for (int i = 0; i < length; i++)
        {
            input[i] = input[i] > tau ? input[i] : 0;
        }
    }

    __device__ void sigmoid(float *input, size_t length)
    {
        for (int i = 0; i < length; i++)
        {
            input[i] = 1 / (1 + exp(-1 * input[i]));
        }
    }

    __device__ float *mm(float *a, float *b, size_t shape_a_first, size_t shape_a_second, size_t shape_b_first, size_t shape_b_second)
    {
        if (shape_a_second != shape_b_first)
        {
            return NULL;
        }
        float *c = new float(shape_a_first * shape_b_second);

        for (int b_j = 0; b_j < shape_b_second; b_j++)
        {
            for (int a_i = 0; a_i < shape_a_first; a_i++)
            {
                float c_val = 0;
                for (int a_b = 0; a_b < shape_a_second; a_b++)
                {
                    c_val += *(a + a_i * shape_a_second + a_b) * *(b + a_b * shape_b_second + b_j);
                }
                *(c + a_i * shape_b_second + b_j) = c_val;
            }
        }
        return c;
    }

    __device__ void mm(float *a, float *b, float *c, size_t shape_a_first, size_t shape_a_second, size_t shape_b_first, size_t shape_b_second)
    {
        if (shape_a_second != shape_b_first)
        {
            return;
        }

        for (int b_j = 0; b_j < shape_b_second; b_j++)
        {
            for (int a_i = 0; a_i < shape_a_first; a_i++)
            {
                float c_val = 0;
                for (int a_b = 0; a_b < shape_a_second; a_b++)
                {
                    c_val += *(a + a_i * shape_a_second + a_b) * *(b + a_b * shape_b_second + b_j);
                }
                *(c + a_i * shape_b_second + b_j) = c_val;
            }
        }
    }
    __device__ void mm_euclidean_distance(float *a, float *b, float *c, size_t shape_first, size_t shape_second)
    {

        for (int i = 0; i < shape_first; i++)
        {
            for (int j = 0; j < shape_second; j++)
            {
                *(c + i * shape_second + j) = pow(*(a + i * shape_second + j) - *(b + i * shape_second + j), 2);
            }
        }
    }
    __device__ float *mm_euclidean_distance(float *a, float *b, size_t shape_first, size_t shape_second)
    {

        float *c = new float[shape_first * shape_second];
        for (int i = 0; i < shape_first; i++)
        {
            for (int j = 0; j < shape_second; j++)
            {
                *(c + i * shape_second + j) = pow(*(a + i * shape_second + j) - *(b + i * shape_second + j), 2);
            }
        }
        return c;
    }

    __device__ float euclidean_distance(char *attr_l, char *attr_r, int length_)
    {
        size_t i = 0;
        float *attr_l_f, *attr_r_f;
        attr_l_f = (float *)attr_l;
        attr_r_f = (float *)attr_r;
        float dis = 0;
        for (i = 0; i < length_; i++)
        {
            dis += (attr_l_f[i] - attr_r_f[i]) * (attr_l_f[i] - attr_r_f[i]);
        }
        return dis;
    }

    __device__ bool eq(char *attr_l, char *attr_r)
    {
        size_t i = 0;
        size_t len_r = 0, len_l = 0;
        while (*(attr_l + i) != '\0' && *(attr_r + i) != '\0')
        {
            i++;
            len_l++;
            len_r++;
            if (*(attr_l + i) != *(attr_r + i))
            {
                return false;
            }
        }
        while (*(attr_l + len_l) != '\0' && len_l < 256)
        {
            len_l++;
        }
        while (*(attr_r + len_r) != '\0' && len_r < 256)
        {
            len_r++;
        }
        if (len_l != len_r)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    __device__ double lev_jaro_ratio(size_t len1, const char *string1,
                                     size_t len2, const char *string2, int *tag)
    {
        size_t i, j, halflen, trans, match, to;
        //    size_t *idx;
        double md;
        if (len1 == 0 || len2 == 0)
        {
            if (len1 == 0 && len2 == 0)
                return 1.0;
            return 0.0;
        }
        if (len1 > len2)
        {
            const char *b;

            b = string1;
            string1 = string2;
            string2 = b;

            i = len1;
            len1 = len2;
            len2 = i;
        }

        halflen = (len1 + 1) / 2;
        size_t *idx = (size_t *)malloc(sizeof(size_t) * len1);
        memset(idx, 0, sizeof(size_t) * len1);
        double result = 0;
        if (!idx)
            result = -1.0;

        match = 0;
        // if (tag == 0)
        // {
        //     return 0;
        // }
        for (i = 0; i < halflen; i++)
        {
            for (j = 0; j <= i + halflen; j++)
            {
                if (string1[j] == string2[i] && !idx[j])
                {
                    match++;
                    idx[j] = match;
                    break;
                }
            }
        }
        to = len1 + halflen < len2 ? len1 + halflen : len2;
        //  if (tag == 0)
        //  {
        //      return 0;
        //  }
        for (i = halflen; i < to; i++)
        {
            for (j = i - halflen; j < len1; j++)
            {
                if (string1[j] == string2[i] && !idx[j])
                {
                    match++;
                    idx[j] = match;
                    break;
                }
            }
        }
        // if (tag == 0)
        // {
        //     return 0;
        // }
        if (!match)
        {
            result = 0.0;
        }
        else
        {
            i = 0;
            trans = 0;
            for (j = 0; j < len1; j++)
            {
                if (idx[j])
                {
                    i++;
                    if (idx[j] != i)
                        trans++;
                }
            }
            md = (double)match + 1;
            //double sim_ = (md / *len1 + md / *len2 + 1.0 - trans / md / 2.0) / 3.0;
            result = (md / len1 + md / len2 + 1.0) / 3.0;
        }
        free(idx);
        return (double)result;
    }
    /*
__device__ double lev_jaro_winkler_ratio(size_t len1, const char *string1,
        size_t len2, const char *string2,
        double pfweight)
{
    double j;
    size_t p, m;

    j = lev_jaro_ratio(len1, string1, len2, string2);
    m = len1 < len2 ? len1 : len2;
    for (p = 0; p < m; p++)
    {
        if (string1[p] != string2[p])
            break;
    }
    j += (1.0 - j) * p * pfweight;
    return j > 1.0 ? 1.0 : j;
}
*/
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
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        pair<int, int> d_ML_mm_shape,
        const size_t tuple_size,
        float *d_sim,
        int *offset)
    {
        /*
    * Clearn-Clearn ER.
    */
        size_t bid = blockIdx.x;
        size_t tid_in_block = blockIdx.y * blockDim.x + threadIdx.x;
        size_t tid_global = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
        size_t p_aid = threadIdx.y;
        size_t original_tid = tid_global;

        char *tuple_l, *tuple_r;
        size_t start_point_ = 0, start_point_eid = d_attr_start_point[0], attr_len_eid = d_attr_len_arr[0];
        int num_match = 0;
        float sim_ = 0, threshold = 0;
        int len_attr_val_l = 0;
        int len_attr_val_r = 0;
        size_t aid = d_rule_aid[p_aid];
        int tmp_tag = 0;
        extern __shared__ int shared_tag[];
        char attr_val_l[1024];
        char attr_val_r[1024];
        float *output = new float;
        float *euclidean_distance = new float[64 * 1];
        while (tid_global < data_size_l)
        {
            tuple_l = (d_tuples_l + tid_global * line_size);
            memcpy(attr_val_l, (tuple_l + d_attr_start_point[aid]), d_attr_len_arr[aid]);

            for (int i = 0; i < data_size_r; i++)
            {
                if (0)
                {
                    continue;
                }
                else
                {
                    if (p_aid == 0)
                    {
                        *(shared_tag + original_tid) = 1;
                    }

                    d_sim[p_aid + 30] = p_aid;
                    __syncthreads();
                    tuple_r = (d_tuples_r + i * line_size);
                    memcpy(attr_val_r, (tuple_r + d_attr_start_point[aid]), d_attr_len_arr[aid]);
                    if (d_rule_threshold[p_aid] == 1.0)
                    {
                        tmp_tag = (int)eq(attr_val_l, attr_val_r);
                        d_sim[p_aid + 40] = tmp_tag + 40;
                        d_sim[p_aid + 50] = tmp_tag + 50;
                        atomicAnd((shared_tag + original_tid), tmp_tag);
                    }
                    __syncthreads();
                    if (*(shared_tag + original_tid))
                    {
                        if (aid == *d_ML_aid)
                        {
                            mm_euclidean_distance((float *)attr_val_l, (float *)attr_val_r, euclidean_distance, 64, 1);

                            relu(euclidean_distance, 64, 0);
                            mm(euclidean_distance, d_ML_mm, output, 1, 64, 64, 1);
                            *output += bias;
                            tmp_tag = *output > d_rule_threshold[p_aid] ? 1 : 0;
                        }

                        else
                        {
                            len_attr_val_r = (int)cuda_len(attr_val_r);
                            len_attr_val_l = (int)cuda_len(attr_val_l);
                            sim_ = (float)lev_jaro_ratio(len_attr_val_l, attr_val_l, len_attr_val_r, attr_val_r, shared_tag);
                            tmp_tag = sim_ > d_rule_threshold[p_aid] ? 1 : 0;
                        }

                        atomicAnd((shared_tag + original_tid), tmp_tag);
                    }

                    __syncthreads();
                    if (p_aid == 0 && shared_tag[original_tid] == true)
                    {
                        int local_offset = atomicAdd(offset, 1);
                        memcpy((d_match + ((local_offset)) * attr_len_eid * 2), (tuple_l + start_point_eid), attr_len_eid);
                        memcpy((d_match + ((local_offset)) * attr_len_eid * 2 + attr_len_eid), (tuple_r + start_point_eid), attr_len_eid);
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            tid_global += gridDim.y * gridDim.x * blockDim.x;
        }
        //d_sim[49] = 50;
    }
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
        int *offset)
    {
        /*
    * Clearn-Clearn ER.
    */
        size_t bid = blockIdx.x;
        size_t tid_in_block = blockIdx.y * blockDim.x + threadIdx.x;
        size_t tid_global = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
        size_t p_aid = threadIdx.y;
        size_t original_tid = tid_global;

        char *tuple_l, *tuple_r;
        size_t start_point_ = 0, start_point_eid = d_attr_start_point[0], attr_len_eid = d_attr_len_arr[0];
        int num_match = 0;
        float sim_ = 0, threshold = 0;
        int len_attr_val_l = 0;
        int len_attr_val_r = 0;
        size_t aid = d_rule_aid[p_aid];
        bool tmp_tag = 0;
        extern __shared__ int shared_tag[];
        //__shared__ int shared_tag_[4096];
        char attr_val_l[128];
        char attr_val_r[128];
        int local_offset = 0;

        while (tid_global < data_size_l)
        {
            tuple_l = (d_tuples_l + tid_global * line_size);
            memcpy(attr_val_l, (tuple_l + d_attr_start_point[aid]), d_attr_len_arr[aid]);
            for (int i = 0; i < data_size_r; i++)
            {
                if (p_aid == 0)
                {
                    *(shared_tag + original_tid) = true;
                }
                __syncthreads();
                tuple_r = (d_tuples_r + i * line_size);
                memcpy(attr_val_r, (tuple_r + d_attr_start_point[aid]), d_attr_len_arr[aid]);
                if (d_rule_threshold[p_aid] == 1.0)
                {

                    tmp_tag = eq(attr_val_l, attr_val_r);
                    atomicAnd((shared_tag + original_tid), tmp_tag);
                }
                __syncthreads();
                if (*(shared_tag + original_tid) && d_rule_threshold[p_aid] != 1.0 && p_aid == 1)
                {
                    len_attr_val_r = (int)cuda_len(attr_val_r);
                    len_attr_val_l = (int)cuda_len(attr_val_l);
                    float sim_ = 0;
                    sim_ = (float)lev_jaro_ratio(len_attr_val_l, attr_val_l, len_attr_val_r, attr_val_r, shared_tag);
                    tmp_tag = sim_ > d_rule_threshold[p_aid] ? true : false;
                    atomicAnd((shared_tag + original_tid), tmp_tag);
                }

                if (p_aid == 0 && shared_tag[original_tid] == true)
                {
                    int local_offset = atomicAdd(offset, 1);
                    memcpy((d_match + ((local_offset)) * 64 * 2), (tuple_l + start_point_eid), attr_len_eid);
                    memcpy((d_match + ((local_offset)) * 64 * 2 + 64), (tuple_r + start_point_eid), attr_len_eid);
                }
                else
                {
                    continue;
                }
            }
            tid_global += gridDim.y * gridDim.x * blockDim.x;
        }

        //*offset = 1000;
        //*offset = 11;
    }
    __global__ void dirtyER(
        char *d_tuples,
        char *d_match,
        const size_t line_size,
        const size_t data_size,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        int *offset)
    {
        /*
    * Dirty ER. with blocking
    */
        size_t bid = blockIdx.x;
        size_t tid_in_block = blockIdx.y * blockDim.x + threadIdx.x;
        size_t tid_global = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
        size_t p_aid = threadIdx.y;
        size_t original_tid = tid_global;
        clock_t start_point, end_point;

        char *tuple_l, *tuple_r;
        size_t start_point_ = 0, start_point_eid = d_attr_start_point[0], attr_len_eid = d_attr_len_arr[0];
        int num_match = 0;
        float sim_ = 0, threshold = 0;
        int len_attr_val_l = 0;
        int len_attr_val_r = 0;
        size_t aid = d_rule_aid[p_aid];
        int tmp_tag = 0;
        extern __shared__ int shared_tag[];
        //__shared__ char attr_val_l[1024];
        //__shared__ char attr_val_r[1024];
        char attr_val_l[128];
        char attr_val_r[128];
        int local_offset;

        while (tid_global < data_size)
        {
            tuple_l = (d_tuples + tid_global * line_size);
            memcpy(attr_val_l, (tuple_l + d_attr_start_point[aid]), d_attr_len_arr[aid]);

            for (int i = tid_global; i < data_size; i++)
            {
                if (i == tid_global)
                {
                    continue;
                }
                else
                {
                    if (p_aid == 0)
                    {
                        *(shared_tag + threadIdx.x) = 1;
                    }
                    __syncthreads();
                    tuple_r = (d_tuples + i * line_size);
                    memcpy(attr_val_r, (tuple_r + d_attr_start_point[aid]), d_attr_len_arr[aid]);
                    if (d_rule_threshold[p_aid] == 1.0)
                    {
                        tmp_tag = (int)eq(attr_val_l, attr_val_r);
                        //tmp_tag = (int)eq( (tuple_l + d_attr_start_point[aid]) , (tuple_r + d_attr_start_point[aid]));
                        atomicAnd((shared_tag + threadIdx.x), tmp_tag);
                    }
                    __syncthreads();
                    if (*(shared_tag + threadIdx.x))
                    {
                        len_attr_val_r = (int)cuda_len(attr_val_r);
                        len_attr_val_l = (int)cuda_len(attr_val_l);

                        //sim_ = (float)lev_jaro_ratio(len_attr_val_l, attr_val_l, len_attr_val_r, attr_val_r);
                        sim_ = (float)lev_jaro_ratio(len_attr_val_l, attr_val_l, len_attr_val_r, attr_val_r, (shared_tag));
                        //sim_ = (float)lev_jaro_ratio(len_attr_val_l, (tuple_l + d_attr_start_point[aid]), len_attr_val_r, (tuple_r + d_attr_start_point[aid]), (shared_tag));

                        tmp_tag = sim_ > d_rule_threshold[p_aid] ? 1 : 0;
                        atomicAnd((shared_tag + threadIdx.x), tmp_tag);
                    }

                    __syncthreads();
                    if (p_aid == 0 && shared_tag[threadIdx.x] == true)
                    {
                        int local_offset = atomicAdd(offset, 1);
                        memcpy((d_match + ((local_offset)) * attr_len_eid * 2), (tuple_l + start_point_eid), attr_len_eid);
                        memcpy((d_match + ((local_offset)) * attr_len_eid * 2 + attr_len_eid), (tuple_r + start_point_eid), attr_len_eid);
                        //memcpy((d_match + ((local_offset)) * 64 * 2), (tuple_l + start_point_eid), 64);
                        //memcpy((d_match + ((local_offset)) * 64 * 2 + attr_len_eid), (tuple_r + start_point_eid), 64);
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            tid_global += gridDim.y * gridDim.x * blockDim.x;
        }
        //*offset = 1000;
    }

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
        char *h_match,
        char *d_match,
        int *d_offset,
        int *h_offset,
        std::pair<size_t, size_t> dim_grid,
        std::pair<size_t, size_t> dim_block,
        int size_tag,
        size_t max_match,
        cudaStream_t *stream,
        int *signal_vec,
        int iblock,
        int *signal)
    {

        dim3 block(dim_block.first, dim_block.second);
        dim3 grid(dim_grid.first, dim_grid.second);
        cudaStreamSynchronize(*stream);
        clearnClearnER<<<grid, block, size_tag * sizeof(int)>>>(
            d_tuples_l,
            d_tuples_r,
            d_match,
            line_size,
            data_size_l,
            data_size_r,
            d_rule_aid,
            d_rule_threshold,
            rule_size,
            d_attr_len_arr,
            d_attr_start_point,
            tuple_size,
            d_sim,
            d_offset);

        *signal = iblock;
        signal_vec[iblock] = 1;
        //*h_offset = 1;
        // cudaStreamSynchronize(*stream);
        // cudaMemcpy(h_offset, d_offset, sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "###d_offset: " << *h_offset << "####"<<endl;
        // cudaMemcpyAsync(h_offset, d_offset, sizeof(int), cudaMemcpyDeviceToHost, *stream);
        // cudaMemcpyAsync(h_sim, d_sim, sizeof(float) * 64, cudaMemcpyDeviceToHost, *stream);
        // cudaMemcpyAsync(h_match, d_match, sizeof(char) * 2 * 64 * max_match, cudaMemcpyDeviceToHost, *stream);
    }

    __host__ void dirtyHyper(
        char *d_tuples,
        int line_size,
        size_t data_size,
        size_t *d_rule_aid,
        float *d_rule_threshold,
        const size_t rule_size,
        size_t *d_attr_len_arr,
        size_t *d_attr_start_point,
        const size_t tuple_size,
        float *d_sim,
        float *h_sim,
        char *h_match,
        char *d_match,
        int *d_offset,
        int *h_offset,
        std::pair<size_t, size_t> dim_grid,
        std::pair<size_t, size_t> dim_block,
        int size_tag,
        size_t max_match,
        cudaStream_t *stream,
        int *signal_vec,
        int iblock,
        int *signal)
    {

        dim3 block(dim_block.first, dim_block.second);
        dim3 grid(dim_grid.first, dim_grid.second);
        cudaStreamSynchronize(*stream);
        //TEST<<<grid, block, size_tag * sizeof(int), *stream>>>(d_offset);

        //dirtyER<<<grid, block, size_tag * sizeof(int), *stream>>>(
        dirtyER<<<grid, block, size_tag * sizeof(int)>>>(
            d_tuples,
            d_match,
            line_size,
            data_size,
            d_rule_aid,
            d_rule_threshold,
            rule_size,
            d_attr_len_arr,
            d_attr_start_point,
            tuple_size,
            d_sim,
            d_offset);

        *signal = iblock;
        signal_vec[iblock] = 1;
    }
} //namespace Hyper