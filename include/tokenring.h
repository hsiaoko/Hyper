#ifndef CUER_TOKENRING_H_
#define CUER_ROKENRING_H_
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cuda_runtime.h>
#include <ctime>
#include <unistd.h>
#include <cmath>
#include "data_loader.h"
#include "core.h"
#include <stdio.h>
#include <time.h>
#include "matcher.h"
#include "tools.h"
#include "ML.h"
#include <mpi.h>
#include <condition_variable>

using namespace std;
namespace CUER
{
    class TokenRing
    {
    private:
        int ngpus;
        int init_ngpus;
        int dilation_threshold;
        int nstream;
        int line_size;
        int tuple_size;
        size_t *attr_start_point;
        std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l;
        std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_r;
        std::vector<size_t> *threads_pool; // the number of reminder avaliable threadss
        std::vector<char *> *d_tuples_vec_l, *d_tuples_vec_r;
        float *h_rule_threshold;
        size_t *d_rule_aid, *d_attr_len_arr, *d_attr_start_point, result_offset;
        float *d_sim, *h_sim, *d_rule_threshold;
        std::pair<size_t, size_t> dim_grid;
        std::pair<size_t, size_t> dim_block;
        size_t rule_size;

    public:
        int nblocking;
        size_t *attr_len_arr;
        size_t pinned_bytes;
        std::vector<char *> *h_tuples_vec_l, *h_tuples_vec_r;
        std::vector<float *> *d_sim_vec, *h_sim_vec;
        std::vector<char *> *d_match_vec;
        std::vector<char *> *h_match_vec;
        std::map<int, cudaStream_t *> *stream_map;
        std::vector<int *> *h_offset_vec;
        std::vector<int *> *d_offset_vec;
        std::condition_variable repo_not_empty;
        std::condition_variable repo_not_full;
        std::vector<int> *threads_cost_vec;

        std::map<size_t, size_t> *iblock_igpu; // iblock, igpu
        std::map<size_t, int> *igpu_threads;   // igpu, threads
        std::map<size_t, size_t> *iblock_cost;
        int *signal_vec;
        int signal;
        std::mutex mtx;
        cudaStream_t *stream;
        int max_match;
        TokenRing(int ngpus,
                  int init_ngpus,
                  std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l,
                  std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_r,
                  int line_size,
                  int tuple_size,
                  size_t rule_size,
                  size_t *attr_len_arr,
                  size_t *attr_start_point,
                  int max_match);

        TokenRing(int ngpus,
                  int init_ngpus,
                  std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l,
                  int line_size,
                  int tuple_size,
                  size_t rule_size,
                  size_t *attr_len_arr,
                  size_t *attr_start_point,
                  int max_match);

        void RunHostProducer(size_t rule_size, size_t *h_rule_aid, float *h_rule_threshold, bool isDirtyER);
        static int hash(int key, int bin_num);
        static int get_next_bin(int bin_id, int bin_num);
    };
} //namespace CUER

#endif //CUER_TOKENRING_H_