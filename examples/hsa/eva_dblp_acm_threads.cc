#include <cuda_runtime.h>
#include <time.h>
#include <string>
#include "../../include/data_loader.h"
#include "../../include/tools.h"
#include "../../include/matcher.h"
#include "../../include/core.h"
#include "../../include/tokenring.h"
#include "rapidcsv.h"
#include <iostream>
#include <sstream>
#include <assert.h>
#include <pthread.h>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <thread>

//#define MAX_NUM_TUPLES 2048
using namespace std;

struct parameter
{
    CUER::TokenRing *tokenring;
    size_t *prule_size;
    size_t *h_rule_aid;
    float *h_rule_threshold;
};

void HostReducer(int c, CUER::TokenRing *tokenring)
{
    std::vector<std::pair<string, string>> *dup = new std::vector<std::pair<string, string>>;
    int nstream = 0;
    std::map<int, cudaStream_t *>::iterator iter = tokenring->stream_map->end();

    while (tokenring->signal < tokenring->nblocking-1)
    {
        sleep(0.3);
        continue;
    }

    int *stream_flag = (int *)malloc(sizeof(int) * tokenring->nblocking);
    while (nstream < tokenring->nblocking)
    {
        for (iter = tokenring->stream_map->begin(); iter != tokenring->stream_map->end(); iter++)
        {
            if (tokenring->signal_vec[iter->first] && stream_flag[iter->first] != 1)
            {
                cudaError_t err = cudaStreamQuery(*iter->second);
                if (err == cudaSuccess)
                {
                    //cout << "start for()" << iter->first << "signal: " << tokenring->signal_vec[iter->first] << "err: " << err << endl;
                    cudaMemcpyAsync(tokenring->h_offset_vec->at(iter->first), tokenring->d_offset_vec->at(iter->first), sizeof(int), cudaMemcpyDeviceToHost, *iter->second);
                    cudaMemcpyAsync(tokenring->h_match_vec->at(iter->first), tokenring->d_match_vec->at(iter->first), sizeof(char) * 64 * 2 * tokenring->max_match, cudaMemcpyDeviceToHost, *iter->second);

                    stream_flag[iter->first] = 1;
                    size_t cost = tokenring->iblock_cost->find(iter->first)->second;
                    size_t igpu = tokenring->iblock_igpu->find(iter->first)->second;
                    size_t reminder_threads = tokenring->igpu_threads->find(igpu)->second;
                    tokenring->igpu_threads->find(igpu)->second += cost;
                    nstream++;
                }
                else
                {
                    sleep(0.1);
                }
            }
        }
    }
    cudaError_t err = cudaGetLastError();
    //cout << cudaGetErrorString(err) << ":" << err << endl;
    cudaDeviceSynchronize();
    int total_offset = 0;
    for (int i = 0; i < tokenring->h_offset_vec->size(); i++)
    {
        total_offset += *tokenring->h_offset_vec->at(i);
        cout<<"num_matchs: "<<total_offset<<endl;
        for (int j = 0; j < *tokenring->h_offset_vec->at(i); j++)
        {
            dup->push_back(make_pair((tokenring->h_match_vec->at(i) + 2 * j * 64), (tokenring->h_match_vec->at(i) + 2 * j * 64 + 64)));
            //cout << "match:" << (tokenring->h_match_vec->at(i) + 2 * j * 64) << "," << (tokenring->h_match_vec->at(i) + 2 * j * 64 + 64) << endl;
        }
    }
    cout<<dup->size()<<endl;
    for(auto iter:*dup) {
        cout<<iter.first<<", "<<iter.second<<endl;
    }
    std::string gt_path = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/origin/DBLP-ACM_perfectMapping.csv";
    //CUER::getF1(gt_path, dup);
}

int main(int argc, char *argv[])
{
    clock_t start, finish;
    start = clock();

    int rank, nprocs, size, source, namelen;
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int ngpus;
    int num_blocking = 7;
    size_t max_match = 1000;
    rank = 0;
    cudaGetDeviceCount(&ngpus);
    for (int i = 0; i < ngpus; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d, has compute capability %d, %d.\n", i, prop.major, prop.minor);
    }
    int init_ngpus = 1;
    CUER::CSVLoader *csv_l = new CUER::CSVLoader("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/origin/DBLP2.csv");
    CUER::CSVLoader *csv_r = new CUER::CSVLoader("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/origin/ACM.csv");
    //CUER::CSVLoader *csv_l = new CUER::CSVLoader("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/imdb_omdb/filtered/imdb_filtered_mini.csv");
    //CUER::CSVLoader *csv_r = new CUER::CSVLoader("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/imdb_omdb/filtered/omdb_filtered_mini.csv");

    std::vector<string> *head = new std::vector<string>;
    head->push_back("1");
    head->push_back("2");
    head->push_back("3");
    head->push_back("4");
    head->push_back("5");
    csv_l->read2(false, ',', head);
    csv_r->read2(false, ',', head);
    int data_size = csv_l->tuples_->size() + csv_r->tuples_->size();

    vector<pair<size_t, size_t>> *bid = new vector<pair<size_t, size_t>>;
    bid->push_back(make_pair(4, 0));
    bid->push_back(make_pair(1, 3));

    csv_l->blocking(num_blocking, bid);
    csv_r->blocking(num_blocking, bid);

    int max_cmp = INT_MIN, min_cmp = INT_MAX, cmp, sum_cmp = 0;

    sum_cmp = (sum_cmp / num_blocking);

    printf("avg: %d, min: %d, max: %d\n", sum_cmp, min_cmp, max_cmp);
    size_t tuple_size = 5;
    size_t *attr_len_arr = (size_t *)malloc(sizeof(size_t) * tuple_size);
    size_t *attr_start_point = (size_t *)malloc(sizeof(size_t) * tuple_size);
    attr_len_arr[0] = 64;
    attr_len_arr[1] = 64;
    attr_len_arr[2] = 64;
    attr_len_arr[3] = 64;
    attr_len_arr[4] = 8;

    for (int i = 0; i < tuple_size; i++)
    {
        int index = 0;
        for (int j = 0; j < i; j++)
        {
            index += attr_len_arr[j];
        }
        attr_start_point[i] = index;
    }

    int line_size = 0;
    for (int i = 0; i < tuple_size; i++)
    {
        line_size += attr_len_arr[i];
    }

    size_t *h_rule_aid;
    float *h_rule_threshold;
    size_t rule_size = 3;
    h_rule_aid = (size_t *)malloc(sizeof(size_t) * rule_size);
    h_rule_threshold = (float *)malloc(sizeof(float) * rule_size);

    h_rule_aid[0] = 4;
    h_rule_threshold[0] = 1.0;
    h_rule_aid[1] = 1;
    h_rule_threshold[1] = 0.9;
    h_rule_aid[2] = 2;
    h_rule_threshold[2] = 0.9;
    CUER::TokenRing *tokenring = new CUER::TokenRing(
        ngpus,
        init_ngpus,
        csv_l->blocking_vec_,
        csv_r->blocking_vec_,
        line_size,
        tuple_size,
        rule_size,
        attr_len_arr,
        attr_start_point,
        max_match);

    cout << "h_match_vec: " << tokenring->h_match_vec->size() << endl;

    std::thread t(HostReducer, 3, tokenring); // 3

    tokenring->RunHostProducer(rule_size, h_rule_aid, h_rule_threshold, false);
    t.join(); //

    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("response time %f seconds\n", duration);

    return 0;
}