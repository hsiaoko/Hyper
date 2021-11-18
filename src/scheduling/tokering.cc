#include "../../include/tokenring.h"
using namespace std;
namespace CUER
{

    TokenRing::TokenRing(int ngpus,
                         int init_ngpus,
                         std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l,
                         std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_r,
                         int line_size,
                         int tuple_size,
                         size_t rule_size,
                         size_t *attr_len_arr,
                         size_t *attr_start_point,
                         int max_match)
    {
        this->pinned_bytes = 0;
        this->ngpus = ngpus;
        this->init_ngpus = init_ngpus;
        this->blocking_vec_l = blocking_vec_l;
        this->blocking_vec_r = blocking_vec_r;
        this->nblocking = blocking_vec_l->size();
        this->nstream = 0;
        this->tuple_size = tuple_size;
        this->line_size = line_size;
        this->attr_len_arr = attr_len_arr;
        this->rule_size = rule_size;
        this->max_match = max_match;
        this->attr_start_point = attr_start_point;
        this->stream_map = new std::map<int, cudaStream_t *>;
        this->d_match_vec = new std::vector<char *>;
        this->h_match_vec = new std::vector<char *>;
        this->d_offset_vec = new std::vector<int *>,
        this->h_offset_vec = new std::vector<int *>;
        this->d_tuples_vec_l = new std::vector<char *>;
        this->d_tuples_vec_r = new std::vector<char *>;
        this->h_tuples_vec_l = new std::vector<char *>;
        this->h_tuples_vec_r = new std::vector<char *>;
        this->d_sim_vec = new std::vector<float *>;
        this->h_sim_vec = new std::vector<float *>;
        this->iblock_igpu = new std::map<size_t, size_t>;
        this->igpu_threads = new std::map<size_t, int>;
        this->iblock_cost = new std::map<size_t, size_t>;
        this->signal = 0;
        this->signal_vec = (int *)malloc(sizeof(int) * this->nblocking);
        for (int i = 0; i < this->nblocking; i++)
        {
            float *h_sim;
            int *h_offset;
            cudaHostAlloc(&h_offset, sizeof(int), cudaHostAllocDefault);
            this->h_offset_vec->push_back(h_offset);
            *h_offset = 0;

            cudaHostAlloc(&h_sim, sizeof(float) * 64, cudaHostAllocDefault);
            this->h_sim_vec->push_back(h_sim);
            char *h_match;
            cudaHostAlloc(&h_match, sizeof(char) * 64 * 2 * this->max_match, cudaHostAllocDefault);
            this->h_match_vec->push_back(h_match);
        }
        cout << "STREAM NUM:" << this->stream_map->size() << endl;
        for (int i = 0; i < this->ngpus; i++)
        {
            this->igpu_threads->insert(make_pair(i, 5120 * 32));
        }
        cout << "Tokenring()" << endl;
    }

    TokenRing::TokenRing(int ngpus,
                         int init_ngpus,
                         std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l,
                         int line_size,
                         int tuple_size,
                         size_t rule_size,
                         size_t *attr_len_arr,
                         size_t *attr_start_point,
                         int max_match)
    {
        this->pinned_bytes = 0;
        this->ngpus = ngpus;
        this->init_ngpus = init_ngpus;
        this->blocking_vec_l = blocking_vec_l;
        this->nblocking = blocking_vec_l->size();
        this->nstream = 0;
        this->tuple_size = tuple_size;
        this->line_size = line_size;
        this->attr_len_arr = attr_len_arr;
        this->rule_size = rule_size;
        this->max_match = max_match;
        this->attr_start_point = attr_start_point;
        this->stream_map = new std::map<int, cudaStream_t *>;
        this->d_match_vec = new std::vector<char *>;
        this->h_match_vec = new std::vector<char *>;
        this->d_offset_vec = new std::vector<int *>,
        this->h_offset_vec = new std::vector<int *>;
        this->d_tuples_vec_l = new std::vector<char *>;
        this->d_tuples_vec_r = new std::vector<char *>;
        this->h_tuples_vec_l = new std::vector<char *>;
        this->h_tuples_vec_r = new std::vector<char *>;
        this->d_sim_vec = new std::vector<float *>;
        this->h_sim_vec = new std::vector<float *>;
        this->iblock_igpu = new std::map<size_t, size_t>;
        this->igpu_threads = new std::map<size_t, int>;
        this->iblock_cost = new std::map<size_t, size_t>;
        this->signal = 0;
        this->signal_vec = (int *)malloc(sizeof(int) * this->nblocking);
        for (int i = 0; i < this->nblocking; i++)
        {
            float *h_sim;
            int *h_offset;
            cudaHostAlloc(&h_offset, sizeof(int), cudaHostAllocDefault);
            this->h_offset_vec->push_back(h_offset);
            *h_offset = 0;

            cudaHostAlloc(&h_sim, sizeof(float) * 64, cudaHostAllocDefault);
            this->h_sim_vec->push_back(h_sim);
            char *h_match;
            cudaHostAlloc(&h_match, sizeof(char) * 64 * 2 * this->max_match, cudaHostAllocDefault);
            this->h_match_vec->push_back(h_match);
        }
        //cout << "STREAM NUM:" << this->stream_map->size() << endl;
        for (int i = 0; i < this->ngpus; i++)
        {
            this->igpu_threads->insert(make_pair(i, 5120 * 32));
        }
    }

    void TokenRing::RunHostProducer(size_t rule_size, size_t *h_rule_aid, float *h_rule_threshold, bool isDirtyER)
    {
        clock_t stream_finish, stream_start;
        char *h_tuples_l, *h_tuples_r, *d_tuples_l, *d_tuples_r, *d_match;
        int iblock = 0;
        size_t *d_rule_aid, *d_attr_len_arr, *d_attr_start_point, result_offset;
        float *d_rule_threshold;
        for (int i = 0; i < rule_size; i++)
        {
            cout << "|" << h_rule_aid[i] << "-" << h_rule_threshold[i] << "|" << endl;
        }
        while (iblock < this->nblocking)
        {
            cout << "this->pinned_bytes:" << this->pinned_bytes << endl;
            /*
            * Token Ring
            */
            bool tag = true;
            int *d_offset, *h_offset;
            int igpu = 0;

            // while (tag)
            // {
            int threads_cost;
            if (isDirtyER == true)
            {
                threads_cost = this->blocking_vec_l->at(iblock)->size() * this->blocking_vec_l->at(iblock)->size();
            }
            else
            {
                threads_cost = this->blocking_vec_l->at(iblock)->size() > this->blocking_vec_r->at(iblock)->size() ? this->blocking_vec_l->at(iblock)->size() * this->blocking_vec_l->at(iblock)->size() : this->blocking_vec_r->at(iblock)->size() * this->blocking_vec_r->at(iblock)->size();
            }

            igpu = this->hash(iblock, this->init_ngpus);
            //cout << "GPU: " << igpu << endl;
            int count = 0, max_threads = INT_MIN, max_gpu = -1;
            while (this->igpu_threads->find(igpu)->second < threads_cost)
            {
                igpu = this->get_next_bin(igpu, this->init_ngpus);
                count++;
                if (max_threads < this->igpu_threads->find(igpu)->second)
                {
                    max_gpu = igpu;
                    max_threads = this->igpu_threads->find(igpu)->second;
                }
                if (count > (this->init_ngpus))
                {
                    if (max_threads > 0 && init_ngpus < ngpus)
                    {
                        cout << "!!!ADD GPU!!!" << this->init_ngpus << endl;
                        this->init_ngpus++;
                    }
                    else
                    {
                        igpu = max_gpu;
                        break;
                    }
                }
            }

            int reminder_threads = this->igpu_threads->find(igpu)->second - threads_cost;
            this->iblock_igpu->insert(make_pair(iblock, igpu));
            this->iblock_cost->insert(make_pair(iblock, threads_cost));
            this->igpu_threads->insert(make_pair(igpu, reminder_threads));
            this->igpu_threads->find(igpu)->second = reminder_threads;

            cout << "iblock: " << iblock << ", "
                 << "igpu: " << igpu << ", "
                 << "cost: " << threads_cost << ", "
                 << "reminder: " << reminder_threads << endl;

            cudaSetDevice(igpu);
            //cudaDeviceReset(igpu);

            cudaStream_t *pstream = new cudaStream_t;
            cudaStreamCreate(pstream);
            this->stream_map->insert(make_pair(iblock, pstream));

            stream_start = clock();
            stream_finish = clock();
            double stream_duration = (double)(stream_finish - stream_start) / CLOCKS_PER_SEC;
            printf("stream time %f seconds\n", stream_duration);
            //cout << "igpu:" << igpu << endl;
            std::map<int, cudaStream_t *>::iterator iter;
            iter = this->stream_map->find(iblock);
            if (isDirtyER == false)
            {

                size_t nBytes_l = sizeof(char) * this->blocking_vec_l->at(iblock)->size() * this->line_size;
                size_t nBytes_r = sizeof(char) * this->blocking_vec_r->at(iblock)->size() * this->line_size;

                iter = this->stream_map->find(iblock);
                //cout << "MAIN " << iblock << " : blocksize::" << this->blocking_vec_l->at(iblock)->size() << this->blocking_vec_l->at(iblock)->size() << "iter->first: " << iter->first << endl;
                cudaError_t error_t_l = cudaHostAlloc(&h_tuples_l, nBytes_l, cudaHostAllocPortable);
                cudaError_t error_t_r = cudaHostAlloc(&h_tuples_r, nBytes_r, cudaHostAllocPortable);
                cout << "error_t: " << error_t_r << ", " << error_t_l << ", " << nBytes_l << ", " << nBytes_r << endl;

                //cudaError_t error_t_l = cudaHostAlloc(&h_tuples_l, nBytes_l, cudaHostAllocDefault);
                //cudaError_t error_t_r = cudaHostAlloc(&h_tuples_r, nBytes_r, cudaHostAllocDefault);
                this->pinned_bytes += nBytes_l;
                this->pinned_bytes += nBytes_r;
                cout << "this->pinned_bytes:" << this->pinned_bytes << endl;
                //cudaFree(h_tuples_l);
                //cudaFree(h_tuples_r);

                //if (error_t_l != cudaSuccess && error_t_r != cudaSuccess)
                //{
                //    sleep(0.1);
                //    error_t_l = cudaHostAlloc(&h_tuples_l, nBytes_l, cudaHostAllocDefault);
                //    error_t_r = cudaHostAlloc(&h_tuples_r, nBytes_r, cudaHostAllocDefault);
                //    cout << "ERRPR!!!!" << iblock <<endl;
                //    iblock++;
                //    continue;
                //}
                cudaMalloc((void **)&d_tuples_l, nBytes_l);
                cudaMalloc((void **)&d_tuples_r, nBytes_r);
                CUER::CSVLoader::tuples2arr(this->blocking_vec_l->at(iblock), this->attr_len_arr, this->attr_start_point, this->line_size, h_tuples_l);
                CUER::CSVLoader::tuples2arr(this->blocking_vec_r->at(iblock), this->attr_len_arr, this->attr_start_point, this->line_size, h_tuples_r);
                this->h_tuples_vec_l->push_back(h_tuples_l);
                this->h_tuples_vec_r->push_back(h_tuples_r);
                this->d_tuples_vec_l->push_back(d_tuples_l);
                this->d_tuples_vec_r->push_back(d_tuples_r);
                cudaError_t err_match = cudaMalloc((void **)&d_match, sizeof(char) * 64 * 2 * this->max_match);
                this->d_match_vec->push_back(d_match);

                cudaMemcpyAsync(d_tuples_l, h_tuples_l, nBytes_l, cudaMemcpyHostToDevice, *iter->second);
                cudaMemcpyAsync(d_tuples_r, h_tuples_r, nBytes_r, cudaMemcpyHostToDevice, *iter->second);

                //cudaStreamSynchronize(*iter->second);
                int grid_num = threads_cost / (rule_size * 8 * 16) > 16 ? 16 : threads_cost / (rule_size * 8 * 16);
                std::pair<size_t, size_t> dim_grid = make_pair(grid_num, 16);
                //std::pair<size_t, size_t> dim_grid = make_pair(16, 16);
                std::pair<size_t, size_t> dim_block = make_pair(16, rule_size);

                dim3 block(dim_block.first, dim_block.second);
                dim3 grid(dim_grid.first, dim_grid.second);
                int size_tag = dim_grid.first * dim_grid.second * dim_block.first;
                cudaError_t err = cudaGetLastError();
                cout << cudaGetErrorString(err) << ":" << err << endl;
                cout << "grid: " << dim_grid.first << ", " << dim_grid.second << ", dim: " << dim_block.first << ", " << dim_block.second << "share_mem" << size_tag << endl;

                cudaMalloc((void **)&d_attr_len_arr, sizeof(size_t) * this->tuple_size);
                cudaMalloc((void **)&d_attr_start_point, sizeof(size_t) * this->tuple_size);

                //cudaMemcpyAsync(d_attr_len_arr, this->attr_len_arr, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice, *iter->second);
                //cudaMemcpyAsync(d_attr_start_point, this->attr_start_point, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice, *iter->second);
                cudaMemcpy(d_attr_len_arr, this->attr_len_arr, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_attr_start_point, this->attr_start_point, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice);

                cudaMalloc((void **)&d_rule_threshold, sizeof(float) * this->rule_size);
                cudaMalloc((void **)&d_rule_aid, sizeof(size_t) * this->rule_size);

                cudaMemcpyAsync(d_rule_threshold, h_rule_threshold, sizeof(float) * rule_size, cudaMemcpyHostToDevice, *iter->second);
                cudaMemcpyAsync(d_rule_aid, h_rule_aid, sizeof(size_t) * rule_size, cudaMemcpyHostToDevice, *iter->second);
                cudaMalloc((void **)&d_sim, sizeof(float) * 64);

                cudaMalloc((void **)&d_offset, sizeof(int));
                //initialData((float*)d_offset, sizeof(int));
                this->d_offset_vec->push_back(d_offset);
                cout << "-------" << endl;
                CUER::clearnclearnHyper(
                    d_tuples_l,
                    d_tuples_r,
                    this->line_size,
                    blocking_vec_l->at(iblock)->size(),
                    blocking_vec_r->at(iblock)->size(),
                    d_rule_aid,
                    d_rule_threshold,
                    this->rule_size,
                    d_attr_len_arr,
                    d_attr_start_point,
                    this->tuple_size,
                    d_sim,
                    this->h_sim_vec->at(iblock),
                    this->h_match_vec->at(iblock),
                    d_match,
                    this->d_offset_vec->at(iblock),
                    this->h_offset_vec->at(iblock),
                    dim_grid,
                    dim_block,
                    size_tag,
                    this->max_match,
                    iter->second,
                    this->signal_vec,
                    iblock,
                    &this->signal);
                //cudaStreamSynchronize(*iter->second);
                cout << "###" << endl;
                // this->threads_pool->at(igpu) -= (dim_grid.first * dim_grid.second * dim_block.first * dim_block.second);
            }
            else
            {
                size_t nBytes_l = sizeof(char) * this->blocking_vec_l->at(iblock)->size() * this->line_size;

                iter = this->stream_map->find(iblock);
                //cout << "MAIN " << iblock << " : blocksize::" << this->blocking_vec_l->at(iblock)->size() << this->blocking_vec_l->at(iblock)->size() << "iter->first: " << iter->first << endl;
                //               cudaError_t error_t_l = cudaHostAlloc(&h_tuples_l, nBytes_l, cudaHostAllocDefault);
                cudaError_t error_t_l = cudaHostAlloc(&h_tuples_l, nBytes_l, cudaHostAllocPortable);
                cout << "error_t: "
                     << ", " << error_t_l << nBytes_l << ", " << endl;
                cout << cudaGetErrorString(error_t_l) << endl;
                cudaMalloc((void **)&d_tuples_l, nBytes_l);
                //cout << "tuples2array" << nBytes_l << endl;
                CUER::CSVLoader::tuples2arr(this->blocking_vec_l->at(iblock), this->attr_len_arr, this->attr_start_point, this->line_size, h_tuples_l);
                cudaError_t err = cudaGetLastError();
                cout << cudaGetErrorString(err) << ":" << err << endl;
                this->h_tuples_vec_l->push_back(h_tuples_l);
                this->d_tuples_vec_l->push_back(d_tuples_l);
                cudaError_t err_match = cudaMalloc((void **)&d_match, sizeof(char) * 64 * 2 * this->max_match);
                this->d_match_vec->push_back(d_match);

                cudaMemcpyAsync(d_tuples_l, h_tuples_l, nBytes_l, cudaMemcpyHostToDevice, *iter->second);
                int grid_num = threads_cost / (rule_size * 8 * 16) > 16 ? 16 : threads_cost / (rule_size * 8 * 16);
                std::pair<size_t, size_t> dim_grid = make_pair(grid_num + 1, 16);
                //std::pair<size_t, size_t> dim_grid = make_pair(16, 16);
                std::pair<size_t, size_t> dim_block = make_pair(16, rule_size);

                //cout << "grid: " << dim_grid.first << ", " << dim_grid.second << ", dim: " << dim_block.first << ", " << dim_block.second << endl;
                dim3 block(dim_block.first, dim_block.second);
                dim3 grid(dim_grid.first, dim_grid.second);
                int size_tag = dim_grid.first * dim_grid.second * dim_block.first;

                cudaMalloc((void **)&d_attr_len_arr, sizeof(size_t) * this->tuple_size);
                cudaMalloc((void **)&d_attr_start_point, sizeof(size_t) * this->tuple_size);

                //cudaMemcpyAsync(d_attr_len_arr, this->attr_len_arr, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice, *iter->second);
                //cudaMemcpyAsync(d_attr_start_point, this->attr_start_point, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice, *iter->second);
                cudaMemcpy(d_attr_len_arr, this->attr_len_arr, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_attr_start_point, this->attr_start_point, sizeof(size_t) * this->tuple_size, cudaMemcpyHostToDevice);

                cudaMalloc((void **)&d_rule_threshold, sizeof(float) * this->rule_size);
                cudaMalloc((void **)&d_rule_aid, sizeof(size_t) * this->rule_size);

                //cudaMemcpyAsync(d_rule_threshold, h_rule_threshold, sizeof(float) * rule_size, cudaMemcpyHostToDevice, *iter->second);
                //cudaMemcpyAsync(d_rule_aid, h_rule_aid, sizeof(size_t) * rule_size, cudaMemcpyHostToDevice, *iter->second);
                cudaMemcpy(d_rule_threshold, h_rule_threshold, sizeof(float) * rule_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_rule_aid, h_rule_aid, sizeof(size_t) * rule_size, cudaMemcpyHostToDevice);
                cudaMalloc((void **)&d_sim, sizeof(float) * 64);

                cudaMalloc((void **)&d_offset, sizeof(int));
                //initialData((float*)d_offset, sizeof(int));
                this->d_offset_vec->push_back(d_offset);

                CUER::dirtyHyper(
                    d_tuples_l,
                    this->line_size,
                    blocking_vec_l->at(iblock)->size(),
                    d_rule_aid,
                    d_rule_threshold,
                    this->rule_size,
                    d_attr_len_arr,
                    d_attr_start_point,
                    this->tuple_size,
                    d_sim,
                    this->h_sim_vec->at(iblock),
                    this->h_match_vec->at(iblock),
                    d_match,
                    this->d_offset_vec->at(iblock),
                    this->h_offset_vec->at(iblock),
                    dim_grid,
                    dim_block,
                    size_tag,
                    this->max_match,
                    iter->second,
                    this->signal_vec,
                    iblock,
                    &this->signal);
            }
            tag = false;
            iblock++;
        }

        cout << "END inputRsssing" << endl;
    }

    int TokenRing::hash(int key, int bin_num)
    {
        unsigned int seed = 174321;
        unsigned int hash = (key * seed) >> 3;
        return (hash & 0x7FFFFFFF) % bin_num;
    }

    int TokenRing::get_next_bin(int bin_id, int bin_num)
    {
        bin_id++;
        return bin_id % bin_num;
    }

}