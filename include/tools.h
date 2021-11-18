#ifndef CUER_TOOLS_H_
#define CUER_TOOLS_H_
#include <map>
#include <vector>
#include <string.h>
#include <string>
#include <ext/hash_map>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
//#include "../include/rapidcsv.h"
#include "rapidcsv.h"
using namespace __gnu_cxx;
using namespace std;
using namespace rapidcsv;

namespace CUER
{

static void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate); // 输出的是GPU的时钟频率
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

static int hash(const char *str)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*str)
    {
        hash = hash * seed + (*str++);
    }
    return hash & 0x7FFFFFFF;
}

static int hash(const std::string str_)
{
    char *str = (char *)malloc(sizeof(char) * 512);
    memset(str, '\0', sizeof(str));
    strcpy(str, str_.c_str());
    unsigned int seed = 11565371;
    unsigned int hash = 13311221;
    while (*str)
    {
        hash = hash * seed + (*str++);
        hash = (hash + seed) * seed;
    }
    //free(str);
    return hash & 0x7FFFFFFF;
}

static int prefixHash(const std::string str_, size_t num_prefix)
{

    char *str = (char *)malloc(sizeof(char) * num_prefix + 1);
    string str2 = str_.substr(0, num_prefix);
    memset(str, '\0', sizeof(char) * num_prefix + 1);
    strcpy(str, str2.c_str());
    unsigned int seed = 11565371;
    unsigned int hash = 13171;
    while (*str)
    {
        hash = hash * seed + (*str++);
        hash = (hash + seed) * seed;
    }
    return hash & 0x7FFFFFFF;
}

static void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

static char *tuples2vec(std::vector<std::vector<std::string> *> *tuples, std::vector<int> *attr_len_arr)
{
    std::vector<int> *attr_start_point = new std::vector<int>;
    for (int i = 0; i < attr_len_arr->size(); i++)
    {
        int index = 0;
        for (int j = 0; j < i; j++)
        {
            index += attr_len_arr->at(j);
        }
        attr_start_point->push_back(index);
    }

    int line_size = 0;
    for (int i = 0; i < attr_len_arr->size(); i++)
    {
        line_size += attr_len_arr->at(i);
    }
    char *tuples_arr = (char *)malloc(sizeof(char) * line_size * tuples->size());
    memset(tuples_arr, 0, sizeof(char) * line_size * tuples->size());
    for (int i = 0; i < tuples->size(); i++)
    {
        for (int j = 0; j < tuples->at(i)->size(); j++)
        {
            memcpy(((tuples_arr + i * line_size) + attr_start_point->at(j)), tuples->at(i)->at(j).c_str(), tuples->at(i)->at(j).length() + 1);
        }
    }
    // for(int i = 0; i < tuples->size(); i++) {
    //     line = (tuples_arr + i*line_size);
    //     cout<<line+attr_start_point->at(0)<<", "<<line+attr_start_point->at(1)<<", "<<line+attr_start_point->at(2)<<", "<<line+attr_start_point->at(3)<<", "<<line+attr_start_point->at(4)<<", "<<line+attr_start_point->at(5)<<", "<<line+attr_start_point->at(6)<<", "<<line+attr_start_point->at(7)<<endl;
    // }
    return tuples_arr;
}

static char *tuples2vec(std::vector<std::vector<std::string> *> *tuples,
                        size_t *attr_len_arr,
                        size_t *attr_start_point,
                        size_t tuple_size,
                        size_t line_size)
{

    char *tuples_arr = (char *)malloc(sizeof(char) * line_size * tuples->size());
    memset(tuples_arr, 0, sizeof(char) * line_size * tuples->size());
    for (int i = 0; i < tuples->size(); i++)
    {
        for (int j = 0; j < tuples->at(i)->size(); j++)
        {
            memcpy(((tuples_arr + i * line_size) + attr_start_point[j]), tuples->at(i)->at(j).c_str(), tuples->at(i)->at(j).length() + 1);
        }
    }
    return tuples_arr;
}

static void showTuples(std::vector<std::vector<std::string> *> *tuples)
{
    for (int i = 0; i < tuples->size(); i++)
    {
        for (int j = 0; j < tuples->at(i)->size(); j++)
        {
            cout << tuples->at(i)->at(j) << ", ";
        }
        cout << endl;
    }
}

static void showTuple(char *tuple_arr, size_t *attr_start_point, size_t tuple_size)
{
    cout << tuple_arr << endl;
    for (int i = 0; i < tuple_size; i++)
    {
        cout << tuple_arr + attr_start_point[i] << ", ";
    }
}
static void showTuple(char *tuple_arr, std::vector<int> *attr_start_point)
{
    cout << tuple_arr << endl;
    for (int i = 0; i < attr_start_point->size(); i++)
    {
        cout << tuple_arr + attr_start_point->at(i) << ", ";
    }
}
static void showTuples(char *tuples_arr, int tuples_num, std::vector<int> *attr_len_arr)
{
    char *line;
    int line_size = 0;
    for (int i = 0; i < attr_len_arr->size(); i++)
    {
        line_size += attr_len_arr->at(i);
    }
    std::vector<int> *attr_start_point = new std::vector<int>;
    for (int i = 0; i < attr_len_arr->size(); i++)
    {
        int index = 0;
        for (int j = 0; j < i; j++)
        {
            index += attr_len_arr->at(j);
        }
        attr_start_point->push_back(index);
    }

    for (int i = 0; i < tuples_num; i++)
    {
        line = (tuples_arr + i * line_size);
        cout << line + attr_start_point->at(0) << ", " << line + attr_start_point->at(1) << ", " << line + attr_start_point->at(2) << ", " << line + attr_start_point->at(3) << ", " << line + attr_start_point->at(4) << ", " << line + attr_start_point->at(5) << ", " << line + attr_start_point->at(6) << ", " << line + attr_start_point->at(7) << endl;
    }
}

static void showTuples(char *tuples_arr, size_t tuples_num, size_t *attr_start_point, size_t line_size)
{
    char *line;

    for (int i = 0; i < tuples_num; i++)
    {
        line = (tuples_arr + i * line_size);
        cout << line + attr_start_point[0] << ", " << line + attr_start_point[1] << ", " << line + attr_start_point[2] << ", " << line + attr_start_point[3] << ", " << line + attr_start_point[4] << ", " << line + attr_start_point[5] << ", " << line + attr_start_point[6] << ", " << line + attr_start_point[7] << endl;
    }
}

static vector<vector<std::string> *> *read(bool read_head, char split_symbol, std::string csv_path_)
{
    std::ifstream fp(csv_path_);
    vector<vector<std::string> *> *tuples = new vector<vector<std::string> *>;
    string line;
    if (read_head == false)
    {
        getline(fp, line);
    }
    vector<string> *tuple;
    while (getline(fp, line))
    {
        //循环读取每行数据
        tuple = new vector<std::string>;
        const char *mystart = line.c_str();
        //const char * mystart = new char * line.length();
        //memcpy(line.c_str)
        //准备解析行 - 开始是字段开始的位置
        bool instring{false};
        for (const char *p = mystart; *p; p++)
        {
            //遍历字符串
            if (*p == '"')
            {
                //如果我们是btw双引号
                instring = !instring;
            }
            else if (*p == ',' && !instring)
            {
                //如果逗号超出双引号
                tuple->push_back(string(mystart, p - mystart));
                //保持字段
                mystart = p + 1; //并开始解析下一个
            }
        }
        tuple->push_back(string(mystart));
        tuples->push_back(tuple);
    }
    return tuples;
}

// static bool findVec(std::vector<pair<string, string>> * gt, pair<string, string> compare_pair){
//     bool tag = false;
//     for(int i = 0; i < gt->size();i++){
//         cout<<gt->at(i).first<<", "<<gt->at(i).second<<endl;
//     }
//     cout<<"findVec"<<endl;
//     return tag;
// }
static void getF1(std::string gt_path, std::vector<std::pair<string, string>> *dup)
{
    std::map<string, vector<string> *> *gt = new std::map<string, vector<string> *>;
    std::map<string, vector<string> *>::iterator iter;
    vector<vector<std::string> *> *tuples = new vector<vector<std::string> *>;

    rapidcsv::Document doc(gt_path);
    std::vector<string> *col_l = new std::vector<string>;
    std::vector<string> *col_r = new std::vector<string>;
    *(col_l) = doc.GetColumn<string>(0);
    *(col_r) = doc.GetColumn<string>(1);

    for (int i = 0; i < col_l->size(); i++)
    {
        iter = gt->find(col_l->at(i));
        if (iter == gt->end())
        {
            vector<string> *r_vec = new vector<string>;
            r_vec->push_back(col_r->at(i));
            gt->insert(pair<string, vector<string> *>(col_l->at(i), r_vec));
        }
        else
        {
            iter->second->push_back(col_r->at(i));
        }
    }
    size_t tp_ = 0;
    bool tag = false;
    for (int i = 0; i < dup->size(); i++)
    {
        //      tag = false;
        std::string l = dup->at(i).first;
        std::string r = dup->at(i).second;
        cout<< l << ", " << r << endl;
        iter = gt->find(l);
        if (iter != gt->end())
        {
            for (int j = 0; j < iter->second->size(); j++)
            {
                if (iter->second->at(j) == r)
                {
                    tp_++;
                    tag = true;
                }
            }
        }
        iter = gt->find(r);
        if (iter != gt->end())
        {
            for (int j = 0; j < iter->second->size(); j++)
            {
                if (iter->second->at(j) == l)
                {
                    tp_++;
                    tag = true;
                }
            }
        }
        if(!tag) {
            cout<<"error matches: "<< l<<", "<<r<<endl;
        }
    }
    size_t gt_ = col_l->size();
    size_t dup_ = dup->size();
    size_t fp_ = dup_ - tp_;
    size_t fn_ = gt_ - tp_;
    float prec = (float)tp_ / (float)(tp_ + fp_);
    float rec = (float)tp_ / (float)(tp_ + fn_);
    float f1 = 2 * (prec * rec) / (prec + rec);
    printf("dup: %d, gt: %d, tp: %d, fp: %d, fn: %d -> f1: %f, prec: %f, recall: %f\n", dup_, gt_, tp_, fp_, fn_, f1, prec, rec);
}
static void getF1(size_t gt_, std::vector<std::pair<string, string>> *dup)
{
    size_t tp_ = 0;
    for (int i = 0; i < dup->size(); i++)
    {
        if (dup->at(i).first == dup->at(i).first)
        {
            tp_++;
        }
    }
    size_t dup_ = dup->size();
    size_t fp_ = dup_ - tp_;
    size_t fn_ = gt_ - tp_;
    float prec = (float)tp_ / (float)(tp_ + fp_);
    float rec = (float)tp_ / (float)(tp_ + fn_);
    float f1 = 2 * (prec * rec) / (prec + rec);
    printf("dup: %d, gt: %d, tp: %d, fp: %d, fn: %d -> f1: %f, prec: %f, recall: %f\n", dup_, gt_, tp_, fp_, fn_, f1, prec, rec);
}
template <typename T>
void showArray(T &t, int arr_len)
{
    cout << " >>> ";
    for (int i = 0; i < arr_len; i++)
    {
        cout << t[i] << ", ";
    }
    cout << endl;
}

} //end namespace CUER
#endif //CUER_TOOLs_H_
