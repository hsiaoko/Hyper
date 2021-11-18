#ifndef CUER_STRUCT_H_
#define CUER_STRUCT_H_
#include <map>
#include <vector>
#include <string.h>
#include <string>
#include <ext/hash_map>
using namespace __gnu_cxx;
using namespace std;
namespace CUER
{

struct str_hash
{
    size_t operator()(const string &s) const
    {
        return __stl_hash_string(s.c_str());
    }
};

struct str_compare
{
    int operator()(const string &a, const string &b) const
    {
        return (a == b);
    }
};

typedef hash_map<string, int, str_hash, str_compare> StrMap;


struct Message
{
    int relation_id;
    int tuple_size;
    int num_mrls;
    int *mrls_id;
    char *tuple;
};

struct Entity
{
    int relation_id;
    int tuple_size;
    std::map<int, int> *mrls_id;
    std::vector<std::string> *tuple;
};

struct Entity2
{
    int relation_id;

    std::map<int, std::vector<int>* > *mrls_id;
    //mapping node id, vec of mrls_id;

    std::map<int, std::vector<int> *> *worker_id_set;
    //mapping node id, vec of worker id;

    std::string tid;
    std::map<int, int> *mapping_node_id; // <id, -1>
};


struct RelationRAM
{
    std::vector<std::vector<std::string> *> *tuples;
    std::vector<std::string> *schema;
    int num_tuples;
    std::vector<int> *connected_attr_pos_in_tuple;
    std::vector<int> *connected_attr_dim_in_hypercube;
    int num_connected_attr;
    int num_dim_hypercube;
};

struct Block
{
    int *worker_id;
    int block_id;
    int *hypercube_id;
    int num_tuples;
    std::vector<int> *relation_id_set;
    //std::vector<std::vector<std::vector<std::string> *> *> *work_space;
    std::vector<std::vector<std::vector<std::string> *> *> *story_space;
    std::vector<std::vector<std::vector<std::string> *> *>  work_space;
};

struct Block2
{
    int block_id;
    int num_tuples;
// std::vector<std::vector<int>*> * tid_set; // vec to store tid for each relation.
    std::map<int, std::vector<int>*> *tid_set;
    // <relation id, vec of tids>
};

struct MRL
{
    std::string file_name;
    int mrl_id;
    std::vector<std::vector<int> *> *pre_equalities;
    std::vector<std::vector<int> *> *pre_mls;
    std::vector<int> *conseq_equality;
    std::vector<std::vector<int> *> *pre_relations;
    std::vector<std::vector<int> *> *pre_relations_origin;
    std::string conseq_option;
    int num_relations;
    std::vector<int> *relation_id_set;
};

struct RelationInfo
{
    int relation_id;
    std::vector<int> *connected_attr_pos_in_tuple;
    std::vector<int> *connected_attr_dim_in_hypercube;
    std::string relation_path;
    int num_connected_attr;
    int num_dim_hypercube;
};




struct PairCount
{
    std::pair<int, int> *pair_relation_attr_pos;
    int count;
};


}//namespace CUER
#endif //CUER_STRUCT_H_
