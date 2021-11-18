#include "../../include/ML.h"
//#include "../lib/REENet/include/core.h"

using namespace std;
namespace CUER
{
/*
ML::ML(std::vector<int> *aid, std::vector<string> *bin_pth, std::vector<string> *model_pth)
{
std::vector<reenet::REEModule *> *vec_reemodules = new std::vector<reenet::REEModule *>;

for (int i = 0; i < 2; i++)
{
    cout << i << endl;
    cout << bin_pth->at(i) << ", " << model_pth->at(i) << endl;
    vec_reemodules->push_back(new reenet::REEModule(bin_pth->at(i), model_pth->at(i)));
    // reenet::REEModule reemodule = reenet::REEModule("/home/LAB/zhuxk/project/CUER/lib/REENet/models/embeding/dblp_acm_authors.bin", "/home/LAB/zhuxk/project/CUER/lib/REENet/models/classifiers/rnn_linear_authors_seq.pt");
    // reenet::REEModule reemodule2 = reenet::REEModule("/home/LAB/zhuxk/project/CUER/lib/REENet/models/embeding/dblp_acm_title.bin", "/home/LAB/zhuxk/project/CUER/lib/REENet/models/classifiers/rnn_linear_title_seq.pt");
}
this->vec_reemodules = vec_reemodules;
this->vec_aid_ = aid;
this->embeding_size_ = 64;
//this->tuples_ = tuples_;
}
*/
ML::ML(std::map<int, std::vector<float *>*> *map_attr_embeding_2D, std::map<int, string> *bin_pth, std::map<int, string> *model_pth)
{
    cout << "ML()" << endl;
    std::map<int, reenet::REEModule *> *map_reemodules = new std::map<int, reenet::REEModule *>;
    std::map<int, std::vector<float *>*>::iterator iter;
    size_t i = 0;
    for (iter = map_attr_embeding_2D->begin(); iter != map_attr_embeding_2D->end(); iter++)
    {
        map_reemodules->insert(make_pair(iter->first, new reenet::REEModule(bin_pth->find(iter->first)->second, model_pth->find(iter->first)->second)));
        i++;
    }

    this->map_reemodules = map_reemodules;
    this->bin_pth = bin_pth;
    this->model_pth = model_pth;
    this->map_attr_embeding_2D_ = map_attr_embeding_2D;
    this->embeding_size_ = 64;
}

ML::ML()
{
    printf("ML()\n");
}

float *ML::attr_embeding(size_t aid, std::string str_)
{
    std::map<int, reenet::REEModule *>::iterator iter = this->map_reemodules->find(aid);
    if (iter == this->map_reemodules->end())
    {
        printf("attr model not find\n");
        throw 0;
    }
    reenet::REEModule *reeModel = iter->second;
    float *attr_embeding = reeModel->rnn_embeding(str_, 64);
    return attr_embeding;
}
/*
std::vector<std::vector<float *>*> *ML::attrs_embeding(std::vector<std::vector<std::string>*> *tuples_)
{

    std::vector<std::vector<float *>*> *vec_attr_embeding_2D = new std::vector<std::vector<float *>*>;
    for (int i = 0 ; i < vec_aid_->size(); i++)
    {
        size_t aid = vec_aid_->at(i);
        vector<float *> *vec_attr_embeding = new vector<float *>;
        for (int tid = 0; tid < tuples_->size(); tid++)
        {
            string attr_val = tuples_->at(tid)->at(aid);
            // string attr_val_l = "Raghu Ramakrishnan, S. Sudarshan, Divesh Srivastava, Praveen Seshadri";
            // string attr_val_r = "Raghu Ramakrishnan, Divesh Srivastava, S. Sudarshan, Praveen Seshadri";
            // string attr_val_r = "A. M. Ouksel, A. Sheth";
            // i = 1;
            float *attr_embeding = vec_reemodules->at(i)->rnn_embeding(attr_val, embeding_size_);
            vec_attr_embeding->push_back(attr_embeding);
            //  float * attr_embeding_l = this->vec_reemodules->at(i)->rnn_embeding(attr_val_l, this->embeding_size_);
            //  float * attr_embeding_r = this->vec_reemodules->at(i)->rnn_embeding(attr_val_r, this->embeding_size_);
            //  float dis = ML::euclidean_distance(attr_embeding_l, attr_embeding_r, this->embeding_size_);
            //  cout<<"dis: "<<dis<<endl;
            //show_embeding(attr_embeding, this->embeding_size_);
        }
        vec_attr_embeding_2D->push_back(vec_attr_embeding);
        this->map_attr_embeding_2D_->find(aid)->second = vec_attr_embeding;
    }

    cout << this->map_attr_embeding_2D_->find(1)->second->size() << endl;
    //this->vec_attr_embeding_2D = vec_attr_embeding_2D;
    return vec_attr_embeding_2D;
}
*/
void ML::show_embeding(float *eb, size_t eb_size)
{
    cout << endl;
    for (int i = 0; i < eb_size; i++)
    {
        cout << eb[i] << ", ";
    }
    cout << endl;
}


float ML::euclidean_distance(float *eb_1, float *eb_r, int eb_size)
{
    float dis = 0;
    for ( int i = 0; i < eb_size; i++)
    {
        dis += (eb_1[i] - eb_r[i]) * (eb_1[i] - eb_r[i]);
    }
    return dis;
}


void ML::relu(float *input, size_t length, float tau)
{
    for (int i = 0; i < length; i++)
    {
        input[i] = input[i] > tau ? input[i] : 0;
    }
}

void ML::sigmoid(float *input, size_t length)
{
    for (int i = 0; i < length; i++)
    {
        cout << input[i] << endl;
        input[i] = 1 / (1 + exp(-1 * input[i]));
    }

}

float *ML::mm(float *a, float *b, std::pair<size_t, size_t> shape_a, std::pair<size_t, size_t> shape_b)
{
    if (shape_a.second != shape_b.first)
    {
        return NULL;
    }
    float *c = new float(shape_a.first * shape_b.second);

    for (int b_j = 0; b_j < shape_b.second; b_j++)
    {
        for (int a_i = 0; a_i < shape_a.first; a_i++)
        {
            float c_val = 0;
            for (int a_b = 0; a_b < shape_a.second; a_b++)
            {
                c_val += *(a + a_i * shape_a.second + a_b) * *(b + a_b * shape_b.second + b_j);
                //c_val += a[a_i][a_b] * b[a_b][b_j];
            }
            cout << c_val << endl;
            //*(c + b_j* shape_a.first + a_i) = c_val;
            *(c +  a_i * shape_b.second + b_j) = c_val;
        }
    }
    return c;
}

float *ML::mm_euclidean_distance(float *a, float *b, std::pair<size_t, size_t> shape)
{

    float *c = new float(shape.first * shape.second);
    for (int i = 0; i < shape.first; i++)
    {
        for (int j = 0; j < shape.second; j++)
        {
            *(c + i * shape.second + j) = pow(*(a + i * shape.second + j) - * (b + i * shape.second + j), 2);
        }
    }
    return c;
}


float ML::euclidean_distance(char *attr_l, char *attr_r, int length_)
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


} //namespace CUER