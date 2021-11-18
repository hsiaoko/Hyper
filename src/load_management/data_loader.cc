/* ******************************************************
// Copyright (c) 2020 Beihang University. All rights reserved.
// License(BSD/GPL/MIT)
// Author       : Xiaoke Zhu
// Last modified: 2020-05-14 09:14
// Email        : xiaoke.zhu@outlook.com
// Filename     : loader.cc
// This is ...
  ***************************************************** */
#include "../../include/data_loader.h"
#include "rapidcsv.h"
using namespace std;
namespace CUER
{

    void showArray(char *array, size_t len_)
    {
        for (int i = 0; i < len_; i++)
        {
            cout << array[i];
        }
        cout << endl;
    }

    CSVLoader::CSVLoader(std::string csv_path)
    {
        this->csv_path_ = csv_path;
    }
    std::vector<string> *split_str(string s)
    {
        int n = s.size();
        for (int i = 0; i < n; ++i)
        {
            if (s[i] == ',')
            {
                s[i] = ' ';
            }
        }
        istringstream out(s);
        string str;
        std::vector<string> *str_vec = new std::vector<string>;
        while (out >> str)
        {
            str_vec->push_back(str);
        }
        return str_vec;
    }

    bool CSVLoader::read(bool read_head, char split_symbol)
    {
        std::ifstream fp(this->csv_path_);
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
        this->tuples_ = tuples;
        //cout<<"sad:"<<this->tuples_->size()<<endl;
    }

    bool CSVLoader::read2(bool read_head, char split_symbol, std::vector<string> *head)
    {
        std::ifstream fp(this->csv_path_);
        vector<vector<std::string> *> *tuples = new vector<vector<std::string> *>;
        vector<vector<std::string> *> *col_vec = new vector<vector<std::string> *>;
        string line;
        getline(fp, line);
        cout << this->csv_path_ << endl;
        rapidcsv::Document doc(this->csv_path_);
        std::vector<string> *head_vec = split_str(line);

        for (int i = 0; i < head_vec->size(); i++)
        {
            std::vector<string> *col = new std::vector<string>;
            while (head_vec->at(i).find("\"") != head_vec->at(i).npos)
            {
                head_vec->at(i).replace(head_vec->at(i).find("\""), 1, "");
            }
            *(col) = doc.GetColumn<string>(i);
            col_vec->push_back(col);
        }
        //std::vector<string> idDBLP = doc.GetColumn<string>("idDBLP");
        //std::cout << "Read " <<col_vec->size()<<", " <<col_vec->at(0)->size() << " values." << std::endl;

        vector<std::string> *tuple;
        for (int i = 0; i < col_vec->at(0)->size(); i++)
        {
            tuple = new vector<std::string>;
            for (int j = 0; j < col_vec->size(); j++)
            {
                tuple->push_back(col_vec->at(j)->at(i));
            }
            tuples->push_back(tuple);
        }
        for (int i = 0; i < col_vec->size(); i++)
        {
            col_vec->at(i)->erase(col_vec->at(i)->begin(), col_vec->at(i)->end());
        }
        free(col_vec);
        this->tuples_ = tuples;
        this->data_size_ = tuples->size();
    }

    void CSVLoader::ShowTuple(std::vector<std::string> *tuple)
    {
        for (int i = 0; i < tuple->size(); i++)
        {
            //   cout<<endl;
            //printf("%s ", tuple->at(i).c_str());
            cout << tuple->at(i) << "|";
        }
        printf("\n\n");
    }

    void CSVLoader::ShowCSV()
    {
        for (int i = 0; i < this->tuples_->size(); i++)
        {
            // for(int j = 0; j < this->tuples_->at(i)->size(); j++){
            //     cout<<this->tuples_->at(i)->at(j)<<"|";
            // }
            cout << "\n"
                 << this->tuples_->at(i)->at(0) << "|" << this->tuples_->at(i)->at(1) << "|" << this->tuples_->at(i)->at(2) << "|" << this->tuples_->at(i)->at(3) << "|" << this->tuples_->at(i)->at(4) << "|" << this->tuples_->at(i)->at(5) << "|" << this->tuples_->at(i)->at(6) << "|" << this->tuples_->at(i)->at(7) << endl;
        }
        cout << endl;
    }

    void CSVLoader::ShowCSV(size_t k)
    {
        for (int i = 0; i < k; i++)
        {
            ShowTuple(this->tuples_->at(i));
        }
        cout << endl;
    }

    void CSVLoader::ShowCSV(std::vector<std::vector<std::string> *> *tuples, size_t k)
    {
        size_t t = 0;
        t = tuples->size() < k ? tuples->size() : k;

        for (int i = 0; i < t; i++)
        {
            ShowTuple(tuples->at(i));
        }
        cout << endl;
    }

    void CSVLoader::blocking(
        size_t num_blocks,
        std::vector<pair<size_t, size_t>> *aid //<aid, num_prefix>
    )
    {
        std::vector<std::vector<std::string> *> *tuples = this->tuples_;
        this->blocking_vec_ = new std::vector<std::vector<std::vector<std::string> *> *>;
        for (int i = 0; i < num_blocks; i++)
        {
            this->blocking_vec_->push_back(new std::vector<std::vector<std::string> *>);
        }

        int hash_val = 0, bid = 0;
        for (int i = 0; i < tuples->size(); i++)
        {
            hash_val = 0;
            for (int j = 0; j < aid->size(); j++)
            {
                if (aid->at(j).second == 0)
                {
                    hash_val += CUER::hash(tuples->at(i)->at(aid->at(j).first));
                }
                else if (aid->at(j).second > 0)
                {
                    hash_val += CUER::prefixHash(tuples->at(i)->at(aid->at(j).first), aid->at(j).second);
                }
            }
            bid = hash_val % num_blocks;
            this->blocking_vec_->at(bid)->push_back(tuples->at(i));
        }
        std::vector<std::vector<std::vector<std::string> *> *>::iterator iter;
        for (iter = this->blocking_vec_->begin(); iter != this->blocking_vec_->end();)
        {
            if ((*iter)->size() == 0)
            {
                iter = this->blocking_vec_->erase(iter);
                //iter = blocking_vec_->begin();
            }
            else
            {
                iter++;
            }
        }
        cout << "BLOCK:::::::" << this->blocking_vec_->size() << endl;
    }

    void CSVLoader::ShowBlockingVec(size_t k)
    {
        if (this->blocking_vec_->size() == 0)
        {
            printf("empty blokcing_vec !%d");
        }
        for (int i = 0; i < this->blocking_vec_->size(); i++)
        {
            printf("blocking %d, hold %d tuples, show top %d;\n", i, this->blocking_vec_->at(i)->size(), k);
            this->ShowCSV(this->blocking_vec_->at(i), k);
        }
    }

    char *CSVLoader::tuples2arr(
        std::vector<std::vector<std::string> *> *tuples,
        size_t *attr_len_arr,
        size_t *attr_start_point,
        size_t line_size,
        char *tuples_arr)
    {
        //cout << "tuple:size" << tuples->size() << endl;
        for (int i = 0; i < tuples->size(); i++)
        {
            for (int j = 0; j < tuples->at(i)->size(); j++)
            {
                //cout << j << ":" << tuples->at(i)->at(j) << " ";
                if (tuples->at(i)->at(j).length() < 1)
                {
                    *((tuples_arr + i * line_size) + attr_start_point[j] + tuples->at(i)->at(j).length()) = '\0';
                }
                else
                {
                    memcpy(((tuples_arr + i * line_size) + attr_start_point[j]), tuples->at(i)->at(j).c_str(), tuples->at(i)->at(j).length());
                    *((tuples_arr + i * line_size) + attr_start_point[j] + tuples->at(i)->at(j).length()) = '\0';
                }
                //cout << "j:" << j << ", " << tuples->at(i)->at(j) << "len:" << tuples->at(i)->at(j).length() << endl;
                //cout << j << ((tuples_arr + i * line_size) + attr_start_point[j]) << " ";
            }
            //cout << i<<"/"<< tuples->size()  <<endl;
            //break;
        }
        //showTuples(tuples_arr, tuples->size(), attr_start_point, line_size);
        //cout<<"end()"<<endl;
        // cout<<"-------------#@#!#"<<endl;
        // for(int i = 0; i < tuples->size(); i++) {
        //     line = (tuples_arr + i*line_size);
        //     cout<<line+attr_start_point->at(0)<<", "<<line+attr_start_point->at(1)<<", "<<line+attr_start_point->at(2)<<", "<<line+attr_start_point->at(3)<<", "<<line+attr_start_point->at(4)<<", "<<line+attr_start_point->at(5)<<", "<<line+attr_start_point->at(6)<<", "<<line+attr_start_point->at(7)<<endl;
        // }
    }

    char *CSVLoader::tuples2arr(
        std::vector<std::vector<std::string> *> *tuples,
        CUER::ML *ml,
        size_t *attr_len_arr,
        size_t *attr_start_point,
        size_t line_size,
        char *tuples_arr)
    {
        memset(tuples_arr, 0, line_size * tuples->size());
        std::map<int, std::vector<float *> *>::iterator iter;
        for (int i = 0; i < tuples->size(); i++)
        {

            for (int j = 0; j < tuples->at(i)->size(); j++)
            {
                //cout<<tuples->at(i)->at(j).c_str()<<"length: "<<tuples->at(i)->at(j).length()<<"start point" << attr_start_point[j]<<","<< attr_start_point[2]<<endl;
                iter = ml->map_attr_embeding_2D_->find(j);
                if (iter == ml->map_attr_embeding_2D_->end())
                {
                    memcpy(((tuples_arr + i * line_size) + attr_start_point[j]), tuples->at(i)->at(j).c_str(), tuples->at(i)->at(j).length());
                    *((tuples_arr + i * line_size) + attr_start_point[j] + tuples->at(i)->at(j).length()) = '\0';
                    //cout << "attr: "<<j<<" "<<(tuples_arr + i * line_size) + attr_start_point[j] << ", " << endl;
                }
                else
                {
                    //memcpy(((tuples_arr + i * line_size) + attr_start_point[j]), tuples->at(i)->at(j).c_str(), tuples->at(i)->at(j).length());
                    float *eb;
                    if (tuples->at(i)->at(j).length() > 0)
                    {

                        eb = ml->attr_embeding(j, tuples->at(i)->at(j));
                    }
                    else
                    {
                        eb = ml->attr_embeding(j, "?");
                    }
                    memcpy(((tuples_arr + i * line_size) + attr_start_point[j]), (char *)eb, sizeof(float) * 64);
                    //ML::show_embeding((float *)((tuples_arr + i * line_size) + attr_start_point[j]), 64);
                }
            }
            //cout << "################################" << endl;
        }
    }

    void CSVLoader::freeTuples()
    {
        for (int i = 0; i < this->tuples_->size(); i++)
        {
            //for(int j = 0; j< this->tuples_->at(i)->size(); j++){
            this->tuples_->at(i)->clear();
            //}
        }
    }

    void CSVLoader::freeBlockingVec()
    {
        for (int i = 0; i < this->blocking_vec_->size(); i++)
        {
            for (int j = 0; j < this->blocking_vec_->at(i)->size(); j++)
            {
                this->blocking_vec_->at(i)->at(j)->clear();
            }
        }
    }
    void CSVLoader::getRMSE(std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l, std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_r)
    {
        std::vector<int> *compare_cuer = new std::vector<int>;
        std::vector<int> *compare_mapreduce = new std::vector<int>;
        size_t sum_map = 0, sum_cuer = 0, avg_map = 0, avg_cuer = 0;
        for (int i = 0; i < blocking_vec_l->size(); i++)
        {
            size_t comp_cuer = blocking_vec_l->at(i)->size() > blocking_vec_r->at(i)->size() ? blocking_vec_l->at(i)->size() : blocking_vec_r->at(i)->size();
            size_t comp_map = blocking_vec_l->at(i)->size() * blocking_vec_r->at(i)->size();
            cout << "map: " << comp_map << ", " << comp_cuer << endl;
            sum_map += comp_map;
            sum_cuer += comp_cuer;
            compare_mapreduce->push_back(comp_map);
            compare_cuer->push_back(comp_cuer);
        }
        avg_map = sum_map / blocking_vec_l->size();
        avg_cuer = sum_cuer / blocking_vec_l->size();
        size_t cuer_dev = 0, map_dev = 0;
        for (int i = 0; i < blocking_vec_l->size(); i++)
        {
            size_t comp_cuer = blocking_vec_l->at(i)->size() > blocking_vec_r->at(i)->size() ? blocking_vec_l->at(i)->size() : blocking_vec_r->at(i)->size();
            size_t comp_map = blocking_vec_l->at(i)->size() * blocking_vec_r->at(i)->size();
            cuer_dev += (comp_cuer - avg_cuer) * (comp_cuer - avg_cuer);
            map_dev += (comp_map - avg_map) * (comp_map - avg_map);
        }
        float rmse_cuer = 0, rmse_map = 0;
        rmse_cuer = sqrt(cuer_dev / blocking_vec_l->size());
        rmse_map = sqrt(map_dev / blocking_vec_l->size());

        cout << setprecision(7) << "remse cuer:" << rmse_cuer << " rmse map" << rmse_map << endl;
        return;
    }

} //namespace CUER