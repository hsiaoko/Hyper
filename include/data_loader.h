/* ******************************************************
// Copyright (c) 2020 Beihang University. All rights reserved.
// License(BSD/GPL/MIT)
// Author       : Xiaoke Zhu
// Last modified: 2020-05-14 09:10
// Email        : xiaoke.zhu@outlook.com
// Filename     : loader.h
// This is ...
  ***************************************************** */

#ifndef PER_LOADER_H_
#define PER_LOADER_H_
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
//#include "sqlite3.h"
#include <math.h>

#include "struct.h"
#include "tools.h"
#include "ML.h"
namespace CUER
{

//   class DBLoader
//   {
//   private:
//       std::string dbPath_;
//       char *zErrMsg_ = 0;
//       sqlite3 *db_;
//       int rc_;
//       static int count;
//       std::vector<int> *num_tuples_in_relations_;

//   public:
//       DBLoader(std::string dbPath);
//       DBLoader();
//       ~DBLoader();
//       static int CallBack2Vec(void *data, int argc, char **argv, char **azColName);
//       static int CallBackRelation2CSV(void *data, int argc, char **argv, char **azColName);
//       static int CallBackStatistic(void *data, int argc, char **argv, char **azColName);
//       static sqlite3 *CreateDB(char *db_path);
//       static int CreateTable(sqlite3 *db, std::vector<char *> *);

//       void ExecSQL(std::string sql);
//       int ExecSQLStatistic(std::string sql);
//       void ExecRelation2CSV(std::string sql, std::string csv_path);
//       void ReadeDB(std::string sql, std::vector<std::vector<std::string> *> *tuples);
//   };

class CSVLoader
{
private:

public:
    std::string csv_path_;
    size_t data_size_;
    std::vector<std::vector<std::string>*> *tuples_;
    std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_;

    CSVLoader(std::string csv_path);

    void ShowTuple(std::vector<std::string> *tuple);
    void ShowCSV();
    void ShowCSV(size_t k);
    void ShowCSV(std::vector<std::vector<std::string> *> *tuples, size_t k);
    bool read(bool read_head, char split_symbol);
    bool read2(bool read_head, char split_symbol, std::vector<string> *head);
    void blocking(
        size_t num_blocks,
        std::vector<pair<size_t, size_t>> *aid //<aid, num_prefix>
    );
    void freeTuples();
    void freeBlockingVec();
    void ShowBlockingVec(size_t k);
    static char *tuples2arr(
        std::vector<std::vector<std::string>*> *tuples,
        size_t *attr_len_arr,
        size_t *attr_start_point,
        size_t line_size,
        char *tuples_arr);
    static char *tuples2arr(
        std::vector<std::vector<std::string>*> *tuples,
        CUER::ML *ml,
        size_t *attr_len_arr,
        size_t *attr_start_point,
        size_t line_size,
        char *tuples_arr);
    static void getRMSE(std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_l, std::vector<std::vector<std::vector<std::string> *> *> *blocking_vec_r);
};

} //end namespace CUER
#endif //PER_LOADER_H_
