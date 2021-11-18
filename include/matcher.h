#ifndef CUER_MATCHER_H_
#define CUER_MATCHER_H_
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include "struct.h"
#include "tools.h"

namespace CUER
{
class Matcher
{
private:
public:
    Matcher(std::vector<pair<int, float>> *attr);
    static bool callMatch(std::vector<pair<int, float>> *rule,
                          char *tuple_l, char *tuple_r,
                          std::vector<int> *attr_len_arr,
                          std::vector<int> *attr_start_point, int line_size);

    static __device__ bool callMatch(
        char *tuple_l,
        char *tuple_r,
        size_t *rule_aid,
        float *rule_threshold,
        size_t *attr_len_attr,
        size_t *attr_start_point,
        size_t line_size,
        size_t rule_size
    );

    static double Jaccard(std::string str1, std::string str2);
    static double jws(std::string str1, std::string str2);
    static double JaroWinkler(std::string str1, std::string str2);
    static int MatchSize(std::string str1, std::string str2);

    static int LevenshteinDistance(const std::string source, const std::string target);
    static void Split(const std::string &src, const std::string &separator, std::vector<std::string> &dest);
    static double lev_jaro_ratio(size_t len1, const char *string1, size_t len2, const char *string2);
    static double lev_jaro_winkler_ratio(size_t len1, const char *string1, size_t len2, const char *string2, double pfweight);
    static bool eq(char *attr_l, char *attr_r);
};

} // namespace CUER
#endif //CUER_MATCHER_H_
