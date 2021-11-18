#include "../../include/matcher.h"

using namespace std;
namespace CUER
{
bool Matcher::callMatch(std::vector<pair<int, float>> *rule,
                        char *tuple_l, char *tuple_r,
                        std::vector<int> *attr_len_arr,
                        std::vector<int> *attr_start_point,
                        int line_size)
{

    // showTuple(tuple_l, attr_start_point);
    //showTuple(tuple_r, attr_start_point);
    size_t tag = 0;
    for (int i = 0; i < rule->size(); i++)
    {
        size_t aid = rule->at(i).first;
        double threshold =  rule->at(i).second;
        size_t start_point = attr_start_point->at(aid);
        char *attr_l = tuple_l + attr_start_point->at(aid);
        char *attr_r = tuple_r + attr_start_point->at(aid);
        if (threshold != 1)
        {
            // cout<<"# "<<attr_l<<" , "<<attr_r<<" sim: "<<jws(attr_l, attr_r)<<"thre: "<<threshold<<endl;
            if (jws(attr_l, attr_r) < threshold)
            {
                //   cout<<attr_l<<" != "<<attr_r<<endl<<" sim: "<<jws(attr_l, attr_r)<<endl;
                break;
            }
            else
            {
                tag += jws(attr_l, attr_r) >= threshold ? 1 : 0;
            }
        }
        else
        {
            // cout<<"# "<<attr_l<<" , "<<attr_r<<endl;
            if (eq(attr_l, attr_r))
            {
                tag += 1;
            }
            else
            {
                //   cout<<attr_l<<" = "<<attr_r<<endlhttps://research.google/outreach/phd-fellowship/recipients/
            }
        }
    }
    //cout<<(int)c&b<<endl;
    //cout<<"tag:"<<tag<<','<<rule->size()<<endl;
    return tag == rule->size() ? true : false;
}


__device__ bool Matcher::callMatch(
    char *tuple_l,
    char *tuple_r,
    size_t *rule_aid,
    float *rule_threshold,
    size_t *attr_len_attr,
    size_t *attr_start_point,
    //size_t line_size,
    size_t tuple_size,
    size_t rule_size
)
{
    size_t tag = 0;
    for (int i = 0; i < rule_size; i++)
    {
        size_t aid = rule_aid[i];
        double threshold =  rule_threshold[i];
        size_t start_point = attr_start_point[aid];
        char *attr_l = tuple_l + attr_start_point[aid];
        char *attr_r = tuple_r + attr_start_point[aid];
        if (threshold != 1)
        {
            if (jws(attr_l, attr_r) < threshold)
            {
                break;
            }
            else
            {
                tag += jws(attr_l, attr_r) >= threshold ? 1 : 0;
            }
        }
        else
        {
            if (eq(attr_l, attr_r))
            {
                tag += 1;
            }
            else
            {
                break;
            }
        }
    }
    return tag == rule_size ? true : false;
}

bool Matcher::eq(char *attr_l, char *attr_r)
{
    size_t i = 0;
    bool tag = true;
    while ( *(attr_l + i) != '\0' && *(attr_r + i) != '\0')
    {
        i++;
    }
    return tag;

}

double Matcher::Jaccard(std::string str1, std::string str2)
{
    // for title attribution.
    std::vector<std::string> str1_set;
    std::vector<std::string> str2_set;
    int len_c = 0;
    if (str1.length() + str2.length() == 0)
    {
        return 1.0;
    }
    else
    {
        Split(str1, ",", str1_set);
        Split(str2, ",", str2_set);
        for (int i = 0; i < str1_set.size(); i++)
        {
            for (int j = 0; j < str2_set.size(); j++)
            {
                len_c += str1_set.at(i) == str2_set.at(j) ? 1 : 0;
            }
        }
        return (double)len_c / (str1_set.size() + str2_set.size() - len_c);
    }
}

double Matcher::jws(std::string str1, std::string str2)
{
    std::vector<std::string> str1_set;
    std::vector<std::string> str2_set;
    int len_c = 0;
    if (str1.length() + str2.length() == 0)
    {
        return 1.0;
    }
    else
    {
        Split(str1, ", ", str1_set);
        Split(str2, ", ", str2_set);
        for (int i = 0; i < str1_set.size(); i++)
        {
            for (int j = 0; j < str2_set.size(); j++)
            {
                len_c += lev_jaro_winkler_ratio(str1_set.at(i).size(), str1_set.at(i).c_str(), str2_set.at(j).size(), str2_set.at(j).c_str(), 0.3) > 0.8 ? 1 : 0;
            }
        }
        return (double)len_c / (str1_set.size() + str2_set.size() - len_c);
    }
}

double Matcher::JaroWinkler(std::string str1, std::string str2)
{
    // for others
    if (str1.length() == 0 && str2.length() == 0)
    {
        return 1.0;
    }
    else
    {
        double sim = ((double)MatchSize(str1, str2) / (double)str1.length() + (double)MatchSize(str1, str2) / (double)str2.length() + ((double)MatchSize(str1, str2) - (double)LevenshteinDistance(str1, str2)) / (double)MatchSize(str1, str2));
        sim /= 3;
        //       return (0.5 / str1.length() + 0.5 / str2.length() + (2 - LevenshteinDistance(str1, str2)) / 2);
        return sim;
    }
}

int Matcher::MatchSize(std::string str1, std::string str2)
{
    int count = 0;
    for (int i = 0; i < str1.length() && i < str2.length(); i++)
    {
        if (str1[i] == str2[i])
        {
            count++;
        }
    }
    return count;
}

int Matcher::LevenshteinDistance(const std::string source, const std::string target)
{
    int n = source.length();
    int m = target.length();
    if (m == 0)
        return n;
    if (n == 0)
        return m;
    typedef vector<vector<int>> Tmatrix;
    Tmatrix matrix(n + 1);
    for (int i = 0; i <= n; i++)
        matrix[i].resize(m + 1);
    for (int i = 1; i <= n; i++)
        matrix[i][0] = i;
    for (int i = 1; i <= m; i++)
        matrix[0][i] = i;
    for (int i = 1; i <= n; i++)
    {
        const char si = source[i - 1];
        for (int j = 1; j <= m; j++)
        {
            const char dj = target[j - 1];
            int cost;
            if (si == dj)
            {
                cost = 0;
            }
            else
            {
                cost = 1;
            }
            const int above = matrix[i - 1][j] + 1;
            const int left = matrix[i][j - 1] + 1;
            const int diag = matrix[i - 1][j - 1] + cost;
            matrix[i][j] = min(above, min(left, diag));
        }
    }
    return matrix[n][m];
}

void Matcher::Split(const std::string &src, const std::string &separator, std::vector<std::string> &dest)
{
    //参数1：要分割的字符串；参数2：作为分隔符的字符；参数3：存放分割后的字符串的vector向量
    std::string str = src;
    std::string substring;
    string::size_type start = 0, index;
    dest.clear();
    index = str.find(separator, start);
    do
    {
        if (index != string::npos)
        {
            substring = str.substr(start, index - start);
            dest.push_back(substring);
            start = index + separator.size();
            index = str.find(separator, start);
            if (start == string::npos)
                break;
        }
    }
    while (index != string::npos);
    //the last part
    substring = str.substr(start);
    dest.push_back(substring);
}

double Matcher::lev_jaro_ratio(size_t len1, const char *string1,
                               size_t len2, const char *string2)
{
    size_t i, j, halflen, trans, match, to;
    size_t *idx;
    double md;

    if (len1 == 0 || len2 == 0)
    {
        if (len1 == 0 && len2 == 0)
            return 1.0;
        return 0.0;
    }
    /* make len1 always shorter (or equally long) */
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
    idx = (size_t *)calloc(len1, sizeof(size_t));
    if (!idx)
        return -1.0;

    /* The literature about Jaro metric is confusing as the method of assigment
            * of common characters is nowhere specified.  There are several possible
            * deterministic mutual assignments of common characters of two strings.
            * We use earliest-position method, which is however suboptimal (e.g., it
            * gives two transpositions in jaro("Jaro", "Joaro") because of assigment
            * of the first `o').  No reasonable algorithm for the optimal one is
            * currently known to me. */
    match = 0;
    /* the part with allowed range overlapping left */
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
    /* the part with allowed range overlapping right */
    to = len1 + halflen < len2 ? len1 + halflen : len2;
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
    if (!match)
    {
        free(idx);
        return 0.0;
    }
    /* count transpositions */
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
    free(idx);

    md = (double)match;
    return (md / len1 + md / len2 + 1.0 - trans / md / 2.0) / 3.0;
}

double Matcher::lev_jaro_winkler_ratio(size_t len1, const char *string1,
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
} //namespace CUER