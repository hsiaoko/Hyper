#include "../../include/fixpoint_model.h"


namespace CUER
{
int FixpointModel::factorial(size_t n)
{
    register int i, f = 1;
    for (i = 1; i <= n; i++)
        f *= i;

    return f;
}

int FixpointModel::summation(size_t n)
{
    register int i, f = 0;
    for (i = 1; i <= n; i++)
        f += i;

    return f;
}
int FixpointModel::get_comparisons(size_t num_tuples, size_t num_threads)
{
    if (num_threads >= num_tuples)
    {
        printf("# num. of threads >= num. of tuples\n");
        return 0;
    }
    size_t max_cmp_num = 0;
    for ( int i = 0; i < num_tuples; i += num_threads)
    {
        //   cout<<"tid: "<<i<<endl;
        max_cmp_num += (num_tuples - 1 - i);
        //   cout<<(num_tuples - 1 - i)<<endl;
        //for(int j =0; j );
        //size_t cmp_num = FixpointModel::summation((num_tuples - 1) - i);
    }
    cout << max_cmp_num << endl;
}
float *FixpointModel::get_ratio(size_t *allocation, size_t len_)
{
    size_t min_ = 99999;
    for(int  i = 0; i  < len_;i++){
        min_ = allocation[i]<min_? allocation[i]: min_;
    }
    float *ratio_ = new float[len_];
    for (int i = 0; i  < len_; i++)
    {
        ratio_[i] = ((float)allocation[i] / (float)min_);
    }
    return ratio_;
}
} //namespace CUER