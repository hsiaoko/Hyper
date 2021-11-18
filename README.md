# Overview

Hyper is a CPU-GPU parallel System for Entity Resolution which is an implementation of the paper "Parallelizing Sequential Entity Resolution with a Multicore Machine". It implements an implicitly parallel programming model, where the programmer only need to provide a matching rule and three sequential algorithms (i.e. Embeding, Blocking, Match). Hyper is designed so that the programmer does not have to deal with low-level parallel CUDA programming constructs such as threads, locks, barriers, condition variables, etc.
This repository is only a prototype code; the product-quality code will coming soon. 

Hyper is released under the MIT license.

#Building Hyper
You can checkout the latest release by typing (in a terminal):
git clone https://github.com/hsiaoko/HypER

#Dependencies
At the minimum, Hyper depends on the following software:
  + A modern C++ compiler compliant with the C++-17 standard (gcc >= 7, Intel >= 19.0.1)
  + CUDA (>=11.0)
  + cuDNN: v8.2.0
  + CMake (>= 3.2)


# Compiling Hyper & Running Hyper
## Compiling Hyper
We use CMake to streamline building, testing and installing Hyper. In the following, we will highlight some common commands.
Let's assume that SRC_DIR is the directory where the source code for Galois resides, and you wish to build Galois in some BUILD_DIR. Run the following commands to set up a build directory:

```c++
SRC_DIR=`pwd` # Or top-level Hyper source dir
BUILD_DIR=<path-to-your-build-dir>
> mkdir -p $BUILD_DIR
> cmake ..
> make
```

## Running Hyper
If you System is based on SLURM please move submit.sh file to your $BUILD_DIR. And then submit your task by sbatch command,
e.g.
```
> mv $SRC_DIR/submit.sh
> sbatch submit.sh
```
Otherwise,
```
./hyper
```
if your enviroment is not SLURM.

# Demo: Entity Resolution on Songs Dataset

```c++
//Step 1: read csv
CUER::CSVLoader *csv_l = new CUER::CSVLoader("$SRC_DIR/dataset/msd_cleared_utf_8_mini.csv");
std::vector<string> *head = new std::vector<string>;
    head->push_back("1");
    head->push_back("2");
    head->push_back("3");
    head->push_back("4");
    head->push_back("5");
    csv_l->read2(false, ',', head);

//Step 2: blocking (plaste your Blocking code to CUER::CSVLoader::blocking()->std::vector<std::vector<std::vector<std::string> *> *> partitions)
vector<pair<size_t, size_t>> *bid = new vector<pair<size_t, size_t>>;
bid->push_back(make_pair(7, 0));
csv_l->blocking(num_blocking, bid);

//Step 3: Set Schema of the dataset (here each atom of a tuple apply 64 bytes)
size_t tuple_size = 5;
size_t *attr_len_arr = (size_t *)malloc(sizeof(size_t) * tuple_size);
size_t *attr_start_point = (size_t *)malloc(sizeof(size_t) * tuple_size);
attr_len_arr[0] = 64;
attr_len_arr[1] = 64;
attr_len_arr[2] = 64;
attr_len_arr[3] = 64;
attr_len_arr[4] = 64;
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
//Step 4: Set matching rule (here Match(t.atom1 s.atom1) and Match(t.atom2 s.atom2) and t.atom4 s.atom4 -> t.id = s.id)
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

//Step 5: run Hyper
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

std::thread t(HostReducer, 3, tokenring);

tokenring->RunHostProducer(rule_size, h_rule_aid, h_rule_threshold, false); // here we used lev_jaro_ratio as Match function, see include/core.h file to plaste your Match function.
t.join();
```

#Contact Us
For bugs, please raise an issue on GiHub. Questions and comments are also welcome at the Hyper users mailing list: zhuxk@buaa.edu.cn,  hsiaoko.chu@gmail.com

Any contributions you make are greatly appreciated!
