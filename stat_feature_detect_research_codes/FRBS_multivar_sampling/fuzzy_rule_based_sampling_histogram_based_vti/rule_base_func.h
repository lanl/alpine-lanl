#include <cpp_headers.h>

using namespace std;

class Membership_func
{
public:
    float mean;
    float sigma;  
    string rule_label;
};

class Rules
{
public:
    vector<Membership_func> inputmfs;
    vector<float> out_params;
    string membership_func_type;
    int rule_id;
};

class Rule_Based_System
{
public:
    vector<Rules> rules;    
    string fuzzy_system_type;
    int num_rules;
    int num_input_dim;
};