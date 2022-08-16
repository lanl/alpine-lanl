#include <cpp_headers.h>
#include <rule_base_func.h>
#include <glm_headers.h>

using namespace std;

class Feature_vector
{

public:
    vector<float> feature_vec;
};

vector<float> split(string str, string sep)
{
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    vector<float> arr;
    current=strtok(cstr,sep.c_str());
    while(current != NULL){
        arr.push_back(atof(current));
        current=strtok(NULL, sep.c_str());
    }
    return arr;
}

//Evaluates a gaussian membership function with an input value
float gmf(float val, float mean, float sigma)
{
    float ret;

    if(val==mean)
        ret = 1.0;
    else if(sigma>0.0)
        ret = exp(-(((val-mean)*(val-mean))/(2*sigma*sigma)));
    else
        ret = 0.0;       

    return ret; 
}

////////////////////////////////////////////////////////////////////////
// Evaluate a single rule on a single input and returns a float value
// return = firing strength of the rule computed by product of memberships
////////////////////////////////////////////////////////////////////////

//hardcoded version for 2 var only
/*glm::vec2 evaluate_single_rule(Rules R, Feature_vector data)
{
    glm::vec2 ret;
    float consequent=0;
    float fire[2];
    float final=0;
    float feature[2];
    feature[0] = data.feature_vec[0];
    feature[1] = data.feature_vec[1];

    //solve antecedent part to get membership evaluation
    for(int qq=0;qq<R.inputmfs.size();qq++)
    {
        fire[qq] = gmf(feature[qq],R.inputmfs[qq].mean,R.inputmfs[qq].sigma);
    }

    //compute final firing strength for the rule using product operator
    final = fire[0]*fire[1];

    //compute the consequent part
    consequent = feature[0]*R.out_params[0] + feature[1]*R.out_params[1] + R.out_params[2];

    ret.x = final;
    ret.y = consequent;

    return ret;
}*/

//generalized for multivar: uses c++ vector
/*glm::vec2 evaluate_single_rule(Rules R, Feature_vector data)
{
    glm::vec2 ret;
    int num = data.feature_vec.size();
    float consequent=0;
    float final=0;
    float fire[num];    
    float feature[num];
    
    for(int i=0;i<num;i++)
        feature[i] = data.feature_vec[i];

    //solve antecedent part to get membership evaluation
    for(int qq=0;qq<R.inputmfs.size();qq++)
        fire[qq] = gmf(feature[qq],R.inputmfs[qq].mean,R.inputmfs[qq].sigma);

    //compute final firing strength for the rule using product operator
    final=1;
    for(int i=0;i<num;i++)
        final = final*fire[i];

    for(int i=0;i<num;i++)
        consequent += feature[i]*R.out_params[i];
    consequent += R.out_params[num];

    ret.x = final;
    ret.y = consequent;

    return ret;
}*/

//generalized for multivar: uses float* and some performance is improved
glm::vec2 evaluate_single_rule(Rules R, float* data, int numPts)
{
    glm::vec2 ret;
    int num = numPts;
    float consequent=0;
    float final=0;
    float fire[num];    
    float feature[num];

    //solve antecedent part to get membership evaluation
    for(int qq=0;qq<R.inputmfs.size();qq++)
        fire[qq] = gmf(data[qq],R.inputmfs[qq].mean,R.inputmfs[qq].sigma);

    //compute final firing strength for the rule using product operator
    final=1;
    for(int i=0;i<num;i++)
        final = final*fire[i];

    for(int i=0;i<num;i++)
        consequent += data[i]*R.out_params[i];
    consequent += R.out_params[num];

    ret.x = final;
    ret.y = consequent;

    return ret;
}

////////////////////////////////////////////////////////////////////////
// Returns a floating value where return value is the evaluated value from
// the fuzzy system for the input data point
////////////////////////////////////////////////////////////////////////

//generalized for multivar: uses c++ vector
/*float evaluate_rulebase(Rule_Based_System rulebase, Feature_vector data)
{
    float ret;

    vector<glm::vec2> eval_vals;
    float num=0;
    float denom=0;

    //Evaluate the rulebase with each data input
    for(int qq=0;qq<rulebase.num_rules;qq++)
    {
        eval_vals.push_back(evaluate_single_rule(rulebase.rules[qq],data));
    }

    for(int qq=0;qq<eval_vals.size();qq++)
    {
        num += eval_vals[qq].x*eval_vals[qq].y; //fire strength*consequent added
        denom += eval_vals[qq].x; // firing strength added
    }

    if(denom>0)
        ret = num/denom;
    else
        ret=0;

    return ret;
}*/

//generalized for multivar: uses float* and some performance is improved
float evaluate_rulebase(Rule_Based_System rulebase, float* data, int numPts)
{
    float ret;

    vector<glm::vec2> eval_vals;
    float num=0;
    float denom=0;

    //Evaluate the rulebase with each data input
    for(int qq=0;qq<rulebase.num_rules;qq++)
    {
        eval_vals.push_back(evaluate_single_rule(rulebase.rules[qq],data,numPts));
    }

    for(int qq=0;qq<eval_vals.size();qq++)
    {
        num += eval_vals[qq].x*eval_vals[qq].y; //fire strength*consequent added
        denom += eval_vals[qq].x; // firing strength added
    }

    if(denom>0)
        ret = num/denom;
    else
        ret=0;

    return ret;
}