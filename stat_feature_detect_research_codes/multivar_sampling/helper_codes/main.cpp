#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <glm/glm.hpp>

using namespace std;

#define ARR_DIM 128 
int ipvarid=3;
int opvarid=0;

int numOutParam=23;//39  //hardcoded now
int numParam = 55; //55      //hardcoded now
int numRuns=10000; //10000 //hardcoded now

float *I11,*I12,*I21,*I22,*I31,*I32;
int Array1[ARR_DIM];
int Array2[ARR_DIM]; //Bin
int ArrayComb[ARR_DIM][ARR_DIM]; //nxy
int N = numRuns;

vector < vector < float> > inputParams;
vector < vector < float> > outputParams;
vector<float> maxIpVals;
vector<float> minIpVals;
vector<float> maxOpVals;
vector<float> minOpVals;

glm::vec2 getMaxMin1DVector(vector<float> &vec)
{
    float max = vec[0];
    float min = vec[0];

    for(int i=1;i<vec.size();i++)
    {
        if(max<vec[i])
            max=vec[i];

        if(min>vec[i])
            min=vec[i];
    }

    return glm::vec2(min,max);
}

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

//This function computes all the I-metrices
void compute_metrics()
{
    float prob_of_x_given_y=0.0, prob_of_y_given_x=0.0, prob_of_x=0.0, prob_of_y=0.0;
    int x=0;
 	
 	//Compute I1 and I2
    for(int i=0;i<ARR_DIM;i++)
    {
        for(int j=0;j<ARR_DIM;j++)
        {
 
            if(Array1[i]==0)
            {
                prob_of_y_given_x = 0;
            }
            else
            {
                prob_of_y_given_x = (float)ArrayComb[i][j] / (float)Array1[i];
            }   
 
            prob_of_y = (float)Array2[j] / (float)N;
 
            if(prob_of_y_given_x != 0 && prob_of_y != 0)
            {
                I11[i] += prob_of_y_given_x * log(prob_of_y_given_x / prob_of_y);
            }
            if(prob_of_y_given_x != 0)
            {
                I21[i] += prob_of_y_given_x * log(prob_of_y_given_x);
            }
            if(prob_of_y != 0)
            {
                I21[i] -=  prob_of_y * log(prob_of_y);
            }
 
            if(Array2[i] == 0)
            {
                prob_of_x_given_y = 0;
            }
            else
            {
                prob_of_x_given_y = (float)ArrayComb[j][i] / (float)Array2[i]; // Array2[j]
            }
 
            prob_of_x = (float)Array1[j] / (float)N;
            if(prob_of_x_given_y != 0 && prob_of_x != 0)
            {
                I12[i] += prob_of_x_given_y * log(prob_of_x_given_y / prob_of_x);
            }
            if(prob_of_x_given_y != 0)
            {
                I22[i] += prob_of_x_given_y * log(prob_of_x_given_y);
            }
            if(prob_of_x != 0)
            {
                I22[i] -= prob_of_x * log(prob_of_x);
            }
 
            if(prob_of_y_given_x > 1.0)
            {
                cout<<"Ooopps. value of prob_of_x_given_y is "<<prob_of_y_given_x<<" for i="<<i<<" and j="<<j<<endl;
                getchar();
            }
 
            if(prob_of_x_given_y > 1.0)
            {
                cout<<"Ooopps. value of prob_of_x_given_y is "<<prob_of_x_given_y<<" for i="<<i<<" and j="<<j<<endl;
                getchar();
            } 
        } 
    }
 
    //Compute I3
    for(int i=0;i<ARR_DIM;i++)
    {
        for(int j=0;j<ARR_DIM;j++)
        {
            if(Array1[i] == 0)
            {
                prob_of_y_given_x = 0;
            }
            else
            {
                prob_of_y_given_x = (float)ArrayComb[i][j] / (float)Array1[i];
            }
 
            prob_of_y = (float)Array2[j] / (float)N;
            I31[i] += prob_of_y_given_x * I22[j];
 
            if(Array2[i] == 0)
            {
                prob_of_x_given_y = 0;
            }
            else
            {
                prob_of_x_given_y = (float)ArrayComb[j][i] / (float)Array2[i]; // 
            }
            prob_of_x = (float)Array1[j] / (float)N;
            I32[i] += prob_of_x_given_y * I21[j];       
        }
    }
}

void compute_histograms()
{
	int binid1;
	int binid2;
	
	for(int i=0;i<numRuns;i++)
	{
		float val1 = inputParams[i][ipvarid];
		float val2 = outputParams[i][opvarid];

		val1 = (val1 - minIpVals[ipvarid])/(maxIpVals[ipvarid]-minIpVals[ipvarid]);
		val2 = (val2 - minOpVals[opvarid])/(maxOpVals[opvarid]-minOpVals[opvarid]);
		binid1 = (int)(val1*(ARR_DIM-1));
		binid2 = (int)(val2*(ARR_DIM-1));

		Array1[binid1]++; //nx
        Array2[binid2]++; //ny
        ArrayComb[binid1][binid2]++; //nxy
	}
}

int main(int argc,char **argv) 
{
 	///////////////////////////////////////////////////////////////////////////////////////
    ///Read the input parameter data
    ///////////////////////////////////////////////////////////////////////////////////////
    ifstream readinFis;
    //readinFis.open("data/latte_input.dat");
    readinFis.open("data/input_parameters_0.01.dat");
    string line;
    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, " ");
           
            if(v.size()>0)
            {
                inputParams.push_back(v);
            }
        }
    }

    //extract min and max for each parameter from all the runs
    for(int i=0;i<numParam;i++)
    {
        vector<float> oneParam;
        for(int j=0;j<numRuns;j++)
        {
            oneParam.push_back(inputParams[j][i]);
        }

        glm::vec2 minmax = getMaxMin1DVector(oneParam);
        minIpVals.push_back(minmax.x);
        maxIpVals.push_back(minmax.y);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    ///Read the output parameter data
    ///////////////////////////////////////////////////////////////////////////////////////
    ifstream readoutFis;
    readoutFis.open("data/latte_output_norm_0.01.dat");
    while(!readoutFis.eof())
    {
        getline(readoutFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, " ");

            if(v.size()>0)
            {
                outputParams.push_back(v);
            }
        }
    }

    //extract min and max for each parameter from all the runs
    for(int i=0;i<numOutParam;i++)
    {
        vector<float> oneParam;
        for(int j=0;j<numRuns;j++)
        {
            oneParam.push_back(outputParams[j][i]);
        }

        glm::vec2 minmax = getMaxMin1DVector(oneParam);
        minOpVals.push_back(minmax.x);
        maxOpVals.push_back(minmax.y);
    }

    cout<<"Reading files completed"<<endl;
    cout<<"Range of selected input: "<<minIpVals[ipvarid]<<" "<<maxIpVals[ipvarid]<<endl;
    cout<<"Range of selected output: "<<minOpVals[opvarid]<<" "<<maxOpVals[opvarid]<<endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    I11 = (float *)malloc(ARR_DIM*sizeof(float));
    I12 = (float *)malloc(ARR_DIM*sizeof(float));
    I21 = (float *)malloc(ARR_DIM*sizeof(float));
    I22 = (float *)malloc(ARR_DIM*sizeof(float));
    I32 = (float *)malloc(ARR_DIM*sizeof(float));
    I31 = (float *)malloc(ARR_DIM*sizeof(float));
 
    for(int i=0;i<ARR_DIM;i++)
    {
        I11[i]=0;
        I12[i]=0;
        I21[i]=0;
        I22[i]=0;
        I31[i]=0;
        I32[i]=0;
 
        Array1[i]=0;
        Array2[i]=0;
 
        for(int j=0; j<ARR_DIM; j++)
            ArrayComb[i][j]=0;
    }

    compute_histograms();   
    cout<<"Computing histograms completed"<<endl;      
     
    compute_metrics();      
    cout<<"Computing I-metrics completed"<<endl; 

    for(int i=0;i<ARR_DIM;i++)
     	cout<<i<<" "<<I21[i]<<" "<< minIpVals[ipvarid] + (i/(float)(ARR_DIM-1))*(maxIpVals[ipvarid] - minIpVals[ipvarid])<<endl;       
   	
   	ofstream writeOut;
   	string outfile;
   	stringstream ss1,ss2;
   	ss1<<ipvarid;
   	ss2<<opvarid;
   	outfile = "I2_latte_" + ss1.str() + "_" + ss2.str() + ".csv";
   	writeOut.open(outfile.c_str());

   	for(int i=0;i<ARR_DIM;i++)
     	writeOut<<i<<" "<<I21[i]<<endl;

    writeOut.close(); 

    return 0;
}