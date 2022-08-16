#ifndef SLIC
#define SLIC

#include <cpp_headers.H>
#include <element.H>

using namespace std;

class Slic
{
	
public:

	Slic();
	
	int xdim;
	int ydim;
	int zdim;
    int blockXSize;
	int blockYSize;
	int blockZSize;
	int clusterNum;
    int QUANTIZATION;
	int*** cluster_ids;
    vector<Element> clusterCenters;
    float halt_condition;

	float*** allocate_3d_float_array();
	int*** allocate_3d_int_array();
    void init(int,int,int,int,int,int,float,int);
    void computeSlic(float***);
    int getClosestClusterID(vector<float>, float*);
    int*** getClusterIds();
    vector<Element> getClusterCenters();
    int getClusterNum();
};

#endif 
