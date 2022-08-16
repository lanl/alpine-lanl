#ifndef FEATURE_CLASS_HEADER
#define FEATURE_CLASS_HEADER

#include <cpp_headers.h>
#include <vtk_headers.h>

using namespace std;

struct PointClass
{
  float x,y,z;
};

class Feature_class
{
public:

    int numofpts;
    float bbox[6];
    float cog[3];
    float cbbox[3];
    float mass;  
    vtkSmartPointer<vtkPolyData> surface = vtkSmartPointer<vtkPolyData>::New();
    vector<PointClass> pca_transformed;
    int featureId;  
};

#endif