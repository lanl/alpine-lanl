#include <cpp_headers.h>
#include <vtk_headers.h>

using namespace std;

struct Point
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
    vtkSmartPointer<vtkPolyData> surface;
    vector<Point> pca_transformed;
    int featureId;  
};
