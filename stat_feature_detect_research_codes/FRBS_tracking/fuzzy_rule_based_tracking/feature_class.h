#include <cpp_headers.h>
#include <vtk_headers.h>

using namespace std;

class Feature_class
{
public:

    int numofpts;
    float bbox[6];
    float cog[3];
    float cbbox[3];
    float mass;  
    vtkSmartPointer<vtkPolyData> surface = vtkSmartPointer<vtkPolyData>::New();
    //vtkSmartPointer<vtkPolyData> outer_surface = vtkSmartPointer<vtkPolyData>::New();
    int featureId;  
};
