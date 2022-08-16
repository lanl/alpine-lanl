/////////////////////////////////////////////////////
//Soumya Dutta
//OSU CSE Spring 2017
/////////////////////////////////////////////////////

#ifndef VTK_RENDERER
#define VTK_RENDERER

#include <cpp_headers.h>
#include <vtk_headers.h>
#include <glm/glm.hpp>


using namespace std;

class vtk_renderer
{
    
public:

    //constructor
    vtk_renderer();
    vtkSmartPointer<vtkRenderer> rendererptr;
    vtkSmartPointer<vtkBoxWidget2> boxWidget;
    vtkSmartPointer<vtkSphereWidget2> sphereWidget;
    vtkSmartPointer<vtkActor> outlineActor;
    vtkSmartPointer<vtkActor> pointactor;
    vtkSmartPointer<vtkActor> pointactor1;
    vtkSmartPointer<vtkSphereRepresentation> sphererep;
    vtkSmartPointer<vtkActor> contouractor;
    vtkSmartPointer<vtkActor> geomactor;
    vtkSmartPointer<vtkActor> pointLabelActor;
    vector<glm::vec3> tempPoints;
    vtkSmartPointer<vtkColorTransferFunction> colorTF;

    glm::vec2 uvelRange;
    glm::vec2 entropyRange;
    glm::vec2 temperatureRange;

    //public member  functions
    void render_grid(vtkSmartPointer<vtkImageData>);
    void draw_boxwidget();
    void filter_points(vtkSmartPointer<vtkImageData>,int,int*);
    void draw_spherewidget(double,double*);
    void draw_isosurface(vtkSmartPointer<vtkImageData>,string,double,int);
    void render_geom_surface(vtkSmartPointer<vtkImageData>,string,int);
    void render_blade_id(vtkSmartPointer<vtkImageData>, int);
    void draw_thresholded_region(vtkSmartPointer<vtkImageData>,double,double,int,string,int*);
};

#endif
