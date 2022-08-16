/////////////////////////////////////////////////////
//Soumya Dutta
//OSU CSE 2017
/////////////////////////////////////////////////////

#ifndef PCP_OPENGL_WINDOW_HEADER_H
#define PCP_OPENGL_WINDOW_HEADER_H

#include <cpp_headers.h>
#include <QtOpenGL/QGLWidget>
#include <QMouseEvent>
#include <QMessageBox>
#include <glm/glm.hpp>
#include <vtk_headers.h>

using namespace std;
using namespace glm;

class pcp_openGL_window : public QGLWidget
{
    Q_OBJECT

public:
    explicit pcp_openGL_window(QWidget *parent=0);

    vector<vector<float> > dataInRange;
    vector<glm::vec3> selectedPoints;
    vector<float> selectedStallnessPoints;
    vector<vec2> upperSlider;
    vector<vec2> lowerSlider;
    int height;
    int width;
    int currentT;
    int drawHeight;
    int drawWidth;
    int axisStartX;
    int axisStartY;
    int axisDiff;
    int numVars;

    glm::vec2 uvelRange;
    glm::vec2 entropyRange;
    glm::vec2 temperatureRange;

    vtkSmartPointer<vtkColorTransferFunction> colorTF;
    vtkSmartPointer<vtkColorTransferFunction> load_ctf();
    vector<float> split(string str, string sep);
    vtkSmartPointer<vtkLookupTable> load_colormap();

    void draw_colormap(int startX,int startY);

    void paintGL();
    void initializeGL();
    void resizeGL(int w,int h);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
};

#endif
