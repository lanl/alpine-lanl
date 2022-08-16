#ifndef RenderWindowUISingleInheritance_H
#define RenderWindowUISingleInheritance_H

#include <cpp_headers.h>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QComboBox>
#include <vtk_headers.h>
#include <vtk_renderer.h>
#include <build/ui_RenderWindowUISingleInheritance.h>
#include <glm/glm.hpp>
#include <vtkColorTransferFunction.h>

class RenderWindowUISingleInheritance : public QMainWindow, public Ui::RenderWindowUISingleInheritance
{
    Q_OBJECT

public:

    // Constructor/Destructor
    RenderWindowUISingleInheritance();

    vtkSmartPointer<vtkImageData> data;
    vtk_renderer vtkrenderer;
    int numBlocks;
    int dim[3];
    int initT;
    int currenttimestep;
    glm::vec2 var1Range;
    glm::vec2 var2Range;
    glm::vec2 var3Range;

public slots:

    virtual void slotExit();

private slots:
    void update_renderer();
    void on_filter_points_clicked();
    void on_sphereRadiusSlider_sliderMoved(int position);
    void on_draw_geom_surface_clicked();
    void on_update_rendering_clicked();
    void on_quit_clicked();
    void on_loadData_clicked();
    void on_drawIsosurf_clicked();
    void on_sphereWidget_clicked();
    void on_addSelection_clicked();
    void on_saveSampleData_clicked();
    glm::vec2 getRange(string);

    void on_cleanSampleSet_clicked();

    void on_pushButton_clicked();

private:

    // Designer form
    Ui_RenderWindowUISingleInheritance *ui;
};

#endif
