#include <RenderWindowUISingleInheritance.h>
#include <ui_RenderWindowUISingleInheritance.h>
#include <vtkXMLImageDataReader.h>

using namespace std;

// 0 == isabel, 1 == Nyx, 2 == mfix, 3 == asteoid : Pick which data set
int dataset = 0;

vector<float> split_float(string str, string sep)
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

//Definition of methods
int compute_3d_to_1d_map(int x,int y,int z, int dimx, int dimy, int dimz)
{
    return x + dimx*(y+dimy*z);
}

vtkSmartPointer<vtkImageData> load_data(string fname)
{
    vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(fname.c_str());
    reader->Update();
    return reader->GetOutput();
}

// Constructor
RenderWindowUISingleInheritance::RenderWindowUISingleInheritance()
{
    this->ui = new Ui_RenderWindowUISingleInheritance;
    this->ui->setupUi(this);

    this->numBlocks = 1;

    if(dataset==1)
    {
        this->dim[0] = 256;
        this->dim[1] = 256;
        this->dim[2] = 256;
        this->initT = 400;
    }
    else if(dataset==0)
    {

        this->dim[0] = 250;
        this->dim[1] = 250;
        this->dim[2] = 50;
        this->initT = 25;
    }
    else if(dataset==2)
    {

        this->dim[0] = 128;
        this->dim[1] = 16;
        this->dim[2] = 128;
        this->initT = 150;
    }
    else if(dataset==3)
    {

        this->dim[0] = 300;
        this->dim[1] = 300;
        this->dim[2] = 300;
        this->initT = 100;
    }

    // VTK Renderer
    vtkrenderer.rendererptr = vtkSmartPointer<vtkRenderer>::New();
    this->vtkrenderer.rendererptr->SetBackground(1,1,1);

    // VTK/Qt wedded
    this->ui->qvtkWidget->GetRenderWindow()->AddRenderer(vtkrenderer.rendererptr.GetPointer());

    // Set up action signals and slots
    connect(this->ui->actionExit, SIGNAL(triggered()), this, SLOT(slotExit()));
}

void RenderWindowUISingleInheritance::update_renderer()
{
    this->vtkrenderer.rendererptr->GetRenderWindow()->Render();
}

void RenderWindowUISingleInheritance::slotExit()
{
    qApp->exit();
}

void RenderWindowUISingleInheritance::on_quit_clicked()
{
    qApp->exit();
}

glm::vec2 RenderWindowUISingleInheritance::getRange(string varName)
{
    double range[2];

    this->data->GetPointData()->GetArray(varName.c_str())->GetRange(range);

    glm::vec2 rangevals;
    rangevals.x = range[0];
    rangevals.y = range[1];

    return rangevals;
}

//load data and show grid
void RenderWindowUISingleInheritance::on_loadData_clicked()
{
    stringstream ss;
    int tt = this->initT;
    ss<<tt; //set the timestep to load
    string filename;

    if(dataset==1)
        filename = "/home/soumya/Test_DataSet/multivar_sampling_test_data/Nyx_density_temp_rho_400.vti";
    else if(dataset==0)
        filename = "/home/soumya/Test_DataSet/multivar_sampling_test_data/Isabel_pressure_velocity_qvapor.vti";
    else if(dataset==2)
        filename = "/home/soumya/Test_DataSet/multivar_sampling_test_data/mfix_density_gradient_gradient.vti";
    else if(dataset==3)
        filename = "/home/soumya/Test_DataSet/multivar_sampling_test_data/asteroid/asteroid_28649.vti";

    data = vtkSmartPointer<vtkImageData>::New();
    data = load_data(filename);

    cout<<"Loading is done"<<endl;

    //populate the variable range variables
    ////////////////////////////////////////

     if(dataset==1)
     {
         this->var1Range = this->getRange("logDensity");
         this->var2Range = this->getRange("logTemperature");
         this->var3Range = this->getRange("logRho");
     }
     else if(dataset==0)
     {
         this->var1Range = this->getRange("Pressure");
         this->var2Range = this->getRange("Velocity");
         this->var3Range = this->getRange("QVapor");
     }
     else if(dataset==2)
     {
         this->var1Range = this->getRange("density");
         this->var2Range = this->getRange("gradient");
         this->var3Range = this->getRange("gradient1");
     }
     else if(dataset==3)
     {
         this->var1Range = this->getRange("tev");
         this->var2Range = this->getRange("v02");
         this->var3Range = this->getRange("v03");
     }



    this->vtkrenderer.uvelRange = this->var1Range;
    this->vtkrenderer.entropyRange = this->var2Range;
    this->vtkrenderer.temperatureRange = this->var3Range;

    this->ui->pcp_window->uvelRange = this->var1Range;
    this->ui->pcp_window->entropyRange = this->var2Range;
    this->ui->pcp_window->temperatureRange = this->var3Range;

    //finallly draw the grid and show blade ids
    this->vtkrenderer.render_grid(data);

    this->ui->pcp_window->updateGL();

    update_renderer();
}

void RenderWindowUISingleInheritance::on_filter_points_clicked()
{
    double max,min;

    if(this->ui->selection_checkbox->isChecked())
    {
        this->vtkrenderer.filter_points(data,numBlocks,dim);
        this->ui->pcp_window->dataInRange.clear();

        //process data for pcp
        ////////////////////////////////
        max = this->vtkrenderer.tempPoints[0].x;
        min = this->vtkrenderer.tempPoints[0].x;
        for(int i=0;i<this->vtkrenderer.tempPoints.size();i++)
        {
            vector<float> temp;

            //normalize values for pcp plot
            float uvel = (this->vtkrenderer.tempPoints[i].x-this->var1Range.x)/(this->var1Range.y-this->var1Range.x);
            float entr = (this->vtkrenderer.tempPoints[i].y-this->var2Range.x)/(this->var2Range.y-this->var2Range.x);
            float temperature = (this->vtkrenderer.tempPoints[i].z-this->var3Range.x)/(this->var3Range.y-this->var3Range.x);

            temp.push_back(uvel);
            temp.push_back(entr);
            temp.push_back(temperature);

            this->ui->pcp_window->dataInRange.push_back(temp);
        }
    }

    this->ui->pcp_window->updateGL();
    update_renderer();
}

void RenderWindowUISingleInheritance::on_sphereRadiusSlider_sliderMoved(int position)
{
    double maxr = 100;
    double minr = 1;

    double* center;
    center = this->vtkrenderer.sphererep->GetCenter();

    double radius = minr + (position/100.0)*(maxr-minr);
    this->vtkrenderer.draw_spherewidget(radius,center);
    update_renderer();
}

void RenderWindowUISingleInheritance::on_draw_geom_surface_clicked()
{
    string varName = this->ui->variables->currentText().toStdString();

    //if(this->ui->surf_checkbox->isChecked())
    //    this->vtkrenderer.render_geom_surface(data,varName,36);

    update_renderer();
}

void RenderWindowUISingleInheritance::on_update_rendering_clicked()
{
    double isovalue = this->ui->isoval->toPlainText().toDouble();
    string varName = this->ui->variables->currentText().toStdString();
    double thLow = this->ui->thlow->toPlainText().toDouble();
    double thHigh = this->ui->thhigh->toPlainText().toDouble();

    if(this->ui->isosurf_checkbox->isChecked() && this->vtkrenderer.contouractor!=NULL)
        this->vtkrenderer.draw_isosurface(data,varName,isovalue,36);
    else
        this->vtkrenderer.rendererptr->RemoveActor(this->vtkrenderer.contouractor);

    if(this->ui->selection_checkbox->isChecked())
        this->vtkrenderer.filter_points(data,numBlocks,dim);
    else
        this->vtkrenderer.rendererptr->RemoveActor(this->vtkrenderer.pointactor);

    if(this->ui->th_checkbox->isChecked()&& this->vtkrenderer.pointactor1!=NULL)
        this->vtkrenderer.draw_thresholded_region(data,thLow,thHigh,36,varName,dim);
    else
        this->vtkrenderer.rendererptr->RemoveActor(this->vtkrenderer.pointactor1);

    update_renderer();
}

void RenderWindowUISingleInheritance::on_drawIsosurf_clicked()
{
    double isovalue = this->ui->isoval->toPlainText().toDouble();
    string varName = this->ui->variables->currentText().toStdString();

    if(this->ui->isosurf_checkbox->isChecked())
        this->vtkrenderer.draw_isosurface(data,varName,isovalue,36);

    update_renderer();
}

void RenderWindowUISingleInheritance::on_sphereWidget_clicked()
{
    double center[3] = {0,0,0};
    this->vtkrenderer.draw_spherewidget(5.0,center);
    update_renderer();
}

void RenderWindowUISingleInheritance::on_addSelection_clicked()
{
    vector<float> up;
    vector<float> down;
    up.clear();
    down.clear();
    up.resize(3);
    down.resize(3);

    double stallnessval = this->ui->stallness_val->toPlainText().toDouble();

    up[0] = this->var1Range.x + (this->var1Range.y - this->var1Range.x)*((this->ui->pcp_window->upperSlider[0].y - this->ui->pcp_window->axisStartY)/this->ui->pcp_window->drawHeight);
    down[0] = this->var1Range.x + (this->var1Range.y - this->var1Range.x)*((this->ui->pcp_window->lowerSlider[0].y - this->ui->pcp_window->axisStartY)/this->ui->pcp_window->drawHeight);

    up[1] = this->var2Range.x + (this->var2Range.y - this->var2Range.x)*((this->ui->pcp_window->upperSlider[1].y - this->ui->pcp_window->axisStartY)/this->ui->pcp_window->drawHeight);
    down[1] = this->var2Range.x + (this->var2Range.y - this->var2Range.x)*((this->ui->pcp_window->lowerSlider[1].y - this->ui->pcp_window->axisStartY)/this->ui->pcp_window->drawHeight);

    up[2] = this->var3Range.x + (this->var3Range.y - this->var3Range.x)*((this->ui->pcp_window->upperSlider[2].y - this->ui->pcp_window->axisStartY)/this->ui->pcp_window->drawHeight);
    down[2] = this->var3Range.x + (this->var3Range.y - this->var3Range.x)*((this->ui->pcp_window->lowerSlider[2].y - this->ui->pcp_window->axisStartY)/this->ui->pcp_window->drawHeight);

    float valx,valy,valz;
    for(int i=0;i<this->vtkrenderer.tempPoints.size();i++)
    {
        //get the variable values
        valx = this->vtkrenderer.tempPoints[i].x;
        valy = this->vtkrenderer.tempPoints[i].y;
        valz = this->vtkrenderer.tempPoints[i].z;

        //pcp filter satisfied
        if((valx>=down[0] && valx<=up[0]) && (valy>=down[1] && valy<=up[1]) && (valz>=down[2] && valz<=up[2]))
        {
            this->ui->pcp_window->selectedPoints.push_back(glm::vec3(valx,valy,valz));
            this->ui->pcp_window->selectedStallnessPoints.push_back(stallnessval);
        }
    }

    cout<<"sample set size: "<<this->ui->pcp_window->selectedPoints.size()<<endl;
}

void RenderWindowUISingleInheritance::on_saveSampleData_clicked()
{
    if(this->ui->pcp_window->selectedPoints.size()>0)
    {
        ofstream trainingfilein;
        string ffname = "/home/soumya/Test_DataSet/multivar_sampling_test_data/Isabel_training_data/sample_set_in.txt";
        trainingfilein.open (ffname.c_str());

        ofstream trainingfilein1;
        string ffname1 = "/home/soumya/Test_DataSet/multivar_sampling_test_data/Isabel_training_data/sample_set_out.txt";
        trainingfilein1.open (ffname1.c_str());

        ofstream trainingfilein2;
        string ffname2 = "/home/soumya/Test_DataSet/multivar_sampling_test_data/Isabel_training_data/sample_set_in_out.txt";
        trainingfilein2.open (ffname2.c_str());

        for(int i=0;i<this->ui->pcp_window->selectedPoints.size();i++)
        {
            glm::vec3 val = this->ui->pcp_window->selectedPoints[i];

            //if(val.x < -0.2) // no positive uvel is considered TODO
            {
                trainingfilein<<val.x<<","<<val.y<<","<<val.z<<endl;
                trainingfilein1<<this->ui->pcp_window->selectedStallnessPoints[i]<<endl;

                trainingfilein2<<val.x<<","<<val.y<<","<<val.z<<","<<this->ui->pcp_window->selectedStallnessPoints[i]<<endl;
            }
        }

        trainingfilein.close();
        trainingfilein1.close();
        trainingfilein2.close();
    }
    else
    {
        cout<<"Sample set is empty! Please select points!"<<endl;
    }
}

void RenderWindowUISingleInheritance::on_cleanSampleSet_clicked()
{
    if(this->ui->pcp_window->selectedPoints.size()>0)
    {
        this->ui->pcp_window->selectedPoints.clear();
        cout<<"Sample set cleared!"<<endl;
    }
}

void RenderWindowUISingleInheritance::on_pushButton_clicked()
{
    double thLow = this->ui->thlow->toPlainText().toDouble();
    double thHigh = this->ui->thhigh->toPlainText().toDouble();
    string varName = this->ui->variables->currentText().toStdString();

    if(this->ui->th_checkbox->isChecked())
        this->vtkrenderer.draw_thresholded_region(data,thLow,thHigh,36,varName,dim);

    update_renderer();
}
