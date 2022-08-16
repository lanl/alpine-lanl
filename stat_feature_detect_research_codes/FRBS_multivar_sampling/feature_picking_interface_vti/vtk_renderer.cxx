#include <vtk_renderer.h>
#include <vtkInteractorStyleTrackballCamera.h>

double center_pos[6] = {0,0.1,0,0.1,0,0.1};
double boxBound[6];

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

vtkSmartPointer<vtkColorTransferFunction> load_ctf(double min, double max)
{
    vtkSmartPointer<vtkColorTransferFunction> ctable = vtkSmartPointer<vtkColorTransferFunction>::New();
    string line;
    vector <float> cvals;

    //Read
    ifstream readinFis;
    readinFis.open("../rainbow_ctf.txt");

    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                cvals.push_back(v[0]);
            }
        }
    }

    readinFis.close();

    for(int i=0;i<cvals.size();i=i+4)
    {
        ctable->AddRGBPoint(min+ cvals[i]*(max-min),cvals[i+1],cvals[i+2],cvals[i+3]);
    }

    return ctable;
}

vtk_renderer::vtk_renderer()
{
    this->colorTF = vtkSmartPointer<vtkColorTransferFunction>::New();
}

double dist(double* center, double* point)
{
    double distance_val=0;

    for(int i=0;i<3;i++)
        distance_val += (center[i]-point[i])*(center[i]-point[i]);

    return sqrt(distance_val);
}

// This does the actual work.
// Callback for the boxwidget interaction
class vtkBoxCallback : public vtkCommand, vtk_renderer
{

public:
    static vtkBoxCallback *New()
    {
        return new vtkBoxCallback;
    }

    virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
        vtkBoxWidget2 *boxWidget = reinterpret_cast<vtkBoxWidget2*>(caller);
        double* bb;

        //// Get the actual box coordinates/planes
        //vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        //static_cast<vtkBoxRepresentation*>(boxWidget->GetRepresentation())->GetPolyData(polydata);
        // Display the center of the box
        //double p[3];
        //polydata->GetPoint(14,p); // As per the vtkBoxRepresentation documentation, the 15th point (index 14) is the center of the box
        //std::cout << "Box center: " << p[0] << " " << p[1] << " " << p[2] << std::endl;

        //get the boxwidget bound in physical space
        bb = static_cast<vtkBoxRepresentation*>(boxWidget->GetRepresentation())->GetBounds();
        boxBound[0] = bb[0];
        boxBound[1] = bb[1];
        boxBound[2] = bb[2];
        boxBound[3] = bb[3];
        boxBound[4] = bb[4];
        boxBound[5] = bb[5];
    }

    vtkBoxCallback(){}
};

void vtk_renderer :: render_grid(vtkSmartPointer<vtkImageData> data)
{
    if (data.GetPointer()==NULL)
        return;

    /// ISABEL
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    camera->SetFocalPoint(125,125,25); //center of data
    camera->SetPosition(125,125,-600);
    camera->SetViewUp(0,0,0);

    //    /// NYX
    //    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    //    camera->SetFocalPoint(125,125,125); //center of data
    //    camera->SetPosition(125,125,-600);
    //    camera->SetViewUp(0,0,0);

    //    /// MFIX
    //    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    //    camera->SetFocalPoint(64,8,64); //center of data
    //    camera->SetPosition(64,8,-300);
    //    camera->SetViewUp(1,0,0);

    //    //    /// Asteroid
    //    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    //    camera->SetFocalPoint(150,150,150); //center of data
    //    camera->SetPosition(150,150,-1000);
    //    camera->SetViewUp(0,0,0);

    //Draw outline of the grid
    vtkSmartPointer<vtkOutlineFilter> outlineFilter = vtkSmartPointer<vtkOutlineFilter>::New();
    outlineFilter->SetInputData(data);
    outlineFilter->Update();
    vtkSmartPointer<vtkPolyDataMapper> outlineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    outlineMapper->SetInputConnection(outlineFilter->GetOutputPort(0));

    outlineActor = vtkSmartPointer<vtkActor>::New();
    outlineActor->SetMapper(outlineMapper);
    outlineActor->GetProperty()->SetColor(0.0,0.0,0.0);


    //Visualize Using Renderer
    rendererptr->SetActiveCamera(camera);
    rendererptr->AddActor(outlineActor);
}

void vtk_renderer::filter_points(vtkSmartPointer<vtkImageData> data, int numBlocks,int* dim)
{
    if (data.GetPointer()==NULL)
        return;

    tempPoints.clear();

    vtkSmartPointer<vtkPoints> filteredPoints = vtkSmartPointer<vtkPoints>::New();
    rendererptr->RemoveActor(pointactor);

    if (sphererep.GetPointer()!=NULL)
    {
        double* center = sphererep->GetCenter();
        double radius = sphererep->GetRadius();

        for(int i=0;i<numBlocks;i++)
        {
            double point[3];
            double val[3];

            int index=0;

            for(int p=0;p<dim[2];p++)
                for(int q=0;q<dim[1];q++)
                    for(int r=0;r<dim[0];r++)
                    {
                        point[0] = r;
                        point[1] = q;
                        point[2] = p;

                        //if points are within the sphere
                        if(dist(center,point)<=radius)
                        {
                            // isabel
                            val[0] = data->GetPointData()->GetArray("Pressure")->GetTuple1(index);
                            val[1] = data->GetPointData()->GetArray("Velocity")->GetTuple1(index);
                            val[2] = data->GetPointData()->GetArray("QVapor")->GetTuple1(index);

                            ///// mfix
                            //                            val[0] = data->GetPointData()->GetArray("density")->GetTuple1(index);
                            //                            val[1] = data->GetPointData()->GetArray("gradient")->GetTuple1(index);
                            //                            val[2] = data->GetPointData()->GetArray("gradient1")->GetTuple1(index);


                            //                            /// nyx
                            //                            val[0] = data->GetPointData()->GetArray("logDensity")->GetTuple1(index);
                            //                            val[1] = data->GetPointData()->GetArray("logTemperature")->GetTuple1(index);
                            //                            val[2] = data->GetPointData()->GetArray("logRho")->GetTuple1(index);


                            //                            /// Asteroid
                            //                            val[0] = data->GetPointData()->GetArray("tev")->GetTuple1(index);
                            //                            val[1] = data->GetPointData()->GetArray("v02")->GetTuple1(index);
                            //                            val[2] = data->GetPointData()->GetArray("v03")->GetTuple1(index);

                            filteredPoints->InsertNextPoint(point);
                            tempPoints.push_back(glm::vec3(val[0],val[1],val[2]));
                        }

                        index++;
                    }
        }

        vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();
        pointsPolydata->SetPoints(filteredPoints);

        vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertexFilter->SetInputData(pointsPolydata);
        vertexFilter->Update();

        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        polydata->ShallowCopy(vertexFilter->GetOutput());

        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(polydata);

        pointactor = vtkSmartPointer<vtkActor>::New();
        pointactor->SetMapper(mapper);
        pointactor->GetProperty()->SetPointSize(2);
        pointactor->GetProperty()->SetColor(1,0,0);
        pointactor->GetProperty()->SetOpacity(0.5);

        rendererptr->AddActor(pointactor);
    }
}

void vtk_renderer::draw_spherewidget(double radius,double* center_pos)
{
    sphereWidget = vtkSmartPointer<vtkSphereWidget2>::New();
    sphereWidget->SetInteractor(this->rendererptr->GetRenderWindow()->GetInteractor());
    sphererep = vtkSmartPointer<vtkSphereRepresentation>::New();
    sphereWidget->SetRepresentation(sphererep);
    sphererep->SetCenter(center_pos);
    sphererep->SetRadius(radius);
    sphererep->SetRadialLine(0);
    sphererep->GetSphereProperty()->SetColor(0,0,0);
    sphererep->HandleTextOff(); //removes the text
    sphererep->RadialLineOff(); // removes the red line
    sphereWidget->On();
}

void vtk_renderer::draw_isosurface(vtkSmartPointer<vtkImageData> data, string variable, double isoval, int numBlocks)
{
    if (data.GetPointer()==NULL)
        return;

    rendererptr->RemoveActor(contouractor);

    //set the variable first
    data->GetPointData()->SetActiveAttribute(variable.c_str(), vtkDataSetAttributes::SCALARS);

    vtkSmartPointer<vtkContourFilter> contourFilter = vtkSmartPointer<vtkContourFilter>::New();
    contourFilter->SetInputData(0,data);
    contourFilter->SetValue(0,isoval);
    contourFilter->SetComputeScalars(0);
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer< vtkPolyDataMapper>::New();
    mapper->SetInputConnection((0, contourFilter->GetOutputPort(0)));

    contouractor = vtkSmartPointer<vtkActor>::New();
    contouractor->SetMapper(mapper);
    contouractor->GetProperty()->SetColor(0.0,0.0,1.0);
    contouractor->GetProperty()->SetOpacity(0.5);

    //Visualize Using Renderer
    rendererptr->AddActor(contouractor);
}

void vtk_renderer :: draw_thresholded_region(vtkSmartPointer<vtkImageData> data, double thlow, double thhigh, int numBlocks, string variable, int* dim)
{
    if (data.GetPointer()==NULL)
        return;

    else
    {
        vtkSmartPointer<vtkPoints> filteredPoints = vtkSmartPointer<vtkPoints>::New();
        rendererptr->RemoveActor(pointactor1);

        double point[3];
        double val;

        int index=0;

        for(int p=0;p<dim[2];p++)
            for(int q=0;q<dim[1];q++)
                for(int r=0;r<dim[0];r++)
                {
                    point[0] = r;
                    point[1] = q;
                    point[2] = p;

                    val = data->GetPointData()->GetArray(variable.c_str())->GetTuple1(index);

                    //if points are within range
                    if(val>thlow && val<thhigh)
                    {
                        filteredPoints->InsertNextPoint(point);
                    }

                    index++;
                }


        vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();
        pointsPolydata->SetPoints(filteredPoints);

        vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertexFilter->SetInputData(pointsPolydata);
        vertexFilter->Update();

        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        polydata->ShallowCopy(vertexFilter->GetOutput());

        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(polydata);

        pointactor1 = vtkSmartPointer<vtkActor>::New();
        pointactor1->SetMapper(mapper);
        pointactor1->GetProperty()->SetPointSize(2);
        pointactor1->GetProperty()->SetColor(0,0,1);
        pointactor1->GetProperty()->SetOpacity(0.6);

        rendererptr->AddActor(pointactor1);
    }
}
