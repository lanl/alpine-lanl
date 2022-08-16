//to run: ./volrenderer 128 128 128 /home/soumya/Test_DataSet/vortex/vorts1.data.raw 12.1394 0.0070086 7.0

#include "cpp_headers.h"
#include "vtk_headers.h"
#include "vtkGPUVolumeRayCastMapper.h"

using namespace std;

int numCtrlPts=6;
int split_screen_by = 4;
float **opacityCtrlPointsX;
float **opacityCtrlPointsY;

int main(int argc, char** argv)
{
    int xdim=atoi(argv[1]);
    int ydim=atoi(argv[2]);
    int zdim=atoi(argv[3]);
    char* firstfilename = argv[4];
    float maxi = atof(argv[5]);
    float mini = atof(argv[6]);
    float th = atof(argv[7]);
    int UNSIGNED_RANGE=256;
    float buff1=0.0;
    float valx,valy;
    int windowWidth=800;
    int windowHeight=800;
    double colorLut[3];
    double lookupLut;
    vector <vector <vector <float> > > rawDataStorage;
    float val = ((th - mini)/(maxi-mini))*(UNSIGNED_RANGE-1);

    //Resize the 4D fileStorage Vector
    rawDataStorage.resize(zdim);
    for(int i=0;i<rawDataStorage.size();i++)
        rawDataStorage[i].resize(ydim);

    opacityCtrlPointsX = (float **)malloc(split_screen_by*sizeof(float *));
    for(int i=0;i<split_screen_by;i++)
        opacityCtrlPointsX[i] = (float *)malloc(numCtrlPts*sizeof(float));

    opacityCtrlPointsY = (float **)malloc(split_screen_by*sizeof(float *));
    for(int i=0;i<split_screen_by;i++)
        opacityCtrlPointsY[i] = (float *)malloc(numCtrlPts*sizeof(float));

    vtkSmartPointer<vtkRenderer> ren=vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renWin=vtkSmartPointer<vtkRenderWindow>::New();
    renWin->AddRenderer(ren);

    vtkSmartPointer<vtkRenderWindowInteractor   > iren=vtkSmartPointer<vtkRenderWindowInteractor>::New();
    iren->SetRenderWindow(renWin);

    //Add Axes
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Scale(30.0, 30.0, 30.0);
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    axes->SetUserTransform(transform);
    axes->AxisLabelsOff();

    // Create an image data for the volume file
    vtkSmartPointer<vtkImageData> data = vtkSmartPointer<vtkImageData>::New();
    data = vtkImageData::New();
    data->SetDimensions(xdim,ydim,zdim);
    data->AllocateScalars(VTK_UNSIGNED_CHAR,1); //New imp function in VTK6.x feature, different from VTK5.x

    FILE *fp1 = fopen(firstfilename,"rb");

    //Load vtkimage with raw file data
    for(int p=0; p<zdim; p++)
        for(int q=0; q<ydim; q++)
            for(int r=0; r<xdim; r++)
            {
                fread(&buff1,1,sizeof(buff1),fp1);
                rawDataStorage[p][q].push_back(buff1);
            }

    maxi = mini = rawDataStorage[0][0][0];
    for(int p=0; p<zdim; p++)
        for(int q=0; q<ydim; q++)
            for(int r=0; r<xdim; r++)
            {
                if(maxi<rawDataStorage[p][q][r])
                    maxi = rawDataStorage[p][q][r];

                if(mini>rawDataStorage[p][q][r])
                    mini = rawDataStorage[p][q][r];
            }

    for(int p=0; p<zdim; p++)
        for(int q=0; q<ydim; q++)
            for(int r=0; r<xdim; r++)
            {
                unsigned char* pixel = static_cast<unsigned char*>(data->GetScalarPointer(r,q,p)); // keep the order inside GetScalarPointer in mind important
                pixel[0] = (unsigned char)((rawDataStorage[p][q][r]-mini)*UNSIGNED_RANGE/(maxi-mini));
            }

    rawDataStorage.clear();

    vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
    vtkSmartPointer<vtkSmartVolumeMapper> volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
    volumeMapper->SetBlendModeToComposite(); // composite first
    volumeMapper->SetInputData(data);
    
    // The property describes how the data will look
    volumeProperty->SetColor(0,colorTransferFunction);
    volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

    volumeProperty->ShadeOn(); //ShadeOn() would enable Phong shading. expensive.
    volumeProperty->SetAmbient(0.4);
    volumeProperty->SetDiffuse(1.0);
    volumeProperty->SetSpecular(0.3);
    volumeProperty->SetSpecularPower(30);    

    // Create transfer mapping scalar value to opacity.
	vtkSmartPointer<vtkPiecewiseFunction> opacityTransferFunction = vtkSmartPointer<vtkPiecewiseFunction>::New();
	opacityTransferFunction->AddPoint(0,  0.0);
	opacityTransferFunction->AddPoint(val-1, 0.0);
	opacityTransferFunction->AddPoint(val, 0.7);
	opacityTransferFunction->AddPoint(255.0, 1.0);

	volumeProperty->SetScalarOpacity(0,opacityTransferFunction);

    // The volume holds the mapper and the property and can be used to position/orient the volume
    vtkSmartPointer<vtkVolume> vol = vtkSmartPointer<vtkVolume> :: New();
    vol->SetMapper(volumeMapper);
    vol->SetProperty(volumeProperty);

    // Create a lookup table to show variability  // use a traditional RGB scale .. how? TODO
    vtkSmartPointer<vtkLookupTable> Lut = vtkSmartPointer<vtkLookupTable>::New();
    Lut->SetHueRange(0.7,0.0);
    Lut->SetValueRange(1.0, 1.0);
    Lut->SetSaturationRange(1.0, 1.0);
    Lut->SetTableRange (0.0,1.0);
    Lut->SetNumberOfColors(UNSIGNED_RANGE);
    Lut->Build();

    float colorValues[6] = {0, 51, 102, 153, 204, 255};
    //float colorValues[4] = {0, 125, 255}; //Use this for pure RGB color map

    //Fixed Color TF
    for(int i=0;i<numCtrlPts;i++)
    {
        lookupLut = colorValues[i]/(float)UNSIGNED_RANGE;
        Lut->GetColor(lookupLut,colorLut);
        colorTransferFunction->AddRGBPoint(colorValues[i],colorLut[0],colorLut[1],colorLut[2]);
    }
    //done setting up color transfer func

    vtkSmartPointer<vtkTextActor> textActor = vtkSmartPointer<vtkTextActor>::New();
    string names = "Set Volume Title";
    textActor->SetPosition(240.0,10.0);
    textActor->GetTextProperty()->SetFontSize (18);
    textActor->GetTextProperty()->SetBold(2);
    textActor->GetTextProperty()->SetItalic(2.5);
    textActor->GetTextProperty()->SetColor(0.0,0.0,0.0);
    textActor->SetInput (names.c_str());

    ren->AddActor(textActor);
    ren->AddVolume(vol);
    //ren->AddActor(axes);
    ren->SetBackground(1,1,1);
    renWin->SetSize(windowWidth,windowHeight);
    renWin->SetWindowName("Set Window Title Here");
    renWin->Render();

    iren->Initialize();
    renWin->Render();
    iren->Start();

    return 0;
}
