#include <cpp_headers.h>
#include <glm_headers.h>
#include <vtk_headers.h>
#include <vtkTransformFilter.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */  

using namespace std;

int main(int argc, char** argv)
{
    /* initialize random seed: */
  srand (time(NULL));

    // to turn off vtk warning messages
    vtkObject::GlobalWarningDisplayOff();

    //Get the file name
    string inputFilename;
    //string path = "/home/soumya/Test_DataSet/vortex/vti/";
    string path = "../";
    int initStep=1;
    stringstream tt;
    tt<<initStep;
    //inputFilename = path + "vortex_" + tt.str() + ".vti";
    inputFilename = path + "multi_spheres.vti";
    int threshold = 227;
    int dim[3] = {128,128,128};
    double center[3] = {0,0,0};
    double new_center[3] = {0,0,0};
    double bbox[6] = {0,0,0,0,0,0,};
    double new_bbox[6] = {0,0,0,0,0,0,};

    //Read the raw field OR the source data
    vtkSmartPointer<vtkXMLImageDataReader> reader1 = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader1->SetFileName(inputFilename.c_str());
    reader1->Update();

    //Threshold the field to extract different features
    vtkSmartPointer<vtkThreshold> thresholding = vtkSmartPointer<vtkThreshold>::New();
    thresholding->ThresholdByUpper(threshold);
    thresholding->SetInputData(reader1->GetOutput());

    //Find connected components
    vtkSmartPointer<vtkConnectivityFilter> segmentation = vtkSmartPointer<vtkConnectivityFilter>::New();
    segmentation->SetInputConnection( thresholding->GetOutputPort() );
    segmentation->SetExtractionModeToAllRegions();
    segmentation->ColorRegionsOn();
    segmentation->Update();

    vtkSmartPointer<vtkUnstructuredGrid> ug = vtkSmartPointer<vtkUnstructuredGrid>::New();
    ug->ShallowCopy(segmentation->GetOutput());
    int num_segments = segmentation->GetNumberOfExtractedRegions() ;

    vtkSmartPointer<vtkTransformFilter> transformFilter = vtkSmartPointer<vtkTransformFilter>::New();

    vtkSmartPointer<vtkUnstructuredGrid> ug1 = vtkSmartPointer<vtkUnstructuredGrid>::New();

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(dim[0],dim[1],dim[2]);
    imageData->AllocateScalars(VTK_FLOAT, 1);

    vtkDataArray *transformed_data;
    vtkDataArray *transformed_point_ary;

    for(int i=0; i<num_segments; i++)
    {
        //if(i==12) //pick which object
        {
            vtkSmartPointer<vtkThreshold> thresholding1 = vtkSmartPointer<vtkThreshold>::New();
            thresholding1->SetInputData(ug);
            thresholding1->ThresholdBetween(i,i);
            thresholding1->Update();

            vtkSmartPointer<vtkUnstructuredGrid> object = vtkSmartPointer<vtkUnstructuredGrid>::New();
            object->ShallowCopy(thresholding1->GetOutput());
            object->GetBounds(bbox);
            center[0] = bbox[0] + (bbox[1]-bbox[0])/2.0;
            center[1] = bbox[2] + (bbox[3]-bbox[2])/2.0;
            center[2] = bbox[4] + (bbox[5]-bbox[4])/2.0;

            cout<<"center before transform: "<<center[0]<<" "<<center[1]<<" "<<center[2]<<endl;

            vtkSmartPointer<vtkTransform> translation1 = vtkSmartPointer<vtkTransform>::New();
            translation1->PreMultiply();
            translation1->Translate(15,0,0);
            // translation1->Translate(center[0],center[1],center[2]);
            // translation1->RotateZ(45);
            // translation1->Translate(-center[0],-center[1],-center[2]);
            transformFilter->SetInputData(object);
            transformFilter->SetTransform(translation1);
            transformFilter->Update();

            ug1 = vtkUnstructuredGrid::SafeDownCast ( transformFilter->GetOutput() );
            transformed_data = ug1->GetPointData()->GetArray("ImageScalars");
            transformed_point_ary = ug1->GetPoints()->GetData();

            ug1->GetBounds(new_bbox);
            new_center[0] = new_bbox[0] + (new_bbox[1]-new_bbox[0])/2.0;
            new_center[1] = new_bbox[2] + (new_bbox[3]-new_bbox[2])/2.0;
            new_center[2] = new_bbox[4] + (new_bbox[5]-new_bbox[4])/2.0;
            cout<<"new center after transform: "<<new_center[0]<<" "<<new_center[1]<<" "<<new_center[2]<<endl;

            for(int p=0; p<dim[2]; p++)
                for(int q=0; q<dim[1]; q++)
                    for(int r=0; r<dim[0]; r++)
                    {
                        float* pixel = static_cast<float*>(imageData->GetScalarPointer(r,q,p));
                        pixel[0] = rand() % 100 + 1;//100.0;
                    }

            for(int j=0; j<object->GetPoints()->GetNumberOfPoints(); j++)
            {
                double pos[3];
                double val;

                transformed_point_ary->GetTuple(j,pos);
                transformed_data->GetTuple(j,&val);
                int mm = ceil(pos[0]);
                int nn = ceil(pos[1]);
                int oo = ceil(pos[2]);

                float* pixel = static_cast<float*>(imageData->GetScalarPointer(mm,nn,oo));
                pixel[0] = (float)val;

                mm = floor(pos[0]);
                nn = floor(pos[1]);
                oo = floor(pos[2]);

                float* pixel1 = static_cast<float*>(imageData->GetScalarPointer(mm,nn,oo));
                pixel1[0] = (float)val;

                mm = round(pos[0]);
                nn = round(pos[1]);
                oo = round(pos[2]);

                float* pixel2 = static_cast<float*>(imageData->GetScalarPointer(mm,nn,oo));
                pixel2[0] = (float)val;
            }
        }
    }

    // //Write file
    // vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    // writer->SetFileName("output6.vtu");
    // writer->SetInputData(transformFilter->GetOutput());
    // writer->Write();

    vtkSmartPointer<vtkXMLImageDataWriter> writer1 =  vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer1->SetFileName("output5.vti");
    writer1->SetInputData(imageData);
    writer1->Write();

    return 0;
}
