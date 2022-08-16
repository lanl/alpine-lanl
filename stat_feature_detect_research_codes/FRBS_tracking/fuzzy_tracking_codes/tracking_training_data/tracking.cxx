#include <cpp_headers.h>
#include <vtkSmartPointer.h>
#include <vtkImageMedian3D.h>
#include <vtkImageData.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLImageDataReader.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkThreshold.h>
#include <vtkConnectivityFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkOutlineFilter.h>
#include <vtkDataArray.h>
#include <vtkDelaunay3D.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkXMLPolyDataWriter.h>
#include <map>
#include <glm_headers.h>
#include <feature_class.h>

using namespace std;

class Feature_vector
{
public:
    int id;
    float feature_vec[4];
};

int main(int argc, char** argv)
{
    // To turn off vtk warning messages
    vtkObject::GlobalWarningDisplayOff();

    float xyzf[3];
    float vv=0.9; float uu = 0.1;
    int isabel_data = 0; //TODO
    int neg_data = 0; //TODO
    int mfix_data=0;
    int massTH = 100; // only applicable for MFIX=500

    //    //Combustion
    //    int xdim=240; //TODO
    //    int ydim=360; //TODO
    //    int zdim=60; //TODO
    //    int timeSteps=10;//TODO
    //    int initStep=50; //TODO
    //    float threshold = 350000;
    //    xyzf[0] = 60;
    //    xyzf[1] = 135;
    //    xyzf[2] = 42.5;
    //    const int MIN_POINTS = 25;
    //    string path = "/home/soumya/Test_DataSet/Combustion/vorticity_vti/";
    //    float neighborhood = 0.1;

    //Vortex: works fine
    int xdim=128; //TODO
    int ydim=128; //TODO
    int zdim=128; //TODO
    int timeSteps=10;//TODO
    int initStep=1; //TODO
    float threshold = 7;
    xyzf[0] = 63;
    xyzf[1] = 55;
    xyzf[2] = 2.5;
    string path = "/Users/sdutta/Data/vortex_vti/";
    float neighborhood = 0.4;
    int tstep_window_multiplier = 1;

    //    //Isabel lambda2
    //    int xdim=250; //TODO
    //    int ydim=250; //TODO
    //    int zdim=50; //TODO
    //    int timeSteps=10;//TODO
    //    int initStep=1; //TODO
    //    float threshold = -1.0;
    //    xyzf[0] = 142;
    //    xyzf[1] = 60;
    //    xyzf[2] = 7.5;
    //    const int MIN_POINTS = 50;
    //    string path = "/home/soumya/Test_DataSet/Isabel_vortex/vti/";
    //    float neighborhood = 10.0;

    // //MFiX feature similarity: feature example 1
    // int xdim=128; //TODO
    // int ydim=16; //TODO
    // int zdim=128; //TODO
    // int timeSteps=10;//TODO
    // string path = "/Users/sdutta/Data/MFIX_bubble_fields_localbound/";
    // int tstep_window_multiplier = 100;
    // float threshold = 10;
    // float neighborhood = 1.0;
    // int initStep=175; //TODO // gives good training result
    // xyzf[0] = 0.0283007;
    // xyzf[1] = 0.0015998;
    // xyzf[2] = 0.0101638;

    // // int initStep=255; //TODO
    // // xyzf[0] = 0.0283007;
    // // xyzf[1] = 0.0015998;
    // // xyzf[2] = 0.0418226; 

    const int MIN_POINTS = massTH;
    string arrName = "ImageScalars";

    /////////////////////////////////////////////////////////////////////////
    float maxdist = sqrt(zdim*zdim + ydim*ydim + xdim*xdim);
    int num_segments;
    vector<Feature_class> init_feature_list;

    int index=0; //feature index
    system("rm *.vti");
    system("rm *.vtu");
    system("rm *.csv");

    //Get the file name
    string inputFilename;
    stringstream tt;

    if(!mfix_data)
        tt<<initStep;
    else
        tt<<(initStep*tstep_window_multiplier);

    ////TODO
    //inputFilename = path + "tornado_lambda2_" + tt.str() + ".vti";
    inputFilename = path + "vortex_" + tt.str() + ".vti";
    //inputFilename = path + "combustion_vorticity_" + tt.str() + ".vti";
    //inputFilename = path + "isabel_lambda2_" + tt.str() + ".vti";
    //inputFilename = path + "fcc" + tt.str() + ".vti";

    //Read the probability field generated
    vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(inputFilename.c_str());
    reader->Update();

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(xdim,ydim,zdim);
    imageData->AllocateScalars(VTK_FLOAT, 1);
    imageData = reader->GetOutput();

    //Threshold the field to extract different features
    vtkSmartPointer<vtkThreshold> thresholding = vtkSmartPointer<vtkThreshold>::New();
    if(neg_data || mfix_data)
        thresholding->ThresholdByLower( threshold );
    else
        thresholding->ThresholdByUpper( threshold );
    thresholding->SetInputData( imageData );

    //Find connected components
    vtkSmartPointer<vtkConnectivityFilter> segmentation = vtkSmartPointer<vtkConnectivityFilter>::New();
    segmentation->SetInputConnection(thresholding->GetOutputPort());
    segmentation->SetExtractionModeToAllRegions();
    segmentation->ColorRegionsOn();
    segmentation->Update();

    vtkSmartPointer<vtkUnstructuredGrid> ug = vtkSmartPointer<vtkUnstructuredGrid>::New();
    ug->ShallowCopy(segmentation->GetOutput());
    num_segments = segmentation->GetNumberOfExtractedRegions() ;
    cout << "Extracted regions at initial time has "<< num_segments << " segments.."<< endl;

    //Second level threshold on unstructured grid: extract individual features
    for(int i=0; i<num_segments; i++)
    {
        double bbox[6] = {0,0,0,0,0,0};
        float cog[3] = {0,0,0};
        int numofpts=0;
        float datatotal=0;

        vtkSmartPointer<vtkThreshold> thresholding1 = vtkSmartPointer<vtkThreshold>::New();
        thresholding1->SetInputData(ug);
        thresholding1->ThresholdBetween(i,i);
        thresholding1->Update();

        vtkSmartPointer<vtkUnstructuredGrid> object = vtkSmartPointer<vtkUnstructuredGrid>::New();
        object->ShallowCopy(thresholding1->GetOutput());
        object->GetBounds(bbox);

        vtkDataArray *data = object->GetPointData()->GetArray(arrName.c_str());
        vtkDataArray *point_ary = object->GetPoints()->GetData();

        //Iterate over all points of a selected feature
        for(int j=0; j<object->GetPoints()->GetNumberOfPoints(); j++)
        {
            double pos[3];
            double val;

            point_ary->GetTuple(j,pos);
            data->GetTuple(j,&val);
            cog[0] += pos[0]; cog[1] += pos[1]; cog[2] += pos[2];
            numofpts++;
            datatotal +=fabs(val);
        }

        //Only push if satisfy threshold// ignore small fragments
        if(numofpts > MIN_POINTS)
        {
            Feature_class f;
            f.numofpts = numofpts;
            f.cog[0] = cog[0]/numofpts; f.cog[1] = cog[1]/numofpts; f.cog[2] = cog[2]/numofpts;
            f.bbox[0] = bbox[0]; f.bbox[1] = bbox[1]; f.bbox[2] = bbox[2]; f.bbox[3] = bbox[3]; f.bbox[4] = bbox[4]; f.bbox[5] = bbox[5];
            f.cbbox[0] = bbox[0] + (bbox[1]-bbox[0])/2.0;
            f.cbbox[1] = bbox[2] + (bbox[3]-bbox[2])/2.0;
            f.cbbox[2] = bbox[4] + (bbox[5]-bbox[4])/2.0;

            f.featureId = i;
            f.mass = datatotal;

            //push the feature to list
            init_feature_list.push_back(f);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Processing initial time step ends here
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //open the output training data file
    ofstream trainingfilein;
    string ffname1 = "training_feature_props_in.csv";
    trainingfilein.open (ffname1.c_str());

    ofstream trainingfileout;
    string ffname2 = "training_feature_props_out.csv";
    trainingfileout.open (ffname2.c_str());

    ofstream trainingfileref;
    string ffname3 = "training_feature_props_all.csv";
    trainingfileref.open (ffname3.c_str());

    ofstream trainingfileref1;
    string ffname4 = "training_feature_props_all_no_header.csv";
    trainingfileref1.open (ffname4.c_str());

    trainingfileref<<"x,y,z,w,o"<<endl;

    Feature_class fstar;
    cout<<"feature list size: "<<init_feature_list.size()<<endl;

    //decide which feature to track
    vector<float> dd;
    float temp=0;

    for(int i=0; i<init_feature_list.size(); i++)
    {
        //Compute distance between centers
        temp = 0;
        for(int j=0;j<3;j++)
            temp += (init_feature_list[i].cog[j] - xyzf[j])*(init_feature_list[i].cog[j] - xyzf[j]);

        dd.push_back(sqrt(temp));
    }

    float mind=dd[0];
    index=0;
    for(int i=0;i<dd.size();i++)
    {
        if(dd[i]<mind)
        {
            mind = dd[i];
            index=i;
        }
    }

    fstar = init_feature_list[index];
    cout<<"init target feature center: "<<fstar.cog[0]<<" "<<fstar.cog[1]<<" "<<fstar.cog[2]<<endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Iterate over each time step and track the selected feature finally!!
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    if(!mfix_data)
    {
        for(int ii=initStep+1; ii<initStep+timeSteps;ii++)
        {
            vector<Feature_class> current_feature_list;
            vector<float> distance_cog;
            vector<float> distance_mass;
            vector<float> distance_blob;
            vector<float> distance_cbbox;
            int match = 0;
            float temp=0;
            map<float,Feature_vector> feature_prop;

            //Generate the file name
            stringstream pp;
            pp<<ii;

            ////TODO
            //inputFilename = path + "tornado_lambda2_" + pp.str() + ".vti";
            inputFilename = path + "vortex_" + pp.str() + ".vti";
            //inputFilename = path + "combustion_vorticity_" + pp.str() + ".vti";
            //inputFilename = path + "isabel_lambda2_" + pp.str() + ".vti";

            //Read the probability field generated
            vtkSmartPointer<vtkXMLImageDataReader> reader1 = vtkSmartPointer<vtkXMLImageDataReader>::New();
            reader1->SetFileName(inputFilename.c_str());
            reader1->Update();

            //Initialize image data
            vtkSmartPointer<vtkImageData> imageData1 = vtkSmartPointer<vtkImageData>::New();
            imageData1->SetDimensions(xdim,ydim,zdim);
            imageData1->AllocateScalars(VTK_FLOAT, 1);
            imageData1 = reader1->GetOutput();

            //Threshold the field to extract different features
            vtkSmartPointer<vtkThreshold> thresholding_1 = vtkSmartPointer<vtkThreshold>::New();
            if(neg_data || mfix_data)
                thresholding_1->ThresholdByLower( threshold );
            else
                thresholding_1->ThresholdByUpper( threshold );
            thresholding_1->SetInputData( imageData1 );

            //Find connected components
            vtkSmartPointer<vtkConnectivityFilter> segmentation1 = vtkSmartPointer<vtkConnectivityFilter>::New();
            segmentation1->SetInputConnection( thresholding_1->GetOutputPort() );
            segmentation1->SetExtractionModeToAllRegions();
            segmentation1->ColorRegionsOn();
            segmentation1->Update();

            vtkSmartPointer<vtkUnstructuredGrid> ug1 = vtkSmartPointer<vtkUnstructuredGrid>::New();
            ug1->ShallowCopy(segmentation1->GetOutput());
            num_segments = segmentation1->GetNumberOfExtractedRegions();

            //Second level threshold on unstructured grid: extract individual features
            for(int i=0; i<num_segments; i++)
            {
                stringstream qq;
                double bbox[6] = {0,0,0,0,0,0,};
                float cog[3] = {0,0,0};
                int numofpts=0;
                float dataTot=0;

                vtkSmartPointer<vtkThreshold> thresholding11 = vtkSmartPointer<vtkThreshold>::New();
                thresholding11->SetInputData(ug1);
                thresholding11->ThresholdBetween(i,i);
                thresholding11->Update();

                //object1 is a specific feature
                vtkSmartPointer<vtkUnstructuredGrid> object1 = vtkSmartPointer<vtkUnstructuredGrid>::New();
                object1->ShallowCopy(thresholding11->GetOutput());
                object1->GetBounds(bbox);

                vtkDataArray *data1 = object1->GetPointData()->GetArray(arrName.c_str());
                vtkDataArray *point_ary1 = object1->GetPoints()->GetData();

                //Iterate over all points of a selected feature
                dataTot=0;
                for(int j=0; j<object1->GetPoints()->GetNumberOfPoints(); j++)
                {
                    double pos[3];
                    double val;

                    point_ary1->GetTuple(j,pos);
                    data1->GetTuple(j,&val);
                    cog[0] += pos[0]; cog[1] += pos[1]; cog[2] += pos[2];
                    numofpts++;
                    dataTot += fabs(val);
                }

                if(numofpts > MIN_POINTS)
                {
                    Feature_class f1;
                    f1.numofpts = numofpts;
                    f1.cog[0] = cog[0]/(numofpts); f1.cog[1] = cog[1]/(numofpts); f1.cog[2] = cog[2]/(numofpts);
                    f1.bbox[0] = bbox[0]; f1.bbox[1] = bbox[1]; f1.bbox[2] = bbox[2]; f1.bbox[3] = bbox[3]; f1.bbox[4] = bbox[4]; f1.bbox[5] = bbox[5];
                    f1.cbbox[0] = bbox[0] + fabs(bbox[0]-bbox[1])/2.0;
                    f1.cbbox[1] = bbox[2] + fabs(bbox[2]-bbox[3])/2.0;
                    f1.cbbox[2] = bbox[4] + fabs(bbox[4]-bbox[5])/2.0;
                    f1.featureId = i;
                    f1.mass = dataTot;

                    //push the feature to list
                    current_feature_list.push_back(f1);
                }
            }

            //////////////////////////////////////////////////////////////////////////////////////////////////
            // Find match/correspondence
            //////////////////////////////////////////////////////////////////////////////////////////////////
            float feature1,feature2,feature3,feature4;

            for(int i=0; i<current_feature_list.size(); i++)
            {
                //Compute distance between centers
                /////////////////////////////////////////////////////
                temp = 0;
                for(int j=0;j<3;j++)
                    temp += (current_feature_list[i].cog[j] - fstar.cog[j])*(current_feature_list[i].cog[j] - fstar.cog[j]);

                distance_cog.push_back(sqrt(temp)/maxdist);
                feature1 = sqrt(temp)/maxdist;

                //Compute change of mass
                //////////////////////////////////////////////////////
                temp = 0;
                temp = fabs(current_feature_list[i].mass - fstar.mass)/fstar.mass;
                distance_mass.push_back(temp);
                feature2 = temp;

                //Compute change of total number of voxels/ Volume
                ////////////////////////////////////////////////////////
                temp = 0;
                temp = fabs(current_feature_list[i].numofpts - fstar.numofpts)/fstar.numofpts;
                distance_blob.push_back(temp);
                feature3 = temp;

                //Compute distance between cbbox
                ////////////////////////////////////////////////////////
                temp = 0;
                for(int j=0;j<3;j++)
                    temp += (current_feature_list[i].cbbox[j] - fstar.cbbox[j])*(current_feature_list[i].cbbox[j] - fstar.cbbox[j]);

                distance_cbbox.push_back(sqrt(temp)/maxdist);
                feature4 = sqrt(temp)/maxdist;

                //Create feature vector
                Feature_vector fvector;
                fvector.feature_vec[0] = feature1;
                fvector.feature_vec[1] = feature2;
                fvector.feature_vec[2] = feature3;
                fvector.feature_vec[3] = feature4;
                fvector.id = i;

                feature_prop.insert(pair<float,Feature_vector>(feature1,fvector));
            }

            //Find the one which matches closest
            float mindist=distance_cog[0];
            for(int i=0;i<distance_cog.size();i++)
            {
                if(distance_cog[i]<mindist)
                {
                    mindist = distance_cog[i];
                    match=i;
                }
            }

            // This map orders the objects with most probable order from closest to farthest
            map<float,Feature_vector>::iterator it = feature_prop.begin();
            int jj=0;

            for (it=feature_prop.begin(); it!=feature_prop.end(); ++it)
            {
                //Feature_vector data = it->second;
                feature1 = it->second.feature_vec[0];
                feature2 = it->second.feature_vec[1];
                feature3 = it->second.feature_vec[2];
                feature4 = it->second.feature_vec[3];

                if(feature1<neighborhood) // for making the training more effective, only consider target and a few non-target features
                {
                    if(jj==0)
                    {
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        // create input+output training data samples
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //trainingfileref<<ii<<" "<<jj<<" "<<feature1<<" "<<feature2<<" "<<feature3<<" "<<feature4<<" "<< vv<<endl;
                        trainingfileref<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<vv<<endl;

                        trainingfileref1<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<vv<<endl;

                        trainingfilein<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<endl;

                        trainingfileout<<vv<<endl;
                    }
                    else
                    {
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        // create input+output training data samples
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //trainingfileref<<ii<<" "<<jj<<" "<<feature1<<" "<<feature2<<" "<<feature3<<" "<<feature4<<" "<< uu<<endl;
                        trainingfileref<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<uu<<endl;

                        trainingfileref1<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<uu<<endl;

                        trainingfilein<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<endl;

                        trainingfileout<<uu<<endl;
                    }
                }
                jj++;
            }
            
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //write volumes out
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            int mm,nn,oo;
            //Initialize image data
            vtkSmartPointer<vtkImageData> imageData2 = vtkSmartPointer<vtkImageData>::New();
            imageData2->SetDimensions(xdim,ydim,zdim);
            imageData2->AllocateScalars(VTK_FLOAT, 1);

            int idd = current_feature_list[match].featureId;
            //cout<<"matched_feature size at time: "<<ii<<"  is: "<<current_feature_list[match].numofpts<<endl;
            vtkSmartPointer<vtkThreshold> thresholding12 = vtkSmartPointer<vtkThreshold>::New();
            thresholding12->SetInputData(ug1);
            thresholding12->ThresholdBetween(idd,idd);
            thresholding12->Update();

            vtkSmartPointer<vtkUnstructuredGrid> object2 = vtkSmartPointer<vtkUnstructuredGrid>::New();
            object2->ShallowCopy(thresholding12->GetOutput());

            vtkDataArray *data2 = object2->GetPointData()->GetArray(arrName.c_str());
            vtkDataArray *point_ary2 = object2->GetPoints()->GetData();

            for(int p=0; p<zdim; p++)
                for(int q=0; q<ydim; q++)
                    for(int r=0; r<xdim; r++)
                    {
                        float* pixel = static_cast<float*>(imageData2->GetScalarPointer(r,q,p));
                        float* pixel1 = static_cast<float*>(imageData1->GetScalarPointer(r,q,p));

                        if(isabel_data)
                        {
                            //for isabel: TODO
                            if(pixel1[0] >= threshold)
                                pixel[0] = 100;
                            else
                                pixel[0] = -pixel1[0];
                        }
                        else
                        {
                            //for others: TODO
                            if(pixel1[0] >= threshold)
                                pixel[0] = 0;
                            else
                                pixel[0] = pixel1[0];
                        }
                    }

            for(int j=0; j<object2->GetPoints()->GetNumberOfPoints(); j++)
            {
                double pos[3];
                double val;

                point_ary2->GetTuple(j,pos);
                data2->GetTuple(j,&val);
                mm = (int)pos[0];
                nn = (int)pos[1];
                oo = (int)pos[2];

                float* pixel = static_cast<float*>(imageData2->GetScalarPointer(mm,nn,oo));
                pixel[0] = (float)val;
            }

            //isolated feature volume
            string outfile = "feature_time_" + pp.str() + ".vti";
            vtkSmartPointer<vtkXMLImageDataWriter> writer =  vtkSmartPointer<vtkXMLImageDataWriter>::New();
            writer->SetFileName(outfile.c_str());
            writer->SetInputData(imageData2);
            writer->Write();
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            cout<<"Matched feature at time: "<<ii<< " is: "<<match<< " num of pts changed: "<<distance_blob[match]<< " mass changed: "<<distance_mass[match]<<endl;

            //Reassign current feature to matched feature
            fstar = current_feature_list[match];
        }
    }
    else
    {
        for(int ii=initStep+1; ii<(initStep+timeSteps);ii=ii+1)
        {
            vector<Feature_class> current_feature_list;
            vector<float> distance_cog;
            vector<float> distance_mass;
            vector<float> distance_blob;
            vector<float> distance_cbbox;
            int match = 0;
            float temp=0;
            map<float,Feature_vector> feature_prop;

            //Generate the file name
            stringstream pp;
            //pp<<ii;
            int actual_tstep = ii*tstep_window_multiplier;

            pp<<(actual_tstep);

            ////TODO
            inputFilename = path + "fcc" + pp.str() + ".vti";

            //Read the probability field generated
            vtkSmartPointer<vtkXMLImageDataReader> reader1 = vtkSmartPointer<vtkXMLImageDataReader>::New();
            reader1->SetFileName(inputFilename.c_str());
            reader1->Update();

            //Initialize image data
            vtkSmartPointer<vtkImageData> imageData1 = vtkSmartPointer<vtkImageData>::New();
            imageData1->SetDimensions(xdim,ydim,zdim);
            imageData1->AllocateScalars(VTK_FLOAT, 1);
            imageData1 = reader1->GetOutput();

            //Threshold the field to extract different features
            vtkSmartPointer<vtkThreshold> thresholding_1 = vtkSmartPointer<vtkThreshold>::New();
            if(neg_data || mfix_data)
                thresholding_1->ThresholdByLower( threshold );
            else
                thresholding_1->ThresholdByUpper( threshold );
            thresholding_1->SetInputData( imageData1 );

            //Find connected components
            vtkSmartPointer<vtkConnectivityFilter> segmentation1 = vtkSmartPointer<vtkConnectivityFilter>::New();
            segmentation1->SetInputConnection( thresholding_1->GetOutputPort() );
            segmentation1->SetExtractionModeToAllRegions();
            segmentation1->ColorRegionsOn();
            segmentation1->Update();

            vtkSmartPointer<vtkUnstructuredGrid> ug1 = vtkSmartPointer<vtkUnstructuredGrid>::New();
            ug1->ShallowCopy(segmentation1->GetOutput());
            num_segments = segmentation1->GetNumberOfExtractedRegions();

            //Second level threshold on unstructured grid: extract individual features
            for(int i=0; i<num_segments; i++)
            {
                stringstream qq;
                double bbox[6] = {0,0,0,0,0,0,};
                float cog[3] = {0,0,0};
                int numofpts=0;
                float dataTot=0;

                vtkSmartPointer<vtkThreshold> thresholding11 = vtkSmartPointer<vtkThreshold>::New();
                thresholding11->SetInputData(ug1);
                thresholding11->ThresholdBetween(i,i);
                thresholding11->Update();

                //object1 is a specific feature
                vtkSmartPointer<vtkUnstructuredGrid> object1 = vtkSmartPointer<vtkUnstructuredGrid>::New();
                object1->ShallowCopy(thresholding11->GetOutput());
                object1->GetBounds(bbox);

                vtkDataArray *data1 = object1->GetPointData()->GetArray(arrName.c_str());
                vtkDataArray *point_ary1 = object1->GetPoints()->GetData();

                //Iterate over all points of a selected feature
                dataTot=0;
                for(int j=0; j<object1->GetPoints()->GetNumberOfPoints(); j++)
                {
                    double pos[3];
                    double val;

                    point_ary1->GetTuple(j,pos);
                    data1->GetTuple(j,&val);
                    cog[0] += pos[0]; cog[1] += pos[1]; cog[2] += pos[2];
                    numofpts++;
                    dataTot += fabs(val);
                }

                if(numofpts > MIN_POINTS)
                {
                    Feature_class f1;
                    f1.numofpts = numofpts;
                    f1.cog[0] = cog[0]/(numofpts); f1.cog[1] = cog[1]/(numofpts); f1.cog[2] = cog[2]/(numofpts);
                    f1.bbox[0] = bbox[0]; f1.bbox[1] = bbox[1]; f1.bbox[2] = bbox[2]; f1.bbox[3] = bbox[3]; f1.bbox[4] = bbox[4]; f1.bbox[5] = bbox[5];
                    f1.cbbox[0] = bbox[0] + fabs(bbox[0]-bbox[1])/2.0;
                    f1.cbbox[1] = bbox[2] + fabs(bbox[2]-bbox[3])/2.0;
                    f1.cbbox[2] = bbox[4] + fabs(bbox[4]-bbox[5])/2.0;
                    f1.featureId = i;
                    f1.mass = dataTot;

                    //push the feature to list
                    current_feature_list.push_back(f1);
                }
            }

            //////////////////////////////////////////////////////////////////////////////////////////////////
            // Find match/correspondence
            //////////////////////////////////////////////////////////////////////////////////////////////////
            float feature1,feature2,feature3,feature4;

            for(int i=0; i<current_feature_list.size(); i++)
            {
                //Compute distance between centers
                /////////////////////////////////////////////////////
                temp = 0;
                for(int j=0;j<3;j++)
                    temp += (current_feature_list[i].cog[j] - fstar.cog[j])*(current_feature_list[i].cog[j] - fstar.cog[j]);

                distance_cog.push_back(sqrt(temp)/maxdist);
                feature1 = sqrt(temp)/maxdist;

                //Compute change of mass
                //////////////////////////////////////////////////////
                temp = 0;
                temp = fabs(current_feature_list[i].mass - fstar.mass)/fstar.mass;
                distance_mass.push_back(temp);
                feature2 = temp;

                //Compute change of total number of voxels/ Volume
                ////////////////////////////////////////////////////////
                temp = 0;
                temp = fabs(current_feature_list[i].numofpts - fstar.numofpts)/fstar.numofpts;
                distance_blob.push_back(temp);
                feature3 = temp;

                //Compute distance between cbbox
                ////////////////////////////////////////////////////////
                temp = 0;
                for(int j=0;j<3;j++)
                    temp += (current_feature_list[i].cbbox[j] - fstar.cbbox[j])*(current_feature_list[i].cbbox[j] - fstar.cbbox[j]);

                distance_cbbox.push_back(sqrt(temp)/maxdist);
                feature4 = sqrt(temp)/maxdist;

                //Create feature vector
                Feature_vector fvector;
                fvector.feature_vec[0] = feature1;
                fvector.feature_vec[1] = feature2;
                fvector.feature_vec[2] = feature3;
                fvector.feature_vec[3] = feature4;
                fvector.id = i;

                if(current_feature_list[i].mass > massTH)    
                    feature_prop.insert(pair<float,Feature_vector>(feature1,fvector));
            }

            //Find the one which matches closest
            float mindist=distance_cog[0];
            for(int i=0;i<distance_cog.size();i++)
            {
                if(distance_cog[i]<mindist)
                {
                    mindist = distance_cog[i];
                    match=i;
                }
            }

            // This map orders the objects with most probable order from closest to farthest
            map<float,Feature_vector>::iterator it = feature_prop.begin();
            int jj=0;

            for (it=feature_prop.begin(); it!=feature_prop.end(); ++it)
            {
                //Feature_vector data = it->second;
                feature1 = it->second.feature_vec[0];
                feature2 = it->second.feature_vec[1];
                feature3 = it->second.feature_vec[2];
                feature4 = it->second.feature_vec[3];

                if(feature1<neighborhood) // for making the training more effective, only consider target and a few non-target features
                {
                    if(jj==0)
                    {
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        // create input+output training data samples
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //trainingfileref<<ii<<" "<<jj<<" "<<feature1<<" "<<feature2<<" "<<feature3<<" "<<feature4<<" "<< vv<<endl;
                        trainingfileref<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<vv<<endl;

                        trainingfileref1<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<vv<<endl;

                        trainingfilein<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<endl;

                        trainingfileout<<vv<<endl;
                    }
                    else
                    {
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        // create input+output training data samples
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //trainingfileref<<ii<<" "<<jj<<" "<<feature1<<" "<<feature2<<" "<<feature3<<" "<<feature4<<" "<< uu<<endl;
                        trainingfileref<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<uu<<endl;

                        trainingfileref1<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<uu<<endl;

                        trainingfilein<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<endl;

                        trainingfileout<<uu<<endl;
                    }
                }
                jj++;
            }
            
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //write volumes out
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            int mm,nn,oo;
            //Initialize image data
            vtkSmartPointer<vtkImageData> imageData2 = vtkSmartPointer<vtkImageData>::New();
            imageData2->SetDimensions(xdim,ydim,zdim);
            imageData2->SetSpacing(imageData1->GetSpacing());
            imageData2->AllocateScalars(VTK_FLOAT, 1);

            int idd = current_feature_list[match].featureId;
            //cout<<"matched_feature size at time: "<<ii<<"  is: "<<current_feature_list[match].numofpts<<endl;
            vtkSmartPointer<vtkThreshold> thresholding12 = vtkSmartPointer<vtkThreshold>::New();
            thresholding12->SetInputData(ug1);
            thresholding12->ThresholdBetween(idd,idd);
            thresholding12->Update();

            vtkSmartPointer<vtkUnstructuredGrid> object2 = vtkSmartPointer<vtkUnstructuredGrid>::New();
            object2->ShallowCopy(thresholding12->GetOutput());

            vtkDataArray *data2 = object2->GetPointData()->GetArray(arrName.c_str());
            vtkDataArray *point_ary2 = object2->GetPoints()->GetData();

            double* bounds;
            bounds = imageData2->GetBounds();

            for(int p=0; p<zdim; p++)
                for(int q=0; q<ydim; q++)
                    for(int r=0; r<xdim; r++)
                    {
                        float* pixel = static_cast<float*>(imageData2->GetScalarPointer(r,q,p));
                        double* pixel1 = static_cast<double*>(imageData1->GetScalarPointer(r,q,p));
                        {
                            if(pixel1[0] > threshold)
                            {
                                pixel[0] = pixel1[0];
                            }
                            if (pixel1[0] <= threshold)
                            {
                                pixel[0] = threshold+5;
                            }
                        }
                    }

            for(int j=0; j<object2->GetPoints()->GetNumberOfPoints(); j++)
            {
                double pos[3];
                double val;

                point_ary2->GetTuple(j,pos);
                data2->GetTuple(j,&val);

                pos[0] = ((pos[0]-bounds[0])/(bounds[1]-bounds[0]))*(xdim-1);
                pos[1] = ((pos[1]-bounds[2])/(bounds[3]-bounds[2]))*(ydim-1);
                pos[2] = ((pos[2]-bounds[4])/(bounds[5]-bounds[4]))*(zdim-1);


                mm = (int)pos[0];
                nn = (int)pos[1];
                oo = (int)pos[2];

                float* pixel = static_cast<float*>(imageData2->GetScalarPointer(mm,nn,oo));
                pixel[0] = (float)val;
            }

            //isolated feature volume
            string outfile = "feature_time_" + pp.str() + ".vti";
            vtkSmartPointer<vtkXMLImageDataWriter> writer =  vtkSmartPointer<vtkXMLImageDataWriter>::New();
            writer->SetFileName(outfile.c_str());
            writer->SetInputData(imageData2);
            writer->Write();
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            cout<<"Matched feature at time: "<<actual_tstep<< " is: "<<match<< " num of pts changed: "<<distance_blob[match]<< " mass changed: "<<distance_mass[match]<<endl;

            //Reassign current feature to matched feature
            fstar = current_feature_list[match];
        }
    }

    trainingfilein.close();
    trainingfileout.close();
    trainingfileref.close();
    trainingfileref1.close();

    return 0;
}
