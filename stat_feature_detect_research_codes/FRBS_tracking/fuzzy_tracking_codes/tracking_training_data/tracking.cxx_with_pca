
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
#include <glm_headers.h>
#include <feature_class.h>
#include <opencv2/opencv.hpp>

using namespace std;

class Feature_vector
{
    public:
        int id;
        float feature_vec[5];
};

int find_intersection(vector<Point> first, vector<Point> second)
{
  int count=0;
  
  for(int i=0;i<first.size();i++)
  {
    for(int j=0;j<second.size();j++)
    {
        if((first[i].x == second[j].x) && (first[i].y == second[j].y) && (first[i].z == second[j].z))
        {
          count++;  
          break;    
        }
    }
  }

  return count;
}

float dice_sim(vector<Point> first, vector<Point> second)
{
  float dice_sim_val=0;
  int intersect = find_intersection(first,second);
  dice_sim_val = (2.0*intersect)/(float)(first.size()+second.size());
  
  if(dice_sim_val>1)
  	cout<<intersect<<" "<<first.size()<<" "<<second.size()<<endl;

  return dice_sim_val;
}

float jaccard_sim(vector<Point> first, vector<Point> second)
{
  float jac_sim_val=0;
  int intersect = find_intersection(first,second);
  jac_sim_val = intersect/(float)(first.size()+second.size()-intersect);
  return jac_sim_val;
}

//PCA function
void PCA_analysis(const cv::Mat& pcaset, int maxComponents,const cv::Mat& testset, cv::Mat& compressed)
{
    cv::PCA pca(pcaset, // pass the data
            cv::Mat(), // we do not have a pre-computed mean vector,
            // so let the PCA engine to compute it
            cv::PCA::DATA_AS_ROW, // indicate that the vectors
            // are stored as matrix rows
            // (use PCA::DATA_AS_COL if the vectors are
            // the matrix columns)
            maxComponents // specify, how many principal components to retain
            );
    // if there is no test data, just return the computed basis, ready-to-use
    if( !testset.data )
    {
        cout<<"error here"<<endl;
        exit(0);
    }


    CV_Assert( testset.cols == pcaset.cols );
    compressed.create(testset.rows, maxComponents, testset.type());

    for( int i = 0; i < testset.rows; i++ )
    {
        cv::Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
        // compress the vector, the result will be stored
        // in the i-th row of the output matrix
        pca.project(vec, coeffs);
        // and then reconstruct it
        pca.backProject(coeffs, reconstructed);
        // and measure the error
        //printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
    }
}

int main(int argc, char** argv)
{
    //To turn off vtk warning messages
    //vtkObject::GlobalWarningDisplayOff();

    float xyzf[3];
    float vv=0.9; float uu = 0.1;
    int isabel_data = 0; //TODO
    int neg_data = 0; //TODO

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

    //Vortex
    int xdim=128; //TODO
    int ydim=128; //TODO
    int zdim=128; //TODO
    int timeSteps=10;//TODO
    int initStep=1; //TODO
    float threshold = 7;
    xyzf[0] = 63;
    xyzf[1] = 55;
    xyzf[2] = 2.5;
    const int MIN_POINTS = 0;
    string path = "/home/soumya/Test_DataSet/vortex/vti/";
    float neighborhood = 0.3;

   //Isabel lambda2
   // int xdim=250; //TODO
   // int ydim=250; //TODO
   // int zdim=50; //TODO
   // int timeSteps=10;//TODO
   // int initStep=1; //TODO
   // float threshold = -1.0;
   // xyzf[0] = 142;
   // xyzf[1] = 60;
   // xyzf[2] = 7.5;
   // const int MIN_POINTS = 50;
   // string path = "/home/soumya/Test_DataSet/Isabel_vortex/vti/";
   // float neighborhood = 10.0;
   // isabel_data=1;
   // neg_data=1;

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
    tt<<initStep;

    //inputFilename = path + "tornado_lambda2_" + tt.str() + ".vti";
    inputFilename = path + "vortex_" + tt.str() + ".vti";
    //inputFilename = path + "combustion_vorticity_" + tt.str() + ".vti";
    //inputFilename = path + "isabel_lambda2_" + tt.str() + ".vti";

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
    if(neg_data)
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
        double bbox[6] = {0,0,0,0,0,0,};
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

        vtkDataArray *data = object->GetPointData()->GetArray("ImageScalars");
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

            ///////////
        	// Do PCA 
        	///////////
        	cv::Mat compressed;
        	cv::Mat time_line_mat(object->GetPoints()->GetNumberOfPoints(),3,CV_32FC1);
        	for(int j=0; j<object->GetPoints()->GetNumberOfPoints(); j++)
	        {
	            double pos[3];
	            point_ary->GetTuple(j,pos);
	            time_line_mat.at<float>(j,0) = pos[0];
            	time_line_mat.at<float>(j,1) = pos[1];
            	time_line_mat.at<float>(j,2) = pos[2];
	        }
	        //Do PCA: compressed matrix will have the required values
        	PCA_analysis(time_line_mat, 3, time_line_mat, compressed);
        	for(int k=0;k<compressed.rows;k++)
        	{
        		Point pt;
        		pt.x=round(compressed.at<float>(k,0));
        		pt.y=round(compressed.at<float>(k,1));
        		pt.z=round(compressed.at<float>(k,2));
        		f.pca_transformed.push_back(pt);
        	}

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
    cout<<"initi target feature center: "<<fstar.cog[0]<<" "<<fstar.cog[1]<<" "<<fstar.cog[2]<<endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Iterate over each time step and track the selected feature finally!!
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for(int ii=initStep+1; ii<initStep+timeSteps;ii++)
    {
        vector<Feature_class> current_feature_list;
        vector<float> distance_cog;
        vector<float> distance_mass;
        vector<float> distance_blob;
        vector<float> distance_cbbox;
        vector<float> distance_shape;
        int match = 0;
        float temp=0;
        map<float,Feature_vector> feature_prop;

        //Generate the file name
        //inputFilename,outfile;
        stringstream pp;
        pp<<ii;
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
        if(neg_data)
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

            vtkDataArray *data1 = object1->GetPointData()->GetArray("ImageScalars");
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

                ///////////
	        	// Do PCA 
	        	///////////
	        	cv::Mat compressed;
	        	cv::Mat time_line_mat(object1->GetPoints()->GetNumberOfPoints(),3,CV_32FC1);
	        	for(int j=0; j<object1->GetPoints()->GetNumberOfPoints(); j++)
		        {
		            double pos[3];
		            point_ary1->GetTuple(j,pos);
		            time_line_mat.at<float>(j,0) = pos[0];
	            	time_line_mat.at<float>(j,1) = pos[1];
	            	time_line_mat.at<float>(j,2) = pos[2];
		        }
		        //Do PCA: compressed matrix will have the required values
	        	PCA_analysis(time_line_mat, 3, time_line_mat, compressed);
	        	for(int k=0;k<compressed.rows;k++)
	        	{
	        		Point pt;
	        		pt.x=round(compressed.at<float>(k,0));
	        		pt.y=round(compressed.at<float>(k,1));
	        		pt.z=round(compressed.at<float>(k,2));
	        		f1.pca_transformed.push_back(pt);
	        	}

                //push the feature to list
                current_feature_list.push_back(f1);
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Find match/correspondence
        //////////////////////////////////////////////////////////////////////////////////////////////////
        float feature1,feature2,feature3,feature4,feature5;

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

            //Compute overlap with PCA transformed objects
            ////////////////////////////////////////////////////////
            feature5 = dice_sim(fstar.pca_transformed,current_feature_list[i].pca_transformed);
            distance_shape.push_back(feature5);

            //Create feature vector
            Feature_vector fvector;
            fvector.feature_vec[0] = feature1;
            fvector.feature_vec[1] = feature2;
            fvector.feature_vec[2] = feature3;
            fvector.feature_vec[3] = feature4;
            fvector.feature_vec[4] = feature5;
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
            feature5 = it->second.feature_vec[4];

            if(feature1<neighborhood) // for making the training more effective, only consider target and a few non-target features
            {
                if(jj==0)
                {
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    // create input+output training data samples
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    //trainingfileref<<ii<<" "<<jj<<" "<<feature1<<" "<<feature2<<" "<<feature3<<" "<<feature4<<" "<< vv<<endl;

                    trainingfileref<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<feature5<<","<< vv<<endl;

                    trainingfilein<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<feature5<<endl;

                    trainingfileout<<vv<<endl;
                }
                else
                {
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    // create input+output training data samples
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    //trainingfileref<<ii<<" "<<jj<<" "<<feature1<<" "<<feature2<<" "<<feature3<<" "<<feature4<<" "<< uu<<endl;

                    trainingfileref<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<feature5<<","<< uu<<endl;

                    trainingfilein<<feature1<<","<<feature2<<","<<feature3<<","<<feature4<<","<<feature5<<endl;

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
        vtkSmartPointer<vtkThreshold> thresholding12 = vtkSmartPointer<vtkThreshold>::New();
        thresholding12->SetInputData(ug1);
        thresholding12->ThresholdBetween(idd,idd);
        thresholding12->Update();

        vtkSmartPointer<vtkUnstructuredGrid> object2 = vtkSmartPointer<vtkUnstructuredGrid>::New();
        object2->ShallowCopy(thresholding12->GetOutput());

        vtkDataArray *data2 = object2->GetPointData()->GetArray("ImageScalars");
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

        cout<<"Matched feature at time: "<<ii<< " is: "<<match<< " num of pts changed: "
            <<distance_blob[match]<< " mass changed: "<<distance_mass[match]<<" pca shape changed: "<<distance_shape[match]<<endl;

        //Reassign current feature to matched feature
        fstar = current_feature_list[match];
    }

    trainingfilein.close();
    trainingfileout.close();
    trainingfileref.close();

    return 0;
}