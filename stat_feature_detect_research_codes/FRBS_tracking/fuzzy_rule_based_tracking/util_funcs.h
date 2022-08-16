#ifndef UTIL_FUNCS_HEADER
#define UTIL_FUNCS_HEADER

#include <cpp_headers.h>
#include <vtk_headers.h>
#include <feature_class.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;

// put color on polydata points
// Setup colors
unsigned char red[3] = {255, 0, 0};
unsigned char green[3] = {0, 255, 0};
unsigned char blue[3] = {0, 0, 255};

const int NUM_PTS = 25;// 75 = combustion , 25/75 = vortex, 25 isabel
float neighborhood = 0.25; // 0.08 for combustion, 0.25 vortex, 0.5 isabel
int neg_data = 0; //TODO
int isabel_data = 0; //TODO

//time function
double start_t,end_t,glob_t_pca,glob_t_find_min_point;
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

//PCA function
void PCA_analysis(const Mat& pcaset, int maxComponents,const Mat& testset, Mat& compressed)
{
    PCA pca(pcaset, // pass the data
            Mat(), // we do not have a pre-computed mean vector,
            // so let the PCA engine to compute it
            PCA::DATA_AS_ROW, // indicate that the vectors
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
        Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
        // compress the vector, the result will be stored
        // in the i-th row of the output matrix
        pca.project(vec, coeffs);
        // and then reconstruct it
        pca.backProject(coeffs, reconstructed);
        // and measure the error
        //printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
    }
}

glm::vec3 get_min_point(vector<glm::vec3> first_feature, vector<glm::vec3> second_feature)
{
    glm::vec3 minpoint;
    vector < vector <float> > distance_array;
    int mini,minj;
    float minval=0;

    for(int i=0;i<first_feature.size();i++)
    {
        vector<float> dist;
        float val=0;
        for(int j=0;j<second_feature.size();j++)
        {
            val = 0;

            //compute distance
            for(int k=0;k<3;k++)
            {
                val += (first_feature[i].x - second_feature[j].x)*(first_feature[i].x - second_feature[j].x)
                        + (first_feature[i].y - second_feature[j].y)*(first_feature[i].y - second_feature[j].y)
                        + (first_feature[i].z - second_feature[j].z)*(first_feature[i].z - second_feature[j].z);
            }

            val = sqrt(val);
            dist.push_back(val);
        }

        distance_array.push_back(dist);
    }

    mini = minj = 0;
    minval = distance_array[0][0];
    for(int i=0;i<first_feature.size();i++)
    {
        for(int j=0;j<second_feature.size();j++)
        {
            if(distance_array[i][j]<minval)
            {
                minval = distance_array[i][j];
                mini = i;
                minj = j;
            }
        }
    }

    minpoint.x = (first_feature[mini].x - second_feature[minj].x);
    minpoint.y = (first_feature[mini].y - second_feature[minj].y);
    minpoint.z = (first_feature[mini].z - second_feature[minj].z);

    return minpoint;
}

glm::vec3 get_center(vector<glm::vec3> feature)
{
    glm::vec3 center;
    center.x = 0;
    center.y = 0;
    center.z = 0;

    for(int i=0;i<feature.size();i++)
    {
        center.x += feature[i].x;
        center.y += feature[i].y;
        center.z += feature[i].z;
    }

    center.x /= feature.size();
    center.y /= feature.size();
    center.z /= feature.size();

    return center;
}

float distance_func(glm::vec3 v1, glm::vec3 v2)
{
    float dist=0;
    dist  = sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y) + (v1.z-v2.z)*(v1.z-v2.z));
    return dist;
}

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

//Evaluates a gaussian membership function on a input value
float gmf(float val, float mean, float sigma)
{
    float vall;

    if(sigma>0.0)
        vall = exp(-(((val-mean)*(val-mean))/(2*sigma*sigma)));
    else
        vall = 0.0;

    return vall;
}

float tnorm_product(float *fire, int len)
{
    float val = 1;

    for(int i=0;i<len;i++)
        val = val*fire[i];

    return val;
}

float tnorm_min(float *fire, int len)
{
    float val = fire[0];
    for(int i=0;i<len;i++)
    {
        if(val>fire[i])
            val=fire[i];
    }

    return val;
}

///////////////////////////////////////////////////////////////////////////
// Evaluate a single rule on a single input and returns a glm::vec2
// vec2.x = firing strength of the rule computed by product of memberships
// vec2.y = id of the feature
///////////////////////////////////////////////////////////////////////////
glm::vec2 evaluate_single_rule(Rules R, Feature_vector data)
{
    glm::vec2 ret;
    float consequent=0;
    const int len_feature_vec = data.feature_vec.size();
    float fire[len_feature_vec];
    float final=0;
    float feature[len_feature_vec];

    for(int i=0;i<len_feature_vec;i++)
        feature[i] = data.feature_vec[i];

    //Compute antecedent part: use product rule
    for(int qq=0;qq<R.inputmfs.size();qq++)
        fire[qq] = gmf(feature[qq],R.inputmfs[qq].mean,R.inputmfs[qq].sigma);   // mean and sigma passed in reversed order

    //get the firing strength
    final = tnorm_product(fire,len_feature_vec);

    //compute the consequent part
    for(int i=0;i<len_feature_vec;i++)
        consequent += feature[i]*R.out_params[i]; //Trying to multiply weights here in this step

    consequent = consequent + R.out_params[R.out_params.size()-1]; //add the last one

    ret.x = final;
    ret.y = consequent;

    return ret;
}

////////////////////////////////////////////////////////////////////////
// Returns a glm::vec2 where
// vec2.x = evaluated value
// vec2.y = id of the feature
////////////////////////////////////////////////////////////////////////
glm::vec2 evaluate_rulebase(Rule_Based_System rulebase, Feature_vector data)
{
    glm::vec2 ret;

    vector<glm::vec2> eval_vals;
    float num=0;
    float denom=0;

    //Evaluate the rulebase with each data input
    for(int qq=0;qq<rulebase.num_rules;qq++)
    {
        eval_vals.push_back(evaluate_single_rule(rulebase.rules[qq],data));
    }

    for(int qq=0;qq<eval_vals.size();qq++)
    {
        num += eval_vals[qq].x*eval_vals[qq].y;
        denom += eval_vals[qq].x;
    }

    if(denom==0) // this means no match is found so return 0
    {
        //cout<<"ops.. "<<num<<" "<<denom<<" "<<data.id<<endl;
        //cout<<"Feature death or something else has happened! This feature can not be tracked anymore! exiting"<<endl;
        ret.x = 0.0;
        ret.y = data.id;
    }
    else
    {
        ret.x = num/denom;
        ret.y = data.id;
    }

    return ret;
}

//Find best math (max response from FRS) for correspondence
/////////////////////////////////////////////////////////////
glm::vec2 find_matched_index(vector<glm::vec2> evaluated_vals, int timestep)
{
    glm::vec2 ret;
    int id=0;
    float val=0;

    id = (int)evaluated_vals[0].y;
    val = evaluated_vals[0].x;
    for(int qq=0;qq<evaluated_vals.size();qq++)
    {
        if(val < evaluated_vals[qq].x)
        {
            val = evaluated_vals[qq].x;
            id = (int)evaluated_vals[qq].y;
        }
    }

    ret.x = val;
    ret.y = id;

    return ret;
}


vtkSmartPointer<vtkImageData> read_data(string filename)
{
    //Read the raw field OR the source data
    vtkSmartPointer<vtkXMLImageDataReader> reader1 = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader1->SetFileName(filename.c_str());
    reader1->Update();
    return reader1->GetOutput();
}

//currently not in use
vtkSmartPointer<vtkPolyData> make_surface_from_ugrid(vtkSmartPointer<vtkUnstructuredGrid> data)
{
    //    vtkSmartPointer<vtkDelaunay3D> delaunay3D = vtkSmartPointer<vtkDelaunay3D>::New();
    //    delaunay3D->SetInputData(data);
    //    delaunay3D->Update();

    //    vtkSmartPointer<vtkGeometryFilter> geometryFilter = vtkSmartPointer<vtkGeometryFilter>::New();
    //    geometryFilter->SetInputData(delaunay3D->GetOutput());
    //    geometryFilter->Update();

    //    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    //    triangleFilter->SetInputData(geometryFilter->GetOutput());
    //    triangleFilter->Update();

    vtkSmartPointer<vtkGeometryFilter> geometryFilter = vtkSmartPointer<vtkGeometryFilter>::New();
    geometryFilter->SetInputData(data);
    geometryFilter->Update();

    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(geometryFilter->GetOutput());
    triangleFilter->Update();

    return triangleFilter->GetOutput();
}

//currently not in use
vtkSmartPointer<vtkPolyData> make_surface_from_polydata(vtkSmartPointer<vtkPolyData> data)
{
    //    vtkSmartPointer<vtkDelaunay3D> delaunay3D = vtkSmartPointer<vtkDelaunay3D>::New();
    //    delaunay3D->SetInputData(data);
    //    delaunay3D->Update();

    //    vtkSmartPointer<vtkGeometryFilter> geometryFilter = vtkSmartPointer<vtkGeometryFilter>::New();
    //    geometryFilter->SetInputData(delaunay3D->GetOutput());
    //    geometryFilter->Update();

    //    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    //    triangleFilter->SetInputData(geometryFilter->GetOutput());
    //    triangleFilter->Update();

    vtkSmartPointer<vtkGeometryFilter> geometryFilter = vtkSmartPointer<vtkGeometryFilter>::New();
    geometryFilter->SetInputData(data);
    geometryFilter->Update();

    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(geometryFilter->GetOutput());
    triangleFilter->Update();

    return triangleFilter->GetOutput();
}

vector<Feature_class> locate_target_feature(string filename,int *index,float threshold, float *xyzf, int *dim, int timestep)
{
    int num_segments;
    vector<Feature_class> init_feature_list;

    //Read the probability field generated
    vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    int *dim1 =reader->GetOutput()->GetDimensions();
    dim[0] = dim1[0];
    dim[1] = dim1[1];
    dim[2] = dim1[2];

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(dim1[0],dim1[1],dim1[2]);
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
    segmentation->SetInputConnection( thresholding->GetOutputPort() );
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
        if(numofpts > NUM_PTS)
        {
            //cout<<"dawd"<<endl;
            Feature_class f;
            f.numofpts = numofpts;
            f.cog[0] = cog[0]/numofpts; f.cog[1] = cog[1]/numofpts; f.cog[2] = cog[2]/numofpts;
            f.bbox[0] = bbox[0]; f.bbox[1] = bbox[1]; f.bbox[2] = bbox[2]; f.bbox[3] = bbox[3]; f.bbox[4] = bbox[4]; f.bbox[5] = bbox[5];
            f.cbbox[0] = bbox[0] + fabs(bbox[0]-bbox[1])/2.0; f.cbbox[1] = bbox[2] + fabs(bbox[2]-bbox[3])/2.0; f.cbbox[2] = bbox[4] + fabs(bbox[4]-bbox[5])/2.0;
            f.featureId = i;
            f.mass = datatotal;

            vtkSmartPointer<vtkPolyData> surf = make_surface_from_ugrid(object);

            f.surface->ShallowCopy(surf); //store the points only

            //push the feature to list
            init_feature_list.push_back(f);
        }
    }

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
    *index=0;
    for(int i=0;i<dd.size();i++)
    {
        if(dd[i]<mind)
        {
            mind = dd[i];
            *index=i;
        }
    }
    dd.clear();

    cout<<"mathced_index is: "<<*(index)<<endl;


    vtkSmartPointer<vtkXMLPolyDataWriter> writer =  vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName("detected_initial_target.vtp");
    writer->SetInputData(init_feature_list[*index].surface);
    writer->Write();

    return init_feature_list;
}


void construct_feature_list(vtkSmartPointer<vtkUnstructuredGrid> ug1,vector<Feature_class> *current_feature_list, int num_segments, int timestep)
{
    //Second level threshold on unstructured grid: extract individual features
    for(int i=0; i<num_segments; i++)
    {
        double bbox[6] = {0,0,0,0,0,0};
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

        if(numofpts > NUM_PTS)
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

            vtkSmartPointer<vtkPolyData> surf = make_surface_from_ugrid(object1);

            f1.surface->ShallowCopy(surf);

            //push the feature to list
            current_feature_list->push_back(f1);
        }
    }
}


//writes output unstructured grid and segments features from data
vtkSmartPointer<vtkUnstructuredGrid> segment_feature_from_data(int *num_segments, float threshold, vtkImageData *imageData1, int ii)
{
    int *dim;
    dim = imageData1->GetDimensions();

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

    //Extract the unstructured grid data from connected component result
    vtkSmartPointer<vtkUnstructuredGrid> ug1 = vtkSmartPointer<vtkUnstructuredGrid>::New();
    ug1->ShallowCopy(segmentation1->GetOutput());
    *(num_segments) = segmentation1->GetNumberOfExtractedRegions();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Write file for testing only
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    stringstream ss;
    ss<<ii;
    string outfile = "outugrid" + ss.str() + ".vtu";
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writerout = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writerout->SetFileName(outfile.c_str());
    writerout->SetInputData(ug1);
    writerout->Write();

    return ug1;
}



void generate_output_volume(int ii, vtkSmartPointer<vtkImageData> imageData1, vector<Feature_class> current_feature_list,
                            vtkSmartPointer<vtkUnstructuredGrid> ug1, int match, float threshold)
{
    int mm,nn,oo;
    int *dim;
    dim =imageData1->GetDimensions();

    //Initialize image data
    vtkSmartPointer<vtkImageData> imageData2 = vtkSmartPointer<vtkImageData>::New();
    imageData2->SetDimensions(dim[0],dim[1],dim[2]);
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

    for(int p=0; p<dim[2]; p++)
        for(int q=0; q<dim[1]; q++)
            for(int r=0; r<dim[0]; r++)
            {
                float* pixel = static_cast<float*>(imageData2->GetScalarPointer(r,q,p));
                pixel[0] = 0;
            }

    for(int p=0; p<dim[2]; p++)
        for(int q=0; q<dim[1]; q++)
            for(int r=0; r<dim[0]; r++)
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
    stringstream pp;
    pp<<ii;
    string outfile = "feature_time_" + pp.str() + ".vti";
    vtkSmartPointer<vtkXMLImageDataWriter> writer =  vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(outfile.c_str());
    writer->SetInputData(imageData2);
    writer->Write();

}



void generate_output_volume_split(int ii, vtkSmartPointer<vtkImageData> imageData1, vector<Feature_class> current_feature_list,
                                  vtkSmartPointer<vtkUnstructuredGrid> ug1, int id1, int id2, float threshold)
{

    int mm,nn,oo;
    int *dim;
    dim =imageData1->GetDimensions();

    double pos[3];
    double val;

    //Initialize image data
    vtkSmartPointer<vtkImageData> imageData2 = vtkSmartPointer<vtkImageData>::New();
    imageData2->SetDimensions(dim[0],dim[1],dim[2]);
    imageData2->AllocateScalars(VTK_FLOAT, 1);

    int idd = current_feature_list[id1].featureId;
    vtkSmartPointer<vtkThreshold> thresholding12 = vtkSmartPointer<vtkThreshold>::New();
    thresholding12->SetInputData(ug1);
    thresholding12->ThresholdBetween(idd,idd);
    thresholding12->Update();

    int idd1 = current_feature_list[id2].featureId;
    vtkSmartPointer<vtkThreshold> thresholding22 = vtkSmartPointer<vtkThreshold>::New();
    thresholding22->SetInputData(ug1);
    thresholding22->ThresholdBetween(idd1,idd1);
    thresholding22->Update();

    vtkSmartPointer<vtkUnstructuredGrid> object2 = vtkSmartPointer<vtkUnstructuredGrid>::New();
    object2->ShallowCopy(thresholding12->GetOutput());

    vtkSmartPointer<vtkUnstructuredGrid> object3 = vtkSmartPointer<vtkUnstructuredGrid>::New();
    object3->ShallowCopy(thresholding22->GetOutput());

    vtkDataArray *data2 = object2->GetPointData()->GetArray("ImageScalars");
    vtkDataArray *point_ary2 = object2->GetPoints()->GetData();

    vtkDataArray *data3 = object3->GetPointData()->GetArray("ImageScalars");
    vtkDataArray *point_ary3 = object3->GetPoints()->GetData();

    for(int p=0; p<dim[2]; p++)
        for(int q=0; q<dim[1]; q++)
            for(int r=0; r<dim[0]; r++)
            {
                float* pixel = static_cast<float*>(imageData2->GetScalarPointer(r,q,p));
                pixel[0] = 0;
            }

    for(int p=0; p<dim[2]; p++)
        for(int q=0; q<dim[1]; q++)
            for(int r=0; r<dim[0]; r++)
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

    //update feature volume accoding to split object1
    for(int j=0; j<object2->GetPoints()->GetNumberOfPoints(); j++)
    {
        point_ary2->GetTuple(j,pos);
        data2->GetTuple(j,&val);
        mm = (int)pos[0];
        nn = (int)pos[1];
        oo = (int)pos[2];

        float* pixel = static_cast<float*>(imageData2->GetScalarPointer(mm,nn,oo));
        pixel[0] = (float)val;
    }

    //update feature volume accoding to split object2
    for(int j=0; j<object3->GetPoints()->GetNumberOfPoints(); j++)
    {
        point_ary3->GetTuple(j,pos);
        data3->GetTuple(j,&val);
        mm = (int)pos[0];
        nn = (int)pos[1];
        oo = (int)pos[2];

        float* pixel = static_cast<float*>(imageData2->GetScalarPointer(mm,nn,oo));
        pixel[0] = (float)val;
    }

    //isolated feature volume
    stringstream pp;
    pp<<ii;
    string outfile = "feature_time_" + pp.str() + ".vti";
    vtkSmartPointer<vtkXMLImageDataWriter> writer =  vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(outfile.c_str());
    writer->SetInputData(imageData2);
    writer->Write();

}

//New approach added to the function
void compute_feature_attribute_variations(vector<Feature_class> current_feature_list, map<float,Feature_vector> *feature_prop,
                                          Feature_class fstar, vector<float> *distance_cog, vector<float> *distance_mass,
                                          vector<float> *distance_blob, vector<float> *distance_cbbox, float maxdist)
{

    /////////////////////////////////////////////////////////////////////////
    // Compute feature attributes for use in correspondence system
    /////////////////////////////////////////////////////////////////////////
    float feature1=0;
    float feature2=0;
    float feature3=0;
    float feature4=0;

    for(int i=0; i<current_feature_list.size(); i++)
    {
        //Compute distance between centers
        float temp = 0;
        for(int j=0;j<3;j++)
            temp += (current_feature_list[i].cog[j] - fstar.cog[j])*(current_feature_list[i].cog[j] - fstar.cog[j]);

        distance_cog->push_back(sqrt(temp)/maxdist);
        feature1 = sqrt(temp)/maxdist;

        //Compute chan1ge of mass
        temp = 0;
        temp = fabs(current_feature_list[i].mass - fstar.mass)/fstar.mass;
        distance_mass->push_back(temp);
        feature2 = temp;

        //Compute change of total number of voxels/ Volume
        temp = 0;
        temp = fabs(current_feature_list[i].numofpts - fstar.numofpts)/fstar.numofpts;
        distance_blob->push_back(temp);
        feature3 = temp;

        //Compute distance between cbbox
        temp = 0;
        for(int j=0;j<3;j++)
            temp += (current_feature_list[i].cbbox[j] - fstar.cbbox[j])*(current_feature_list[i].cbbox[j] - fstar.cbbox[j]);

        distance_cbbox->push_back(sqrt(temp)/maxdist);
        feature4 = sqrt(temp)/maxdist;

        //check for a local neighborhood only
        if(feature1 < neighborhood)
        {
            //Create feature vector
            Feature_vector fvector;
            fvector.feature_vec.push_back(feature1);
            fvector.feature_vec.push_back(feature2);
            fvector.feature_vec.push_back(feature3);
            fvector.feature_vec.push_back(feature4);

            fvector.id = i;
            feature_prop->insert(pair<float,Feature_vector>(i,fvector));
        }
    }
}


///////////////////////////////////////////
//Final with opencv PCA approach
///////////////////////////////////////////
map<int,glm::vec2> create_feature_combination(vector<Feature_class> current_feature_list, Feature_class fstar, float maxdist, int timestep, int found_match)
{
    map<int,glm::vec2> feature_comb_map;

    //cout<<"create combine features: "<<timestep<<endl;
    stringstream tt;
    tt<<timestep;
    double p[3];
    glm :: vec3 center;
    vector<glm::vec3> target_feature;

    //Get points out from target feature
    vtkDataArray *point_ary = fstar.surface->GetPoints()->GetData();

    //populate target feature vector
    for(int i=0;i<point_ary->GetNumberOfTuples();i++)
    {
        point_ary->GetTuple(i,p);
        target_feature.push_back(glm::vec3(p[0],p[1],p[2]));
    }

    center = get_center(target_feature);

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Do PCA here on target
    ////////////////////////////////////////////////////////////////////////////////////////////
    Mat compressed_target;
    Mat time_line_mat_target(target_feature.size(),3,CV_32FC1);
    //Populate data for PCA
    for(int k=0;k<target_feature.size();k++)
    {
        time_line_mat_target.at<float>(k,0) = target_feature[k].x;
        time_line_mat_target.at<float>(k,1) = target_feature[k].y;
        time_line_mat_target.at<float>(k,2) = target_feature[k].z;
    }

    //Do PCA: compressed matrix will have the required values
    PCA_analysis(time_line_mat_target, 3, time_line_mat_target, compressed_target);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // generate and write the file for target feature, only for visualization, can be commented later
    //////////////////////////////////////////////////////////////////////////////////////////////////
    vtkSmartPointer< vtkPolyData> polydata_target = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> points_target = vtkSmartPointer< vtkPoints >::New();
    vtkSmartPointer<vtkCellArray> vertices_target = vtkSmartPointer<vtkCellArray>::New();
    vtkIdType ptid_target[1];

    for(int k=0; k<compressed_target.rows; k++)
    {
        ptid_target[0] = points_target->InsertNextPoint(compressed_target.at<float>(k,0),
                                                        compressed_target.at<float>(k,1),
                                                        compressed_target.at<float>(k,2));
        vertices_target->InsertNextCell(1,ptid_target);
    }

    polydata_target->SetPoints(points_target);
    polydata_target->SetVerts(vertices_target);
    //polydata_target->SetPolys(fstar.surface->GetPolys()); // important line to get the surface geometry back

    // Write the file for target feature
    /////////////////////////////////////
    string file;
    file = "target_feature_" + tt.str() + ".vtp";
    vtkSmartPointer<vtkXMLPolyDataWriter> writer =  vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(file.c_str());
    writer->SetInputData(polydata_target);
    writer->Write();

    /////////////////////////////////////////////////////////
    //Select combination of two
    /////////////////////////////////////////////////////////

    int seq_id=0;
    glob_t_pca=0;
    glob_t_find_min_point=0;
    for(int i=0; i<current_feature_list.size(); i++)
    {
        glm :: vec3 center1;
        vector<glm::vec3> first_feature;

        vtkDataArray *point_ary1 = current_feature_list[i].surface->GetPoints()->GetData();

        for(int k=0;k<point_ary1->GetNumberOfTuples();k++)
        {
            point_ary1->GetTuple(k,p);
            first_feature.push_back(glm::vec3(p[0],p[1],p[2]));
        }

        center1 = get_center(first_feature);

        for(int j=i+1; j<current_feature_list.size(); j++)
        {
            glm :: vec3 center2;
            vector<glm::vec3> second_feature;
            vector<glm::vec3> combined_feature;

            //copy first feature points to combined feature
            combined_feature = first_feature;

            vtkDataArray *point_ary2 = current_feature_list[j].surface->GetPoints()->GetData();
            for(int k=0;k<point_ary2->GetNumberOfTuples();k++)
            {
                point_ary2->GetTuple(k,p);
                second_feature.push_back(glm::vec3(p[0],p[1],p[2]));
            }

            center2 = get_center(second_feature);

            /////////////////////////////////////////////////
            //consider search in a local neighborhood
            /////////////////////////////////////////////////
            float distance1 = distance_func(center,center1);
            distance1 = distance1/maxdist;

            float distance2 = distance_func(center,center2);
            distance2 = distance2/maxdist;

            if(distance1 < neighborhood && distance2 < neighborhood) //consider a local neighborhood value depends on dataset
            {
                //Find the translation factor between the pair of feature points
                start_t = rtclock();
                glm::vec3 min_point_distance = get_min_point(first_feature,second_feature);
                end_t = rtclock();
                glob_t_find_min_point += end_t - start_t;

                stringstream seqid;
                seqid << seq_id;
                ofstream comb_file_ptr;
                string comb_file;

                //translate points in second feature wrt to min distance point and merge to combine feature
                for(int k=0;k<second_feature.size();k++)
                {
                    combined_feature.push_back(glm::vec3(second_feature[k].x + min_point_distance.x,
                                                         second_feature[k].y + min_point_distance.y,
                                                         second_feature[k].z + min_point_distance.z));
                }

                ////////////////////////////////////////////////////////////////////////////////////////////
                // Do PCA here
                ////////////////////////////////////////////////////////////////////////////////////////////
                Mat compressed_new;
                Mat time_line_mat_new(combined_feature.size(),3,CV_32FC1);
                //Populate data for PCA
                for(int k=0;k<combined_feature.size();k++)
                {
                    time_line_mat_new.at<float>(k,0) = combined_feature[k].x;
                    time_line_mat_new.at<float>(k,1) = combined_feature[k].y;
                    time_line_mat_new.at<float>(k,2) = combined_feature[k].z;
                }

                //Do PCA: compressed matrix will have the required values
                start_t = rtclock();
                PCA_analysis(time_line_mat_new, 3, time_line_mat_new, compressed_new);
                end_t = rtclock();
                glob_t_pca += end_t - start_t;

                ////////////////////////////////////////////////////////////////////////////////////////////
                //Join the two features only for visualization.. can comment later
                ////////////////////////////////////////////////////////////////////////////////////////////
                vtkSmartPointer<vtkPolyData> polydata_new = vtkSmartPointer<vtkPolyData>::New();
                vtkSmartPointer<vtkPoints> points_new = vtkSmartPointer< vtkPoints >::New();
                vtkSmartPointer<vtkCellArray> vertices_new = vtkSmartPointer<vtkCellArray>::New();
                vtkIdType ptid_new[1];
                //color array
                vtkSmartPointer<vtkFloatArray> colors = vtkSmartPointer<vtkFloatArray>::New();
                colors->SetNumberOfComponents(1);
                colors->SetName ("Colors");

                //populate polydata
                for(int k=0; k<compressed_new.rows; k++)
                {
                    ptid_new[0] = points_new->InsertNextPoint(compressed_new.at<float>(k,0),
                                                              compressed_new.at<float>(k,1),
                                                              compressed_new.at<float>(k,2));
                    vertices_new->InsertNextCell(1,ptid_new);

                    //assign the pieces in different colors
                    if(k<first_feature.size())
                        colors->InsertNextTuple1(0.0);
                    else
                        colors->InsertNextTuple1(255.0);
                }

                vtkSmartPointer<vtkTransform> translation1 = vtkSmartPointer<vtkTransform>::New();
                translation1->Translate(min_point_distance.x,min_point_distance.y,min_point_distance.z);
                vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                transformFilter->SetInputData(current_feature_list[j].surface);
                transformFilter->SetTransform(translation1);
                transformFilter->Update();
                vtkSmartPointer<vtkAppendPolyData> appendedData = vtkSmartPointer<vtkAppendPolyData> :: New();
                appendedData->AddInputData(current_feature_list[i].surface); // feature 1
                appendedData->AddInputData(transformFilter->GetOutput()); // feature 2 after translation
                appendedData->Update(); // features are glued

                polydata_new->SetPoints(points_new);
                polydata_new->SetVerts(vertices_new);
                //polydata_new->SetPolys(appendedData->GetOutput()->GetPolys()); // get the geometry info
                polydata_new->GetPointData()->SetScalars(colors);

                ////write vtp
                ///////////////////
                stringstream xx,yy;
                xx<<i;
                yy<<j;
                string file = "out_text_" + xx.str() + "_" + yy.str() + "_" + tt.str() + ".vtp";
                vtkSmartPointer<vtkXMLPolyDataWriter> writer =  vtkSmartPointer<vtkXMLPolyDataWriter>::New();
                writer->SetFileName(file.c_str());
                writer->SetInputData(polydata_new);
                writer->Write();

                /////////////////////////////////////////////////////////////////////////////////////////
                // Write the csv file of PCA output: first two components which will be used for Jaccard
                /////////////////////////////////////////////////////////////////////////////////////////
                comb_file = "/home/soumya/comb_features/comb_feature_" + tt.str()  + "_" + seqid.str() + ".csv";
                comb_file_ptr.open(comb_file.c_str());
                comb_file_ptr<<"x"<<","<<"y"<<","<<"z"<<endl;

                for(int k=0;k<compressed_new.rows;k++)
                    comb_file_ptr<<round(compressed_new.at<float>(k,0))<<","<<round(compressed_new.at<float>(k,1))<<","<<round(compressed_new.at<float>(k,2))<<endl;

                comb_file_ptr.close();

                feature_comb_map.insert(pair<int,glm::vec2>(seq_id,glm::vec2(i,j)));
                seq_id++;
            }
        }
    }

    cout<<"time for all loop PCA: "<<glob_t_pca<<" secs"<<endl;
    cout<<"time for all loop find_min_point: "<<glob_t_find_min_point<<" secs"<<endl;

    return feature_comb_map;
}

#endif
