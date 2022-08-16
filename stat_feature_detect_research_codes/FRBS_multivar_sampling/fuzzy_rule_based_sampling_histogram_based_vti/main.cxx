#include <cpp_headers.h>
#include <glm_headers.h>
#include <utils.h>
#include <vtk_headers.h>

using namespace std;

////Isabel settings/////////////////////////////////
string var1 = "Pressure";
string var2 = "Velocity";
string var3 = "QVapor";
string inputdata  = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/data/Isabel_pressure_velocity_qvapor.vti";
string membership_in = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/membership_functions/isabel/inputmfs.txt";
string membership_out = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/membership_functions/isabel/outputmfs.txt";
string sampled_outfile = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/output/isabel_sampled.vtp";
////////////////////////////////////////////////////

///// //MFIX settings/////////////////////////////////
//string var1 = "density";
//string var2 = "gradient";
//string var3 = "gradient1";
//string inputdata  = "/home/soumya/Test_DataSet/multivar_sampling_test_data/mfix_density_gradient_gradient.vti";
////////////////////////////s////////////////////////

///// //NYX settings/////////////////////////////////
//string var1 = "logDensity";
//string var2 = "logTemperature";
//string var3 = "logRho";
//string inputdata  = "/home/soumya/Test_DataSet/multivar_sampling_test_data/Nyx_density_temp_rho_400.vti";
////////////////////////////////////////////////////

///// //ASTEROID settings/////////////////////////////////
//string var1 = "tev";
//string var2 = "v02";
//string var3 = "v03";
//string inputdata  = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/data/asteroid_28649.vti";
//string membership_in = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/membership_functions/asteroid/inputmfs.txt";
//string membership_out = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/membership_functions/asteroid/outputmfs.txt";
//string sampled_outfile = "/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/output/asteroid_28649_sampled.vtp";
////////////////////////////////////////////////////


int dim[3];
const int BIN=128;
float samp_fraction=5.0; //TODO
const int rule_num = 3; //TODO
const int input_dim_num = 3; //TODO
long totNumPts,ptsToSample;

double clkbegin=0;
double clkend=0;
double writeclkbegin=0;
double writeclkend=0;
double writeTime=0;
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


class Point{

public:

    int v1;
    int v2;
    int v3;

    Point(int v1, int v2, int v3)
    {
        this->v1 = v1;
        this->v2 = v2;
        this->v3 = v3;
    }
};

//Definition of methods
int compute_3d_to_1d_map(int x,int y,int z, int dimx, int dimy, int dimz)
{
    
    return x + dimx*(y+dimy*z);
}

void write_polydata(vtkSmartPointer<vtkPolyData> data)
{
    string fname = sampled_outfile;
    vtkSmartPointer<vtkXMLPolyDataWriter> mbwriter = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    mbwriter->SetInputData(data);
    mbwriter->SetEncodeAppendedData(0);
    mbwriter->SetFileName(fname.c_str());
    mbwriter->Write();
}

vtkSmartPointer<vtkImageData> load_data(string fname)
{
    vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(fname.c_str());
    reader->Update();
    return reader->GetOutput();
}

void insertCorner(int index, vtkSmartPointer<vtkImageData> mb, double var1_range[2], double var2_range[2], double var3_range[2],
float*** BinAcceptance, vtkSmartPointer<vtkPoints> newPoints,
vtkSmartPointer<vtkFloatArray> sampled_arr, vtkSmartPointer<vtkFloatArray> sampled_arr_var1,
vtkSmartPointer<vtkFloatArray> sampled_arr_var2, vtkSmartPointer<vtkFloatArray> sampled_arr_var3)
{
    double* pts;
    pts = mb->GetPoint(index);
    float var1_val = mb->GetPointData()->GetArray(var1.c_str())->GetTuple1(index);
    float var2_val = mb->GetPointData()->GetArray(var2.c_str())->GetTuple1(index);
    float var3_val = mb->GetPointData()->GetArray(var3.c_str())->GetTuple1(index);
    int binid1 = (int)(((var1_val-var1_range[0])/(var1_range[1]-var1_range[0]))*(BIN-1));
    int binid2 = (int)(((var2_val-var2_range[0])/(var2_range[1]-var2_range[0]))*(BIN-1));
    int binid3 = (int)(((var3_val-var3_range[0])/(var3_range[1]-var3_range[0]))*(BIN-1));
    float rett = BinAcceptance[binid1][binid2][binid3];

    newPoints->InsertNextPoint(pts);
    sampled_arr->InsertNextTuple1(rett);
    sampled_arr_var1->InsertNextTuple1(var1_val);
    sampled_arr_var2->InsertNextTuple1(var2_val);
    sampled_arr_var3->InsertNextTuple1(var3_val);

    //cout<<"retval: "<<rett<<" index: "<<index<<endl;
}

int main(int argc, char** argv)
{
    /* initialize random seed: */
    srand (time(NULL));

    float temp_rule_matrix1[rule_num][input_dim_num];
    float temp_rule_matrix2[rule_num][input_dim_num];
    string line;
    vector<glm::vec2> temp_rule_matrix;

    //Read the parameters of the trained fuzzy rule based system
    ///////////////////////////////////////////////////////////////////
    ifstream readoutFis;
    readoutFis.open(membership_out.c_str());

    //Read outmfs
    getline(readoutFis, line);
    vector<float> outparamvals = split(line, ",");

    //Read inmfs
    ifstream readinFis;
    readinFis.open(membership_in.c_str());

    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                temp_rule_matrix.push_back(glm::vec2(v[0],v[1]));
            }
        }
    }

    int ij=0;
    for(int qq=0;qq<input_dim_num;qq++)
    {
        for(int jj=0;jj<rule_num;jj++)
        {
            glm::vec2 v21 = temp_rule_matrix[ij++];
            temp_rule_matrix1[jj][qq] = v21.x; // sigma vals
            temp_rule_matrix2[jj][qq] = v21.y; // mean vals
        }
    }

    temp_rule_matrix.clear();

    ///////////////////////////////////////////////
    //Fuzzy rule based inference system creation
    ///////////////////////////////////////////////
    Rule_Based_System rulebase;
    rulebase.num_rules = rule_num;
    rulebase.num_input_dim = input_dim_num;
    rulebase.fuzzy_system_type = "TSK";

    for(int qq=0;qq<rule_num;qq++)
    {
        Rules rule;
        rule.membership_func_type = "GMF";
        
        for(int jj=0;jj<input_dim_num;jj++)
        {
            Membership_func mm;
            mm.sigma = temp_rule_matrix1[qq][jj]; // sigma vals
            mm.mean = temp_rule_matrix2[qq][jj]; // mean vals
            rule.inputmfs.push_back(mm);
        }

        for(int jj=0;jj<=input_dim_num;jj++)
        {
            rule.out_params.push_back(outparamvals[jj]);
        }

        rulebase.rules.push_back(rule);
    }

    //////////////////////////////////////////////////////////////////////////////////
    //At this point the rule based system is ready to use
    //////////////////////////////////////////////////////////////////////////////////

    string ffname = "timing.txt";
    ofstream globfptr;
    globfptr.open(ffname.c_str(),ios::out);
    float *datavals;
    datavals = (float *)malloc(sizeof(float)*input_dim_num);

    int ***Histogram;
    float ***predictedBinVals;
    float ***BinAcceptance;

    predictedBinVals = (float ***)malloc(BIN*sizeof(float **));
    for(int i=0;i<BIN;i++)
    {
        predictedBinVals[i] = (float **)malloc(BIN*sizeof(float *));
        for(int j=0;j<BIN;j++)
            predictedBinVals[i][j] = (float *)malloc(BIN*sizeof(float));
    }

    BinAcceptance = (float ***)malloc(BIN*sizeof(float **));
    for(int i=0;i<BIN;i++)
    {
        BinAcceptance[i] = (float **)malloc(BIN*sizeof(float *));
        for(int j=0;j<BIN;j++)
            BinAcceptance[i][j] = (float *)malloc(BIN*sizeof(float));
    }

    Histogram = (int ***)malloc(BIN*sizeof(int **));
    for(int i=0;i<BIN;i++)
    {
        Histogram[i] = (int **)malloc(BIN*sizeof(int *));
        for(int j=0;j<BIN;j++)
            Histogram[i][j] = (int *)malloc(BIN*sizeof(int));
    }

    for(int i=0;i<BIN;i++)
    {
        for(int j=0;j<BIN;j++)
        {
            for(int k=0;k<BIN;k++)
            {
                Histogram[i][j][k] = 0;
                BinAcceptance[i][j][k]=0;
            }
        }
    }

    cout<<"processing data"<<endl;

    //Now load data here for inference
    //////////////////////////////////////////////////////
    vtkSmartPointer<vtkImageData> mb = vtkSmartPointer<vtkImageData>::New();

    //Loads the data into a vtk multiblock dataset
    clkbegin = rtclock();
    mb = load_data(inputdata);
    clkend = rtclock();
    globfptr<<clkend-clkbegin<<" ";

    mb->GetDimensions(dim);
    totNumPts = dim[0]*dim[1]*dim[2];
    ptsToSample = (totNumPts*samp_fraction)/100.0;

    cout<<"data loading time: "<<clkend-clkbegin<<" secs"<<endl;

    clkbegin = rtclock();
    double var1_range[2];
    double var2_range[2];
    double var3_range[2];
    double *temp;

    //find the ranges for the variables
    temp = mb->GetPointData()->GetArray(var1.c_str())->GetRange();
    var1_range[0] = temp[0]; var1_range[1] = temp[1];
    temp = mb->GetPointData()->GetArray(var2.c_str())->GetRange();
    var2_range[0] = temp[0]; var2_range[1] = temp[1];
    temp = mb->GetPointData()->GetArray(var3.c_str())->GetRange();
    var3_range[0] = temp[0]; var3_range[1] = temp[1];

    temp = mb->GetPointData()->GetArray(var1.c_str())->GetRange();
    var1_range[0]=temp[0];
    var1_range[1]=temp[1];

    temp = mb->GetPointData()->GetArray(var2.c_str())->GetRange();
    var2_range[0]=temp[0];
    var2_range[1]=temp[1];

    temp = mb->GetPointData()->GetArray(var3.c_str())->GetRange();
    var3_range[0]=temp[0];
    var3_range[1]=temp[1];

    //////////////////////////////////////////////////////////////////////////////
    //compute multivariate histogram
    int index=0;
    for(int r=0;r<dim[2];r++)
        for(int q=0;q<dim[1];q++)
            for(int p=0;p<dim[0];p++)
            {
                index = compute_3d_to_1d_map(p,q,r,dim[0],dim[1],dim[2]);
                float var1_val = mb->GetPointData()->GetArray(var1.c_str())->GetTuple1(index);
                float var2_val = mb->GetPointData()->GetArray(var2.c_str())->GetTuple1(index);
                float var3_val = mb->GetPointData()->GetArray(var3.c_str())->GetTuple1(index);

                int binid1 = (int)(((var1_val-var1_range[0])/(var1_range[1]-var1_range[0]))*(BIN-1));
                int binid2 = (int)(((var2_val-var2_range[0])/(var2_range[1]-var2_range[0]))*(BIN-1));
                int binid3 = (int)(((var3_val-var3_range[0])/(var3_range[1]-var3_range[0]))*(BIN-1));

                Histogram[binid1][binid2][binid3]++;
            }

    clkend = rtclock();
    globfptr<<clkend-clkbegin<<endl;
    cout<<"histogram computation time: "<<clkend-clkbegin<<" secs"<<endl;

    //perform inference for all the bin centers
    clkbegin = rtclock();
    for(int i=0;i<BIN;i++)
    {
        for(int j=0;j<BIN;j++)
        {
            for(int k=0;k<BIN;k++)
            {
                datavals[0] = var1_range[0] + (k/(float)(BIN-1))*(var1_range[1]-var1_range[0]);
                datavals[1] = var2_range[0] + (j/(float)(BIN-1))*(var2_range[1]-var2_range[0]);
                datavals[2] = var3_range[0] + (i/(float)(BIN-1))*(var3_range[1]-var3_range[0]);

                float rett = evaluate_rulebase(rulebase,datavals,3);

                if(rett>1.0) //clamp to 1 if greater than 1
                    rett=1.0;
                else if(rett<0) //clamp to 0 if less than 0
                    rett=0.0;

                predictedBinVals[k][j][i] = rett;
            }
        }
    }
    clkend = rtclock();
    cout<<"inference for all bin centers time: "<<clkend-clkbegin<<" secs"<<endl;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //    float maxval=0;
    //    index=0;
    //    for(int r=0;r<dim[2];r++)
    //        for(int q=0;q<dim[1];q++)
    //            for(int p=0;p<dim[0];p++)
    //            {
    //                index = compute_3d_to_1d_map(p,q,r,dim[0],dim[1],dim[2]);
    //                datavals[0] = mb->GetPointData()->GetArray(var1.c_str())->GetTuple1(index);
    //                datavals[1] = mb->GetPointData()->GetArray(var2.c_str())->GetTuple1(index);
    //                datavals[2] = mb->GetPointData()->GetArray(var3.c_str())->GetTuple1(index);

    //                float rett = evaluate_rulebase(rulebase,datavals,3);

    //                if (maxval < rett)
    //                    maxval=rett;
    //            }

    //    cout<<"max prediction: "<<maxval<<endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //estimate sampling percentage
    float val=0;
    for(int i=0;i<BIN;i++)
    {
        for(int j=0;j<BIN;j++)
        {
            for(int k=0;k<BIN;k++)
            {
                val = val + Histogram[k][j][i]*predictedBinVals[k][j][i];
            }
        }
    }

    float frac = val / ptsToSample;

    for(int i=0;i<BIN;i++)
    {
        for(int j=0;j<BIN;j++)
        {
            for(int k=0;k<BIN;k++)
            {
                if(Histogram[k][j][i]>0)
                    BinAcceptance[k][j][i] = ((Histogram[k][j][i]*predictedBinVals[k][j][i])/frac)/Histogram[k][j][i];
                else
                    BinAcceptance[k][j][i]=0.0;
            }
        }
    }

    // perform final inference for all points and sample based on histogram
    clkbegin = rtclock();

    vtkSmartPointer<vtkPolyData> pdata = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> newPoints = vtkSmartPointer<vtkPoints>::New();

    //create stallness array
    vtkSmartPointer<vtkFloatArray> sampled_arr = vtkSmartPointer<vtkFloatArray>::New();
    sampled_arr->SetName("predicted_acceptance");
    sampled_arr->SetNumberOfComponents(1);

    vtkSmartPointer<vtkFloatArray> sampled_arr_var1 = vtkSmartPointer<vtkFloatArray>::New();
    sampled_arr_var1->SetName(var1.c_str());
    sampled_arr_var1->SetNumberOfComponents(1);

    vtkSmartPointer<vtkFloatArray> sampled_arr_var2 = vtkSmartPointer<vtkFloatArray>::New();
    sampled_arr_var2->SetName(var2.c_str());
    sampled_arr_var2->SetNumberOfComponents(1);

    vtkSmartPointer<vtkFloatArray> sampled_arr_var3 = vtkSmartPointer<vtkFloatArray>::New();
    sampled_arr_var3->SetName(var3.c_str());
    sampled_arr_var3->SetNumberOfComponents(1);

    int id=0;
    double* pts;
    for(int r=0;r<dim[2];r++)
        for(int q=0;q<dim[1];q++)
            for(int p=0;p<dim[0];p++)
            {
                index = compute_3d_to_1d_map(p,q,r,dim[0],dim[1],dim[2]);
                float var1_val = mb->GetPointData()->GetArray(var1.c_str())->GetTuple1(index);
                float var2_val = mb->GetPointData()->GetArray(var2.c_str())->GetTuple1(index);
                float var3_val = mb->GetPointData()->GetArray(var3.c_str())->GetTuple1(index);

                int binid1 = (int)(((var1_val-var1_range[0])/(var1_range[1]-var1_range[0]))*(BIN-1));
                int binid2 = (int)(((var2_val-var2_range[0])/(var2_range[1]-var2_range[0]))*(BIN-1));
                int binid3 = (int)(((var3_val-var3_range[0])/(var3_range[1]-var3_range[0]))*(BIN-1));

                float rett = BinAcceptance[binid1][binid2][binid3];
                double random_val = ((double) rand() / (RAND_MAX));

                pts = mb->GetPoint(id);
                if(random_val<=rett)
                {
                    newPoints->InsertNextPoint(pts);
                    sampled_arr->InsertNextTuple1(rett);

                    sampled_arr_var1->InsertNextTuple1(var1_val);
                    sampled_arr_var2->InsertNextTuple1(var2_val);
                    sampled_arr_var3->InsertNextTuple1(var3_val);
                }

                id++;
            }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //explicitely insert the corner data points for python linear interpolation to be correctly working

    ///////////////////////////////////////////////////////
    /// \brief Get corners
    ///
    vector<Point> corners;

    Point c(0,0,0);
    corners.push_back(c);

    Point c1(0,0,dim[2]-1);
    corners.push_back(c1);

    Point c2(0,dim[1]-1,0);
    corners.push_back(c2);

    Point c3(0,dim[1]-1,dim[2]-1);
    corners.push_back(c3);

    Point c4(dim[0]-1,0,0);
    corners.push_back(c4);

    Point c5(dim[0]-1,0,dim[2]-1);
    corners.push_back(c5);

    Point c6(dim[0]-1,dim[1]-1,0);
    corners.push_back(c6);

    Point c7(dim[0]-1,dim[1]-1,dim[2]-1);
    corners.push_back(c7);

    // insert the corners
    for (int ii=0; ii<corners.size();ii++)
    {
        Point currentP = corners[ii];

        index = compute_3d_to_1d_map(currentP.v1,currentP.v2,currentP.v3,dim[0],dim[1],dim[2]);
        insertCorner(index,mb,var1_range,var2_range,var3_range,BinAcceptance, newPoints, sampled_arr,
                     sampled_arr_var1, sampled_arr_var2, sampled_arr_var3);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////

    pdata->SetPoints(newPoints.GetPointer());
    pdata->GetPointData()->AddArray(sampled_arr.GetPointer());
    pdata->GetPointData()->AddArray(sampled_arr_var1.GetPointer());
    pdata->GetPointData()->AddArray(sampled_arr_var2.GetPointer());
    pdata->GetPointData()->AddArray(sampled_arr_var3.GetPointer());

    clkend = rtclock();
    globfptr<<clkend-clkbegin<<endl;
    cout<<"prediction time: "<<clkend-clkbegin<<" secs"<<endl;

    ///write file out with sampled data
    /////////////////////////////////////////////
    clkbegin = rtclock();
    write_polydata(pdata);
    clkend = rtclock();
    globfptr<<clkend-clkbegin<<endl;

    cout<<"num points sampled: "<<pdata->GetNumberOfPoints()<<endl;

    cout<<"output I/O time: "<<clkend-clkbegin<<" secs"<<endl;

    globfptr.close();

    free(predictedBinVals);
    free(Histogram);
    free(BinAcceptance);

    return 0;
}
