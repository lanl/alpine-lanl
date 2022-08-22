//Sample run command: ./slic data.vti
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vtk_headers.h>
#include <slic.h>
#include <sys/time.h>

using namespace std;

//Some control params
string ArrName = "ImageScalars";
int QUANTIZATION=1;
int blockXSize=3;
int blockYSize=3;
int blockZSize=3;
float halt_cond=0.0005;
double TH=1;
double TH_size=25;
double rtclock();
double clkbegin, clkend;
double t=0;
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

//writes a vti cluster_ids file and also a run length encoded compressed raw cluster_ids file
void writeVtiOClusterutput(int*** cluster_info, int xdim, int ydim, int zdim, vtkSmartPointer<vtkImageData> raw_data, string fname)
{
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    //imageData->AllocateScalars(VTK_INT,1);

    imageData->SetDimensions(raw_data->GetDimensions());
    imageData->SetSpacing(raw_data->GetSpacing());
    imageData->SetOrigin(raw_data->GetOrigin());

    vtkSmartPointer<vtkIntArray> simdata = vtkSmartPointer<vtkIntArray>::New();
    simdata->SetNumberOfComponents(1);
    simdata->SetName("ClusterIds");

    int ii=0;
    //Fill every entry of the image data with cluster id
    for (int z = 0; z < zdim; z++)
        for (int y = 0; y < ydim; y++)
            for (int x = 0; x < xdim; x++)
            {
                //double val = log(output->GetPointData()->GetArray("simdata")->GetTuple1(i));
                //int* pixel = static_cast<int*>(imageData->GetScalarPointer(x,y,z));
                //pixel[0] = (int)cluster_info[x][y][z];

                int val = cluster_info[x][y][z];
                simdata->InsertTuple1(ii++,val);
            }
    imageData->GetPointData()->AddArray(simdata);

    //Write the output
    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    stringstream ss;
    ss<<blockXSize;
    //string fname = "cluster.vti";
    writer->SetFileName(fname.c_str());
    writer->SetDataModeToBinary();
    writer->SetInputData(imageData);
    writer->Write();
}

vtkSmartPointer<vtkImageData> readVtiRawData(char* filename)
{
    /////////////////////////////////////////////////////////////////////
    //Open data file and read the data
    // Read the file
    vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(filename);
    reader->Update();
    return reader->GetOutput();
}

//normalize data field for SLIC
float*** normalize_data(vtkSmartPointer<vtkImageData> data,int xdim, int ydim, int zdim)
{
    float*** array;
    double val=0;

    array = (float ***)malloc(xdim*sizeof(float **));
    for(int i=0;i<xdim;i++)
    {
        array[i] = (float **)malloc(ydim*sizeof(float *));
        for(int j=0;j<ydim;j++)
            array[i][j] = (float *)malloc(zdim*sizeof(float));
    }

    double* range = data->GetPointData()->GetArray(ArrName.c_str())->GetRange();

    int ii=0;
    for(int k=0;k<zdim;k++)
        for(int j=0;j<ydim;j++)
            for(int i=0;i<xdim;i++)
            {
                val = data->GetPointData()->GetArray(ArrName.c_str())->GetTuple1(ii);
                array[i][j][k] = (val-range[0])/(range[1]-range[0]);
                ii++;
            }

    return array;
}

int *** mergeIds_byTH(vector<Element> finalClusterCenters, int*** cluster_ids, double TH, double* dataRange,int xdim, int ydim, int zdim)
{
    int*** array;

    array = (int ***)malloc(xdim*sizeof(int **));
    for(int i=0;i<xdim;i++)
    {
        array[i] = (int **)malloc(ydim*sizeof(int *));
        for(int j=0;j<ydim;j++)
            array[i][j] = (int *)malloc(zdim*sizeof(int));
    }

    for(int k=0;k<zdim;k++)
        for(int j=0;j<ydim;j++)
            for(int i=0;i<xdim;i++)
            {
                int id = cluster_ids[i][j][k];
                double avgVal = dataRange[0] + finalClusterCenters[id].var1*(dataRange[1]-dataRange[0]);

                //array[i][j][k] = avgVal;

                 if(avgVal<= TH)
                     array[i][j][k] = 1;
                 else
                     array[i][j][k]  = 0;
            }

    return array;
}

int *** mergeIds_avgVal(vector<Element> finalClusterCenters, int*** cluster_ids, double TH, double* dataRange,int xdim, int ydim, int zdim)
{
    int*** array;

    array = (int ***)malloc(xdim*sizeof(int **));
    for(int i=0;i<xdim;i++)
    {
        array[i] = (int **)malloc(ydim*sizeof(int *));
        for(int j=0;j<ydim;j++)
            array[i][j] = (int *)malloc(zdim*sizeof(int));
    }

    for(int k=0;k<zdim;k++)
        for(int j=0;j<ydim;j++)
            for(int i=0;i<xdim;i++)
            {
                int id = cluster_ids[i][j][k];
                double avgVal = dataRange[0] + finalClusterCenters[id].var1*(dataRange[1]-dataRange[0]);

                array[i][j][k] = avgVal;

//                 if(avgVal<= TH)
//                     array[i][j][k] = 1;
//                 else
//                     array[i][j][k]  = 0;
            }

    return array;
}

int *** mergeIds_by_size_val(vector<Element> finalClusterCenters, int*** cluster_ids, double TH, double* dataRange,int xdim, int ydim, int zdim)
{
    int*** array;

    array = (int ***)malloc(xdim*sizeof(int **));
    for(int i=0;i<xdim;i++)
    {
        array[i] = (int **)malloc(ydim*sizeof(int *));
        for(int j=0;j<ydim;j++)
            array[i][j] = (int *)malloc(zdim*sizeof(int));
    }

    for(int k=0;k<zdim;k++)
        for(int j=0;j<ydim;j++)
            for(int i=0;i<xdim;i++)
            {
                int id = cluster_ids[i][j][k];
                double avgVal = dataRange[0] + finalClusterCenters[id].var1*(dataRange[1]-dataRange[0]);

                 if(avgVal<= TH && finalClusterCenters[id].numElem >= TH_size)
                     array[i][j][k] = 1;
                 else
                     array[i][j][k]  = 0;
            }

    return array;
}

////////////////////////////////////////////////////////////////////////////////
//main function
int main(int argc,char* argv[])
{
    //initialize random seed
    srand (time(NULL));

    ///////////////////////////////////////////////////////////////////////
    vtkSmartPointer<vtkImageData> raw_data1;
    float ***normalized_raw_data;
    raw_data1 = readVtiRawData(argv[1]);

    int tstep = atoi(argv[2]);

    double* dataRange = raw_data1->GetPointData()->GetArray(ArrName.c_str())->GetRange();
    int* dims = raw_data1->GetDimensions();
    int xdim = dims[0];
    int ydim = dims[1];
    int zdim = dims[2];

    normalized_raw_data = normalize_data(raw_data1,xdim,ydim,zdim);
    cout<<"Data reading done"<<endl;

    //////////////////////////////////////////////////////////////////////
    //Estimate Slic Clusters
    cout<<"Computing Slic"<<endl;
    Slic s;
    s.init(xdim,ydim,zdim,blockXSize,blockYSize,blockZSize,halt_cond,QUANTIZATION);
    clkbegin = rtclock();
    s.computeSlic(normalized_raw_data); //SLIC is computed on normalized data
    int*** cluster_ids = s.getClusterIds();
    clkend = rtclock();
    cout<<"Slic is done and time taken: "<<clkend - clkbegin<<" secs"<<endl;

    ////////////////////////////////////////////////////////////////////////////////
    // process cluster data and merge into two broad groups based on some data threshold
    vector<Element> finalClusterCenters = s.getClusterCenters();
    int*** merged_cluster_ids1 = mergeIds_byTH(finalClusterCenters,cluster_ids,TH,dataRange,xdim, ydim, zdim);
    int*** merged_cluster_ids2 = mergeIds_avgVal(finalClusterCenters,cluster_ids,TH,dataRange,xdim, ydim, zdim);
    int*** merged_cluster_ids3 = mergeIds_by_size_val(finalClusterCenters,cluster_ids,TH,dataRange,xdim, ydim, zdim);

    //write output files
    /////////////////////////
    cout<<"Writing files out"<<endl;
    //writeVtiOClusterutput(merged_cluster_ids1, xdim, ydim, zdim,raw_data1,"thresholded_field.vti");
    //writeVtiOClusterutput(merged_cluster_ids2, xdim, ydim, zdim,raw_data1, "averaged_field.vti");
    //writeVtiOClusterutput(merged_cluster_ids3, xdim, ydim, zdim,raw_data1, "threshold_size_field.vti");

    stringstream ss;
    ss<<tstep;
    string fname = "../out/slic_clusters/cluster_" + ss.str() + ".vti";
    //string fname = "cluster.vti";
    writeVtiOClusterutput(cluster_ids, xdim, ydim, zdim,raw_data1,fname.c_str());

    return 0;
}
