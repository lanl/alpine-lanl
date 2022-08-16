#include <vtk_headers.h>
#include <cpp_headers.h>
#include <sys/time.h>

using namespace std;

const float percentageToKeep = 5;
string ArrName1 = "Pressure";
string ArrName2 = "QVapor";
string dataset = "isabel_pressure_qva";

/////////////////////////
// specific to sampling
int Bin = 128;
float *ratioHist;
int pointsToretain=0;
int pointsPerBin;
int *Array1; 
int *Array2; 
int **ArrayComb; 

// Timing function
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

struct sortHistogram
{
    int freq;
    int binId;

    sortHistogram(int k, int s) : freq(k), binId(s) {}

    bool operator < (const sortHistogram& hist) const
    {
        return (freq < hist.freq);
    }
};

struct sortPMIHistogram
{
    float pmi;
    int binId;

    sortPMIHistogram(float k, int s) : pmi(k), binId(s) {}

    bool operator < (const sortPMIHistogram& hist) const
    {
        return (pmi < hist.pmi);
    }
};

int compute_3d_to_1d_map(int x,int y,int z, int dimx, int dimy, int dimz)
{

    return x + dimx*(y+dimy*z);
}

int compute_2d_to_1d_map(int x, int y, int Bin)
{

    return y + (x*Bin);
}

int compute_1d_to_2d_map(int index, int Bin1, int Bin2)
{

    return index/Bin + Bin2;
}

int main(int argc,char* argv[])
{
	//initialize random seed
    srand (time(NULL));

    stringstream per;
    per<<percentageToKeep;

    ///////////////////////////////////////////////////////////////////////
    // Read data here
    vtkSmartPointer<vtkImageData> raw_data1;
    raw_data1 = readVtiRawData(argv[1]);

    vtkSmartPointer<vtkImageData> raw_data2;
    raw_data2 = readVtiRawData(argv[2]);

    int* dims = raw_data1->GetDimensions();
    double* spaceing = raw_data1->GetSpacing();

    cout<<"Data reading done"<<endl;

    //cout<<compute_2d_to_1d_map(0,0,Bin)<<endl;
    //cout<<compute_2d_to_1d_map(0,1,Bin)<<endl;
    //cout<<compute_2d_to_1d_map(1,0,Bin)<<endl;
    //cout<<1/Bin<<" "<<1%Bin<<endl;

    //////////////////////////////////////////////////////////////////////////////////////
    // Compute Histogram and PMI 
    //////////////////////////////////////////////////////////////////////////////////////

    cout<<"Computing Joint Histogram"<<endl;
    clkbegin = rtclock();

    double* range1 = raw_data1->GetPointData()->GetArray(ArrName1.c_str())->GetRange();
    double* range2 = raw_data2->GetPointData()->GetArray(ArrName2.c_str())->GetRange();
    long int numOfPoints = raw_data1->GetPointData()->GetArray(ArrName1.c_str())->GetNumberOfTuples();

    cout<<"number of data points: "<<numOfPoints<<endl;

    //Allocate memory
    Array1 = (int *)malloc(Bin*sizeof(int));
    Array2 = (int *)malloc(Bin*sizeof(int));
    ArrayComb = (int **)malloc(Bin*sizeof(int *));
    for(int i=0;i<Bin;i++)
        ArrayComb[i] = (int *)malloc(Bin*sizeof(int));

    for(int i=0;i<Bin;i++)
    {
        Array1[i]=0;
        Array2[i]=0;

        for(int j=0;j<Bin;j++)
        {
            ArrayComb[i][j] = 0;
        }
    }

    for(int i=0;i<numOfPoints;i++)
    {
        double val1 = raw_data1->GetPointData()->GetArray(ArrName1.c_str())->GetTuple1(i);
        double val2 = raw_data2->GetPointData()->GetArray(ArrName2.c_str())->GetTuple1(i);
        int binid1 = (int)(((val1-range1[0])/(range1[1]-range1[0]))*(Bin-1));
        int binid2 = (int)(((val2-range2[0])/(range2[1]-range2[0]))*(Bin-1));

        Array1[binid1]++; 
        Array2[binid2]++; 
        ArrayComb[binid1][binid2]++; 
    }

    //Create Histogram
    vector < sortPMIHistogram > PMIHist;
    for(int i=0;i<Bin*Bin;i++)
        PMIHist.push_back(sortPMIHistogram(0.0, i));

    //Calculatre PMI Here*************************************/
    for(int i=0;i<Bin;i++)
    {
        for(int j=0;j<Bin;j++)
        {
            double joint_prob_xy = ArrayComb[i][j] /(float) numOfPoints;
            double prob_of_x = Array1[i]/(float)numOfPoints;
            double prob_of_y = Array2[j]/(float)numOfPoints;
            double pmi_val=0;

            if(prob_of_x > 0.0 && prob_of_y > 0.0 && joint_prob_xy > 0.0)
            {
                pmi_val = log2(joint_prob_xy/(prob_of_x*prob_of_y));       
                
                //if(pmi_val<0)
                //pmi_val=0;        
            }
            else
            {
            	pmi_val=0;
            }

            int index = compute_2d_to_1d_map(i,j,Bin);
            PMIHist[index].pmi = pmi_val;            
        }
    }

    free(Array1);
    free(Array2);
    free(ArrayComb);

    ///////////////////////////////////////////////////////
    // PMI-based sampling is done here 

    //// Sort PMI histogram first
    sort(PMIHist.begin(), PMIHist.end());
    double min = PMIHist[0].pmi;
    double max = PMIHist[Bin*Bin-1].pmi;

    cout<<"max min pmi vals are: "<<max<<" "<<min<<endl;

    // for(int i=0;i<Bin*Bin;i++)
    // 	cout<<PMIHist[i].pmi<<" "<<PMIHist[i].binId<<endl;

    clkend = rtclock();
    cout<<"PMI Histogram is done and time taken: "<<clkend - clkbegin<<" secs"<<endl;    

    pointsToretain = (numOfPoints*percentageToKeep)/100.0;
    pointsPerBin = pointsToretain/(Bin*Bin);
    cout<<"Number of points to be sampled: "<<pointsToretain<<" and number of points to be picked per bin: "<<pointsPerBin<<endl;

    //Sample points based on sampling function
    ratioHist = (float *)malloc(Bin*Bin*sizeof(float));

    //Normalize PMI vals and use that as importance function
    for(int i=0;i<Bin*Bin;i++)
    {
    	//PMIHist[i].pmi = (PMIHist[i].pmi -min)/(max-min);
    	ratioHist[i] = (PMIHist[i].pmi -min)/(max-min);
    	cout<<i<<" "<<ratioHist[i]<<" "<<PMIHist[i].pmi<<endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // Sample points from global importance function
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkDoubleArray> var1 = vtkSmartPointer<vtkDoubleArray>::New();
    var1->SetName("pressure");
    vtkSmartPointer<vtkDoubleArray> var2 = vtkSmartPointer<vtkDoubleArray>::New();
    var2->SetName("qvapor");

    //Use PMI vals to sample
    int ii=0;
    for(int k=0;k<dims[2];k++)
        for(int j=0;j<dims[1];j++)
            for(int i=0;i<dims[0];i++)
            {
                double val1 = raw_data1->GetPointData()->GetArray(ArrName1.c_str())->GetTuple1(ii);
        		double val2 = raw_data2->GetPointData()->GetArray(ArrName2.c_str())->GetTuple1(ii);
        		int binid1 = (int)(((val1-range1[0])/(range1[1]-range1[0]))*(Bin-1));
        		int binid2 = (int)(((val2-range2[0])/(range2[1]-range2[0]))*(Bin-1));

        		int index = compute_2d_to_1d_map(binid1,binid2,Bin);
                double rand_val = ((double) rand() / (RAND_MAX));

                if(rand_val > ratioHist[index])
                {
                    points->InsertNextPoint(double(i)*spaceing[0],double(j)*spaceing[1],double(k)*spaceing[2]);
                    var1->InsertNextTuple(&val1);
                    var2->InsertNextTuple(&val2);
                }

                ii++;
            }

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->GetPointData()->AddArray(var1);
    polydata->GetPointData()->AddArray(var2);
    polydata->SetPoints(points);


    //write output files
    /////////////////////////
    cout<<"Writing pmi sampled files out"<<endl;

    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    string fname = "/Users/sdutta/Desktop/" + dataset + "_" + per.str() + "_percent_sampled_hist.vtp";
    writer->SetFileName(fname.c_str());
    writer->SetInputData(polydata);
    writer->SetDataModeToBinary();
    writer->Write();     

	return 0;
}
