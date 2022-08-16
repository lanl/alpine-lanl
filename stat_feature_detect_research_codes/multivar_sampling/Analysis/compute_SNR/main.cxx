#include <vtk_headers.h>
#include <cpp_headers.h>

using namespace std;

string varname = "ImageFile";
string varname1 = "ImageScalars";

int compute_3d_to_1d_map(int x,int y,int z, int dimx, int dimy, int dimz)
{
    
    return x + dimx*(y+dimy*z);
}

vtkSmartPointer<vtkImageData> readVtiField(char* filename)
{
    vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(filename);
    reader->Update();
    return reader->GetOutput();
}

void computeLocalStats(vtkSmartPointer<vtkImageData> rawData, vtkSmartPointer<vtkImageData> sampledField,float* stats, int* bounds)
{
    double mean_raw=0;
    double stdev_raw=0;
    double mean_sampled=0;
    double stdev_sampled=0;
    double mean_error=0;
    double stdev_error=0;
 
    int numPoints = (bounds[1]-bounds[0])*(bounds[3]-bounds[2])*(bounds[5]-bounds[4]);

    int* dims;
    dims = sampledField->GetDimensions();
 
    for(int i=bounds[4];i<bounds[5];i++)
        for(int j=bounds[2];j<bounds[3];j++)
            for(int k=bounds[0];k<bounds[1];k++)
            {
                int index = compute_3d_to_1d_map(k,j,i,dims[0],dims[1],dims[2]);
                float pixel = sampledField->GetPointData()->GetArray(varname1.c_str())->GetTuple1(index);
                float val = rawData->GetPointData()->GetArray(varname.c_str())->GetTuple1(index);

		if (!isnan(val))
		{
 
                mean_raw += val;
                mean_sampled +=pixel;
                mean_error += fabs(pixel-val);
		}
		else
		{
		cout<<val<<endl;
		}
            }

    mean_raw /= numPoints;
    mean_sampled /= numPoints;
    mean_error /= numPoints;
 
    for(int i=bounds[4];i<bounds[5];i++)
        for(int j=bounds[2];j<bounds[3];j++)
            for(int k=bounds[0];k<bounds[1];k++)
            {
                int index = compute_3d_to_1d_map(k,j,i,dims[0],dims[1],dims[2]);
                float pixel = sampledField->GetPointData()->GetArray(varname1.c_str())->GetTuple1(index);
                float val = rawData->GetPointData()->GetArray(varname.c_str())->GetTuple1(index);
 		
		if (!isnan(val))
		{
                stdev_raw += (val-mean_raw)*(val-mean_raw);
                stdev_sampled += (pixel-mean_sampled)*(pixel-mean_sampled);
                stdev_error += (fabs(pixel-val)-mean_error)*(fabs(pixel-val)-mean_error);
		}
		else
		{
		cout<<val<<endl;
		}
            }
 
    stdev_raw = sqrt(stdev_raw/numPoints);
    stdev_sampled = sqrt(stdev_sampled/numPoints);
    stdev_error = sqrt(stdev_error/numPoints);
 
    stats[0] = mean_raw;
    stats[1] = stdev_raw;
    stats[2] = mean_sampled;
    stats[3] = stdev_sampled;
    stats[4] = 20*log10(stdev_raw/stdev_error);
}

void computGlobalStats(vtkSmartPointer<vtkImageData> rawData, vtkSmartPointer<vtkImageData> sampledField, float* stats)
{
    double mean_raw=0;
    double stdev_raw=0;
    double mean_sampled=0;
    double stdev_sampled=0;
    double mean_error=0;
    double stdev_error=0;

    int* dims;
    dims = sampledField->GetDimensions();
    int nanc = 0;
    for(int i=0;i<dims[2]*dims[1]*dims[0];i++)
    {
        float pixel = sampledField->GetPointData()->GetArray(varname1.c_str())->GetTuple1(i);
        float val = rawData->GetPointData()->GetArray(varname.c_str())->GetTuple1(i);
	
    	if (!isnan(pixel))
    	{
            mean_raw += val;
            mean_sampled +=pixel;
            mean_error += fabs(pixel-val);
    	}
        else
        {
            nanc++;
        }
    }

    cout<<"Percentage of nans: "<<(nanc/(float)(dims[2]*dims[1]*dims[0]))*100<<endl;

    mean_raw /= dims[0]*dims[1]*dims[2];
    mean_sampled /= dims[0]*dims[1]*dims[2];
    mean_error /= dims[0]*dims[1]*dims[2];

    for(int i=0;i<dims[2]*dims[1]*dims[0];i++)
    {
        float pixel = sampledField->GetPointData()->GetArray(varname1.c_str())->GetTuple1(i);
        float val = rawData->GetPointData()->GetArray(varname.c_str())->GetTuple1(i);
	
    	if (!isnan(pixel))
    	{
            stdev_raw += (val-mean_raw)*(val-mean_raw);
            stdev_sampled += (pixel-mean_sampled)*(pixel-mean_sampled);
            stdev_error += (fabs(pixel-val)-mean_error)*(fabs(pixel-val)-mean_error);
    	}
        
    }

    stdev_raw = sqrt(stdev_raw/(dims[0]*dims[1]*dims[2]));
    stdev_sampled = sqrt(stdev_sampled/(dims[0]*dims[1]*dims[2]));
    stdev_error = sqrt(stdev_error/(dims[0]*dims[1]*dims[2]));

    stats[0] = mean_raw;
    stats[1] = stdev_raw;
    stats[2] = mean_sampled;
    stats[3] = stdev_sampled;
    stats[4] = 20*log10(stdev_raw/stdev_error);
}


int main(int argc,char* argv[])
{
    //initialize random seed
    srand (time(NULL));

    //read data
    vtkSmartPointer<vtkImageData> rawData =  vtkSmartPointer<vtkImageData>::New();
    rawData = readVtiField(argv[1]);

    vtkSmartPointer<vtkImageData> sampledData =  vtkSmartPointer<vtkImageData>::New();
    sampledData = readVtiField(argv[2]);

    float stats[5] = {0,0,0,0,0};
    computGlobalStats(rawData,sampledData,stats);
    cout<<"global raw mean: "<<stats[0]<<" global raw stdev: "<<stats[1]<<endl;
    cout<<"global sampled mean: "<<stats[2]<<" global sampled stdev: "<<stats[3]<<endl;
    cout<<"global signal to noise: "<<stats[4]<<endl;

   // float localStats[4] = {0,0,0,0};
   // int bounds[6] = {115,160,100,150,0,50}; // for Pressure, Velocity, QVapor
   //int bounds[6] = {0,100,200,299,100,210}; // for tev
   //int bounds[6] = {130,230,100,200,50,250}; // for v02
   //int bounds[6] = {10,100,220,299,120,180}; // for v03
   // computeLocalStats(rawData,sampledData,localStats,bounds);
   // cout<<"local raw mean: "<<localStats[0]<<" local raw stdev: "<<localStats[1]<<endl;
   // cout<<"local sampled mean: "<<localStats[2]<<" local sampled stdev: "<<localStats[3]<<endl;
   // cout<<"local signal to noise: "<<localStats[4]<<endl;
    
    return 0;
}
