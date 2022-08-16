#include <slic.H>

using namespace std;

int min_val(int x,int y)
{
    if(x>=y)
        return y;
    else
        return x;
}

Slic :: Slic()
{
    this->xdim=0;
    this->ydim=0;
    this->zdim=0;
    this->blockXSize=0;
    this->blockYSize=0;
    this->blockZSize=0;
    this->clusterNum=0;
    this->halt_condition=0;
    this->QUANTIZATION=1;
}

void Slic :: init(int xdim,int ydim, int zdim, int blockx, int blocky, int blockz,float halt,int QUANT)
{
    this->xdim=xdim;
    this->ydim=ydim;
    this->zdim=zdim;
    this->blockXSize=blockx;
    this->blockYSize=blocky;
    this->blockZSize=blockz;
    this->clusterNum = (xdim*ydim*zdim)/(blockx*blocky*blockz);
    this->halt_condition=halt;
    this->QUANTIZATION=QUANT;
}

float*** Slic :: allocate_3d_float_array()
{
    float*** array;

    array = (float ***)malloc(xdim*sizeof(float **));
    for(int i=0;i<xdim;i++)
    {
        array[i] = (float **)malloc(ydim*sizeof(float *));
        for(int j=0;j<ydim;j++)
            array[i][j] = (float *)malloc(zdim*sizeof(float));
    }

    return array;
}

int*** Slic :: allocate_3d_int_array()
{
    int*** array;

    array = (int ***)malloc(xdim*sizeof(int **));
    for(int i=0;i<xdim;i++)
    {
        array[i] = (int **)malloc(ydim*sizeof(int *));
        for(int j=0;j<ydim;j++)
            array[i][j] = (int *)malloc(zdim*sizeof(int));
    }

    return array;
}

int Slic ::  getClosestClusterID(vector<float> list, float *distval)
{
    int id=0;
    float val=list[0];
    for(int i=1;i<list.size();i++)
    {
        if(list[i]>val)
        {
            val = list[i];
            id=i;
        }
    }

    *distval = val;
    return id;
}

int*** Slic ::  getClusterIds()
{
    return this->cluster_ids;
}

//Scale back to actual values before returing
vector<Element> Slic :: getClusterCenters()
{
    vector<Element> clusCenters;

    for(int i=0;i<this->clusterCenters.size();i++)
    {
        Element e;
        e.x = this->clusterCenters[i].x;
        e.y = this->clusterCenters[i].y;
        e.z = this->clusterCenters[i].z;
        //e.var1 = this->clusterCenters[i].var1/QUANTIZATION;
        e.var1 = this->clusterCenters[i].var1;
        e.numElem = this->clusterCenters[i].numElem;
        clusCenters.push_back(e);
    }

    return clusCenters;
}

int Slic :: getClusterNum()
{
    return this->clusterNum;
}

void Slic :: computeSlic(float*** raw_data1)
{
    int iter=0;
    float*** distances;
    int*** processedElem;
    float epsilon = 9e+8;
    int iterLimit =150;
    float weight=0.1;

    distances = this->allocate_3d_float_array();
    processedElem = this->allocate_3d_int_array();
    this->cluster_ids = this->allocate_3d_int_array();

    for(int k=0;k<zdim;k++)
        for(int j=0;j<ydim;j++)
            for(int i=0;i<xdim;i++)
            {
                processedElem[i][j][k]=0;
                distances[i][j][k]=9e+10;
                this->cluster_ids[i][j][k]=-1;
            }

    //Initialize cluster centers uniformely at the beginning
    int count=0;
    for(int k=0;k<zdim;k=k+blockZSize)
        for(int j=0;j<ydim;j=j+blockYSize)
            for(int i=0;i<xdim;i=i+blockXSize)
            {
                Element e;
                e.x = min_val(i+blockXSize/2.0,xdim-1); //added
                e.y = min_val(j+blockYSize/2.0,ydim-1); //added
                e.z = min_val(k+blockZSize/2.0,zdim-1); //added

                //e.var1 = raw_data1[i][j][k]*QUANTIZATION;
                e.var1 = raw_data1[i][j][k];
                this->clusterCenters.push_back(e);

                cout<< count<<" "<<e.x
                            <<" "<<e.y 
                            <<" "<<e.z
                            <<" "<<e.var1<<endl;


                 count++;
            }

    this->clusterNum = this->clusterCenters.size(); //added
    cout<<"actual number of clusters: "<<this->clusterNum<<endl;

    while((epsilon > halt_condition) && (iter < iterLimit))
    {
        for(int k=0;k<zdim;k++)
            for(int j=0;j<ydim;j++)
                for(int i=0;i<xdim;i++)
                {
                    processedElem[i][j][k]=0;
                }  

        int count1=0;
        #pragma omp parallel for //works fine
        // Iterate for every cluster center
        for(int i=0;i<this->clusterCenters.size();i++)
        {
            count1=0;
            int xleft = (int)(this->clusterCenters[i].x - blockXSize);
            int xright = (int)(this->clusterCenters[i].x + blockXSize);

            int yleft = (int)(this->clusterCenters[i].y - blockYSize);
            int yright = (int)(this->clusterCenters[i].y + blockYSize);

            int zleft = (int)(this->clusterCenters[i].z - blockZSize);
            int zright = (int)(this->clusterCenters[i].z + blockZSize);

            //Loop locally on the points of each cluster center
            for(int ii=xleft;ii<=xright;ii++)
                for(int jj=yleft;jj<=yright;jj++)
                    for(int kk=zleft;kk<=zright;kk++)
                    {
                        //Make sure points are inside the bounds
                        if(ii>=0 && ii<xdim && jj>=0 & jj<ydim && kk>=0 && kk<zdim)
                        {
                            Element currentPt;
                            currentPt.x=ii;
                            currentPt.y=jj;
                            currentPt.z=kk;
                            //currentPt.var1 = raw_data1[ii][jj][kk]*QUANTIZATION;
                            currentPt.var1 = raw_data1[ii][jj][kk];

                            float dist = this->clusterCenters[i].getDistanceEuclid(this->clusterCenters[i],currentPt,weight,blockXSize);

                            if(distances[ii][jj][kk]>dist) //original constraint
                            {
                                distances[ii][jj][kk] = dist;
                                this->cluster_ids[ii][jj][kk] = i;
                            }

                            processedElem[ii][jj][kk] = 1;
                            count1++;
                        }
                    }
        }

        vector<Element> tempClusterCenters;
        tempClusterCenters.resize(clusterNum);

        // int count=0;
        // for(int k=0;k<zdim;k++)
        //     for(int j=0;j<ydim;j++)
        //         for(int i=0;i<xdim;i++)
        //         {
        //             if(processedElem[i][j][k]==0)
        //             {
        //                 count++;
        //             }
        //         }        

        //Assign unprocessed points to the nearest cluster
        for(int k=0;k<zdim;k++)
            for(int j=0;j<ydim;j++)
                for(int i=0;i<xdim;i++)
                {
                    if(processedElem[i][j][k]==0)
                    {
                        vector<float> distVector;
                        distVector.resize(this->clusterCenters.size());

                        for(int p=0;p<this->clusterCenters.size();p++)
                        {
                            Element e;
                            e.x = i;
                            e.y = j;
                            e.z = k;
                            //e.var1 = raw_data1[i][j][k]*QUANTIZATION;
                            e.var1 = raw_data1[i][j][k];
                            distVector[p] = e.getDistanceEuclid(this->clusterCenters[p],e,weight,blockXSize);
                        }

                        float finaldist=0;
                        this->cluster_ids[i][j][k] = getClosestClusterID(distVector,&finaldist);
                        distances[i][j][k] = finaldist;
                        processedElem[i][j][k] = 1;
                        distVector.clear();
                    }
                }

        int count=0;
        for(int k=0;k<zdim;k++)
            for(int j=0;j<ydim;j++)
                for(int i=0;i<xdim;i++)
                {
                    if(processedElem[i][j][k]==0)
                    {
                        count++;
                    }
                }        

        // Compute new cluster centers
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(int k=0;k<zdim;k++)
            for(int j=0;j<ydim;j++)
                for(int i=0;i<xdim;i++)
                {
                    tempClusterCenters[this->cluster_ids[i][j][k]].x += i;
                    tempClusterCenters[this->cluster_ids[i][j][k]].y += j;
                    tempClusterCenters[this->cluster_ids[i][j][k]].z += k;
                    tempClusterCenters[this->cluster_ids[i][j][k]].var1 += raw_data1[i][j][k]; //raw_data1[i][j][k]*QUANTIZATION;
                    tempClusterCenters[this->cluster_ids[i][j][k]].numElem++;
                }

        #pragma omp parallel for
        for(int i=0;i<tempClusterCenters.size();i++)
        {
            if(tempClusterCenters[i].numElem>0)
            {
                tempClusterCenters[i].x = tempClusterCenters[i].x/tempClusterCenters[i].numElem;
                tempClusterCenters[i].y = tempClusterCenters[i].y/tempClusterCenters[i].numElem;
                tempClusterCenters[i].z = tempClusterCenters[i].z/tempClusterCenters[i].numElem;
                tempClusterCenters[i].var1 = tempClusterCenters[i].var1/tempClusterCenters[i].numElem;
            }
        }

        // Compute the epsilon for the current iteration
        epsilon=0;
        for(int i=0;i<this->clusterCenters.size();i++)
        {
            epsilon += this->clusterCenters[i].getDistanceEuclid(this->clusterCenters[i],tempClusterCenters[i],weight,blockXSize);
        }

        //Assign new cluster centers for next round
        this->clusterCenters = tempClusterCenters;

        tempClusterCenters.clear();

        //if (count>0)
        cout<<"Iteration: "<<iter<<" epsilon is: "<<epsilon<<" count: "<<count<<" count1: "<<count1<<endl;

        iter++;
    }

}


