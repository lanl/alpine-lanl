////////////////////////////////////////////////////////////////////////////////////
//// In situ Global histogram based sampling with prorities to low frequency values
////////////////////////////////////////////////////////////////////////////////////
float* computeHistSampling(float* data, int xdim, int ydim, int zdim, int cx, int cy, int cz, int *arrLength)
{  	
    //initialize random seed
    srand (time(NULL));

    // Get the rank and size of the process
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Status Stat;
    
    //Find the local max and mins
    int numPts = xdim*ydim*zdim;
    float localMax = data[0];
    float localMin = data[0];
    for(int i=1;i<numPts;i++)
    {
        if(localMax < data[i])
            localMax = data[i];

        if(localMin > data[i])
            localMin = data[i];
    }
        
    // send the local min to root
    MPI_Send(&localMin, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);    
    //compute glomal min
    if(rank==0)
    {
      float minVals[size];
      minVals[0] = localMin;

      for(int i=1;i<size;i++)
      {
          float val=0;
          MPI_Recv(&val, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &Stat);
          minVals[i] = val;
      }
 
      globMin = minVals[0];
      for(int i=1;i<size;i++)
      {
          if(globMin > minVals[i])
              globMin = minVals[i];   
      }             
    }

    // send the local max to root
    MPI_Send(&localMax, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    //compute glomal max
    if(rank==0)
    {
      float maxVals[size];
      maxVals[0] = localMax;

      for(int i=1;i<size;i++)
      {
          float val=0;
          MPI_Recv(&val, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &Stat);
          maxVals[i] = val;
      }
 
      globMax = maxVals[0];
      for(int i=1;i<size;i++)
      {
          if(globMax > maxVals[i])
              globMax = maxVals[i];   
      }             
    }     
    
    // broadcast global max min to all	
    MPI_Bcast(&globMin, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&globMax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);     
    //cout<<"rank,min,max,globMin is: "<<rank<<" "<<localMin<<" "<<localMax<<" "<<globMin<<" "<<globMax<<endl; 
    
    //initialize histograms with zeros
    for(int i=0;i<BIN;i++)
	{
     	localHist[i]=0;
	    globHist[i]=0;
	}
	
    //compute the local histogram
    for(int i=0;i<numPts;i++)
    {
      int val = (int)((data[i] - globMin)/(float)(globMax-globMin)*(BIN-1));
      localHist[val]++;
    }
  
   // Reduce the localhist into a globalhist at rank 0
   MPI_Reduce(localHist,globHist,BIN,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);  

   //RatioHist will be the importance function. So allocate memory for it first
   ratioHist = (float *)malloc(BIN*sizeof(float));
	
   if(rank==0)
   {
     long int numOfPoints=0;
     
    //now compute the importance function for sampling      
     vector < sortHistogram > Hist;
     for(int i=0;i<BIN;i++)
	{
	    Hist.push_back(sortHistogram(globHist[i], i));
	    numOfPoints = numOfPoints+globHist[i];
	}

    //sort the global histogram first
    std::sort(Hist.begin(), Hist.end());

    pointsToretain = (numOfPoints*percentageToKeep)/100.0;
    pointsPerBin = pointsToretain/BIN;

    //Compute the sampling/importance function    
    int currentBinFreq=0;
    int binCounter=0;
    int BinsLeft=BIN;
    do
    {
        if(Hist[binCounter].freq <= pointsPerBin)
        {
            ratioHist[Hist[binCounter].binId] = 1.0;
            BinsLeft--;
            pointsToretain = pointsToretain - Hist[binCounter].freq;
            binCounter++;
            pointsPerBin = (int) pointsToretain/BinsLeft;
            currentBinFreq = Hist[binCounter].freq;
        }
    }
    while (currentBinFreq <= pointsPerBin);

     

    Hist.clear();	    
   }

   // Broadcast the importance/sampling function to all
   MPI_Bcast(ratioHist, BIN, MPI_FLOAT, 0, MPI_COMM_WORLD); 

   /////////////////////////////////////////////////////////////////////////////////	
   // At this point all the processors have the global importance function with them
	   
   int pointsToBeSampled = ceil(((xdim*ydim*zdim)*percentageToKeep)/100.00);
   float* SampledData;
   SampledData = (float *)malloc((4*pointsToBeSampled+1)*sizeof(float)); //4 becasue (x,y,z,v)
           	
   // Now sample points based on the global importance function
   int iter=0;
   int idx=1;
   int counter=0;
   for(int k=0;k<xdim;k++)
    for(int j=0;j<ydim;j++)
      for(int i=0;i<zdim;i++)
	    {	
            if(counter < pointsToBeSampled)
            {
  		        int binid = (int)(((data[iter]-globMin)/(globMax-globMin))*(BIN-1));
  		        double rand_val = ((double) rand() / (RAND_MAX));
  		        if(rand_val <= ratioHist[binid])
  		        {
  		             SampledData[idx] = cz+i;
  		             SampledData[idx+1] = cy+j;
  		             SampledData[idx+2] = cx+k;
  		             SampledData[idx+3] = data[iter];
  		             idx = idx+4;
  		             counter++;
  		        }                                 
            }
           else
               goto end_loop; // exit from the nested loop and go to label end_loop

           iter++;                       
	    }

   end_loop:
 
   SampledData[0] = counter;	

   *arrLength = 4*counter+1;
   return SampledData;
}

////////////////////////////////
//// In situ Regular sampling 
////////////////////////////////
float* computeRegularSampling(float* data, int xdim, int ydim, int zdim, int cx, int cy, int cz, int *arrLength)
{ 

    int pointSkippingWindow = 100/percentageToKeep;    
    int pointsToBeSampled = ceil(((xdim*ydim*zdim)*percentageToKeep)/100.00);
    float* SampledData;
    SampledData = (float *)malloc((4*pointsToBeSampled+1)*sizeof(float)); //4 becasue (x,y,z,v)
           
    SampledData[0] = pointsToBeSampled;

    // Extract samples regularly  
    int iter=0;
    int idx=1;
    for(int k=0;k<xdim;k++)
     for(int j=0;j<ydim;j++)
      for(int i=0;i<zdim;i++)
      { 
                if(iter % pointSkippingWindow == 0)
                {                    
                     SampledData[idx] = cz+i;
                     SampledData[idx+1] = cy+j;
                     SampledData[idx+2] = cx+k;
                     SampledData[idx+3] = data[iter];
                     idx = idx+4;
                }
                                 
    iter++;
      }

   *arrLength = 4*pointsToBeSampled+1;
   return SampledData;
}

////////////////////////////////
//// In situ Random sampling 
////////////////////////////////
float* computeRandomSampling(float* data, int xdim, int ydim, int zdim, int cx, int cy, int cz, int *arrLength)
{ 
    //initialize random seed
    srand (time(NULL));
  
    int numPts = xdim*ydim*zdim;
    int pointsToBeSampled = ceil(((xdim*ydim*zdim)*percentageToKeep)/100.00);
           
    //Generate random indexes to be sampled from
    int *randIndexes;   
    randIndexes = (int *)malloc(sizeof(int)*pointsToBeSampled);
    for(int i=0;i<pointsToBeSampled;i++)
    randIndexes[i] = rand() % numPts;  
    
    float* SampledData;
    SampledData = (float *)malloc((4*pointsToBeSampled+1)*sizeof(float)); //4 becasue (x,y,z,v)
   
    // Extract samples randomly
    SampledData[0] = pointsToBeSampled;
    int idarray=1;
    for(int i=0;i<pointsToBeSampled;i++)     
    { 
       //Assuming iteration order : Z --> Y --> X
       int idx = randIndexes[i];
       int zIndex = idx % zdim;
       int yIndex = (idx / zdim) % ydim;
       int xIndex = idx / (ydim * zdim);

       SampledData[idarray] = cz+xIndex;
       SampledData[idarray+1] = cy+yIndex;
       SampledData[idarray+2] = cx+zIndex;
       SampledData[idarray+3] = data[idx];
       idarray = idarray+4;
    }

   *arrLength = 4*pointsToBeSampled+1;
   return SampledData;
}

////////////////////////////////////////////////////////
//// In situ Stratified Random sampling 
////////////////////////////////////////////////////////
int compute_3d_to_1d_map(int x,int y,int z, int dimx, int dimy, int dimz)
{

    return x + dimx*(y+dimy*z);
}

float* computeStratifiedRandomSampling(float* data, int xdim, int ydim, int zdim, int cx, int cy, int cz, int *arrLength)
{ 
    //initialize random seed
    srand (time(NULL));
   
    int blockx = blockSize;
    int blocky = blockSize;
    int blockz = blockSize;
  
    int numPts = xdim*ydim*zdim;
    int pointsToBeSampled = ceil(((xdim*ydim*zdim)*percentageToKeep)/100.00);
           
    int numStrata = numPts/(blockx*blocky*blockz);   
    int PtsPerStrata = ceil(pointsToBeSampled/(float)numStrata);

   float* SampledData;
   SampledData = (float *)malloc((4*(PtsPerStrata*numStrata)+1)*sizeof(float)); //4 becasue (x,y,z,v)

   int idarray=1;
   for(int i=0;i<xdim;i=i+blockx)
      for(int j=0;j<ydim;j=j+blocky)
         for(int k=0;k<zdim;k=k+blockz)
    {  
        for(int ss=0;ss<PtsPerStrata;ss++)
            {
                int zIndex = rand() % blockz + k;
                int yIndex = rand() % blocky + j;
                int xIndex = rand() % blockx + i;

                int idx = compute_3d_to_1d_map(xIndex,yIndex,zIndex,xdim,ydim,zdim);

                SampledData[idarray] = cz+xIndex;
          SampledData[idarray+1] = cy+yIndex;
          SampledData[idarray+2] = cx+zIndex;
          SampledData[idarray+3] = data[idx];
          idarray = idarray+4;
            }
    
    }
  
   SampledData[0] = PtsPerStrata*numStrata;
   *arrLength = 4*(PtsPerStrata*numStrata)+1;
   return SampledData;
}

