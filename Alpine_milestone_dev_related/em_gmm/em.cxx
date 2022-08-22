#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

double TINY = 1e-6;
double ZEPS = 1e-10;
double eps =  1e-6;

double * alloc_vector(int cols)
{

    return (double *) malloc(sizeof(double) * cols);
}

double ** alloc_matrix(int rows, int cols)
{
    double ** matrix = (double **) malloc(sizeof(double *) * rows);
    for (int i  =  0; i  < rows; i++)
        matrix[i] = alloc_vector(cols);

    return matrix;
}

int check_convergence(double fmax, double fmin, double ftol)
{
	double delta = fabs(fmax - fmin);
	double accuracy = (fabs(fmax) + fabs(fmin)) * ftol;
	return (delta < (accuracy + ZEPS));
}

double eval_gaussian_density(double mean, double sigma, double val)
{
	double prob=0.0;
    double exponent=0.0;

    if(sigma>0)
    {
        exponent = (val-mean)/sigma;
        exponent = -(exponent*exponent)/2.0;
        prob =  ( 1.0 / (sigma*sqrt( 2.0*M_PI))) * exp(exponent);
    }
    else
    {
    	if(fabs(val-mean) < ZEPS)
    		prob=1.0;
    	else
    		prob=0.0;
    }

    return prob;
}

void update_parameters(int n, double * data, int k, double * prob, double * mean, double * sd, double ** class_prob)
{
	int i, j;
    
	//Update weights first
    for (int j = 0; j < k; j++)
    {
        prob[j] = 0.0;
        for (int i = 0; i < n; i++)
            prob[j] += class_prob[i][j];

        prob[j] /= n;
    }

    //update mean
    for (int j = 0; j < k; j++)
    {
        mean[j] = 0.0;
        for (int i = 0; i < n; i++)
            mean[j] += data[i] * class_prob[i][j];

        mean[j] /= n * prob[j] + TINY;
    }

    //update standard deviation
    for (int j = 0; j < k; j++)
    {
        sd[j] = 0.0;
        for (int i = 0; i < n; i++)
            sd[j] += (data[i] - mean[j])*(data[i] - mean[j]) * class_prob[i][j];
        sd[j] /= (n * prob[j] + TINY);
        sd[j] = sqrt(sd[j]);
    }
}

double classprob(int j, double x, int k, double *prob, double *mean, double *sd)
{
	double retprob=0;

	double num = prob[j]*eval_gaussian_density(mean[j],sd[j],x);

	double denom=0;
	for(int i=0;i<k;i++)
		denom += prob[i]*eval_gaussian_density(mean[i],sd[i],x); 

	return num/denom;
}

void update_class_prob(int n, double * data, int k, double * prob, double * mean, double * sd, double ** class_prob)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < k; j++)
            class_prob[i][j] = classprob(j, data[i],k, prob, mean, sd);
}

double computeLogLikelihood(int n, double* data,int k, double* prob,double* mean,double* sd)
{
	double llk=0;

	for(int p=0;p<n;p++)
	{
		double val=0;
		for(int q=0;q<k;q++)
		{
			val += prob[q]*eval_gaussian_density(mean[q],sd[q],data[p]);
		}

		llk += log(val);
	}

	return llk/n;
}

double compute_em(int n, double * data, int k, double * prob, double * mean, double * sd, double eps)
{
	//n number of samples
	//k number og Gaussians

	if(n<k)
	{
		cout<<"Number of samples must be larger than number of clusters..."<<endl;
		return 0;
	}

    double llk = 0, prev_llk = 0;
    double **class_prob = alloc_matrix(n, k);
   
    //initial estimate of parameters
    double mean1 = 0.0, sd1 = 0.0;
    
    for (int i = 0; i < n; i++)
        mean1 += data[i];
    mean1 /= n;

    for (int i = 0; i < n; i++)
        sd1 += (data[i] - mean1)*(data[i] - mean1);
    sd1 = sqrt(sd1 / n);

    //assign mean randomely, stdev = sample stdev, equal weight for all gaussians
    for (int j = 0; j < k; j++)
    {
        prob[j] = 1.0 / k;
        mean[j] = data[rand() % n];
        sd[j] = sd1;
    }

    //Do while loop for iterative estimation
    do{
    	//save prev likelihood
	    prev_llk = llk;

        //use Bayes theorem to get posterior prob
        update_class_prob(n, data, k, prob, mean, sd, class_prob);

        //update the parameters with newly estimated probabilities
	    update_parameters(n, data, k, prob, mean, sd, class_prob);
	   
	   	//compute new likelihood
        llk = computeLogLikelihood(n, data, k, prob, mean, sd);

    } while (!check_convergence(llk, prev_llk, eps));

    return llk;
}


int main(int argc,char* argv[])
{
    /* initialize random seed: */
    srand (time(NULL));

    int numGaussians = 2;
    int numSamples = 1000;

    double *mean,*sd,*samples,*weight;

    mean = alloc_vector(numGaussians);
    sd = alloc_vector(numGaussians);
    weight = alloc_vector(numGaussians);
    samples = alloc_vector(numSamples);

    //randomly generate samples for testing
    for(int i=0;i<numSamples;i++)
       	samples[i] = rand()%10+1;
    
    double retval = compute_em(numSamples,samples,numGaussians,weight,mean,sd,eps);

    for(int i=0;i<numGaussians;i++)
    {
    	cout<<"gaussian: "<<i<<" mean: "<<mean[i]<<" sd: "<<sd[i]<<" weight: "<<weight[i]<<endl;
    }    

    return 0;
}