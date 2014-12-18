#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DATASIZE 800  //size of training data and test data
#define ETA 0.01   //learning rate
#define MAX_EPOCHS 100000

typedef struct ENTRY{   //dataset for each sample data
  double x1;
  double x2;
  double desired_value;
} ENTRY;

typedef struct POINT{
  double x;
  double y;
} POINT;

typedef struct NEURON{   
  double output;
  double weight;
  POINT center;
} NEURON;


double gaussian(double distance, double sigma)
{
  return exp(- distance*distance / (2*sigma*sigma));
}

double randomWeight()  //random weight generator between 0 ~ 1
{
  return ((int)rand()%100000)/(float) 100000;
}

double relativeError(double *error, int len)
{
  len = len - 1;
  if(len < 20)
    return 1;
  //keep track of the last 20 Root of Mean Square Errors
  int start1 = len-20;
  int start2 = len-10;

  double error1 = 0;
  double error2 = 0;
  
  int i;
  
  //calculate the average of the first 10 errors
  for(i = start1; i < start1 + 10; i++)
    {
      error1 += error[i];
    }
  double averageError1 = error1 / 10;

  //calculate the average of the second 10 errors
  for(i = start2; i < start2 + 10; i++)
    {
      error2 += error[i];
    }
  double averageError2 = error2 / 10;
  
  double relativeErr = (averageError1 - averageError2)/averageError1;
  return (relativeErr > 0) ? relativeErr : -relativeErr;
}

/*****************create hidden neurons************************************/
void createHiddenNeuronsForFixedCenter(NEURON *neurons, int number, ENTRY *data)
{ 
  int i;
  srand(time(NULL));

  //Bias neuron for output
  neurons[0].output = 1;
  neurons[0].weight= randomWeight();

  for(i = 1; i < number; i++)
    {  
      neurons[i].weight= randomWeight();
      //randomly choose one data set as center position
      int index = (int) (randomWeight() * 100000) % DATASIZE;
      neurons[i].center.x = data[index].x1;
      neurons[i].center.y = data[index].x2;
    }
}

void createHiddenNeuronsForKMeanClusters(NEURON *neurons, int number)
{ 
  int i;
  srand(time(NULL));

  neurons[0].output = 1;
  neurons[0].weight= randomWeight();

  for(i = 1; i < number; i++)
    {  
      neurons[i].weight= randomWeight();
      //randomly initialize the center
      neurons[i].center.x = randomWeight();  
      neurons[i].center.y = randomWeight();
    }
}

double gaussianOutput(POINT point, ENTRY data, double sigma)
{
  double d = sqrt(pow(point.x - data.x1, 2) + pow(point.y - data.x2, 2));
  return gaussian(d, sigma);
}


void printfWeights(NEURON *neurons, int number)
{
  printf("bias weight: %f\n", neurons[0].weight);
  int i;
  for(i = 1; i < number; i++)
    {  
      printf("neuron: %d weight: %f center: %f %f\n ", i, neurons[i].weight, neurons[i].center.x, neurons[i].center.y);
    }
}

double weightsAdjust(NEURON *neurons, int number, double sigma, ENTRY *data)
{
  int i, j;
  double errorSquareSum = 0;
  for(j = 0; j < DATASIZE;  j++)
    {
      for(i = 1; i < number; i++ )    //calculate the gaussian output
	neurons[i].output = gaussianOutput(neurons[i].center, data[j], sigma);

      double y = 0;
      for(i = 0; i < number; i++)    //calculate the induced local field -- the output 
	{
	  y += neurons[i].output * neurons[i].weight;
	}

      double error = data[j].desired_value - y;
      errorSquareSum += error * error / 2;
      
      for(i = 0; i < number; i++)    //adjust the weight
	{
	  neurons[i].weight += ETA * error * neurons[i].output;
	}
    }
 
  return sqrt(errorSquareSum/DATASIZE);
}


double fixedCenter(NEURON *neurons, int number)
{
  int i, j;
  double dmax = 0;
  for(i = 1; i < number; i++)
    {
      for(j = i + 1; j < number; j++)
	{
	  double d = sqrt(pow(neurons[i].center.x - neurons[j].center.x, 2) + pow(neurons[i].center.y - neurons[j].center.y, 2));
	  if(d > dmax)
	    dmax = d;
	}
    }

  double sigma = dmax / sqrt(2*(number-1));
  return sigma;
}

double kMeanClusters(NEURON *neurons, int number, ENTRY *data)
{
  int i, j;
  
  double *error = (double *)malloc(MAX_EPOCHS * sizeof(double));
  int maxlen = 0;
  int epoch = 1;
  do{
    for(i = 0; i < DATASIZE; i++)
      {
	int index;
	double dmin = 0;
	for(j = 1; j < number; j++)  // determine the nearest center
	  {
	    double d = pow(data[i].x1 - neurons[j].center.x , 2) + (data[i].x2 - neurons[j].center.y , 2);
	    if(d < dmin)
	      {
		dmin = d;
		index = j;
	      }
	  }
	//move center
	neurons[j].center.x += ETA * (data[j].x1 - neurons[j].center.x);
	neurons[j].center.y += ETA * (data[j].x2 - neurons[j].center.y);
	error[maxlen] += sqrt(dmin);
      }
    maxlen++;
    epoch++;
  }while(epoch < MAX_EPOCHS && relativeError(error, maxlen) > 1e-5);

  double dmax = 0;
  for(i = 1; i < number; i++)
    {
      for(j = i + 1; j < number; j++)
	{
	  double d = sqrt(pow(neurons[i].center.x - neurons[j].center.x, 2) + pow(neurons[i].center.y - neurons[j].center.y, 2));
	  if(d > dmax)
	    dmax = d;
	}
    }

  double sigma = dmax / sqrt(2*(number-1));
  return sigma;
}


//****************************testing***************************//
double test(NEURON *neurons, int number, double sigma, ENTRY *data)
{
  int i, j;
  double errorSquareSum = 0;
  for(j = 0; j < DATASIZE;  j++)
    {
      for(i = 1; i < number; i++ )    //calculate the gaussian output
	neurons[i].output = gaussianOutput(neurons[i].center, data[j], sigma);

      double y = 0;
      for(i = 0; i < number; i++)    //calculate the induced local field -- the output 
	{
	  y += neurons[i].output * neurons[i].weight;
	}

      double error = data[j].desired_value - y;
      errorSquareSum += error * error / 2;
      printf("%d %f\n", j, y);
    }
  
  return sqrt(errorSquareSum/DATASIZE);
}

//*******************read training data and test data**********************//

void getTrainingAndTestData(int argc, char **path, ENTRY *training, ENTRY *testing)
{
  if(argc != 3)
    {
      printf("Usage: program training_data_file testing_data_file\n");
      exit(0);
    }

  FILE *fp1, *fp2;
  if((fp1 = fopen(path[1], "r")) == NULL)
    {
      printf("cannot open %s\n", path[1]);
      exit(1);
    }
  if((fp2 = fopen(path[2], "r")) == NULL)
    {
      printf("cannot open %s\n", path[2]);
      exit(1);
    }

  int i = 0;
  double num;
  while(i < 800)
   {
     fscanf(fp1, "%lf %lf %lf %lf", &training[i].desired_value, &num, &training[i].x1, &training[i].x2);
     fscanf(fp2, "%lf %lf %lf %lf", &testing[i].desired_value, &num, &testing[i].x1, &testing[i].x2);
     i++;
   }
  fclose(fp1);
  fclose(fp2);
}


//********shuffle the order of presentation to neuron*********//
void swap(ENTRY *data, int i, int j)
{
  ENTRY temp;
  temp.x1 = data[i].x1;
  temp.x2 = data[i].x2;
  temp.desired_value = data[i].desired_value;
  data[i].x1 = data[j].x1;
  data[i].x2 = data[j].x2;
  data[i].desired_value = data[j].desired_value;
  data[j].x1 = temp.x1;
  data[j].x2 = temp.x2;
  data[j].desired_value = temp.desired_value;
}

void shuffle(ENTRY *data, int size)
{
  srand(time(NULL));
  int i;
  for(i = 0; i < size; i++)
    {
      int j = (int)rand()%size;
      swap(data, i, j);
    }
}



//**************************main function***********************//

int main(int argc, char** argv)
{
  int numberOfHiddenNeurons = 5;
  ENTRY *training_data = (ENTRY *)malloc(DATASIZE*sizeof(ENTRY));
  ENTRY *testing_data = (ENTRY *)malloc(DATASIZE*sizeof(ENTRY));
  
  //read training data and testing data from file
  getTrainingAndTestData(argc, argv, training_data, testing_data);  

  int epoch = 1;

  //output data to a file
  FILE *fout;
  if((fout = fopen("1.txt", "w")) == NULL)
    {
      fprintf(stderr, "file open failed.\n");
      exit(1);
    }
  
  //shuffle the order of presenting training data to neural network
  shuffle(training_data, DATASIZE);
  
  //train and test neural network
  double *error = (double *)malloc(MAX_EPOCHS * sizeof(double));
  int maxlen = 0;
  //create neural network for rbf
  NEURON *hiddenNeurons = (NEURON *)malloc((numberOfHiddenNeurons+1) * sizeof(NEURON));
  createHiddenNeuronsForFixedCenter(hiddenNeurons, numberOfHiddenNeurons+1, training_data);
  //createHiddenNeuronsForKMeanClusters(hiddenNeurons, numberOfHiddenNeurons+1);
  double sigma = fixedCenter(hiddenNeurons, numberOfHiddenNeurons+1);
  //double sigma =  kMeanClusters(hiddenNeurons, numberOfHiddenNeurons+1, training_data);
  printf("sigma: %f\n", sigma);
  printfWeights(hiddenNeurons, numberOfHiddenNeurons+1);
  do{
    shuffle(training_data, DATASIZE);
    error[maxlen] = weightsAdjust(hiddenNeurons, numberOfHiddenNeurons+1, sigma, training_data);
    double testError = test(hiddenNeurons, numberOfHiddenNeurons+1, sigma, testing_data);
    //printf("%d %lf %lf\n", epoch, error[maxlen], testError);
    //fprintf(fout, "%d %lf %lf\n", epoch, error[maxlen], testError);
    epoch++;
    maxlen++;
  }while(epoch < MAX_EPOCHS && relativeError(error, maxlen) > 1e-5 );
  
  fclose(fout);
  free(hiddenNeurons);
  free(training_data);
  free(testing_data);
}
