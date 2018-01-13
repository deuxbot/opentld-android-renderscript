#pragma version(1)
#pragma rs java_package_name(com.deuxbot.opentld)

float *output;
int32_t *coords;
int32_t *indices;
int32_t *sum;
float *sqSum;
int32_t cols;

void __attribute__((kernel)) root(int32_t index) 
{		
	int aux = index;
	
	index = indices[index] * 4;

	int brSum = sum[(coords[index + 1] + coords[index + 3]) * cols + coords[index] + coords[index + 2]];
	int blSum = sum[(coords[index + 1] + coords[index + 3]) * cols + coords[index]];
	int trSum = sum[coords[index + 1] * cols + coords[index] + coords[index + 2]];
	int tlSum = sum[coords[index + 1] * cols + coords[index]];
	double brSqSum = sqSum[(coords[index + 1] + coords[index + 3]) * cols + coords[index] + coords[index + 2]];
	double blSqSum = sqSum[(coords[index + 1] + coords[index + 3]) * cols + coords[index]];
	double trSqSum = sqSum[coords[index + 1] * cols + coords[index] + coords[index + 2]];
	double tlSqSum = sqSum[coords[index + 1] * cols + coords[index]];
		
	double boxArea = coords[index + 2] * coords[index + 3];
	double sumMean = (brSum + tlSum - trSum - blSum) / boxArea; 
	double sqSumMean = (brSqSum + tlSqSum - trSqSum - blSqSum) / boxArea;
		
	output[aux] = (float) sqSumMean - sumMean * sumMean;	
}
