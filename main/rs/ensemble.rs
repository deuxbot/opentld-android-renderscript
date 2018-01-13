#pragma version(1)
#pragma rs java_package_name(com.deuxbot.opentld)

int32_t *features;
int32_t *output;
int32_t *x;
int32_t *y;
int8_t *frame;
int32_t numFeatures;
int32_t *scaleIndices;
int32_t numFerns;
int32_t numScales;
int32_t numBoxes;
int32_t frameCols;

void __attribute__((kernel)) root(int index) 
{			
	for(int i = 0; i < numFerns; i++)
	{	
		int fern = 0; 
	
		for(int j = 0; j < numFeatures; j++)
		{
			fern <<= 1;

			int myIndex = (i * numScales * numFeatures * 4) + (scaleIndices[index] * numFeatures * 4) + (j * 4);
			
			int pos1 = (features[myIndex + 1] + y[index]) * 320 + features[myIndex] + x[index];	
			int pos2 = (features[myIndex + 3] + y[index]) * 320 + features[myIndex  + 2] + x[index];
				
			if(frame[pos1] > frame[pos2])
			{
				fern |= 1;	
			}	
			
					
		}		
		
		output[index * numFerns + i] = fern;		
	}
}
