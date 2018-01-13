#pragma version(1)
#pragma rs java_package_name(com.deuxbot.opentld)

float *pPatches;
float *nPatches;
float *patches;
int32_t pPatchesNum;
int32_t nPatchesNum;
int32_t numPatches;
float *output;
int32_t size;

void __attribute__((kernel)) root(int32_t index) 
{	
	float maxP = 0; 
	float maxN = 0;
	float ncc = 0;
	
	for(int i = 0; i < pPatchesNum; i++)
	{	
		double corr = 0;
 		double norm1 = 0;
		double norm2 = 0; 
	  
	    for(int k = 0; k < size; k++)
	    {
	        corr += patches[size * index + k] * pPatches[size * i + k];
	        norm1 += patches[size * index + k] * patches[size * index + k];
	        norm2 += pPatches[size * i + k] * pPatches[size * i + k];
	    }
	    
	    ncc = (float) ((corr / sqrt((float)(norm1 * norm2)) + 1) * 0.5);
	    
	    if(ncc > maxP)
		{
			maxP = ncc;
		}	
	}
	
	for(int i = 0; i < nPatchesNum; i++)
	{	  
	  	double corr = 0;
 		double norm1 = 0;
		double norm2 = 0; 
	   
	    for(int k = 0; k < size; k++)
	    {
	        corr += patches[size * index + k] * nPatches[size * i + k];
	        norm1 += patches[size * index + k] * patches[size * index + k];
	        norm2 += nPatches[size * i + k] * nPatches[size * i + k];
	    }
	    
	    ncc = (float) ((corr / sqrt((float)(norm1 * norm2)) + 1) * 0.5);
	    
	    if(ncc > maxN)
		{
			maxN = ncc;
		}
	}
 	
	float dP = 1 - maxP;	
	float dN = 1 - maxN;	
	float distance = dN / (dN + dP);
				
	output[index] = distance;
}
