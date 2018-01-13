#pragma version(1)
#pragma rs java_package_name(com.deuxbot.opentld)

float P_CONSTRAINT;
float N_CONSTRAINT;
float THRESHOLD;
int32_t *output;
int32_t *coords;
float *posterior;
int4 bb;

void __attribute__((kernel)) root(int32_t box) 
{	
	int index = box * 4;
	float overlap = 0;
	
	if(coords[index] < bb[0] + bb[2])
	{
		if(coords[index + 1] < bb[1] + bb[3])
		{
			if(coords[index] + coords[index + 2]> bb[0])
			{	
				if(coords[index + 1] + coords[index + 3] > bb[1])
				{
					int32_t colIntersection =  min(coords[index] + coords[index + 2], bb[0] + bb[2]) - max(coords[index], bb[0]);
					int32_t rowIntersection =  min(coords[index + 1] + coords[index + 3], bb[1] + bb[3]) - max(coords[index + 1], bb[1]);	    
	    
					int32_t intersection = colIntersection * rowIntersection;
					int32_t area1 = coords[index + 2] * coords[index + 3];
					int32_t area2 = bb[2] * bb[3];
	    
					overlap = intersection / (float)(area1 + area2 - intersection);
				}
			}
	
		}
	}
	
	if(overlap > P_CONSTRAINT && posterior[index] < THRESHOLD)
	{
		output[box] = box;
	}
	else if(overlap < N_CONSTRAINT && posterior[index]  > THRESHOLD)
	{
		output[box] = box;
	}
	else
	{
		output[box] = -2;
	}
}
