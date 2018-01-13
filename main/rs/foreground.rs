#pragma version(1)
#pragma rs java_package_name(com.deuxbot.opentld)

int4 bb;
int32_t *output;
int32_t *coords;

void __attribute__((kernel)) root(int32_t box) 
{	
	int index = box * 4;
	
	if(bb[0] < coords[index])
	{
		if(bb[1] < coords[index + 1])
		{
			if(bb[0] + bb[2] > coords[index] + coords[index + 2])
			{
				if(bb[1] + bb[3] > coords[index + 1] + coords[index + 3])
				{
					output[box] = box; 
				}
			}
		}
	}
}
