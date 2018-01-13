package com.deuxbot.opentld.tld;

public class Feature 
{
	public int x1, y1, x2, y2;
	
	public Feature(int x1, int y1, int x2, int y2) 
	{
		this.x1 = x1;
		this.y1 = y1;
		this.x2 = x2;
		this.y2 = y2;
	}

	// Compare intensity values between the two points	
	public boolean calcFeature(byte[] currentFrame, BoundingBox box) 
	{
		boolean result = false;
	
		int pos1 = (y1 + box.y) * 320 + (x1 + box.x);
		int pos2 = (y2 + box.y) * 320 + (x2 + box.x);
			
		// Point one is brighter than point two?	
		if(currentFrame[pos1] > currentFrame[pos2])
		{
			result = true; 
		}		
		
		return result;		
	}	
	
	public String toString()
	{
		return x1 + " " + y1 + " " + x2 + " " + y2;
	}
}