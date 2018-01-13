package com.deuxbot.opentld.tld;

import java.util.Random;

import org.opencv.core.Size;

public class Fern 
{
	public Feature features[][]; 			// Features[scale][feature]
	private float posteriors[];				// Posterior probabilities
	private int positives[];				// Positive patches
	private int negatives[];  				// Negative patches
	private int numFeatures;				// Number of features
	private int numIndexes;					// Number of Ferns * Features
	private Size scales[];					// Size of scaled boxes

	private int[] featuresRS;
	
	public Fern(int numFeatures, Size[] scales) 
	{
		this.numFeatures = numFeatures;
		this.scales = scales;
	}
	
	public int getNumScales()
	{
		return scales.length;
	}
	
	public int[] getFeaturesRS()
	{		
		return featuresRS;
	}
	
	public float[] getPosteriors()
	{		
		return posteriors;
	}
	
	public void init()
	{
		numIndexes = (int)Math.pow(2, numFeatures);

		initFeatures();
		initPosteriors();
		generateScriptData();
	}
		
	public void initPosteriors()
	{	
		numIndexes = (int)Math.pow(2, numFeatures);
		
		posteriors = new float[numIndexes];
		positives = new int[numIndexes];
		negatives = new int[numIndexes];
	}
	
	// Generate random points for feature locations
	public void initFeatures() 
	{	
		Random rand = new Random();
		features = new Feature[scales.length][numFeatures];

		for (int i = 0; i < numFeatures; i++)
		{
			float x1 = rand.nextFloat();
			float y1 = rand.nextFloat();
			float x2 = rand.nextFloat();
			float y2 = rand.nextFloat();
			
			for (int j = 0; j < scales.length; j++)
			{
				int _x1 = (int) (x1 * scales[j].width);
				int _y1 = (int) (y1 * scales[j].height);
				int _x2 = (int) (x2 * scales[j].width);
				int _y2 = (int) (y2 * scales[j].height);
				
				features[j][i] = new Feature(_x1, _y1, _x2, _y2); 
			}
		}	
	}
		
	// Update the number of P/N patches and posteriors
	public void updatePosterior(int fernIndex, boolean positive)
	{
		if(positive)
		{
			positives[fernIndex]++;
		}
		else
		{
			negatives[fernIndex]++;
		}
		
		posteriors[fernIndex] = (positives[fernIndex]) / (positives[fernIndex] + negatives[fernIndex]);
	}
	
	// Calculate the binary value of Fern
	public int calcFernFeature(byte[] currentFrame, BoundingBox box) 
	{
		int fern = 0;
		
		for(Feature feature : features[box.getScaleIndex()])
		{ 
			fern <<= 1;
			
			if(feature.calcFeature(currentFrame, box)) 
			{
				fern |= 1;
			}
		}
	
		return fern;
	}
	
	// Generate feature vector in form [F1(b1), F2(b1), F3(b1), F4(b1), F1(b2), F2(b2)...]
	private void generateScriptData()
	{
		featuresRS = new int[features.length * features[0].length * 4];
				
		int index = 0;
	
		for(int i = 0; i < features.length; i++)
		{
			for(int j = 0; j < features[0].length; j++)
			{			
				featuresRS[index++] = features[i][j].x1;
				featuresRS[index++] = features[i][j].y1;
				featuresRS[index++] = features[i][j].x2;
				featuresRS[index++] = features[i][j].y2;
			}
		}			
	}
}
