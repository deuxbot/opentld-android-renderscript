package com.deuxbot.opentld.tld;

import java.util.List;

import android.util.Log;

import com.deuxbot.opentld.MainActivity;

public class HierarchicalCluster 
{
	private static final float CUTT_OFF = .5f;	// Distance threshold	
	
	private int numClusters;					// Number of clusters
	
	public HierarchicalCluster()
	{
		numClusters = 0;
	}
	
	public int getClusters()
	{
		return numClusters;
	}
		
	public BoundingBox clusterDetections(List<BoundingBox> confidentDetections) 
	{		
		BoundingBox result = null;
		float[] distances = calcDistances(confidentDetections);
		cluster(confidentDetections, distances);
		
		Log.i(MainActivity.TAG, "Clustering: Number of clusters: " + numClusters);
		
		if(numClusters == 1)	
		{
			result = calcAvgBox(confidentDetections);	// Average the boxes     
		}
		
		return result;
	}
	
	// Calculate the pairwise overlap between all confident boxes
	private float[] calcDistances(List<BoundingBox> confidentDetections) 
	{	
		int size = confidentDetections.size();
		int matTriangle = size * (size - 1) / 2;
		float[] distances = new float[matTriangle];	
		int index = 0;
		
		for(int i = 0; i < size - 1; i++)		
		{
			for(int j = i + 1 ; j < size; j++)	
			{
				float distance = 1 - confidentDetections.get(i).calcOverlap(confidentDetections.get(j));
				distances[index++] = distance;
			}
		}
			
		return distances;	
	}
	
	private void cluster(List<BoundingBox> confidentDetections, float distances[]) 
	{			
		int numDetections = confidentDetections.size();	
		int numDistances = numDetections * (numDetections - 1) / 2;
		int[] distanceUsed = new int[numDistances];			// Cluster distances 
		int[] clusterIndices = new int[numDetections];		// Cluster indices 
		
		int clusterIndex = 1;	// Index of the first cluster
		numClusters = 0;		// Number of initial clusters
		
		if(numDetections == 1)	// There is only one confident box
	    {
			clusterIndices[0] = 1;
	        numClusters++;
	    }
			
		while(true)
		{
			float minDistance = -1;
			int minIndex = -1;
			int distIndex = 0;
			int index1 = -1;
			int index2 = -1;		
			
			// Search for the shortest distance
			for (int i = 0; i < numDetections; i++)
			{
				for (int j = i + 1 ; j < numDetections; j++)
				{
					if ((distanceUsed[distIndex] == 0) && (minIndex == -1 || distances[distIndex] < minDistance))
					{
						minDistance = distances[distIndex];
						minIndex = distIndex;
						index1 = i;
						index2 = j;
					}
					
					distIndex++;
				}
			}
		
	        if(minIndex == -1)
	        {
	            break; // We are done
	        }
			
			distanceUsed[minIndex] = 1;
			
	        // Compare the cluster indexes
	        if(clusterIndices[index1] == 0 && clusterIndices[index2] == 0) 		// Both have no cluster and distance is low
	        {      
	            if(minDistance < CUTT_OFF)	// Distance is short -> put them to the same cluster
	            {
	                clusterIndices[index1] = clusterIndices[index2] = clusterIndex;
	                clusterIndex++;
	                numClusters++;
	            }
	            else    					//Distance is long -> put them to different clusters
	            {
	                clusterIndices[index1] = clusterIndex;
	                clusterIndex++;
	                numClusters++;
	                clusterIndices[index2] = clusterIndex;
	                clusterIndex++;
	                numClusters++;
	            }          
	        }
	        else if(clusterIndices[index1] == 0 && clusterIndices[index2] != 0)	// Second box is in cluster already
	        {
	            if(minDistance < CUTT_OFF)	// Distance is short -> put them to the same cluster
	            {
	                clusterIndices[index1] = clusterIndices[index2];
	            }
	            else     					// Distance is long -> put them to different clusters
	            {
	                clusterIndices[index1] = clusterIndex;
	                clusterIndex++;
	                numClusters++;
	            }
	        }
	        else if(clusterIndices[index1] != 0 && clusterIndices[index2] == 0)	// First box is in cluster already
	        {
	            if(minDistance < CUTT_OFF)	// Distance is short -> put them to the same cluster
	            {
	                clusterIndices[index2] = clusterIndices[index1];
	            }
	            else     					// Distance is long -> put them to different clusters
	            {
	                clusterIndices[index2] = clusterIndex;
	                clusterIndex++;
	                numClusters++;
	            }
	        }
	        else    						// Both indices are in clusters already
	        {
	            if(clusterIndices[index1] != clusterIndices[index2] && minDistance < CUTT_OFF) // Different clusters and distance is short
	            {
	                int oldClusterIndex = clusterIndices[index2];

	                for(int i = 0; i < numDetections; i++)
	                {
	                    if(clusterIndices[i] == oldClusterIndex)
	                    {
	                        clusterIndices[i] = clusterIndices[index1];	//Merge clusters
	                    }
	                }

	                numClusters--;
	            }
	        }
	    }
	}
	
	// Calculate an averaged and compressed box from all confident ones
	private BoundingBox calcAvgBox(List<BoundingBox> confidentDetections) 
	{
		int numIndices = confidentDetections.size();
		BoundingBox avgBox = new BoundingBox();	
		float x = 0;
		float y = 0;
		float width = 0;
		float height = 0;
		float confidence = 0;
		
		for(int i = 0; i < numIndices; i++)
		{
			x += confidentDetections.get(i).x;
		    y += confidentDetections.get(i).y;
		    width += confidentDetections.get(i).width;
		    height += confidentDetections.get(i).height;
		    confidence += confidentDetections.get(i).getConfidence();	  
		}

		x /= numIndices;
		y /= numIndices;
		width /= numIndices;
		height /= numIndices;
		confidence /= numIndices;
		
		avgBox.x = (int) Math.floor(x + 0.5);
		avgBox.y = (int) Math.floor(y + 0.5);
		avgBox.width = (int) Math.floor(width + 0.5);
		avgBox.height = (int) Math.floor(height + 0.5);
		avgBox.setConfidence(confidence);
		
		Log.i(MainActivity.TAG, "Clustering: New box: " + avgBox + " with area " + avgBox.area() + " and confidence " + avgBox.getConfidence());
		
		return avgBox;
	}
}

