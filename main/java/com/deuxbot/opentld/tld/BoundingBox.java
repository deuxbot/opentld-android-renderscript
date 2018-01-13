package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import com.deuxbot.opentld.MainActivity;

import android.util.Log;

public class BoundingBox extends Rect 
{
	private float confidence = -1;	// NNClassifier confidence value
	private float posterior = -1;	// Ensemble Classifier posterior
	private float overlap = -1;		// Overlap to initial bounding box
	private int[] features;			// Feature vector assigned by detector
	private	int scaleIndex;			// Position in the box-scales array
	private Mat normPatch;			// Normalized patch with current frame
	
	public BoundingBox()
	{
		super();
	}
			
	public BoundingBox (int x, int y, int width, int height) 
	{
		super(x, y, width, height);
	}
	
	// Calculate the overlap between another box
	public float calcOverlap(Rect box)
	{
		if(x > box.x + box.width)
		{
			return 0f;	
		}

		if(y > box.y + box.height)
	    {
			return 0f;	
	    }

	    if(x + width < box.x)
	    {
	    	return 0f;	
	    }

	    if(y + height < box.y)
	    {
	    	return 0f;	
	    }
		
		int colIntersection =  Math.min(x + width, box.x + box.width) - Math.max(x, box.x);
	    int rowIntersection =  Math.min(y + height, box.y + box.height) - Math.max(y, box.y);	    
	    
	    int intersection = colIntersection * rowIntersection;
	    int area1 = width * height;
	    int area2 = box.width * box.height;
	    
	    return intersection / (float)(area1 + area2 - intersection);
	}

	// Generate an equally spaced set of points
	public Point[] createGrid(int maxPoints, int margin)
	{
		int stepX = (int) Math.ceil((width - 2 * margin) / maxPoints);
		int stepY = (int) Math.ceil((height - 2 * margin) / maxPoints);	
		List <Point> points = new ArrayList <Point>();
		
		for(int i = y + margin; i < y + height - margin; i += stepY)
		{
			for(int j = x + margin; j < x + width - margin; j += stepX)
			{
				points.add(new Point(j, i));
			}
		}
		
		Log.i(MainActivity.TAG, "Tracker: Points in box: " + points.size() + " with X step = " + stepX + " and Y step = " + stepY);			

		return points.toArray(new Point[points.size()]);
	}

	// Calculate the new (moved and resized) box
	public BoundingBox predictBox(Point[] prevPoints, Point[] nextPoints) 
	{		
		int nPoints = prevPoints.length;
		float xOffset[] = new float[nPoints];
		float yOffset[] = new float[nPoints];
		
		// Distance changes of all points
		for(int i = 0; i < nPoints; i++)
		{
			xOffset[i] = (float) (nextPoints[i].x - prevPoints[i].x);
			yOffset[i] = (float) (nextPoints[i].y - prevPoints[i].y);
		}
		
		// Median of the relative values
		float dX = Util.median(xOffset);
		float dY = Util.median(yOffset);
			
		// Calculate distances of the points
		float[] distance = new float[nPoints * (nPoints - 1) / 2];
		int index = 0;
		
		for(int i = 0; i < nPoints; i++)
		{
			for(int j = i + 1; j < nPoints; j++)
			{
				distance[index++] = Util.euclideanDistance(nextPoints[i], nextPoints[j]) / Util.euclideanDistance(prevPoints[i], prevPoints[j]); // d = d[1] / d[0]
			}
		}
		
		// Scale change is the median of all changes	
		float shift = Util.median(distance);
		
		// Calculate the new box scale changes 
		float s0 = 0.5f * (shift - 1) * width;
		float s1 = 0.5f * (shift - 1) * height;
		
		// Create the new transformed box
		BoundingBox box = new BoundingBox();
		box.x = Math.max(Math.round(x - s0 + dX), 0);
		box.y = Math.max(Math.round(y - s1 + dY), 0);
		box.width = Math.round(width * shift);
		box.height = Math.round(height * shift);
		
		return box;	
	}
	
	// Calculate the intersection between box and frame
	BoundingBox calcIntersect(Mat frame)
	{
		BoundingBox intersection = new BoundingBox();
		
		intersection.x = Math.max(x, 0);
		intersection.y = Math.max(y, 0);
		intersection.width = (int) Math.min(Math.min(frame.cols() - x, width), Math.min(width, br().x));
		intersection.height = (int) Math.min(Math.min(frame.rows() - y, height), Math.min(height, br().y));
		
		return intersection;
	}
	
	public void setConfidence(float confidence) 
	{
		this.confidence = confidence;
	}
	
	public void setPosterior(float posterior) 
	{
		this.posterior = confidence;
	}
	
	public void setScaleIndex(int index) 
	{
		this.scaleIndex = index;
	}
	
	public void setOverlap(float overlap) 
	{
		this.overlap = overlap;	
	}
	
	public void setFeatures(int[] features) 
	{
		this.features = features;	
	}
	
	public void setNormPatch(Mat normPatch) 
	{
		this.normPatch = normPatch;	
	}
	
	public float getConfidence() 
	{
		return confidence;
	}
	
	public float getPosterior() 
	{
		return posterior;
	}
	
	public int getScaleIndex()
	{
		return scaleIndex;
	}
	
	public float getOverlap()
	{
		return overlap;
	}
	
	public int[] getFeatures() 
	{
		return features;	
	}
	
	public Mat getNormPatch() 
	{
		return normPatch;	
	}
}
