package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;

import com.deuxbot.opentld.MainActivity;

import android.content.Context;
import android.util.Log;

public class SlidingWindows implements Iterable <BoundingBox>
{
	public static final float P_CONSTRAINT = .6f;		// Represents high overlap
	public static final float N_CONSTRAINT = .2f;		// Represents low overlap
	public static final int MIN_WINDOW = 15; 			// Minimum window size
	public static final int MIN_AREA = 25; 				// Minimum window area
	private static final float SHIFT = .1f; 			// Set scan windows off by a percentage value of the window dimensions
	
	private List <BoundingBox> boxes;					// Self-explanatory
	private List <Size> boxScales;						// Sizes of scaled boxes
	private BoundingBox bestBox;						// Box with higher overlap	
	
	private int[] boxesCords;							// Coordinates of boxes (x, y, w , h)
	private int[] boxesIndices;							// Indices of boxes 
		
	public SlidingWindows() 
	{
		boxes = new ArrayList<BoundingBox>();		
		boxScales = new ArrayList<Size>();
	}
	
	public void run(Mat frame, Rect initialBox, Context context) 
	{ 		
		List <Integer> coordsList = new ArrayList<Integer>();
		float[] scales = new float[21];
		
		// Sliding window scales 1.2^i / i C [-10, 10]
		for(int i = -10; i <= 10; i++)
		{
			scales[i + 10] = (float)Math.pow(1.2, i);
		}
		
		// Generate boxes applying scale changes to initial one
		for(int i = 0; i < scales.length; i++)
		{
			int width = Math.round(initialBox.width * scales[i]);
			int height = Math.round(initialBox.height * scales[i]);
			int area = height * width;	
			int shiftX = Math.round(initialBox.width * SHIFT);
			int shiftY = Math.round(initialBox.height * SHIFT);
			
			if(isCandidate(area, height, width, frame))
			{
				boxScales.add(new Size(width, height));		
				
				for(int row = 1; row < (frame.rows() - height); row += shiftY)
				{
					for(int col = 1; col < (frame.cols() - width); col += shiftX)
					{
						BoundingBox box = new BoundingBox();
						box.x = col;
						box.y = row;
						box.width = width;
						box.height = height;
						box.setScaleIndex(boxScales.size() - 1);	
						boxes.add(box);					
						
						coordsList.add(col);
						coordsList.add(row);
						coordsList.add(width);
						coordsList.add(height);
					}
				}
			}
		}
		
		// Calculate overlaps with initial bounding box
		calcInitOverlaps(initialBox);
		
		// Generate data for Foreground renderscript
		boxesCords = Util.getArrayFromList(coordsList);	
		boxesIndices = Util.genIndexedArray(boxes.size()); 
	
		Log.e(MainActivity.TAG, "Grid: Number of boxes: " + boxes.size() + " Best box: " + bestBox + " with area " + bestBox.area());
		
	}
	
	// Calculate overlap of all boxes and save which with higher overlap
	public void calcInitOverlaps(Rect box)
	{
		float maxOverlap = 0f;
		
		for(BoundingBox gridBox : boxes)
		{
			gridBox.setOverlap(gridBox.calcOverlap(box));
			
			if(gridBox.getOverlap() > maxOverlap)
			{
				maxOverlap = gridBox.getOverlap();
				bestBox = gridBox;
			}
		}	
	}	
	
	public void calcOverlaps(Rect box)
	{
		for(BoundingBox gridBox : boxes)
		{
			gridBox.setOverlap(gridBox.calcOverlap(box));				
		}	
	}	
	
	public boolean isCandidate(int area, int width, int height, Mat frame)
	{
		if(area >= MIN_AREA && width <= frame.cols() && height <= frame.rows()) 
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	
	public int[] getBoxesCords()
	{
		return boxesCords;
	}

	public int[] getBoxesIndices()
	{
		return boxesIndices;
	}

	public List <BoundingBox> getBoxes()
	{
		return boxes;
	}
	
	public Size[] getScales()
	{
		Size size[] = new Size[boxScales.size()];
		
		return boxScales.toArray(size);
	}
	
	public BoundingBox getBest()
	{
		return bestBox;
	}
	
	@Override
	public Iterator<BoundingBox> iterator() 
	{
		return boxes.iterator();
	}	
}