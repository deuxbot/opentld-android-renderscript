package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.support.v8.renderscript.*;
import android.util.Log;

import com.deuxbot.opentld.MainActivity;
import com.deuxbot.opentld.ScriptC_foreground;

public class ForegroundDetector 
{
	private static final int THRESHOLD = 16;			// Number of pixels of threshold
	private static final float MIN_AREA_OFFSET = .5f;	// Offset for initial box area
	private static final float MAX_AREA_OFFSET = .9f;	// Offset for whole frame area
	private static final float EXPAND_OFFSET = 1.25f;	// Offset to expand foreground box	

	private List <Rect> fgList;							// Rects of objects detected
	private Size resolution;							// Resolution of captured frame
	private double minArea;								// Min area of a foreground rect
	private double maxArea;								// Max area of a foreground rect
	private Mat background;								// Background image (our model)
	public boolean active;								// Indicate if detector is on
	
	private RenderScript mRS;						
	private Allocation mIN;								
	private Allocation mOUT;							
	private int[] inside;								
	private SlidingWindows slWindows;		
	private int[] accepted;
	ScriptC_foreground fgScript;
	
	public ForegroundDetector() 
	{
		fgList = new ArrayList<Rect>(); 
		minArea = 0;
		maxArea = 0;
		active = false;
		background = new Mat();
	}		
	
	public void run(final Mat frame)
	{
		if(!active)	
		{
			accepted = slWindows.getBoxesIndices();
			
			Log.w(MainActivity.TAG, "Foreground detector: No background model specified.");
			
			return;
		}
			
		fgList.clear();								// Remove previous rectangles
						
		Mat absDiff = new Mat(frame.rows(), frame.cols(), frame.type());	
		Mat thresFrame = new Mat(frame.rows(), frame.cols(), frame.type());	
		
		Core.absdiff(background, frame, absDiff); 											// Calculates absolute difference	
		Imgproc.threshold(absDiff, thresFrame, THRESHOLD, 255, Imgproc.THRESH_BINARY);		// Applies a fixed-level threshold			
		List <MatOfPoint>  contours = new ArrayList <MatOfPoint>();	
		Imgproc.findContours(thresFrame, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);	// Find contours
			
	    for(MatOfPoint contour : contours)
		{		    	
	    	if (Imgproc.contourArea(contour) > SlidingWindows.MIN_AREA)
	    	{
	    		Rect rect = Imgproc.boundingRect(contour);	// Create rectangle from contour	
				if(isCandidate(rect))			
				{					
					expandRect(rect);
					fgList.add(rect);					// Add rectangle to detected list
				}
	    	}
		}
	
		for(int i = 0 ; i < fgList.size(); i++)
		{
			Log.i(MainActivity.TAG, "Foreground detector: Object: " + i + " " + fgList.get(i) + " with area: " + fgList.get(i).area());
		}
	}

	// Evaluate if a box is cadidate to add to the list
	private boolean isCandidate(Rect rect)
	{
		boolean candidate = false;
		
		if((rect.area() > (minArea * MIN_AREA_OFFSET)) && rect.area() < maxArea) 
		{
			candidate = true;
		}
		
		return candidate;
	}
	
	// Evaluate if a box is inside of one foreground box
	public boolean isInside(Rect rect) 
	{
		for(Rect fgBox : fgList)
		{		
			if(fgBox.x < rect.x)
			{
				if(fgBox.y < rect.y)
				{
					if(fgBox.x + fgBox.width > rect.x + rect.width)
					{
						if(fgBox.y + fgBox.height > rect.y + rect.height)
						{
							return true;							
						}
					}
				}
			}
		}
		
		return false;
	}

	public void areInside() 
	{	
		if(fgList.isEmpty())
		{
			accepted = null;
			
			return;
		}
	
		Arrays.fill(inside, -2);
		
		for(int i = 0; i < fgList.size(); i++)
		{						
			mOUT.copyFrom(inside);			
			fgScript.set_bb(new Int4 (fgList.get(i).x, fgList.get(i).y, fgList.get(i).width, fgList.get(i).height));							
			fgScript.forEach_root(mIN);													
			mOUT.copyTo(inside);			
		}	
		
		// Save indices of accepted windows		
		inside[inside.length - 1] = -1;
		Arrays.sort(inside);
		int position = Arrays.binarySearch(inside, -1);
		int size = inside.length - position - 1;
		accepted = new int[size];
		System.arraycopy(inside, position + 1, accepted, 0, size);	
	}
	
	// Augment the size of a rect (1 > EXPAND_OFFSET < 2)
	public void expandRect(Rect rect)
	{
		int expW = (int) (rect.width * EXPAND_OFFSET);
		int expH = (int) (rect.height * EXPAND_OFFSET);
		int expX = rect.x - ((expW - rect.width) / 2);
		int expY = rect.y - ((expH - rect.height) / 2);
		
		if(expX > 0)
		{
			rect.x = expX;
		}	
		else
		{
			rect.x = 0;
		}		
		
		if(expY > 0)
		{
			rect.y = expY;	
		}
		else
		{
			rect.y = 0;
		}	
		
		if(expW + rect.x < resolution.width)
		{
			rect.width = expW;
		}	
		else
		{
			rect.width = (int) (resolution.width - rect.x);			
		}
		
		if(expH + rect.y < resolution.height)
		{					
			rect.height = expH;							
		}
		else
		{
			rect.height = (int) (resolution.height - rect.y); 
		}
		
		//Log.i(MainActivity.TAG, "Foreground detector: Box expanded to : " + rect + " with area: " + rect.area());					
	}
		
		
	private void initScript()
	{		
		fgScript = new ScriptC_foreground(mRS);  
			
		mIN = Allocation.createSized(mRS, Element.I32(mRS), slWindows.getBoxesIndices().length);  
		mOUT = Allocation.createSized(mRS, Element.I32(mRS), slWindows.getBoxesIndices().length  + 1);
		Allocation coordsAlloc = Allocation.createSized(mRS, Element.I32(mRS), slWindows.getBoxesCords().length);    
		
		mIN.copyFrom(slWindows.getBoxesIndices());
		coordsAlloc.copyFrom(slWindows.getBoxesCords());
		
		fgScript.bind_output(mOUT);			
		fgScript.bind_coords(coordsAlloc);
		
		inside = new int[slWindows.getBoxesIndices().length + 1]; 			
	}
	
	public int[] getInside()
	{
		return inside;
	}
	
	public int[] getAccepted()
	{
		return accepted;
	}
	
	public int getAcceptedNum()
	{
		if(accepted != null)
		{
			return accepted.length;
		}
		else
		{
			return 0;
		}
	}
	
	public void setRenderScript(RenderScript mRS)
	{
		this.mRS = mRS;
		initScript();
	}
	
	public void setSlidingWindows(SlidingWindows grid)
	{
		this.slWindows = grid;
	}
	
	public void setMinArea(double minArea)
	{
		this.minArea = minArea;
	}
	
	public void setResolution(Size resolution)
	{
		this.resolution = resolution;
		maxArea = resolution.height * resolution.width * MAX_AREA_OFFSET;
	}
	
	public void setBackground(Mat background)
	{
		this.background = background; 
	}
	
	public void setActive(boolean active)
	{
		this.active = active;
	}
	
	public List <Rect> getList()
	{
		return fgList;
	}
}
