package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

import com.deuxbot.opentld.MainActivity;
import com.deuxbot.opentld.ScriptC_variance;

import android.support.v8.renderscript.*;
import android.util.Log;

public class VarianceFilter 
{
	private IntegralImage intImage;				
	private float threshold;
	private SlidingWindows slWindows;
	private ForegroundDetector fgDetector;
	
	private RenderScript mRS;			
	private ScriptC_variance varScript;
	private Allocation mIN;						
	private Allocation mOUT;					
	private Allocation coordsAlloc;			
	private float[] squareSumF;
	
	private float[] variances;
	private int[] accepted;

	public VarianceFilter() 
	{
		intImage = new IntegralImage(this);
		threshold = 0f;
	}
	
	public void setThreshold(Mat frame, BoundingBox box)
	{		
		MatOfDouble stddev = new MatOfDouble();	
		Mat subFrame = frame.submat(box);
		Core.meanStdDev(subFrame, new MatOfDouble(), stddev);			// Calculate mean and standard deviation
		threshold = (float) (Math.pow(stddev.toArray()[0], 2) * 0.5);	// Treshold = Variance result / 2

		Log.i(MainActivity.TAG, "Variance filter: Variance threshold: " + threshold);
	}

	public float calcVariance(BoundingBox box) 
	{		
		int[] sum = intImage.getSum();
		double[] sqSum = intImage.getSquareSum(); 
		int cols = intImage.getSumMat().cols();
			
		int brSum = sum[(box.y + box.height) * cols + box.x + box.width];
		int blSum = sum[(box.y + box.height) * cols + box.x];
		int trSum = sum[box.y * cols + box.x + box.width];
		int tlSum = sum[box.y * cols + box.x];
		double brSqSum = sqSum[(box.y + box.height) * cols + box.x + box.width];
		double blSqSum = sqSum[(box.y + box.height) * cols + box.x];
		double trSqSum = sqSum[box.y * cols + box.x + box.width];
		double tlSqSum = sqSum[box.y * cols + box.x];
		
		double boxArea = box.area();
		double sumMean = (brSum + tlSum - trSum - blSum) / boxArea; 
		double sqSumMean = (brSqSum + tlSqSum - trSqSum - blSqSum) / boxArea;
		
		return (float) (sqSumMean - sumMean * sumMean);		
	}
	
	public void calcVariances() 
	{				
		accepted = null;	
		initScriptData();		
		varScript.forEach_root(mIN);		
		mOUT.copyTo(variances);			
		filter();		
	}	
	
	private void filter()
	{
		List <Integer> varAcceptedList = new ArrayList<Integer>();
		
		if(fgDetector.getAcceptedNum() > 0)
		{		
			for(int i = 0; i < variances.length; i++) 
			{
				if(variances[i] > threshold)
				{
					varAcceptedList.add(fgDetector.getAccepted()[i]);
				}
			}
		}
		else
		{
			for(int i = 0; i < variances.length; i++) 
			{
				if(variances[i] > threshold)
				{
					varAcceptedList.add(slWindows.getBoxesIndices()[i]); 
				}
			}	
		}
		
		accepted = Util.getArrayFromList(varAcceptedList);
	}
	
	private void initScript()
	{					
		varScript = new ScriptC_variance(mRS); 

		coordsAlloc = Allocation.createSized(mRS, Element.I32(mRS), slWindows.getBoxesCords().length);  
		coordsAlloc.copyFrom(slWindows.getBoxesCords());		
		varScript.bind_coords(coordsAlloc);				
	}
	
	public void createSquareSumF(int size)
	{
		squareSumF = new float[size];
	}
	
	private void initScriptData()
	{
		int size = 0;
		int[] inputIndices = null;
		
		if(!fgDetector.getList().isEmpty())	// Calculate boxes accepted by Foreground Detector
		{		
			size = fgDetector.getAccepted().length;			
			inputIndices = Util.genIndexedArray(size);
			
		}
		else					// Calculate all boxes
		{
			size = slWindows.getBoxesIndices().length;
			inputIndices =  slWindows.getBoxesIndices();
		}
		
		for(int i = 0; i < intImage.getSquareSum().length; i++)
		{
			squareSumF[i] = (float)intImage.getSquareSum()[i];
		}

		variances = new float[size];		
		
		mIN = Allocation.createSized(mRS, Element.I32(mRS), size);  
		mOUT = Allocation.createSized(mRS, Element.F32(mRS), size);
		Allocation indicesAlloc = Allocation.createSized(mRS, Element.I32(mRS), size);  
		Allocation sum = Allocation.createSized(mRS, Element.I32(mRS), intImage.getSum().length);  
		Allocation sqSum = Allocation.createSized(mRS, Element.F32(mRS), squareSumF.length);  
				
		mIN.copyFrom(inputIndices);
		
		if(!fgDetector.getList().isEmpty()) 
			indicesAlloc.copyFrom(fgDetector.getAccepted()); 
		else 
			indicesAlloc.copyFrom(slWindows.getBoxesIndices());
			
		sum.copyFrom(intImage.getSum());			
		sqSum.copyFrom(squareSumF);		
		
		varScript.bind_output(mOUT);
		varScript.bind_indices(indicesAlloc);
		varScript.bind_sum(sum);
		varScript.bind_sqSum(sqSum);
		
		varScript.set_cols(intImage.getSumMat().cols());	
	}
	
	public void setSlidingWindows(SlidingWindows slWindows) 
	{
		this.slWindows = slWindows;	
	}
	
	public void setForeground(ForegroundDetector fgDetector) 
	{
		this.fgDetector = fgDetector;	
	}
	
	public void setRenderScript(RenderScript mRS) 
	{
		this.mRS = mRS;	
		initScript();
	}
	
	public float getThreshold()
	{
		return threshold;
	}
	
	public float[] getVariances()
	{
		return variances;
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
		
	public IntegralImage getIntegralImage()
	{
		return intImage;
	}
}

