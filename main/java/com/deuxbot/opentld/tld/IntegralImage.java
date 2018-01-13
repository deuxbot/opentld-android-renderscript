package com.deuxbot.opentld.tld;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;


public class IntegralImage 
{
	private Mat sumMat;					// Integral image
	private Mat squareSumMat;			// Integral image for squared pixel
	private int[] sum;					// Duplicated data in java type
	private double[] squareSum;			// Duplicated data in java type
	private boolean created;
	private VarianceFilter varFilter;
		
	
	public IntegralImage(VarianceFilter varFilter) 
	{
		sumMat = new Mat();
		squareSumMat = new Mat();
		created = false;
		this.varFilter = varFilter;
	}
	
	// Calculate integral images of frame and duplicate data
	public void calcIntegrals(Mat frame)
	{		
		sumMat.create(frame.rows(), frame.cols(), CvType.CV_32F); 
		squareSumMat.create(frame.rows(), frame.cols(), CvType.CV_64F); 	
		Imgproc.integral2(frame, sumMat, squareSumMat);				
		sum = Util.getMatIntArray(sumMat); 
		squareSum = Util.getMatDoubleArray(squareSumMat);
			
		if(!created)
		{
			varFilter.createSquareSumF(squareSum.length);
			created = true;
		}	
	}
	
	public Mat getSumMat()
	{
		return sumMat;
	}
	
	public int[] getSum()
	{
		return sum;
	}
	
	public double[] getSquareSum()
	{
		return squareSum;
	}
		
}
