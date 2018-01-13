package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import com.deuxbot.opentld.MainActivity;

import android.util.Log;
import android.util.Pair;

public class FlowTracker 
{
	private static final Size WINDOW_SIZE = new Size(4, 4);		// Size of quadratic area around the point which is compare
	private static final Size PATCH_SIZE = new Size(10, 10);;	// Size of cross correlation patch
	private static final int MAX_LEVEL = 5; 					// Pyramid maximun level (5 by default)
	private static final int MAX_FB = 10;						// Forward-Backward error threshold
	private static final int MAX_POINTS = 10;					// Number of points generated to track the flow
	private static final int MARGIN = 1;						// Margin of points generated to the box borders
	
	private float fbMedian;										// Median of Forward-Backward error
	private float nccMedian;									// Median of Normalised Correlation Coefficient
	private TermCriteria criteria;								// Termination criteria of the iterative search algorithm
	private Point[] pointMesh;									// Points generated inside box as input to tracker (display)
	private List<Point> apprMesh;								// Points predicted that are approved all thesholds (display)
	
	FlowTracker()
	{
		fbMedian = 0;
		nccMedian = 0;
		criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 20, 0.03);
		pointMesh = null;
		apprMesh = null;
	}

	public BoundingBox track(final Mat prevFrame, final Mat currFrame, final BoundingBox prevBox, NNClassifier nnClassifier) 
	{
		BoundingBox newBox = null;		
	
		if(prevBox == null)
		{
			return null;
		}	
		
		if(prevBox.width <= 0 || prevBox.height <= 0)
		{
			return null;
		}	
				
		Log.i(MainActivity.TAG, "=== TRACKING ===");
		 		 
		// Generate equally spaced set of points from box
		pointMesh = prevBox.createGrid(MAX_POINTS, MARGIN);
		
		// Calculate new points with Lukas-Kanade tracker
		Pair <Point[], Point[]> trackedPoints = lkTrack(prevFrame, currFrame, pointMesh);
		
		if(trackedPoints != null)
		{
			// Transform the bounding box with tracked points 
			newBox = prevBox.predictBox(trackedPoints.first, trackedPoints.second);		
		
			// Calculate tracker confidence for fuse hypothesis
			Mat patch = Util.normalizePatch(currFrame, newBox);										
			float confidence = nnClassifier.classifyPatch(patch);
			newBox.setConfidence(confidence);
				
			Log.i(MainActivity.TAG, "Tracker: Tracked box: " + newBox + " with area " + newBox.area() + " and confidence: " + newBox.getConfidence());
		}
		else
		{
			Log.w(MainActivity.TAG, "Tracker: Unreliable tracking result. Tracked box rejected");
		}
		
		return newBox;
	}
	
	// Tracks points from previous frame to current frame using Lucas-Kanade method
	Pair <Point[], Point[]> lkTrack(final Mat prevFrame, final Mat nextFrame, final Point[] prevPoints) 
	{
		MatOfPoint2f prevPointsMat = new MatOfPoint2f(prevPoints);
		MatOfPoint2f nextPointsMat = new MatOfPoint2f();
		MatOfByte statusMat = new MatOfByte();
		MatOfFloat nccMat = new MatOfFloat();
		MatOfPoint2f fbPointsMat = new MatOfPoint2f();
		MatOfByte fbStatusMat = new MatOfByte();
		MatOfFloat fbMat = new MatOfFloat();
		
		// Lucas-Kanade track 
		Video.calcOpticalFlowPyrLK(prevFrame, nextFrame, prevPointsMat, nextPointsMat, statusMat, nccMat, WINDOW_SIZE, MAX_LEVEL, criteria, 0, 0);

		// Lucas-Kanade retrack (Forward-backward)
		Video.calcOpticalFlowPyrLK(nextFrame, prevFrame, nextPointsMat, fbPointsMat, fbStatusMat, fbMat, WINDOW_SIZE, MAX_LEVEL, criteria, 0, 0);
		
		// Forward-backward error
		float[] fbError = fbMat.toArray();	
		Point[] fbPoints = fbPointsMat.toArray();
		int nPoints = prevPoints.length;
		
		for(int i = 0; i < nPoints; i++)
		{
			fbError[i] = Util.euclideanDistance(prevPoints[i], fbPoints[i]);
		}
		
		// Normalised Correlation Coefficient
		Point[] nextPoints = nextPointsMat.toArray();
		byte[] status = statusMat.toArray();	
		byte[] fbStatus = fbStatusMat.toArray();	
		float[] nccMatch = new float[nPoints]; 
		nccMatch = normCrossCorrelation(prevFrame, nextFrame, prevPoints, nextPoints, status);	

		// Medians of errors and similarity measures
		fbMedian = Util.median(fbError);
		nccMedian = Util.median(nccMatch);
		
		Log.i(MainActivity.TAG, "Tracker: FB Median: " + fbMedian + " NCC Median: " + nccMedian);
		
		// Check if FB median is bigger than stop threshold
		if(fbMedian > MAX_FB)	
		{
			return null;
		}
		else
		{
			// Filter points with medians
			return filterPoints(prevPoints, nextPoints, nccMatch, fbError, status, fbStatus);
		}	
	}
	
	// Filter points ussing medians of FB and NCC
	private Pair <Point[], Point[]> filterPoints(Point[] prevPoints, Point[] nextPoints, float nccSimilarity[], float fbError[], byte status[], byte fbStatus[])
	{ 
		List <Point> approvedPrevPoints = new ArrayList<Point>();
		List <Point> approvedNextPoints = new ArrayList<Point>();
		
		// Filter points with NCC similarity >= NCC median and FB error <= FB median 
		for(int i = 0; i < nextPoints.length; i++)
		{
			if(status[i] == 1 && fbStatus[i] == 1 && nccSimilarity[i] >= nccMedian && fbError[i] <= fbMedian)
			{
				approvedPrevPoints.add(prevPoints[i]);
				approvedNextPoints.add(nextPoints[i]);	
			}
		}
	
		Log.i(MainActivity.TAG, "Tracker: Points: " + prevPoints.length + " NCC and FB approved: " + approvedPrevPoints.size());
		
		// Return filtered points (non-tracked & tracked)	
		int size = approvedPrevPoints.size();
		
		apprMesh = approvedPrevPoints;	// Just display
		
		return new Pair<Point[], Point[]>(approvedPrevPoints.toArray(new Point[size]), approvedNextPoints.toArray(new Point[size]));
	}
	
	// Calculates Normalized Cross Correlation for every point
	private float[] normCrossCorrelation(Mat prevFrame, Mat nextFrame, Point[] prevPoints, Point[] nextPoints, byte[] status)
	{
		Mat prevPatch = new Mat(PATCH_SIZE, CvType.CV_8U);
		Mat nextPatch = new Mat(PATCH_SIZE, CvType.CV_8U);
		Mat result = new Mat(new Size(1, 1), CvType.CV_32F);
		float[] match = new float[prevPoints.length];		
		
		for(int i = 0; i < prevPoints.length; i++)
		{
			if(status[i] == 1)
			{
				Imgproc.getRectSubPix(prevFrame, PATCH_SIZE, prevPoints[i], prevPatch);
				Imgproc.getRectSubPix(nextFrame, PATCH_SIZE, nextPoints[i], nextPatch);
				Imgproc.matchTemplate(prevPatch, nextPatch, result, Imgproc.TM_CCOEFF_NORMED);
				match[i] = Util.getMatFloat(result);
			}
			else
			{
				match[i] = 0f;
			}
		}
		
		return match;
	}
	
	
	public Point[] getPointsMesh()
	{
		return pointMesh;
	}
	
	public List<Point> getApprMesh()
	{
		return apprMesh;
	}
		
}
