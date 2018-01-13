package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;

import android.support.v8.renderscript.RenderScript;
import android.util.Log;
import android.content.Context;

import com.deuxbot.opentld.MainActivity;

public class CascadeDetector 
{
	private SlidingWindows slWindows;
	private ForegroundDetector fgDetector;
	private VarianceFilter varFilter;
	private EnsembleClassifier enClassifier;
	private NNClassifier nnClassifier;
	private HierarchicalCluster cluster;	
	public BoundingBox clustered;	// Display
	private boolean fgActive;
	
	public CascadeDetector()
	{
		slWindows = new SlidingWindows();
		fgDetector = new ForegroundDetector();
		varFilter = new VarianceFilter();
		enClassifier = new EnsembleClassifier();
		nnClassifier = new NNClassifier();
		cluster = new HierarchicalCluster();	
		fgActive = false;
	}
	
	public void init(Mat frame, Rect initialBox, Size res, RenderScript mRS, Context c)
	{
		// Calculate the boxes with this scales
		slWindows.run(frame, initialBox, c);
			
		// Initialize the ferns with their features		
		enClassifier.setRenderScript(mRS);
		enClassifier.init(slWindows.getScales()); 
		
		// Set up Foreground Detector
		fgDetector.setActive(fgActive);
		fgDetector.setMinArea(initialBox.area());
		fgDetector.setResolution(res);	
		fgDetector.setSlidingWindows(slWindows);
		fgDetector.setRenderScript(mRS);
				
		// Set up Variance Filter 
		varFilter.setThreshold(frame, getSlidingWindows().getBest());	
		varFilter.setSlidingWindows(slWindows);
		varFilter.setForeground(fgDetector);
		varFilter.setRenderScript(mRS);
			
		nnClassifier.setRenderScript(mRS);
	}
		
	public BoundingBox detect(Mat currFrame, byte[] currFrameArray) 
	{			
		List<BoundingBox> confidentDetections = new ArrayList<BoundingBox>();	
	
		enClassifier.setCurrentFrame(currFrameArray);
				
		Log.i(MainActivity.TAG, "=== DETECTING ===");			
		
		// Foreground Detector processing
		fgDetector.run(currFrame);			
					
		// Variance Filter processing	
		varFilter.getIntegralImage().calcIntegrals(currFrame);	
				
		//////////////////////////////// OPTIMIZED LOOP ////////////////////////////////
		
		// Foreground Detector	
		fgDetector.areInside();		
		
		// Variance Filter		
		varFilter.calcVariances();
		
		// Ensemble Classifier
		int size = varFilter.getAcceptedNum();
		int[] scaleIndices = new int[size];
		int[] x = new int[size];
		int[] y = new int[size];
		
		for(int i = 0; i < size; i++)	// TODO: This could be referenced, latency is ~1ms thought
		{
			scaleIndices[i] = slWindows.getBoxes().get(varFilter.getAccepted()[i]).getScaleIndex();
			x[i] = slWindows.getBoxes().get(varFilter.getAccepted()[i]).x;
			y[i] = slWindows.getBoxes().get(varFilter.getAccepted()[i]).y;
		}
		
		enClassifier.classifyPatches(varFilter.getAccepted(), scaleIndices, x, y);	
			 
		// Nearest Neighbor Classifier
		int patchSize = (int)(NNClassifier.PATCH_SIZE.width * NNClassifier.PATCH_SIZE.width);
		float[] patches = new float[patchSize * enClassifier.getAcceptedNum()];
		float[] patch = new float[patchSize];
		
		for(int i = 0; i < enClassifier.getAcceptedNum() ; i++)	// Prepare normalized patches for RS
		{		
			Mat normPatch = Util.normalizePatch(currFrame, slWindows.getBoxes().get(enClassifier.getAccepted()[i]));
			patch = Util.getMatFloatArray(normPatch);
			System.arraycopy(patch, 0, patches, i * patchSize, patch.length);
		}
		
		nnClassifier.classifyPatches(enClassifier.getAccepted(), patches);	
		
		for(int i = 0; i < nnClassifier.getAcceptedNum(); i++)
		{
			confidentDetections.add(slWindows.getBoxes().get(nnClassifier.getAccepted()[i]));	
		}
			
		Log.i(MainActivity.TAG, "RS Detector: FG approved: " + fgDetector.getAcceptedNum() + " VAR approved " + varFilter.getAcceptedNum() 
		+ " EN approved " + enClassifier.getAcceptedNum() + " NN approved " + confidentDetections.size()); 
		
		///////////////////////////////////////////////////////////////////////////////////
		
		//////////////////////////////// UNOPTIMIZED LOOP ////////////////////////////////
		/*
		int f = 0, v = 0, e = 0;
		confidentDetections.clear();		 
	
		for(BoundingBox box : slWindows)
		{	
			// Foreground Detector							
			if(fgDetector.isInside(box) || fgDetector.getList().isEmpty())	
			{	f++;							
				// Variance Filter						
				if(varFilter.calcVariance(box) > varFilter.getThreshold())
				{	v++;				
					// Ensemble classifier
					if(enClassifier.classifyPatch(box) > EnsembleClassifier.THRESHOLD)	
					{	e++;						
						// Template matching					
						Mat normPatch = Util.normalizePatch(currFrame, box);	
						box.setNormPatch(normPatch);
						float confidence = nnClassifier.classifyPatch(normPatch); 
						box.setConfidence(confidence);				
						if(confidence > NNClassifier.THETA_TP)
						{								
							confidentDetections.add(box);				
						}											
					}									
				}								
			}			
		}
		
		Log.i(MainActivity.TAG, "Detector: FG approved: " + f + " VAR approved " + v + " ENS approved " + e + " NN approved " +  confidentDetections.size());
		Log.i(MainActivity.TAG, "Detector: Total confident detections: " + confidentDetections.size());		
		*/
		///////////////////////////////////////////////////////////////////////////////////
		
		// Non-maximal suppression			
		return cluster.clusterDetections(confidentDetections);	
	}
	
	public SlidingWindows getSlidingWindows()
	{
		return slWindows;
	}
	
	public ForegroundDetector getForegroundDetector()
	{
		return fgDetector;
	}
	
	public VarianceFilter getVarianceFilter()
	{
		return varFilter;
	}
	
	public EnsembleClassifier getEnsembleClassifier()
	{
		return enClassifier;
	}
	
	public NNClassifier getNNClassifier()
	{
		return nnClassifier;
	}
	
	public HierarchicalCluster getHierarchicalCluster()
	{
		return cluster;
	}
	
	public void setForegroundActive(boolean active)
	{
		fgActive = active;
	}	
}
