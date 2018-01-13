package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;

import com.deuxbot.opentld.MainActivity;
import com.deuxbot.opentld.ScriptC_learning;

import android.content.Context;
import android.support.v8.renderscript.*;
import android.util.Log;

public class TLD 
{
	private FlowTracker tracker;	
	private CascadeDetector detector;
	private BoundingBox prevBox;
	private Size resolution;
	private boolean valid;			// Result validity for learning
	private boolean wasValid;		// Previous validity for learning
	private boolean reset;			// Tracker is reset (for display)
	private Context context;
	
	private RenderScript mRS;
	private ScriptC_learning pnScript;
	private Allocation mIN;
	private Allocation mOUT;
	private int[] accepted; 
	private int[] result;
	private float[] posteriors;
	
	public TLD(Size resolution, RenderScript mRS, Context context) 
	{
		this.tracker = new FlowTracker();
		this.detector = new CascadeDetector();		
		this.prevBox = null;	
		this.valid = false;
		this.wasValid = false;	
		this.reset = false;		
		this.resolution = resolution;
		this.context = context;
		
		this.mRS = mRS;
	}
		
	public void init(Mat frame, Rect box) 
	{			
		Log.i(MainActivity.TAG, "=== INITIALIZING ===");
			
		detector.init(frame, box, resolution, mRS, context); // Initialize cascade detector
					
		prevBox = detector.getSlidingWindows().getBest(); 	// Correct bounding box
				
		// LEARN ===================================================	
		initialLearn(frame, detector.getSlidingWindows());		
		
		initScript(); // Learning RS
	}

	public BoundingBox run(final Mat lastFrame, final Mat currFrame, byte[] currFrameArray)
	{
		wasValid = valid;				
			
		// TRACK  ==================================================		
		BoundingBox trackBox = tracker.track(lastFrame, currFrame, prevBox, detector.getNNClassifier());	
			
		// DETECT  =================================================	
		BoundingBox detectBox = detector.detect(currFrame, currFrameArray);
			
		// FUSE  ===================================================
		BoundingBox fuseBox = fuse(trackBox, detectBox);
						
		// LEARN  ================================================== 		
		if(valid == true)				// Fuse result is valid
		{					
			//learn(currFrame, fuseBox);							
			learnRS(currFrame, fuseBox);				
		}		
		else
		{
			Log.w(MainActivity.TAG, "TLD: Fuse result not valid. Not learning.");
		}
								
		return fuseBox;
	}

	private BoundingBox fuse(BoundingBox tracked, BoundingBox detected) 
	{
		BoundingBox result = null;
	
		valid = false;
		reset = false;
		
		Log.i(MainActivity.TAG, "=== FUSING ===");
		
		if(tracked != null)									// Tracking was reliable			
		{
			// Detector yields only one result with a confidence higher than the result from tracker
			if(detector.getHierarchicalCluster().getClusters() == 1 && detected.getConfidence() > tracked.getConfidence() && detected.calcOverlap(tracked) < 0.5)
			{
				result = detected;	
				Log.i(MainActivity.TAG, "Fusser: Re-initializating tracker");
				reset = true;
			}
			// Detector yields more than one result or one with less confidence than tracker's result
			else
			{
				result = tracked;
				Log.i(MainActivity.TAG, "Fusser: Keeping tracker result");
				
				// Compare tracker confidence with thesholds for decide learning step
				if(tracked.getConfidence() > NNClassifier.THETA_TP)
				{
					valid = true;
				}
				else if(wasValid && tracked.getConfidence() > NNClassifier.THETA_FP)
				{
					valid = true;
				}
			}
		}
		else if(detector.getHierarchicalCluster().getClusters() == 1)	// Tracking was not reliable			
		{
			result = detected;   	
			Log.i(MainActivity.TAG, "Fusser: Re-initializating tracker");
			reset = true;
		}
		
		prevBox = result;										// Save final box for next tracking
				
		return result;						
	}
		
	private void initialLearn(final Mat frame, final SlidingWindows grid) 
	{
		Log.i(MainActivity.TAG, "=== INITIAL LEARNING ===");
		
		detector.getEnsembleClassifier().setCurrentFrame(Util.getMatByteArray(frame));
		
		// Calculate normalized patch (positive) 
		Mat positivePatch = Util.normalizePatch(frame, prevBox);												
		
		// Find positive and negative boxes
		List <BoundingBox> positiveBoxes = new ArrayList <BoundingBox>();	
		List <BoundingBox> negativeBoxes = new ArrayList <BoundingBox>();
		
		for(BoundingBox box : grid)
		{
			if(box.getOverlap() > SlidingWindows.P_CONSTRAINT)
			{
				positiveBoxes.add(box);
			}
			else if(box.getOverlap() < SlidingWindows.N_CONSTRAINT)
			{
				negativeBoxes.add(box); 
			}
		}
		
		Log.i(MainActivity.TAG, "Learn: Positive boxes: " + positiveBoxes.size() + " Negative boxes: " + negativeBoxes.size());
		
		// Sort positive patches by overlap
		Collections.sort(positiveBoxes, new Comparator<BoundingBox>()
		{
			@Override
			public int compare(BoundingBox box1, BoundingBox box2) 
			{
				return Float.compare(box1.getOverlap(), box2.getOverlap());
			}
		});
			
		// Keep a max of 10 positive patches		
		Mat patch = new Mat();
		
		int size = Math.min(positiveBoxes.size(), 10);
		
		Log.i(MainActivity.TAG, "Learn: Learning from " + size + " positive boxes");
				
		// Calculate fern features and learn from positive patches
		for(int i = 0; i < size; i++)
		{
			int[] features = detector.getEnsembleClassifier().calcFeatureVector(positiveBoxes.get(i));
			detector.getEnsembleClassifier().learn(true, features);
		}
		
		Log.i(MainActivity.TAG, "Ensemble Classifier: Total: " + detector.getEnsembleClassifier().getNP() + " positives and " +  detector.getEnsembleClassifier().getNN() + " negatives");
			
		// Shuffle negative patches randomly (Why this need randomness?)
		Random rand = new Random();
		Collections.shuffle(negativeBoxes, rand);
		
		// Keep a max of 100 negative patches
		size = Math.min(negativeBoxes.size(), 100);
		
		Log.i(MainActivity.TAG, "Learn: Learning from " + size + " negative boxes");
		
		List<Mat> negativePatches = new ArrayList<Mat>();
		
		for(int i = 0; i < size; i++)
		{
			patch = Util.normalizePatch(frame, negativeBoxes.get(i)); 
			negativePatches.add(patch);
		}
		
		// Learn from this bounding boxes	
		detector.getNNClassifier().learn(positivePatch, negativePatches);
				
		Log.i(MainActivity.TAG, "NNClassifier: Total " + detector.getNNClassifier().getNP() + " positives and " + detector.getNNClassifier().getNN() + " negatives");
	}
	
	@SuppressWarnings("unused")
	private void learn(final Mat frame, final BoundingBox fuseBox) 
	{			
		Log.i(MainActivity.TAG, "=== LEARNING ===");
		
		// Calculate normalized patch (positive)
		Mat positivePatch = Util.normalizePatch(frame, fuseBox);											
		
		// Find positive and negative patches
		List <BoundingBox> positiveBoxes = new ArrayList <BoundingBox>();	
		List <BoundingBox> negativeBoxes = new ArrayList <BoundingBox>();
				
		for(BoundingBox box : detector.getSlidingWindows())
		{
			if(box.calcOverlap(fuseBox) > SlidingWindows.P_CONSTRAINT && box.getPosterior() < EnsembleClassifier.THRESHOLD)
			{
				positiveBoxes.add(box);
			}
			else if(box.calcOverlap(fuseBox) < SlidingWindows.N_CONSTRAINT && box.getPosterior() > EnsembleClassifier.THRESHOLD)
			{				
				negativeBoxes.add(box);	
			}
		}
	
		Log.i(MainActivity.TAG, "Learn: Positive boxes: " + positiveBoxes.size() + " Negative boxes: " + negativeBoxes.size());	
		
		// Sort positive patches by overlap
		Collections.sort(positiveBoxes, new Comparator<BoundingBox>()
		{
			@Override
			public int compare(BoundingBox box1, BoundingBox box2) 
			{
				return Float.compare(box1.getOverlap(), box2.getOverlap());
			}
		});
		
		// Keep a max of 10 positive patches
		int size = Math.min(positiveBoxes.size(), 10);
		
		Log.i(MainActivity.TAG, "Learn: Learning from " + size + " positive boxes");
		
		// Learn from this positive patches
		for(int i = 0; i < size; i++)
		{
			int[] features = detector.getEnsembleClassifier().calcFeatureVector(positiveBoxes.get(i));
			detector.getEnsembleClassifier().learn(true, features);
		}
		
		// Shuffle negative patches randomly (This again, why randomness?)		
		Random rand = new Random();
		Collections.shuffle(negativeBoxes, rand);
		
		// Keep all negative patches (They are not easily generated)		
		List<Mat> negativePatches = new ArrayList<Mat>();
		
		Log.i(MainActivity.TAG, "Learn: Learning from " + negativeBoxes.size() + " negative boxes");
		
		// Learn from this negative patches
		for(int i = 0; i < negativeBoxes.size(); i++)
		{
			detector.getEnsembleClassifier().learn(false, negativeBoxes.get(i).getFeatures());
		}
	
		Log.i(MainActivity.TAG, "Ensemble Classifier: Total: " + detector.getEnsembleClassifier().getNP() + " positives and " +  detector.getEnsembleClassifier().getNN() + " negatives");

		for(int i = 0; i < negativeBoxes.size(); i++)
		{		
			negativePatches.add(negativeBoxes.get(i).getNormPatch());
		}
					
		// Learn from this bounding boxes
		detector.getNNClassifier().learn(positivePatch, negativePatches);
			
		Log.i(MainActivity.TAG, "NNClassifier: Total " + detector.getNNClassifier().getNP() +" positives and " + detector.getNNClassifier().getNN() + " negatives");
	}	
	
	private void learnRS(final Mat frame, final BoundingBox fuseBox) 
	{			
		Log.i(MainActivity.TAG, "=== LEARNING ===");
		
		// Calculate normalized patch (positive)
		Mat positivePatch = Util.normalizePatch(frame, fuseBox);											
		
		// Find positive and negative patches
		List <BoundingBox> positiveBoxes = new ArrayList <BoundingBox>();	
		List <BoundingBox> negativeBoxes = new ArrayList <BoundingBox>();
		
		overlapFilterRS(fuseBox);
	
		for(int i = 0; i < accepted.length; i++)
		{
			if(posteriors[accepted[i]] < EnsembleClassifier.THRESHOLD)
			{
				positiveBoxes.add(detector.getSlidingWindows().getBoxes().get(accepted[i]));	
			}
			else
			{
				negativeBoxes.add(detector.getSlidingWindows().getBoxes().get(accepted[i]));		
			}
		}
		
		Log.i(MainActivity.TAG, "Learn: Positive boxes: " + positiveBoxes.size() + " Negative boxes: " + negativeBoxes.size());
		
		// Sort positive patches by overlap
		Collections.sort(positiveBoxes, new Comparator<BoundingBox>()
		{
			@Override
			public int compare(BoundingBox box1, BoundingBox box2) 
			{
				return Float.compare(box1.getOverlap(), box2.getOverlap());
			}
		});
		
		// Keep a max of 10 positive patches
		int size = Math.min(positiveBoxes.size(), 10);
		
		Log.i(MainActivity.TAG, "Learn: Learning from " + size + " positive boxes");
		
		// Learn from this positive patches
		for(int i = 0; i < size; i++)
		{
			int[] features = detector.getEnsembleClassifier().calcFeatureVector(positiveBoxes.get(i));
			detector.getEnsembleClassifier().learn(true, features);
		}
		
		// Shuffle negative patches randomly (This again, why randomness?)		
		Random rand = new Random();
		Collections.shuffle(negativeBoxes, rand);
		
		// Keep all negative patches (They are not easily generated)		
		List<Mat> negativePatches = new ArrayList<Mat>();
		
		Log.i(MainActivity.TAG, "Learn: Learning from " + negativeBoxes.size() + " negative boxes");
		
		// Learn from this negative patches
		for(int i = 0; i < negativeBoxes.size(); i++)
		{
			detector.getEnsembleClassifier().learn(false, negativeBoxes.get(i).getFeatures());
		}
				
		Log.i(MainActivity.TAG, "Ensemble Classifier: Total: " + detector.getEnsembleClassifier().getNP() + " positives and " +  detector.getEnsembleClassifier().getNN() + " negatives");

		for(int i = 0; i < negativeBoxes.size(); i++)
		{		
			negativePatches.add(negativeBoxes.get(i).getNormPatch());
		}
					
		// Learn from this bounding boxes
		detector.getNNClassifier().learn(positivePatch, negativePatches);
	
		Log.i(MainActivity.TAG, "NNClassifier: Total " + detector.getNNClassifier().getNP() +" positives and " + detector.getNNClassifier().getNN() + " negatives");
	}	
	
	private void initScript()
	{
		pnScript = new ScriptC_learning(mRS);

		mIN = Allocation.createSized(mRS, Element.I32(mRS), detector.getSlidingWindows().getBoxesIndices().length);
		mOUT = Allocation.createSized(mRS, Element.I32(mRS), detector.getSlidingWindows().getBoxesIndices().length + 1);	
		Allocation coordsAlloc = Allocation.createSized(mRS, Element.I32(mRS), detector.getSlidingWindows().getBoxesCords().length);    
		
		mIN.copyFrom(getDetector().getSlidingWindows().getBoxesIndices());
		coordsAlloc.copyFrom(getDetector().getSlidingWindows().getBoxesCords());
		
		pnScript.bind_output(mOUT);			
		pnScript.bind_coords(coordsAlloc);	
		pnScript.set_P_CONSTRAINT(SlidingWindows.P_CONSTRAINT);
		pnScript.set_N_CONSTRAINT(SlidingWindows.N_CONSTRAINT);
		pnScript.set_THRESHOLD(EnsembleClassifier.THRESHOLD);
		
		result = new int[detector.getSlidingWindows().getBoxesIndices().length + 1]; 		
		posteriors = new float[detector.getSlidingWindows().getBoxesCords().length];
	
	}
	
	private void initScriptData(BoundingBox box)
	{
		Arrays.fill(posteriors, 0);
		
		for(int i = 0; i < detector.getEnsembleClassifier().getAcceptedNum(); i++)
		{
			posteriors[detector.getEnsembleClassifier().getAccepted()[i]] = detector.getSlidingWindows().getBoxes().get(detector.getEnsembleClassifier().getAccepted()[i]).getPosterior();
		}
	
		Allocation postAlloc = Allocation.createSized(mRS, Element.F32(mRS), detector.getSlidingWindows().getBoxesCords().length);    
		postAlloc.copyFrom(posteriors);
		pnScript.bind_posterior(postAlloc);	
		pnScript.set_bb(new Int4(box.x, box.y, box.width, box.height));	
	}
	
		
	private void overlapFilterRS(BoundingBox box)
	{						
		initScriptData(box);
		Arrays.fill(result, -2);
		pnScript.forEach_root(mIN);	
		mOUT.copyTo(result); 		
		result[result.length - 1] = -1;	
		Arrays.sort(result);
		int position = Arrays.binarySearch(result, -1);		
		int size = result.length - position - 1;				
		accepted = new int[size];
		System.arraycopy(result, position + 1, accepted, 0, size);	
	}
	
	
	public FlowTracker getTracker()
	{
		return tracker;
	}
	
	public CascadeDetector getDetector()
	{
		return detector;
	}
	
	public boolean isReset()
	{
		return reset;
	}

}
