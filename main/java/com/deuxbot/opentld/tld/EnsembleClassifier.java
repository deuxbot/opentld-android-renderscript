package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Size;

import android.support.v8.renderscript.*;
import android.util.Log;

import com.deuxbot.opentld.MainActivity;
import com.deuxbot.opentld.ScriptC_ensemble;

public class EnsembleClassifier 
{
	public static final int NUM_FERNS = 10;			// Number of ferns
	public static final int FEATURES_FERN = 13;		// Number of features per fern
	public static final float THRESHOLD = .5f;		// Confidence threshold
	
	public Fern ferns[];							// Array of ferns
	private int positives;							// Number of positive examples
	private int negatives;							// Number of negative examples
	byte[] currentFrame;
	
	private RenderScript mRS;					
	private ScriptC_ensemble enScript;
	private int[] allFeatures;
	private float[] allPosteriors;
	private int[] accepted;
	private float[] confidences; 
	Allocation mIN;  
	Allocation mOUT;
	
	public EnsembleClassifier()
	{
		positives = 0;
		negatives = 0;
	}	
	
	void init(Size[] scales)
	{		
		ferns = new Fern[NUM_FERNS];	
				
		for(int i = 0; i < NUM_FERNS; i++)
		{
			ferns[i] = new Fern(FEATURES_FERN, scales);
			ferns[i].init();
		}
		
		generateScriptData();
		initScript();
		
		Log.i(MainActivity.TAG, "Ensemble classifier: Initialized " + NUM_FERNS + " ferns with " + FEATURES_FERN + " features per fern");
	}
			
	// Calculate average posterior of feature vector
	private float calcConfidence(int[] featureVector)
	{	
		float confidence = 0f;	
		
		for(int i = 0; i < featureVector.length; i++)
		{
			confidence += ferns[i].getPosteriors()[featureVector[i]];
		}
		
		confidence = confidence / featureVector.length;
		
		return confidence;
	}
	
	// Calculate feature vector of patch
	public int[] calcFeatureVector(BoundingBox box) 
	{					
		int[] featureVector = new int[NUM_FERNS];					
		
		for(int i = 0; i < NUM_FERNS; i++)
		{	
			featureVector[i] = ferns[i].calcFernFeature(currentFrame, box);		
		}
				
		return featureVector;
	}
	
	// Classify a patch returning his confidence
	public float classifyPatch(BoundingBox box)
	{
		int[] features = calcFeatureVector(box);			
		float confidence = calcConfidence(features);		
		box.setFeatures(features);
			
		return confidence;
	}
	
	// Update posteriors with a new feature (P/N)
	private void updatePosteriors(int features[], boolean positive)
	{	
		for(int i = 0; i < features.length; i++)
		{
			ferns[i].updatePosterior(features[i], positive);
		}
	}
	
	// Update posterior only if feature pass thresholds
	public void learn(boolean positive, int feature[]) 
	{
		float confidence = calcConfidence(feature);
		
		if((positive && confidence < THRESHOLD) || (!positive && confidence > THRESHOLD))
		{
		    updatePosteriors(feature, positive);
		    
		    if(positive)
		    {
		    	positives++;
		    }
		    else
		    {
		    	negatives++;
		    }
		}
	}
	
	private void generateScriptData()
	{
		int size = ferns[0].getFeaturesRS().length;
		
		allFeatures = new int[size * NUM_FERNS];
						
		for(int i = 0; i < NUM_FERNS; i++)
		{
			System.arraycopy(ferns[i].getFeaturesRS(), 0 , allFeatures, i * size, size);
		}	
		
		size = ferns[0].getPosteriors().length;
				
		allPosteriors = new float[size * NUM_FERNS]; 		
	
		for(int i = 0; i < NUM_FERNS; i++)
		{
			System.arraycopy(ferns[i].getPosteriors(), 0 , allPosteriors, i * size, size);
		}	
	}
	
	private void initScript()
	{					
		enScript = new ScriptC_ensemble(mRS); 
		
		Allocation featuresAlloc = Allocation.createSized(mRS, Element.I32(mRS), allFeatures.length);    			
		featuresAlloc.copyFrom(allFeatures);		
		enScript.bind_features(featuresAlloc);
			
		enScript.set_numFerns(NUM_FERNS);	
		enScript.set_numScales(ferns[0].getNumScales());	
		enScript.set_numFeatures(FEATURES_FERN);	
		//enScript.set_frameCols(frameCols);
	}
	
	private int[] initScriptData(int[] varAccepted, int[] scaleIndices, int[] x, int[] y)
	{
		accepted = null;
		
		int size = varAccepted.length; 
		
		int[] inputIndices = Util.genIndexedArray(size);
		
		confidences = new float[size]; 
		
		int[] result = new int[size * NUM_FERNS];
		
		Allocation frameAlloc = Allocation.createSized(mRS, Element.I8(mRS), currentFrame.length);
		Allocation scaleAlloc = Allocation.createSized(mRS, Element.I32(mRS), size);
		Allocation xAlloc = Allocation.createSized(mRS, Element.I32(mRS), size);
		Allocation yAlloc = Allocation.createSized(mRS, Element.I32(mRS), size);
		mIN = Allocation.createSized(mRS, Element.I32(mRS), size);  
		mOUT = Allocation.createSized(mRS, Element.I32(mRS), size * NUM_FERNS);
		
		frameAlloc.copyFrom(currentFrame);
		scaleAlloc.copyFrom(scaleIndices);
		xAlloc.copyFrom(x);
		yAlloc.copyFrom(y);
		mIN.copyFrom(inputIndices);
		
		enScript.bind_frame(frameAlloc);	
		enScript.bind_scaleIndices(scaleAlloc);		
		enScript.bind_x(xAlloc);
		enScript.bind_y(yAlloc);
		enScript.bind_output(mOUT);		
		enScript.set_numBoxes(size);
		
		return result;
	}
	
	public void classifyPatches(int[] varAccepted, int[] scaleIndices, int[] x, int[] y) 
	{
		if(varAccepted.length <= 0)
		{
			return;
		}
		
		int[] result= initScriptData(varAccepted, scaleIndices, x, y);		
		enScript.forEach_root(mIN);			
		mOUT.copyTo(result);		
		calcConfidenceRS(result, varAccepted.length);		
		filter(varAccepted);	
	}
	
	private void calcConfidenceRS(int[] featureVector, int size)
	{			
		for(int box = 0; box < size; box++)
		{				
			float confidence = 0f;
			
			for(int i = 0; i < NUM_FERNS; i++)
			{	
				confidence += ferns[i].getPosteriors()[featureVector[box * NUM_FERNS + i]];
			}
			
			confidence = confidence / NUM_FERNS;
			
			confidences[box] = confidence;
		}					
	}
	
	private void filter(int[] varAccepted)
	{	
		List <Integer> acceptedList = new ArrayList<Integer>();
		
		for(int i = 0; i < confidences.length; i++) 
		{
			if(confidences[i] > THRESHOLD)
			{
				acceptedList.add(varAccepted[i]);
			}
		}
			
		accepted = Util.getArrayFromList(acceptedList);
	}
	
	public void setRenderScript(RenderScript mRS)
	{
		this.mRS = mRS;
	}
	
	public int getNP()
	{
		return positives;
	}
	
	public int getNN()
	{
		return negatives;
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
	
	public void setCurrentFrame(byte[] currentFrame)
	{
		this.currentFrame = currentFrame;
	}		
}

