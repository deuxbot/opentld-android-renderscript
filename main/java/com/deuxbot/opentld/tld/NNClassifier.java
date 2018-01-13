package com.deuxbot.opentld.tld;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.deuxbot.opentld.ScriptC_nearest;

import android.support.v8.renderscript.*;

public class NNClassifier 
{
	public static final float THETA_FP = .5f;					// Threshold for false ones
	public static final float THETA_TP = .65f;					// Threshold for true ones
	public static final Size PATCH_SIZE = new Size (15, 15);	// New size for patches
	
	private List<Mat> truePositives;							// True positive patches
	private List<Mat> falsePositives;							// False positive patches
		
	private RenderScript mRS;
	private ScriptC_nearest nnScript;
	private Allocation mIN;
	private Allocation mOUT;
	private int[] accepted;
	private float[] confidences;
	private float[] truePositivesArray;
	private float[] falsePositivesArray;
	
	public NNClassifier()
	{
		truePositives = new ArrayList<Mat>();
		falsePositives = new ArrayList<Mat>();
	}
		
	public float classifyPatch(Mat patch) 
	{		
		if(truePositives.isEmpty())
		{
			return 0;	// Is negative
		}
		
		if(falsePositives.isEmpty())
		{
			return 1;	// Is positive
		}	
		
		Mat nccMat = new Mat(1, 1, CvType.CV_32F);
		float maxP = 0; 
		float maxN = 0;
		float ncc = 0;
		
		// Compare patch with true positive patches
		for(int i = 0; i < truePositives.size(); i++)
		{			
			Imgproc.matchTemplate(truePositives.get(i), patch, nccMat, Imgproc.TM_CCORR_NORMED); 						
			ncc = (Util.getMatFloat(nccMat) + 1f) * .5f;	// Normalization to <0,1>					
			
			if(ncc > maxP)
			{
				maxP = ncc;
			}		
		}	  
		
		// Compare patch with false positive patches
		for(int i = 0; i < falsePositives.size(); i++)
		{		
			Imgproc.matchTemplate(falsePositives.get(i), patch, nccMat, Imgproc.TM_CCORR_NORMED); 
			ncc = (Util.getMatFloat(nccMat) + 1f) * .5f;	// Normalization to <0,1>
	
			if(ncc > maxN)
			{
				maxN = ncc;
			}
		}
			
		float dP = 1 - maxP;	// Distance to possitive class
		float dN = 1 - maxN;	// Distance to negative class
		float distance = dN / (dN + dP);
			
		return distance;	
	}
	
	public void learn(Mat positive, List<Mat> negatives) 
	{
		float confidence = 0;
	
		// Evaluate positive patch
		confidence = classifyPatch(positive);
		
		Mat pAdded = null;
		List<Mat> nAdded = new ArrayList<Mat>();
		
		if(confidence <= THETA_TP)
        {
            truePositives.add(positive); 
            pAdded = positive;
        }
		
		// Evaluate negative patches
		for(int i = 0; i < negatives.size(); i++)
	    {
			confidence = classifyPatch(negatives.get(i));
	        
			if(confidence >= THETA_FP)
	        {
	            falsePositives.add(negatives.get(i));
	            nAdded.add(negatives.get(i));
	        }
	    }	
		
		updateExamplesArray(pAdded, nAdded);			
	}
	
	private void updateExamplesArray(Mat positive, List<Mat> negatives) // For RS
	{
		int size = (int) (PATCH_SIZE.width * PATCH_SIZE.height); 
		
		if(positive != null)
		{		
			positive.type();
			
			if(truePositivesArray != null)
			{
				float[] aux = Arrays.copyOf(truePositivesArray, truePositivesArray.length);
				truePositivesArray = new float[truePositivesArray.length + size];
				System.arraycopy(aux, 0, truePositivesArray, 0, aux.length);				
				float[] patch = Util.getMatFloatArray(positive);
				System.arraycopy(patch, 0, truePositivesArray, truePositivesArray.length - size, patch.length);	
			}
			else
			{
				truePositivesArray = new float[size];				
				float[] patch = Util.getMatFloatArray(positive);
				System.arraycopy(patch, 0, truePositivesArray, 0, patch.length);	
			}
		}
		if(!negatives.isEmpty())
		{
			float[] patches = new float[size * negatives.size()];			
			for(int i = 0; i < negatives.size(); i++)
			{
				float[] patch = Util.getMatFloatArray(negatives.get(i));
				System.arraycopy(patch, 0, patches, i * size, patch.length);
			}		
			
			if(falsePositivesArray != null)
			{
				float[] aux = Arrays.copyOf(falsePositivesArray, falsePositivesArray.length);
				falsePositivesArray = new float[falsePositivesArray.length + (size * negatives.size())];
				System.arraycopy(aux, 0, falsePositivesArray, 0, aux.length);
				System.arraycopy(patches, 0, falsePositivesArray, falsePositivesArray.length - (size * negatives.size()), patches.length);		
			}
			else
			{
				falsePositivesArray = new float[size * negatives.size()];
				System.arraycopy(patches, 0, falsePositivesArray, 0, patches.length);		
			}
		}
		
	}
	
	private void initScript()
	{
		nnScript = new ScriptC_nearest(mRS); 					
		nnScript.set_size((int) (PATCH_SIZE.width * PATCH_SIZE.height));
	}

	private void initScriptData(int numPatches, float[] patches) 
	{
		confidences = new float[numPatches];
		
		mIN = Allocation.createSized(mRS, Element.I32(mRS), numPatches); 
		mOUT = Allocation.createSized(mRS, Element.F32(mRS), numPatches); 
		Allocation patchesAlloc = Allocation.createSized(mRS, Element.F32(mRS), patches.length); 	
		Allocation pPatchesAlloc = Allocation.createSized(mRS, Element.F32(mRS), truePositivesArray.length); 
	
		mIN.copyFrom(Util.genIndexedArray(numPatches));
		patchesAlloc.copyFrom(patches);
		pPatchesAlloc.copyFrom(truePositivesArray);
		
		nnScript.bind_patches(patchesAlloc);
		nnScript.bind_pPatches(pPatchesAlloc);
		nnScript.bind_output(mOUT);
		
		nnScript.set_numPatches(numPatches);
		nnScript.set_pPatchesNum(truePositives.size());
		
		if(falsePositivesArray != null)
		{
			Allocation nPatchesAlloc = Allocation.createSized(mRS, Element.F32(mRS), falsePositivesArray.length); 
			nPatchesAlloc.copyFrom(falsePositivesArray);
			nnScript.bind_nPatches(nPatchesAlloc);
			nnScript.set_nPatchesNum(falsePositives.size());
		}
		else
		{
			nnScript.set_nPatchesNum(0);
		}				
	}
	
	public void classifyPatches(int[] enAccepted, float[] patches)
	{
		if(enAccepted == null || enAccepted.length <= 0)
		{
			return;
		}
							
		initScriptData(enAccepted.length, patches);		
		nnScript.forEach_root(mIN);		
		mOUT.copyTo(confidences);		
		filter(enAccepted);		
	}

	public float[] getConfidences() 
	{
		return confidences;
	}
	
	private void filter(int[] enAccepted)
	{	
		List <Integer> nnAcceptedList = new ArrayList<Integer>();
		
		for(int i = 0; i < confidences.length; i++) 
		{
			if(confidences[i] > THETA_TP)
			{
				nnAcceptedList.add(enAccepted[i]);
			}
		}
		
		accepted = Util.getArrayFromList(nnAcceptedList);
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
	
	public int getNP()
	{
		return truePositives.size();
	}
	
	public int getNN()
	{
		return falsePositives.size();
	}

	public int[] getAccepted()
	{
		return accepted;
	}

}

