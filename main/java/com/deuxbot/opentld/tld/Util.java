package com.deuxbot.opentld.tld;

import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Util 
{		
	
	public static float median(float array[])
	{		
		float[] aux = Arrays.copyOf(array, array.length);
		
		Arrays.sort(aux);
		
		return aux[aux.length / 2];
	}
		
	public static float euclideanDistance(Point p1, Point p2)
	{
		double dX = p1.x - p2.x;
		double dY = p1.y - p2.y;
		
		return (float)Math.sqrt(dX * dX + dY * dY);
	}
	
	public static double resizeNormalize(Mat patch, Mat output)
	{	
		MatOfDouble mean = new MatOfDouble();
		MatOfDouble stdev = new MatOfDouble();
		
		Imgproc.resize(patch, output, NNClassifier.PATCH_SIZE);
		Core.meanStdDev(output, mean, stdev);
		output.convertTo(output, CvType.CV_32F);
		Core.subtract(output, new Scalar(mean.toArray()[0]), output);
		
		return stdev.toArray()[0];
	}
	
	public static Mat normalizePatch(Mat frame, BoundingBox box)
	{		
		Rect intersection = box.calcIntersect(frame);
		Mat subFrame = frame.submat(intersection);
		Mat patch = new Mat();
		Util.resizeNormalize(subFrame, patch);	
		
		return patch;
	}
	
	// Kinda parsing OpenCV matrix contents: CV_32S (int), CV_32F (float) and CV_64F (double)
	public static float getMatFloat(Mat mat)
	{		
		float[] buffer = new float[1];
		mat.get(0, 0, buffer);
		
		return buffer[0];
	}
	
	public static int[] getMatIntArray(Mat mat)
	{	
		int size = (int) (mat.total() * mat.channels());	
		int[] array = new int[size];		
		mat.get(0, 0, array);
		
		return array;
	}
		
	public static float[] getMatFloatArray(Mat mat)
	{	
		int size = (int) (mat.total() * mat.channels());	
		float[] array = new float[size];		
		mat.get(0, 0, array);
		
		return array;
	}
	
	public static double[] getMatDoubleArray(Mat mat)
	{	
		int size = (int) (mat.total() * mat.channels());	
		double[] array = new double[size];		
		mat.get(0, 0, array);
		
		return array;
	}
	
	public static byte[] getMatByteArray(Mat mat)
	{
		int size = (int) (mat.total() * mat.channels());
		byte[] array = new byte[size];		
		mat.get(0, 0, array);
		
		return array;
	}	
	
	// Auxiliar functions for RenderScript
	public static int[] getArrayFromList(List <Integer> list)
	{
		int[] array = new int[list.size()];
		
		for(int i = 0; i < list.size(); i++)
		{
			array[i] = list.get(i);
		}
		
		return array;
	}
	
	public static float[] getArrayFromFloatList(List <Float> list)
	{
		float[] array = new float[list.size()];
		
		for(int i = 0; i < list.size(); i++)
		{
			array[i] = list.get(i);
		}
		
		return array;
	}
		
	public static int[] genIndexedArray(int size)
	{
		int[] array = new int[size];
		
		for(int i = 0; i < size; i++)
		{
			array[i] = i;
		}
		
		return array;
	}
}



