package com.deuxbot.opentld;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import org.opencv.android.JavaCameraView;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import com.deuxbot.opentld.tld.BoundingBox;
import com.deuxbot.opentld.tld.SlidingWindows;
import com.deuxbot.opentld.tld.TLD;
import com.deuxbot.opentld.tld.Util;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Point;
import android.support.v8.renderscript.RenderScript;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Toast;
	
@SuppressLint("ShowToast")
public class TLDView extends JavaCameraView implements CvCameraViewListener 
{
	private static final long TOUCH_TIME = 250000000;	 
	private static final int POINTS_RADIUS = 2;
	private static final int RECT_THICKNESS = 2;
	
	private SurfaceHolder surfaceHolder = null; 
	private Rect rectangle = null;
	private Paint paint = new Paint();
	private int canvasYOffset = 0;
    private int canvasXOffset = 0;
    private TLD tld = null;
    private BoundingBox resultBox = null;
    private AtomicReference<Point> firstCorner = new AtomicReference<Point>();
    private byte lastFrameArray[];
    private byte backgroundArray[];
    private boolean takeBackground = false;
    private boolean activeBackground = false;
    private Toast backgroundToast; 
    private Toast rectangleToast; 
    private org.opencv.core.Size FRAME_SIZE = new org.opencv.core.Size(320, 240); // QVGA 
    private org.opencv.core.Size scale = null;
    private Scalar red = new Scalar(255, 0, 0);
    private Scalar yellow = new Scalar(255, 255, 0);
    
    private RenderScript mRS;         
	
	public void setRenderScript(RenderScript renderScript)
	{
		this.mRS = renderScript;
	}
	
	public TLDView(Context context, AttributeSet attrs) 
	{
		super(context, attrs);
		surfaceHolder = getHolder();
		paint.setColor(Color.YELLOW);
		paint.setStrokeWidth(10);
		paint.setStyle(Style.STROKE);
		backgroundToast = Toast.makeText(context, "Background saved", Toast.LENGTH_SHORT);	 
		rectangleToast = Toast.makeText(context, "Rectangle too small", Toast.LENGTH_SHORT);	 

		// Control touches on the screen to take background and create the first bounding box
		setOnTouchListener(new OnTouchListener() 
		{
			long start;
			long end;
			long diff;
			double order = 1000000000.0;

			@Override
			public boolean onTouch(View v, MotionEvent event) 
			{			
				int action = event.getAction();
			    int xPosition = (int)event.getX();
				int yPosition = (int)event.getY();
				boolean done = true;
				Point corner = new Point(xPosition - canvasXOffset, yPosition  - canvasYOffset);
				
				switch(action) 
				{
					case MotionEvent.ACTION_DOWN:	// Touch detected
						start = System.nanoTime();
						firstCorner.set(corner);
						break;
					case MotionEvent.ACTION_UP:		// Touch finalized
						end = System.nanoTime();
						diff = end - start;
						Log.i(MainActivity.TAG, "Camera view: Touch seconds: " + diff/order);
						if(diff < TOUCH_TIME)		// Hyperfast touch
						{
							takeBackground = true;	// Take background
						}
						else
						{
							rectangle = new Rect(firstCorner.get().x, firstCorner.get().y, corner.x - firstCorner.get().x, corner.y - firstCorner.get().y);							
							Log.i(MainActivity.TAG, "Camera view: Points: " + firstCorner + " " + corner);
							Log.i(MainActivity.TAG, "Camera view: Box selected: " + rectangle + " with area " + rectangle.area());
						}
						done = false;
						break;
					case MotionEvent.ACTION_MOVE:	// Touch moving		
						android.graphics.Rect rect = new android.graphics.Rect((int)firstCorner.get().x + canvasXOffset, 
						(int)firstCorner.get().y + canvasYOffset, (int)corner.x + canvasXOffset, (int)corner.y + canvasYOffset);	
						Canvas canvas = surfaceHolder.lockCanvas(rect);
						canvas.drawRect(rect, paint);
						surfaceHolder.unlockCanvasAndPost(canvas);					
						break;
					}
					return done;
				}		
			});			
		}

	@Override
	public void onCameraViewStarted(int width, int height) 
	{
		canvasXOffset =  (getWidth() - width) / 2 ; 	
    	canvasYOffset =  (getHeight() - height) / 2; 
    	scale = new org.opencv.core.Size(getWidth() / FRAME_SIZE.width, getHeight() / FRAME_SIZE.height);
    	Log.i(MainActivity.TAG, "TLDView: View size: " + getWidth() + "x" + getHeight());
    	Log.i(MainActivity.TAG, "TLDView: Camera size: " + width + "x" + height);
    	Log.i(MainActivity.TAG, "TLDView: Canvas offsets: " + canvasXOffset + "x" + canvasYOffset);
	}

	@Override
	public void onCameraViewStopped() {}

	@Override
	public Mat onCameraFrame(Mat inputFrame) 
	{				
		Mat resizedFrame = new Mat();
		Imgproc.resize(inputFrame, resizedFrame, FRAME_SIZE);			
		Mat lastFrame = new Mat(resizedFrame.rows(), resizedFrame.cols(), CvType.CV_8UC1);
		Mat currentFrame = new Mat(resizedFrame.rows(), resizedFrame.cols(),CvType.CV_8UC1);
		Mat background = new Mat(resizedFrame.rows(), resizedFrame.cols(), CvType.CV_8UC1);
	
		try 
		{  				
			if(takeBackground)			// Take background active
			{
				Imgproc.cvtColor(resizedFrame, background, Imgproc.COLOR_RGB2GRAY);			
				backgroundArray = Util.getMatByteArray(background);
				backgroundToast.show();
				takeBackground = false;
				activeBackground = true;
			}
			
			if(rectangle != null)		// Initial box has been selected
			{									
				if(tld == null)			// TLD has not been initialized
				{				        
					Imgproc.cvtColor(resizedFrame, lastFrame, Imgproc.COLOR_RGB2GRAY);		
					Rect scaledRect = scaleDown(rectangle);
					if(Math.min(scaledRect.width, scaledRect.height) < SlidingWindows.MIN_WINDOW)	// Check if a box is smaller than minimum box
					{
						rectangleToast.show();
						rectangle = null;
					}
					else
					{
						tld = new TLD(FRAME_SIZE, mRS, this.getContext());										
						tld.getDetector().setForegroundActive(activeBackground);										
						if(backgroundArray != null) background.put(0, 0, backgroundArray);
						tld.getDetector().getForegroundDetector().setBackground(background);			
						tld.init(lastFrame, scaledRect);	
						lastFrameArray = Util.getMatByteArray(lastFrame);
					}														
				}
				else					// TLD has been initialized
				{														
					Imgproc.cvtColor(resizedFrame, currentFrame, Imgproc.COLOR_RGB2GRAY);
					lastFrame.put(0, 0, lastFrameArray);									
					lastFrameArray = Util.getMatByteArray(currentFrame);										
					resultBox = tld.run(lastFrame, currentFrame, lastFrameArray);								
					printResult(inputFrame, resultBox); 																						
				}			
			}			
		} 		
		catch (Exception e) 
		{
	        Log.e(MainActivity.TAG, "TLDView: Error.", e);
	    }		
		
		return inputFrame;
	}
	
	private void printResult(Mat frame, Rect box)
	{		
		if(box == null)
		{
			return;
		}		
		
		if(tld.isReset())	
		{
			Core.rectangle(frame, scaleUp(box.tl()), scaleUp(box.br()), red, RECT_THICKNESS);			
			//printPoints(frame, tld.getTracker().getPointsMesh(), red);
		}
		else			
		{
			Core.rectangle(frame, scaleUp(box.tl()), scaleUp(box.br()), yellow, RECT_THICKNESS);			
			//printPoints(frame, tld.getTracker().getApprMesh(), yellow);				
		}
	}
	
	// Print functions of tracker and detector results (rects, boxes and points)		
	public void printPoints(Mat frame, org.opencv.core.Point[] points, Scalar color)
	{
		if(points == null)
		{
			return;
		}
			
		for(int i = 0; i < points.length; i++)
		{
			Core.circle(frame, scaleUp(points[i]), POINTS_RADIUS, color, -1);
		}
	}
		
	public void printPoints(Mat frame, List<org.opencv.core.Point> points, Scalar color)
	{
		if(points == null)
		{
			return;
		}
			
		for(org.opencv.core.Point point : points)
		{
			Core.circle(frame, scaleUp(point), POINTS_RADIUS, color, -1);
		}
	}
		
	// Scale functions to resize objects up and down into a from resized frame
	public org.opencv.core.Rect scaleDown(org.opencv.core.Rect rect) 
	{
		return new org.opencv.core.Rect(scaleDown(rect.tl()), scaleDown(rect.br()));
	}
		
	public org.opencv.core.Point scaleDown(org.opencv.core.Point point)
	{		
		return new org.opencv.core.Point((int)(point.x / scale.width), (int)(point.y / scale.height));
	}
			
	public org.opencv.core.Point scaleUp(org.opencv.core.Point point)
	{		
		return new org.opencv.core.Point((int)(point.x * scale.width), (int)(point.y * scale.height));
	}
}
