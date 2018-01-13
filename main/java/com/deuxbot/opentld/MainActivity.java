package com.deuxbot.opentld;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import android.os.Bundle;
import android.support.v8.renderscript.RenderScript;
import android.app.Activity;
import android.util.Log;
import android.view.Menu;
import android.view.SurfaceView;
import android.view.WindowManager;

public class MainActivity extends Activity 
{
	public static final String TAG = "TLD";
	private TLDView tldView;
	
	// OpenCV asynchronous initialization
	private BaseLoaderCallback opencvLoaderCallback = new BaseLoaderCallback(this) 
	{
        @Override
        public void onManagerConnected(int status) 
        {
            switch (status) 
            {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    tldView.enableView();            
                } 
                break;
                
                default:
                {
                    super.onManagerConnected(status);
                } 
                break;
            }
        }
    }; 
    
	@Override
	public void onCreate(Bundle savedInstanceState) 
	{	
		super.onCreate(savedInstanceState);
	     getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
	     setContentView(R.layout.activity_main);
	     tldView = (TLDView) findViewById(R.id.TLDView);
	     tldView.setVisibility(SurfaceView.VISIBLE);
	     tldView.setCvCameraViewListener(tldView);
	     RenderScript renderScript = RenderScript.create(this);		 
		 tldView.setRenderScript(renderScript);	   
	}
	
	@Override
	public boolean onCreateOptionsMenu(Menu menu) 
	{
		getMenuInflater().inflate(R.menu.main, menu);
		
		return true;
	}

	@Override
	public void onPause() 
	{
	    super.onPause();
	   
	    if (tldView != null)
	    {
	    	tldView.disableView();
	    }
	}
	
	public void onDestroy() 
	{
	     super.onDestroy();
	     
	     if (tldView != null)
	     {
	    	 tldView.disableView();
	     }
	 }

	@Override
	public void onResume() 
	{
	    super.onResume();

	    if(!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_8, this, opencvLoaderCallback))
	    {
			Log.e(TAG, "Error: OpenCV loading failed.");
	    }
	}
}
