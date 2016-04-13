#include <stdio.h>  
#include <iostream>  
#include <algorithm>
#include <ctime>
#include <numeric>
#include <fstream>
#include <stdlib.h> 

#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/video/background_segm.hpp>  
#include <opencv2/imgproc/imgproc.hpp>



#include "PFS.h"
using namespace cv;  
using namespace std;  

/*!
 * \brief Get a odd getStructuringElement from
   \param int size
 */
Mat getKernel(int morph_size){
	return getStructuringElement(MORPH_RECT, Size(morph_size*2+1, morph_size*2+1), Point(-1,-1));   
}

/*!
 * \brief Enlarge an ROI rectangle by a specific amount if possible 
 * \param frm The image the ROI will be set on
 * \param boundingBox The current boundingBox
 * \param padding The amount of padding around the boundingbox
 * \return The enlarged ROI as far as possible
 */
Rect enlargeROI(Mat frm, Rect boundingBox, int paddingx, int paddingy) {
    Rect returnRect = Rect(boundingBox.x - paddingx, boundingBox.y - paddingy, boundingBox.width + (paddingx * 2), boundingBox.height + (paddingy * 2));
    if (returnRect.x < 0)returnRect.x = 0;
    if (returnRect.y < 0)returnRect.y = 0;
    if (returnRect.x+returnRect.width >= frm.cols)returnRect.width = frm.cols-returnRect.x;
    if (returnRect.y+returnRect.height >= frm.rows)returnRect.height = frm.rows-returnRect.y;
    return returnRect;
}

/*!
 * \brief Sets and initalizes MOG2. 
 * \param are all internal
 * \return void. 
 */
void setMOG2(Ptr<BackgroundSubtractorMOG2> object){
	//more params
	object->set("nmixtures", 3); //maximum K allowed at any given pixle. 
	object->set("backgroundRatio",0.9);  //1-cf from paper.  
	object->set("varThresholdGen", 9.0); //Tg, determines amont of compoenents. Higher gives less. 
	//pMOG2->set("nShadowDetection", 0);  //detect shadows but reject as background. default is 127
	object->set("fTau", 0.2); //pixles is twice darker is not a shadow. 
	object->set("fCT", 0.05); //complexity reduction param, determines if more guassians should be maintained

	//even more params...
	object->set("fVarInit", 15.0); //variance in image, determines speed of adaptation
	object->set("fVarMin",0.0);
	object->set("fVarMax",25.0);
}

/*!
 * \brief checks if binary mask has non-0 element
 * \parm input is binary mask
 */
bool hasNonZero(Mat binary_input){
	if (countNonZero(binary_input)>0){
		return true;
	}
	return false;
}

/*!
 * \brief returns the largest ROI
 * \parm input is binary mask
 * \parm input Ptr to image where boxes should be overlayed
 * \parm input padding adds extra padding. 
 */
vector<Rect> getROIs(Mat binary_input, Mat same_scale_image, int paddingx, int paddingy){
	if (binary_input.channels()==3){
		cv::cvtColor(binary_input, binary_input, CV_BGR2GRAY);
	}
	Mat ContourImg = binary_input.clone();  
	vector< vector< Point> > contours;  
	findContours(ContourImg, contours,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector< Rect > rects;  
	vector< vector< Point> >::iterator itc= contours.begin();  
	while (itc!=contours.end()){  
	Rect box= boundingRect(Mat(*itc));
			box=enlargeROI(same_scale_image, box, paddingx, paddingy);

			rects.push_back(box);
			rects.push_back(box);
			++itc;
	}  
	//get rid of overlaping rectangels. 
	groupRectangles(rects, 1, 0.01);

	////sort by area
	//sort(rects.begin(), rects.end(), [](Rect a, Rect b){
	//		return b.area()< a.area();
	//});
	return rects;
}

/*!
 * \simple distance estimate. totally BS
 */
vector<Rect> simpleDistanceEstiamte(vector<Rect> input, Mat frame){
	int horizion= frame.size().height/2;
	vector<Rect> elligable;
	vector<Rect>::iterator itc= input.begin();  
	   while (itc!=input.end()){  
		   if ((*itc).br().y > horizion){
		   double part = (double)(*itc).height/frame.size().height;
				if (part >0.2){
				   elligable.push_back(*itc);	
				}
		   }
		   ++itc;
		}  
	return elligable;
}

vector<Rect> realDistanceEstimate(vector<Rect> input, Mat frame, int horizion, int player, int focal_length){
	vector<Rect> elligable;
	vector<Rect> remove;

	double assumed_height= 1.7;
	double assumed_minheight=1.7;
	double to_pixels=1080/5.6;

	//what is the depth of tee?
	int z_tee=focal_length*assumed_height*to_pixels/player;

	//what is infered playerheight 2 meters behind tee?
	int player_min=focal_length*to_pixels*assumed_minheight/(z_tee+2);


	vector<Rect>::iterator itc= input.begin();  
	   while (itc!=input.end()){  
		   //cout << "bottom is: " << (*itc).br().y << " and height: " << (*itc).height << endl; ;
		   if ((*itc).br().y > horizion){
			   if((*itc).height > player_min){
				   elligable.push_back((*itc));
			   }else{
					remove.push_back((*itc));
			   }
		   }else{
				remove.push_back((*itc));
		   }
		   ++itc;
		}  
	//return elligable;
	return remove;
}

/*!
 * \separate foreground from shadow for MOG2
 * \parm input is MOG2 mask
 * \return is pair of foreground and shadow
 */
pair<Mat, Mat> separateMaskMOG2(Mat input_mask){
	   pair<Mat, Mat> par;
	   Mat temp;
	   Mat input_mask_shadow=input_mask.clone();
	   threshold(input_mask_shadow, input_mask_shadow, 100, 255, 0);	
	   threshold(input_mask, input_mask, 150, 255, 0);
	   bitwise_and(input_mask, input_mask_shadow, temp);
	   input_mask_shadow=input_mask_shadow-temp;
	   par.first=input_mask;
	   par.second=input_mask_shadow;
	   return par;
}

/*!
 * \take input as foreground mask, roi, likely foreground, shadow, unlikely foreground and everything else background
 * \these get set ti 1,2,3 and 0 in the result mask. in that order.
 * \return is a mat with same size as input with values 0,1,2,3
 */
Mat getGrabCutMask(Mat Mog_mask, Mat Mog_mask_shadow, Rect ROI){
		    	Mat new_shadow;
				Mat new_foreground;
				Mat clonefgMaskMOG2=Mog_mask.clone();
				Mat invROI=clonefgMaskMOG2(ROI);
				Mat returnmat(Mog_mask.size(), CV_8UC3, cv::Scalar(0));

				invROI.setTo(2);
				threshold(Mog_mask, new_foreground, 127, 1, 0);
				threshold(Mog_mask_shadow, new_shadow, 127, 1, 0);
				return clonefgMaskMOG2-new_foreground + new_shadow;
}

Mat getGrabCutMask2(Mat Mog_mask, Mat Mog_padd, Mat Mog_close, Mat Mog_mask_shadow){
	Mat PB_BG;
	Mat FG;
	Mat PB_FG;
	Mat shadow;
	Mat fill;
	Mat returnmask;

	fill=Mog_mask|Mog_close;
	threshold(Mog_mask, FG, 127, 1, 0);
	threshold(Mog_padd, PB_BG, 127, 2, 0);
	threshold(fill, fill, 127, 1, 0);
	threshold(Mog_mask_shadow, shadow, 127, 1, 0);

	fill=fill-Mog_mask;
	fill=fill-shadow;
	fill.setTo(0, fill == -1); 


	returnmask=PB_BG-FG+fill;

	//double min, max;
	//minMaxLoc(returnmask, &min, &max);
	//cout << "minmax" << min << max;
	return returnmask;
}

/*!
 * \return the eucldian distance between two rect
 */
double boxCenterOffset(Rect one, Rect two){
	Point first((one.tl().x+one.br().x)/2, (one.tl().y+one.br().y)/2);
	Point second((two.tl().x+two.br().x)/2, (two.tl().y+two.br().y)/2);
	return norm(first-second);
}

/*!
 * \return the center of a box
 */
Point boxCenter(Rect one){
	Point first((one.tl().x+one.br().x)/2, (one.tl().y+one.br().y)/2);
	return first;
}

/*!
 * \return the TL of a box
 */
Point boxTopLeft(Point center, Point heightwidth){
	Point TL(center.x-(heightwidth.y/2), center.y-(heightwidth.x/2));
	return TL;
}


Mat drawBoxes(vector<Rect> boxes, Mat image, Scalar colour){
	vector<Rect>::iterator itc= boxes.begin();  
	while (itc!=boxes.end()){  
		rectangle(image, (*itc), colour);
		++itc;
	}  
	return image;
}

void main(){
	//options
	bool record_video=false;
	bool validation=true;
	bool use_depth=false;
	bool use_superres=false;
	bool normalize_light=false;
	int min=1000;
	//downsample
	int downsample_MOG=4;
	//int downsample_GRABCUT=4;

	//assumptionqs for distance estimate
	//for dataset1 - 152
	double f= 12;
	int hor=600/downsample_MOG;
	int play=1100/downsample_MOG;

	////for dataset2 - 114
	//double f= 12;
	//int hor=700/downsample_MOG;
	//int play=650/downsample_MOG;

	////for dataset4
	//double f= 12;
	//int hor=600/downsample_MOG;
	//int play=400/downsample_MOG;

	////MOG2
	Mat fgMaskMOG2; //fg-mask generated by MOG2 method  
	Mat fgMaskMOG2_shadow; //shadow-mask generated by MOG2 method 
	Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

		//params for initialization
		bool bShadowDetection=true;
		int history=500; //divde by 25 for secounds of training. Standard seems to be 200 frames. 
		float Cthr=15.0;
		float learnrate= -1; //automagic is -1 -> WTF is fckning -0.5 from examples!?

		pMOG2 = new BackgroundSubtractorMOG2(history, Cthr,bShadowDetection);
		setMOG2(pMOG2);

		Mat fgMaskMOG22; //fg-mask generated by MOG2 method  
		Mat fgMaskMOG2_shadow2; //shadow-mask generated by MOG2 method 
		Ptr< BackgroundSubtractor> pMOG22; //MOG2 Background subtractor
		pMOG22 = new BackgroundSubtractorMOG2(history, Cthr,bShadowDetection);
		setMOG2(pMOG22);
		Mat frameCropped;
		Mat resizeF2;

	//FRAMES FOR PROGRAM
	Mat frame; //current frame  
	Mat resizeF; //current frame in MOG resolution
	Mat Output; //the returned mask
	Mat boxF=frame.clone(); //draw stuff on the current frame

	//INITIALIZE OUTPUT/INPUT
	VideoCapture stream("Data/Input/syn_0062.avi"); //%TODO can be made into an argument for fuction 
	if(!stream.isOpened()){
        cout  << "Could not open the input video to read "<< endl;
	}
	VideoWriter outputVideo;   // Open the output
	outputVideo.open("Data/Results/video.avi", -1, 25, Size(1920,1080), true);
	if (!outputVideo.isOpened())
	{
		cout  << "Could not open the output video for write "<< endl;
	}
	

	//Grabcut
	Mat fgModel;
	Mat bgModel;

		//params
		int grabcut_itter= 1;
		int grabcut_rate=50; //div by 25 for rate
		float recompute_ratio=0.5;

	//pramas morphological operations 
	Mat element = getKernel(2); //2 //4
	Mat element2 = getKernel(10); //6 //12 NO USING THIS IS IT IS GRABCUT!!!
    Mat element3 = getKernel(1); //1 //2
	Mat element4= getKernel(10);

	//fps stuff
	clock_t current_ticks, delta_ticks;
	clock_t fps = 0;
	vector<double> fpsarray;
	int random=100;

	// unconditional loop
	int fps_itter=0;
	while(true){
		current_ticks = clock();

		//brake if no next frame  
		if(!(stream.read(frame))){
			break; 
		}  
		//quite by q
		if (waitKey(30) >= 0){    
			break;     
		}  

 

	   //downsample
	   resize(frame, resizeF, Size(frame.size().width/downsample_MOG, frame.size().height/downsample_MOG), INTER_NEAREST); 

	   imshow("original", resizeF);

	   //normalize_light
	   if(normalize_light){
			cv::cvtColor(resizeF, resizeF, CV_BGR2Lab);

			// Extract the L channel
			std::vector<cv::Mat> lab_planes(3);
			cv::split(resizeF, lab_planes);  // now we have the L image in lab_planes[0]

			// apply the CLAHE algorithm to the L channel
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
			clahe->setClipLimit(4);
			cv::Mat dst;
			clahe->apply(lab_planes[0], dst);

			// Merge the the color planes back into an Lab image
			dst.copyTo(lab_planes[0]);
			cv::merge(lab_planes, resizeF);

		   // convert back to RGB
		   cv::cvtColor(resizeF, resizeF, CV_Lab2BGR);
		   imshow("normalize colour", resizeF);
	   }

	   ////apply BS
	   pMOG2->operator()(resizeF, fgMaskMOG2, learnrate);  

	   //MOG2separate mask by thresholding 
	   pair<Mat, Mat> Mask=separateMaskMOG2(fgMaskMOG2);
	   fgMaskMOG2= Mask.first;
	   fgMaskMOG2_shadow=Mask.second;
	   Mat pMog;
	   Mat p2Mog;

	   	if(use_superres){
			//1918x1078
			frameCropped = frame(Rect(downsample_MOG/2,downsample_MOG/2,frame.size().width-downsample_MOG/2, frame.size().height-downsample_MOG/2)).clone();
			//478x268
			resize(frameCropped, resizeF2, Size(frame.size().width/downsample_MOG-downsample_MOG/2, frame.size().height/downsample_MOG-downsample_MOG/2), INTER_AREA);  
			pMOG22->operator()(resizeF2, fgMaskMOG2, learnrate);  
			pair<Mat, Mat> Mask=separateMaskMOG2(fgMaskMOG2);
			fgMaskMOG22= Mask.first;
			fgMaskMOG2_shadow2=Mask.second;

			//958x538
			resize(fgMaskMOG22, fgMaskMOG22, Size(frame.size().width*2/downsample_MOG-downsample_MOG/2, frame.size().height*2/downsample_MOG-downsample_MOG/2), INTER_AREA); 
			resize(fgMaskMOG2_shadow2, fgMaskMOG2_shadow2, Size(frame.size().width*2/downsample_MOG-downsample_MOG/2, frame.size().height*2/downsample_MOG-downsample_MOG/2), INTER_AREA); 

			//960x540
			resize(fgMaskMOG2, fgMaskMOG2, Size(frame.size().width*2/downsample_MOG, frame.size().height*2/downsample_MOG), INTER_AREA); 
			resize(fgMaskMOG2_shadow, fgMaskMOG2_shadow, Size(frame.size().width*2/downsample_MOG, frame.size().height*2/downsample_MOG), INTER_AREA); 
			Mat full=fgMaskMOG2.clone();
			full.setTo(0);
			Mat fulls=fgMaskMOG2_shadow.clone();
			fulls.setTo(0);
			
			//958x538
			Rect print =Rect(downsample_MOG/2,downsample_MOG/2,frame.size().width/2-downsample_MOG/2, frame.size().height/2-downsample_MOG/2);
			
			//960x540
			fgMaskMOG22.copyTo(full(print));
			fgMaskMOG2_shadow2.copyTo(fulls(print));
			bitwise_or(fgMaskMOG2, full, fgMaskMOG2);
			bitwise_or(fgMaskMOG2_shadow, fulls, fgMaskMOG2_shadow);

			//478x268
			resize(fgMaskMOG2, fgMaskMOG2, Size(frame.size().width/downsample_MOG, frame.size().height/downsample_MOG), INTER_AREA);  
			resize(fgMaskMOG2_shadow, fgMaskMOG2_shadow, Size(frame.size().width/downsample_MOG, frame.size().height/downsample_MOG), INTER_AREA); 
		}

	   //apply postprocessing
	   morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_OPEN, element);   
	   morphologyEx(fgMaskMOG2, pMog, CV_MOP_DILATE, element4);   
	   morphologyEx(fgMaskMOG2, p2Mog, CV_MOP_CLOSE, element2);   
	   morphologyEx(fgMaskMOG2_shadow, fgMaskMOG2_shadow, CV_MOP_OPEN, element3);   



	   	Mat foregroundMask(resizeF.size(), CV_8UC3, cv::Scalar(0)); //how to rescale this with image?
		//Mat foreground(resizeF.size(), CV_8UC3, cv::Scalar(255,255,255));

		int c=countNonZero(fgMaskMOG2);
		if(c>800){
			Mat use_mask;
			use_mask=getGrabCutMask2(fgMaskMOG2, pMog, p2Mog, fgMaskMOG2_shadow);
			//imshow("input to GRABCUT",use_mask*255/3);
			if (c<min){
				min=c;
			}
			grabCut(resizeF, use_mask, Rect(), bgModel, fgModel, 2, GC_INIT_WITH_MASK);
			use_mask = (use_mask == 1) | (use_mask == 3); //make mask binary
			foregroundMask=use_mask.clone();
		}
		
			//post-process for GRABCUT NOISE
			morphologyEx(foregroundMask, foregroundMask, CV_MOP_OPEN, element3);   

			//post-process for DEPTH???? 
		boxF=resizeF.clone();
		if(use_depth){
		   //Remove elements by depth-estimate. 
		   vector<Rect> ROI_normal= getROIs(foregroundMask, resizeF, 0, 0); //used to be 5 10
		   vector<Rect> remove=ROI_normal;
		   vector<Rect> remove2;
		   vector<Rect> ROI_padd= getROIs(foregroundMask, resizeF, 10, 30); //used to be 5 10

		   ROI_normal=realDistanceEstimate(ROI_normal, foregroundMask, hor, play, f);

		   ROI_padd=realDistanceEstimate(ROI_padd, foregroundMask, hor, play, f);

		   for (int i=0;i<ROI_padd.size();i++){
				ROI_normal.push_back(ROI_padd[i]);
		   }

		   groupRectangles(ROI_normal,1,0.9);
		   for (int i=0; i<remove.size(); i++){
			   for (int j=0; j<ROI_normal.size(); j++){
				   if((remove[i] & ROI_normal[j]).area() > 0){
						remove2.push_back(remove[i]);
				   }
			   }
		   }

		   for (int i=0; i<remove2.size(); i++){
				foregroundMask(remove2[i]).setTo(0);
		   }

		   boxF=drawBoxes(remove2, boxF, Scalar(0,0,255));
		}

	   imshow("remove this", boxF);
	   imshow("graphcutresult", foregroundMask);

	   //scale back to full HD
	   resize(foregroundMask, Output, Size(frame.size().width, frame.size().height), CV_INTER_LINEAR);
	   if (validation){
		resize(fgMaskMOG2, fgMaskMOG2, Size(frame.size().width, frame.size().height), CV_INTER_LINEAR);
	   }
	   //show stuff
	   //imshow("grabCut - LOW RES", foregroundMask);  
	   //frame.copyTo(foreground, Output); //bg pixels are not copied
	   //imshow("foreground - FULL RES", foreground);

	   //write to file every 100 frame
	   if (((fps_itter%random)==0 && validation)){
			imwrite( "Data/Results/" + to_string(fps_itter) + ".jpg", Output);
			imwrite( "Data/Results/" + to_string(fps_itter) + "i.jpg", fgMaskMOG2);
			imwrite( "Data/Results/MOG2+GrabCut" + to_string(fps_itter) + "image.jpg", frame);

	   }

	   if(record_video){
		   //write to video, every frame
		   outputVideo << Output;
	   }

	   //useful for the statistics
	   fps_itter++;
	   delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
	   if(delta_ticks > 0)
       fps = CLOCKS_PER_SEC / delta_ticks;
	   fpsarray.push_back(fps);
	   //cout << fps << endl;
	}

	imwrite( "Data/Results/" + to_string(fps_itter) + ".jpg", Output);
	imwrite( "Data/Results/" + to_string(fps_itter) + "i.jpg", fgMaskMOG2);

	
	//write and compute statistics
	double sum = accumulate(fpsarray.begin(), fpsarray.end(), 0.0);
	double mean = sum / fpsarray.size();

	std::vector<double> diff(fpsarray.size());
	std::transform(fpsarray.begin(), fpsarray.end(), diff.begin(), [mean](double x) { return x - mean; });

	double mm=inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	ofstream myfile;
	myfile.open ("Data/Results/MOG2+GrabCut_results.txt");
	myfile << "fps: ";
	myfile << mean;
	myfile << " +/- ";
	myfile << sqrt(mm / fpsarray.size());
	myfile.close();
}  