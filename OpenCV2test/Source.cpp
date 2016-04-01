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
	object->set("fVarInit",15.0); //variance in image, determines speed of adaptation
	object->set("fVarMin",0.0);
	object->set("fVarMax",30.0);
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
	groupRectangles(rects, 1, 1);

	//sort by area
	sort(rects.begin(), rects.end(), [](Rect a, Rect b){
			return b.area()< a.area();
	});
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
	double assumed_height= 1.8;
	double assumed_minheight=1;
	//what is the depth of tee?
	int z_tee=focal_length*assumed_height/player;
	//what is infered playerheight 2 meters behind tee?
	int player_min=focal_length*assumed_minheight/z_tee;

	vector<Rect>::iterator itc= input.begin();  
	   while (itc!=input.end()){  
		   //cout << "bottom is: " << (*itc).br().y << " and height: " << (*itc).height << endl; ;
		   if ((*itc).br().y > horizion){
			   if((*itc).height > player_min){
				   elligable.push_back((*itc));
			   }
		   }
		   ++itc;
		}  
	return elligable;
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

/*!
 * \perfrom one grabcut with given model, using mask or eval depeding on ratio of existing pixles in foreground
 */
Mat performGrabCut(bool usemask, Mat image, Mat mask, Mat mask_shadow, Mat bgModel, Mat grabCutMask, Mat fgModel, Rect ROI, Rect ROI_last, int itterataions, int itteration, int grabcut_rate, float recompute_ratio){
	//compute the ratio
	double ratio=0; 
	if (itteration!=0){
		Rect temp= ROI | ROI_last;
		ratio= (double) temp.area()/ROI.area();
	}

	//if use mask evaluate if reinitialize or not. 
	if (usemask){
		Mat tempMOG=getGrabCutMask(mask, mask_shadow, ROI);
		if (grabCutMask.channels()==3){
		cv::cvtColor(grabCutMask, grabCutMask, CV_BGR2GRAY);
		}

		tempMOG(ROI).copyTo(grabCutMask(ROI));

		if (itteration%grabcut_rate==0 || ratio<recompute_ratio){
			grabCut(image, grabCutMask, Rect(), bgModel, fgModel, itterataions, GC_INIT_WITH_MASK);
			std::cout << "renitialize model" << std::endl;
		}else{
			grabCut(image, grabCutMask, Rect(), bgModel, fgModel, itterataions, GC_EVAL);
		};
	}else{
		grabCut(image, grabCutMask, ROI, bgModel, fgModel, itterataions, GC_INIT_WITH_RECT); //if asked to do so by paramter use mask
	};
	return grabCutMask;
}

vector<Rect> HOGdetectors(Mat image){
	vector<cv::Rect> found;
	vector<cv::Rect> found2;
	HOGDescriptor hog;
	int stride=8;
	//Typical values for padding include (8, 8), (16, 16), (24, 24), and (32, 32).
	int padding=16;
	//Typical values for scale  are normally in the range [1.01, 1.5]. If you intend on running detectMultiScale  in real-time, this value should be as large as possible without significantly sacrificing detection accuracy.
	double pyr_scale=1.05;
	bool use_mean_shift=false;
	//A GPU example applying the HOG descriptor for people detection can be found at opencv_source_code/samples/gpu/hog.cpp
	//gpu::HOGDescriptor::getDefaultPeopleDetector
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	// 64x128 is default size for people. should not be smaller than that.
	hog.detectMultiScale(image, found, 0, cv::Size(stride,stride), cv::Size(padding,padding), pyr_scale, use_mean_shift);
	//hog.detectMultiScaleROI exist

	int loop=found.size();
	for (int j=0;loop>j;j++){
			//Rect temp=enlargeROI(image, found[j], 10, 50);
			//found2.push_back(temp);
			//found2.push_back(temp);
			found.push_back(found[j]);
			//rectangle(boxF, temp, Scalar(0,255,0));
		}

		groupRectangles(found, 1, 0.20);
		//sort by top left in x
		sort(found.begin(), found.end(), [](Rect a, Rect b){
			return b.tl().x < a.tl().x;
		});

		return found;
		//for (int u=0; ROIs.size() >u; u++){
		//	rectangle(boxF, tempROI[u], Scalar(255,0,0));
		//}
		//imshow("hog detectors", boxF);
}

KalmanFilter initKalman(float x, float y)
{
    KalmanFilter KF(4, 2, 0);
	KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);

	// init...
	KF.statePre.at<float>(0) = x;
	KF.statePre.at<float>(1) = y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	KF.statePost.at<float>(0) = x;
	KF.statePost.at<float>(1) = y;
	KF.statePost.at<float>(2) = 0;
	KF.statePost.at<float>(3) = 0;
		

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));
	return KF;

}

Point kalmanPredict(KalmanFilter KF) 
{
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
    return predictPt;
}

Point kalmanCorrect(KalmanFilter KF, float x, float y)
{
	Mat_<float> measurement(2,1); 
    measurement(0) = x;
    measurement(1) = y;
    Mat estimated = KF.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));
    return statePt;
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

	//downsample
	int downsample_MOG=4;
	int downsample_GRABCUT=4;

	//assumptionqs for distance estimate
	double f= 12;
	int hor=600/downsample_MOG;
	int play=1100/downsample_MOG;

	////MOG2
	Mat fgMaskMOG2; //fg-mask generated by MOG2 method  
	Mat fgMaskMOG2_shadow; //shadow-mask generated by MOG2 method 
	Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

		//params for initialization
		bool bShadowDetection=true;
		int history=1000; //divde by 25 for secounds of training. Standard seems to be 200 frames. 
		float Cthr=30.0;
		float learnrate= -1; //automagic is -1 -> WTF is fckning -0.5 from examples!?

		pMOG2 = new BackgroundSubtractorMOG2(history, Cthr,bShadowDetection);
		setMOG2(pMOG2);

	//FRAMES FOR PROGRAM
	Mat frame; //current frame  
	Mat resizeF; //current frame in MOG resolution
	Mat reconstructF; //current frame in GRABCUT resolution
	Mat Output; //the returned mask
	Mat boxF=frame.clone(); //draw stuff on the current frame

	//INITIALIZE OUTPUT/INPUT
	VideoCapture stream("Data/Input/GUSTAV0152.MOV"); //%TODO can be made into an argument for fuction 
	if(!stream.isOpened()){
		//return -1; //exit

	}
	VideoWriter outputVideo;   // Open the output
	outputVideo.open("Data/Results/video.avi", -1, 25, Size(480,270), true);
	if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write "<< endl;
    }

	//Grabcut
	PFS foregroundclass;
	vector<PFS> foregroundelems;
	Mat fgModel;
	Mat bgModel;

		//params
		int grabcut_itter= 2;
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

		//shift pixles
		//Mat frameCropped;
		//frameCropped = frame(Rect(2,2,frame.size().width-2, frame.size().height-2)).clone();

	   //downsample
	   resize(frame, resizeF, Size(frame.size().width/downsample_MOG, frame.size().height/downsample_MOG), INTER_AREA);  //%TODO is this right or should i create one myself
	   resize(frame, reconstructF,  Size(frame.size().width/downsample_GRABCUT, frame.size().height/downsample_GRABCUT), INTER_AREA);

	   ////apply BS
	   pMOG2->operator()(resizeF, fgMaskMOG2, learnrate);  

	   //MOG2separate mask by thresholding 
	   pair<Mat, Mat> Mask=separateMaskMOG2(fgMaskMOG2);
	   fgMaskMOG2= Mask.first;
	   fgMaskMOG2_shadow=Mask.second;
	   Mat pMog;
	   Mat p2Mog;
	   //apply postprocessing
	   morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_OPEN, element);   
	   morphologyEx(fgMaskMOG2, pMog, CV_MOP_DILATE, element4);   
	   morphologyEx(fgMaskMOG2, p2Mog, CV_MOP_CLOSE, element2);   


	   //morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_CLOSE, element);   
	   morphologyEx(fgMaskMOG2_shadow, fgMaskMOG2_shadow, CV_MOP_OPEN, element3);   

	   //get ROI
	   boxF=resizeF.clone();
	   vector<Rect> ROIs= getROIs(fgMaskMOG2, resizeF, 10, 20); //used to be 5 10
	   boxF=drawBoxes(ROIs, boxF, Scalar(255,0,0));
	   //please note that these paramters are approximate
	   ROIs=realDistanceEstimate(ROIs, resizeF, hor, play, f);
	   boxF=drawBoxes(ROIs, boxF, Scalar(0,0,255));
	   imshow("blue is removed by distance approx", boxF);


	   	Mat foregroundMask(reconstructF.size(), CV_8UC3, cv::Scalar(0)); //how to rescale this with image?
		//Mat foreground(reconstructF.size(), CV_8UC3, cv::Scalar(255,255,255));

		////sort foreground elements from left to right
		//sort(foregroundelems.begin(), foregroundelems.end(), [](PFS a, PFS b){
		//	return b.getROI().tl().x < a.getROI().tl().x;
		//});

		if (!ROIs.empty()){
			Mat use_mask;
			use_mask=getGrabCutMask2(fgMaskMOG2, pMog, p2Mog, fgMaskMOG2_shadow);
			imshow("input to GRABCUT",use_mask*255/3);
			grabCut(resizeF, use_mask, Rect(), bgModel, fgModel, 2, GC_INIT_WITH_MASK);
			use_mask = (use_mask == 1) | (use_mask == 3); //make mask binary

			foregroundMask=use_mask.clone();
		}
		//Mat use_mask;
		//Mat temp_mask;
		//Mat temp_GM;
		//Mat temp_black;

		//if (!ROIs.empty()){
		//   for (int i=0;ROIs.size()>i; i++){
		//	   



		//	   imshow("mask",temp_GM*255/3);

		//	   if (i==0){
		//			temp_mask=temp_GM;
		//		}else{
		//			temp_black=temp_GM.clone();
		//			temp_black.setTo(0);

		//			temp_GM(ROIs[i]).copyTo(temp_black(ROIs[i]));


		//			temp_mask=4*temp_mask + temp_black;

		//			//ensures between 0 and 3
		//			temp_mask.setTo(3, temp_mask > 10); 
		//			temp_mask.setTo(2, temp_mask == 10); 
		//			temp_mask.setTo(1, temp_mask == 9); 
		//			temp_mask.setTo(2, temp_mask == 8); 
		//			temp_mask.setTo(3, temp_mask == 7); 
		//			temp_mask.setTo(1, temp_mask > 3); 
		//		}
		//   }  	
		//	if (temp_mask.channels()==3){
		//		cv::cvtColor(temp_mask, temp_mask, CV_BGR2GRAY);
		//	}
		//	temp_mask.copyTo(use_mask);
		//	grabCut(resizeF, use_mask, Rect(), bgModel, fgModel, 2, GC_INIT_WITH_MASK);
		//	use_mask = (use_mask == 1) | (use_mask == 3); //make mask binary
		//	foregroundMask=use_mask.clone();
		//}else{
		//	foregroundMask.setTo(0);

		//}

		//if (!ROIs.empty()){
		//   for (int i=0;ROIs.size()>i; i++){
		//	   //if detected ROI but no models avalible, add a new one
		//	   if (foregroundelems.size()<(i+1)){
		//			PFS foregroundclass;
		//			foregroundelems.push_back(foregroundclass);
		//	   }
		//	   //grabcut on the model matching with the current mask in order left to right
		//	   Mat bgModel= foregroundelems[i].getBg();
		//	   Mat fgModel= foregroundelems[i].getFg();
		//	   Mat temp= Mat(reconstructF.size(), CV_8UC3, cv::Scalar(0));
		//	   foregroundelems[i].setMask(temp);
		//	   Mat grabCutMask= foregroundelems[i].getMask();
		//	   Rect ROI_last=foregroundelems[i].getlast_ROI();
		//	   grabCutMask=performGrabCut(true, reconstructF, fgMaskMOG2, fgMaskMOG2_shadow, bgModel, grabCutMask, fgModel, ROIs[i], ROI_last, grabcut_itter, fps_itter, grabcut_rate, recompute_ratio);
		//	   foregroundelems[i].update(fgModel, bgModel, grabCutMask, ROIs[i], ROIs[i]);
		//	}
		//};

	 //  //make one mask out of the foreground elements 
	 //  if (!foregroundelems.empty()){
		//   Mat Mask;
		//	for (int j=0;foregroundelems.size()>j;j++){
		//		Mask= foregroundelems[j].getMask();
		//		Mask = (Mask == 1) | (Mask == 3); //make mask binary
	 //           morphologyEx(Mask, Mask, CV_MOP_OPEN, element3);
		//		if (j==0){
		//			foregroundMask=Mask;
		//		}else{
		//		foregroundMask=foregroundMask | Mask;
		//		}
		//	}
	 //  }else{
		//	foregroundMask.setTo(0);
	 //  }

	   fps_itter++;

	   //show stuff
	   //imshow("Origin", reconstructF);  
	   //imshow("MOG2", fgMaskMOG2);  
	   imshow("grabCut (serveral elements)", foregroundMask);  
	   resize(foregroundMask, Output, Size(frame.size().width, frame.size().height), CV_INTER_LINEAR);
	   //reconstructF.copyTo(foreground, foregroundMask); //bg pixels are not copied
	   //imshow("foreground", foreground);
	   if ((fps_itter%random)==0){
	   //random= rand()%100+1;
			imwrite( "Data/Results/MOG2+GrabCut" + to_string(fps_itter) + "foregroundMask.jpg", foregroundMask);
			//imwrite( "Data/Results/MOG2+GrabCut" + to_string(fps_itter) + "foregroundMaskPROCESSED.jpg", fgMaskMOG2);
			imwrite( "Data/Results/MOG2+GrabCut" + to_string(fps_itter) + "image.jpg", resizeF);
			
	   }
	   outputVideo << foregroundMask;
	   //imshow("shadow", fgMaskMOG2_shadow);

	   delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
	   if(delta_ticks > 0)
       fps = CLOCKS_PER_SEC / delta_ticks;
	   fpsarray.push_back(fps);
	   cout << fps << endl;
	}
	
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