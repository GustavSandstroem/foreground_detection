#include <stdio.h>  
#include <iostream>  
#include <algorithm>
#include <ctime>

#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/video/background_segm.hpp>  
#include <opencv2/imgproc/imgproc.hpp>

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
Rect enlargeROI(Mat frm, Rect boundingBox, int padding) {
    Rect returnRect = Rect(boundingBox.x - padding, boundingBox.y - padding, boundingBox.width + (padding * 2), boundingBox.height + (padding * 2));
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
Rect getROI(Mat binary_input, Mat same_scale_image, int padding){
	Rect ROI;
	Mat ContourImg = binary_input.clone();  
	vector< vector< Point> > contours;  
	findContours(ContourImg, contours,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector< Rect > rects;  
	vector< vector< Point> >::iterator itc= contours.begin();  
	while (itc!=contours.end()){  
	Rect box= boundingRect(Mat(*itc));
			//add to vector
			rects.push_back(box);
			rects.push_back(box);
			++itc;
	}  
	//get rid of overlaping rectangels. 
	groupRectangles(rects, 1, 0.2);

	//sort by area
	sort(rects.begin(), rects.end(), [](Rect a, Rect b){
			return b.area()< a.area();
	});

	//if there is. append the largest. 
	if (rects.size()!=0){
		ROI= enlargeROI(same_scale_image, rects.front(), padding);
	}
	return ROI;
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
				invROI.setTo(2);
				threshold(Mog_mask, new_foreground, 127, 1, 0);
				threshold(Mog_mask_shadow, new_shadow, 127, 1, 0);
				return clonefgMaskMOG2-new_foreground + new_shadow;
}

/*!
 * \return the eucldian distance between two rect
 */
double boxCenterOffset(Rect one, Rect two){
	Point first(one.tl().x*(((1-one.br().x)/2)+1), one.br().y*(((1-one.tl().y)/2)+1));
	Point second(two.tl().x*(((1-two.br().x)/2)+1), two.tl().y*(((1-two.tl().y)/2)+1));
	return norm(first-second);
}

void MOG2(){
	//MOG2
	Mat fgMaskMOG2; //fg-mask generated by MOG2 method  
	Mat fgMaskMOG2_shadow; //shadow-mask generated by MOG2 method 
	Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

	//params for initialization
	bool bShadowDetection=true;
	int history=1000;
	float Cthr=16.0;
	float learnrate= -1; //automagic is -1 -> WTF is fckning -0.5 from examples!?

	pMOG2 = new BackgroundSubtractorMOG2(history, Cthr,bShadowDetection);
	setMOG2(pMOG2);

	//Ptr< BackgroundSubtractorGMG> pGMG; //MOG2 Background subtractor  
	//pGMG = new BackgroundSubtractorGMG();

	Mat frame; //current frame  
	Mat resizeF; //current frame in lower resolution for processing
	//Mat reconstructF; //current frame reconstructed in full resoltion %TODO
	VideoCapture stream("GUSTAV0152.MOV"); //%TODO can be made into an argument for fuction 
	if(!stream.isOpened()){
		//return -1; //exit
	}

	//Grabcut
	Mat foregroundMask(Size(frame.size().width/4, frame.size().height/4), CV_8UC3, cv::Scalar(0)); //how to rescale this with image?
	Mat bgModel;
	Mat fgModel;

	//params
	int boxthreshold=100;
	int itterarations= 1;
	int grabcut_rate=25;
	int recompute_area=10000;
	int recompute_distance=20000;

	//pramas morphological operations 
	Mat element = getKernel(2);
	Mat element2 = getKernel(6);
    Mat element3 = getKernel(1);

	clock_t current_ticks, delta_ticks;
	clock_t fps = 0;
	Rect last;

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
	   resize(frame, resizeF, Size(frame.size().width/4, frame.size().height/4), INTER_AREA);  //%TODO is this right or should i create one myself
	   //apply BS
	   pMOG2->operator()(resizeF, fgMaskMOG2, learnrate);  
	   //pGMG->operator()(resizeF, fgMaskMOG2);  

	   //%idea...if foreground is clean for 300 frames. lock background, until time or varaiance in image due to sun and shiet...

	   //MOG2separate mask by thresholding 
	   pair<Mat, Mat> Mask=separateMaskMOG2(fgMaskMOG2);
	   fgMaskMOG2= Mask.first;
	   fgMaskMOG2_shadow=Mask.second;

	   //apply postprocessing
	   morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_OPEN, element);   
	   morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_CLOSE, element);   
	   //erode(fgMaskMOG2, fgMaskMOG2, element3);
	   morphologyEx(fgMaskMOG2_shadow, fgMaskMOG2_shadow, CV_MOP_OPEN, element3);   

	   //get ROI
	   Rect ROI=getROI(fgMaskMOG2, resizeF, 20);

	   //%%%%%%%%%%%%%%%%   GRABCUT  %%%%%%%%%%%%%%%%%%%%%
	   Mat grabCutMask(resizeF.size(), CV_8UC1, cv::GC_BGD); //BGD
	   Mat foreground(resizeF.size(), CV_8UC3, cv::Scalar(255,255,255));

	   if (hasNonZero(fgMaskMOG2)){
		    Mat tempMOG=getGrabCutMask(fgMaskMOG2, fgMaskMOG2_shadow, ROI);
		    tempMOG(ROI).copyTo(grabCutMask(ROI));
		    //resize(grabCutMask, grabCutMask, Size(frame.size().width, frame.size().height), INTER_NEAREST); %todo, scale up
			bool change=false;
		    //show
		    //Mat grabCutShow=grabCutMask.clone();
		    //grabCutShow=(255/3)*grabCutShow;
		    //imshow("input to grabcut", grabCutShow);
			if (fps_itter!=0){
				if (abs(ROI.area()-last.area())>recompute_area){
					change=true;
					cout << "area " << abs(ROI.area()-last.area()) << endl;

				}
				if (boxCenterOffset(ROI, last)>recompute_distance){
					change=true;
					cout << "offset " << boxCenterOffset(ROI, last) << endl;
				}
			}
			last=ROI;
		    if (fps_itter%grabcut_rate==0 || change){
				grabCut(resizeF, grabCutMask, Rect(), bgModel, fgModel, itterarations, GC_INIT_WITH_MASK); //do the cut
				cout << "renitializing cut" << endl;
		    }else{
				grabCut(resizeF, grabCutMask, Rect(), bgModel, fgModel, itterarations, GC_EVAL); //do the cut
		    };
	   }else{
		    grabCutMask.setTo(0);
	   };
	   fps_itter++;

	   //show stuff
	   imshow("Origin", resizeF);  
	   imshow("MOG2", fgMaskMOG2);  
	   foregroundMask = (grabCutMask == 1) | (grabCutMask == 3); //make mask binary
	   morphologyEx(foregroundMask, foregroundMask, CV_MOP_OPEN, element3);   
	   imshow("grabCut", foregroundMask);  
	   resizeF.copyTo(foreground, foregroundMask); //bg pixels are not copied
	   imshow("foreground", foreground);
	   imshow("shadow", fgMaskMOG2_shadow);

	   delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
	   if(delta_ticks > 0)
       fps = CLOCKS_PER_SEC / delta_ticks;
	   cout << "fps: " << fps << endl;
	}
	return;
}  