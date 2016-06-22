
#include "stdafx.h"
#include <stdio.h>  
#include <iostream>  
#include <algorithm>
#include <ctime>
#include <numeric>
#include <fstream>
#include <stdlib.h> 
#include <cmath>  

#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/video/background_segm.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>

#define RWIDTH 480  
#define RHEIGHT 270 

#define OWIDTH 1920
#define OHEIGHT 1080

using namespace std;
using namespace cv;

class WatershedSegmenter{
private:
	cv::Mat markers;
public:
	void setMarkers(cv::Mat& markerImage)
	{
		markerImage.convertTo(markers, CV_32S);
	}

	cv::Mat process(cv::Mat &image)
	{
		cv::watershed(image, markers);
		markers.convertTo(markers, CV_8U);
		return markers;
	}
};

Rect enlargeROI(Mat frm, Rect boundingBox, int paddingx, int paddingy) {
	Rect returnRect = Rect(boundingBox.x - paddingx, boundingBox.y - paddingy, boundingBox.width + (paddingx * 2), boundingBox.height + (paddingy * 2));
	if (returnRect.x < 0)returnRect.x = 0;
	if (returnRect.y < 0)returnRect.y = 0;
	if (returnRect.x + returnRect.width >= frm.cols)returnRect.width = frm.cols - returnRect.x;
	if (returnRect.y + returnRect.height >= frm.rows)returnRect.height = frm.rows - returnRect.y;
	return returnRect;
}

vector<Rect> getROIs(Mat binary_input, Mat same_scale_image, int paddingx, int paddingy){
	if (binary_input.channels() == 3){
		cv::cvtColor(binary_input, binary_input, CV_BGR2GRAY);
	}
	Mat ContourImg = binary_input.clone();
	vector<vector<Point>> contours;
	findContours(ContourImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> rects;
	vector<vector<Point>>::iterator itc = contours.begin();
	while (itc != contours.end()){
		Rect box = boundingRect(Mat(*itc));
		box = enlargeROI(same_scale_image, box, paddingx, paddingy);
		rects.push_back(box);
		++itc;
	}
	return rects;
}

vector<Rect> DistanceEstimate(vector<Rect> input, Mat frame, int horizion, int threshold){
	vector<Rect> elligable;
	vector<Rect> horizion_mode;
	vector<Rect> remove;

	vector<Rect>::iterator itc = input.begin();
	while (itc != input.end()){
		if ((*itc).br().y > horizion){
			if ((*itc).height > threshold){
				elligable.push_back((*itc));
			}
			else{
				remove.push_back((*itc));
				horizion_mode.push_back((*itc));
			}
		}
		else{
			remove.push_back((*itc));
		}
		++itc;
	}
	//return horizion_mode;
	//return elligable;
	return remove;
}

double approxRollingAverage(double avg, double input, int itter) {
	double N = 50;
	if (N > itter){
		avg -= avg / N;
		avg += input / N;
	}
	else{
		avg -= avg / itter;
		avg += input / itter;
	}
	return avg;
}

/*!
* \separate foreground from shadow for MOG2
* \parm input is MOG2 mask
* \return is pair of foreground and shadow
* \this is the GPU version that works 
* \exactly the same but on the GPU. 
*/
pair<gpu::GpuMat, gpu::GpuMat> separateMaskMOG2_gpu(gpu::GpuMat input_mask){
	pair<gpu::GpuMat, gpu::GpuMat> par;
	gpu::GpuMat temp;
	gpu::GpuMat input_mask_shadow = input_mask.clone();
	gpu::threshold(input_mask_shadow, input_mask_shadow, 100, 255, 0);
	gpu::threshold(input_mask, input_mask, 150, 255, 0);
	gpu::bitwise_and(input_mask, input_mask_shadow, temp);
	gpu::subtract(input_mask_shadow, temp, input_mask_shadow);
	par.first = input_mask;
	par.second = input_mask_shadow;
	return par;
}

/*!
* \brief Get a odd getStructuringElement from
\param int size
*/
Mat getKernel(int morph_size){
	return getStructuringElement(MORPH_RECT, Size(morph_size * 2 + 1, morph_size * 2 + 1), Point(-1, -1));
}

Mat drawBoxes(vector<Rect> boxes, Mat image, Scalar colour){
	vector<Rect>::iterator itc = boxes.begin();
	while (itc != boxes.end()){
		rectangle(image, (*itc), colour);
		++itc;
	}
	return image;
}

void mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes)
{
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Mask of original image
	cv::Size scaleFactor(0, 0); // To expand rectangles, i.e. increase sensitivity to nearby rectangles. Doesn't have to be (10,10)--can be anything
	for (int i = 0; i < inputBoxes.size(); i++)
	{
		cv::Rect box = inputBoxes.at(i) + scaleFactor;
		cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED); // Draw filled bounding boxes on mask
	}

	std::vector<std::vector<cv::Point>> contours;
	// Find contours in mask
	// If bounding boxes overlap, they will be joined by this function call
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int j = 0; j < contours.size(); j++)
	{
		outputBoxes.push_back(cv::boundingRect(contours.at(j)));
	}
}

Mat fadeIn(Mat img1, Mat img2, int len){
Mat dst;
	for (int i = 0; i < len; i++){
		float fadein = i / len;
		cv::addWeighted(img1, 1 - fadein, img2, fadein, 0, dst, -1);
	}
return dst;
}

void AlphaBlend(const Mat& imgFore, Mat& imgDst, const Mat& alpha)
{
	vector<Mat> vAlpha;
	Mat imgAlpha3;
	for (int i = 0; i < 3; i++) 
		vAlpha.push_back(alpha);
	merge(vAlpha, imgAlpha3);

	Mat blend = imgFore.mul(imgAlpha3, 1.0 / 255) + imgDst.mul(Scalar::all(255) - imgAlpha3, 1.0 / 255);
	blend.copyTo(imgDst);
}

void setMOG(gpu::MOG2_GPU & pMOG2_g){
	pMOG2_g.history = 500;
	pMOG2_g.varThreshold = 15;
	pMOG2_g.bShadowDetection = true;
	pMOG2_g.backgroundRatio = 0.5; // 0.9 //0.7 
	pMOG2_g.varThreshold = 3;
	pMOG2_g.varThresholdGen = 9.0;
	pMOG2_g.fTau = 0.7;
	pMOG2_g.fCT = 0.05;
	pMOG2_g.fVarInit = 15.0;
	pMOG2_g.fVarMin = 0.0;
	pMOG2_g.fVarMax = 15.0;
}

Mat getGrabCutMask2(Mat Mog_mask, Mat Mog_padd, Mat Mog_close, Mat Mog_mask_shadow){
	Mat PB_BG;
	Mat FG;
	Mat PB_FG;
	Mat shadow;
	Mat fill;
	Mat returnmask;

	fill = Mog_mask | Mog_close;
	threshold(Mog_mask, FG, 127, 1, 0);
	threshold(Mog_padd, PB_BG, 127, 2, 0);
	threshold(fill, fill, 127, 1, 0);
	threshold(Mog_mask_shadow, shadow, 127, 1, 0);

	fill = fill - Mog_mask;
	fill = fill - shadow;
	fill.setTo(0, fill == -1);

	returnmask = PB_BG - FG + fill;



	//double min, max;
	//minMaxLoc(returnmask, &min, &max);
	//cout << "min max" << min << " " << max << endl;
	return returnmask;
}

int main(){	
	////////////////////////////////////////////////////////////////////////
	//functionallity. BOOL. 1 is use. 0 is disable
	bool use_pyrDown = 0;
	bool use_watershead = 0;
	bool use_grabcut = 0;
	bool watershead_native_size = 0;
	bool remove_shadow = 1;
	bool post_open = 1;
	bool use_depth = 1;
	bool use_superres = 0;
	bool use_robust = 0;
	bool lightplot = 0;
	bool normalize_light = 0;
	bool remove_grass = 0;
	bool record = 0;
	bool printout_status = 1;
	bool show_origin = 1;
	bool show_result = 0;
	bool show_markers = 0;

	bool save_result = 0;
	bool save_bg = 0;
	bool save_fg = 0;
	bool tv_average_smooth = 0;
	bool demo = 0;
	bool not_golf = 0;
	bool show_downscaled = 1;



	//152
	double f = 25;
	int hor = 600 / 4;
	int play = 1100 / 4;
	int balldist = 3;
	int camera_height = 1;
	int ball_h = 900;

	//video 142
	//hor = 180 ;
	//play = 600/4;

	//video 0002
	//hor = 150*4;
	//play = 170*4;

	//demo
	hor = 590;
	play = 500;



	//DISTANCE ESTIMATED
	double assumed_height = 1.7;
	double to_pixels = 1080 / 5.6;
	double focal_length = (play*balldist) / assumed_height;

	//thres
	double assumed_minheight = 1;
	int player_min = focal_length*assumed_minheight / (balldist);
	int thres = focal_length*assumed_minheight / (balldist + 2);

	/////////////////////////////////////////////////////////////////////////
	//verify that CUDA works. 
	cout<< "Detected number of CUDA GPUs is " << cv::gpu::getCudaEnabledDeviceCount();
	gpu::DeviceInfo info;
	cout << " of kind " << info.name() << endl;
	cout << "Utilizing CUDA version " << info.majorVersion() << "." << info.minorVersion();
	if (info.isCompatible()){
		cout << " and is therefore compatible with this program! " << endl;
		//these stumps have to be updated if new functionallity that requires CUDA 1.1 or higer is added!
		//then use gpu::DeviceInfo::supports( specify needed version 1.1 1.2, 2.0, 2.1 etc.)
	}
	else{
		cout << " and therefore INcompatible with this program! " << endl;
		cout << "PROGRAM IS ABORTED!" << endl;
		return -1;
	}

	Mat trace = cv::imread("Data/trace.png", CV_LOAD_IMAGE_GRAYSCALE);
	threshold(trace, trace, 254, 255, THRESH_BINARY_INV);
	trace.convertTo(trace, CV_8U);
	//imshow("trace", trace);

	/////////////////////////////////////////////////////////////////////////
	//MOG2
	gpu::MOG2_GPU pMOG2_g(3);
	setMOG(pMOG2_g);

	gpu::GpuMat Mog_Mask_g;
	Mat Mog_Mask;
	Mat Mog_Mask_padded;
	Mat Mog_Mask_extrapadded;


	Mat Mog_Mask_shadow;


	gpu::MOG2_GPU pMOG2_g2(3);
	setMOG(pMOG2_g2);
	gpu::GpuMat Mog_Mask_g2;
	Rect print = Rect(2, 2, 958, 538);
	gpu::GpuMat full;
	gpu::GpuMat fulls;


	/////////////////////////////////////////////////////////////////////////  
	//Intialize videocapture and read imput!
	//device – id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0.
	VideoCapture cap("Data/output.mov");
	if (!cap.isOpened()){
		cout << "INPUT video not opened" << endl;
		return -1;	
	}

	VideoWriter outputVideo;   // Open the output. set 2nd arguemnt to -1 for popup select of codec. 
	//outputVideo.open("video.avi", -1, 25, Size(OWIDTH, OHEIGHT), true);
	Size sizen = Size(OWIDTH, OHEIGHT);
	
	//outputVideo.open("video.avi", CV_FOURCC('I', 'Y', 'U', 'V'), 25, sizen, true);
	//outputVideo.open("Data/new/video.avi", CV_FOURCC('I', 'Y', 'U', 'V'), 25, sizen, true);
	outputVideo.open("Data/resultatfilm.avi", -1, 29, sizen, true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write " << endl;
		return -1;
	}

	/////////////////////////////////////////////////////////////////////////
	Mat o_frame;
	Mat r_frame;
	Mat o_frame_cropped;
	Mat r_frame_cropped;

	Mat result = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);
	Mat markers;
	Mat first = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);
	Mat secound = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);
	Mat third = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);

	//gpu::GpuMat o_frame_gpu;
	gpu::GpuMat r_frame_gpu;

	//gpu::GpuMat o_frame_gpu_cropped;
	gpu::GpuMat r_frame_gpu_cropped;

	//Grabcut
	Mat fgModel;
	Mat bgModel;
	
	ofstream myfile;
	myfile.open("example.txt");

	/////////////////////////////////////////////////////////////////////////  
	cap >> o_frame;
	if (o_frame.empty())
		return 0;

	/////////////////////////////////////////////////////////////////////////  
	unsigned long AAtime = 0, BBtime = 0;
	int itter = 0;
	double fps;
	while (1){
		//start clock
		AAtime = getTickCount();

		//get frame, brake if empty. TODO This should be checked for live video! (make buffer) 
		cap >> o_frame;
		if (o_frame.empty())
			return 0;


		//Resize. It quicker to downsample on GPU and transfer small version to GPU than to vice versa.
		//INTER_AREA will merge many pixles to one. WITh RWIDTH and RHEIGHT => 4x downscale. Prefeable for MOG2 
		//according to reserach by Gustav Sandström Exjobb Protracer 2016. 
		if (use_pyrDown){
		pyrDown(o_frame, r_frame, Size(o_frame.cols / 2, o_frame.rows / 2));
		pyrDown(r_frame, r_frame, Size(r_frame.cols / 2, r_frame.rows / 2));
		}else{
			resize(o_frame, r_frame, Size(RWIDTH, RHEIGHT), INTER_NEAREST);
		}

		if (normalize_light){
			cv::cvtColor(r_frame, r_frame, CV_BGR2Lab);

			// Extract the L channel
			std::vector<cv::Mat> lab_planes(3);
			cv::split(r_frame, lab_planes);  // now we have the L image in lab_planes[0]

			// apply the CLAHE algorithm to the L channel
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
			clahe->setClipLimit(4);
			cv::Mat dst;
			clahe->apply(lab_planes[0], dst);

			// Merge the the color planes back into an Lab image
			dst.copyTo(lab_planes[0]);
			cv::merge(lab_planes, r_frame);

			// convert back to RGB
			cv::cvtColor(r_frame, r_frame, CV_Lab2BGR);
			imshow("normalize colour", r_frame);
		}

		r_frame_gpu.upload(r_frame);

		if (use_superres){
			o_frame_cropped = o_frame(Rect(2, 2, 1918, 1078)).clone();
			resize(o_frame_cropped, r_frame_cropped, Size(478, 268), INTER_NEAREST);
			r_frame_gpu_cropped.upload(r_frame_cropped);
		}
		int meanet;
		cv::Scalar menet;
		if (lightplot){
			Mat r_frame_hsv;
			cvtColor(r_frame, r_frame_hsv, COLOR_BGR2HSV);
			menet=mean(r_frame_hsv);
			//meanet = menet.val[0];
			//myfile << to_string(meanet);
			//myfile << "\n";

			//meanet = menet.val[1];
			//myfile << to_string(meanet);
			//myfile << "\n";

			meanet = menet.val[2];
			myfile << to_string(meanet);
			myfile << "\n";

			//meanet = menet.val[3];
			//myfile << to_string(meanet);
			//myfile << "\n";
		}




		//use MOG2 and separate shadows and foreground. Shadows are tracked by MOG2 with fTau as threshold. Adjust if too much or too little is detected. 
		pMOG2_g.operator()(r_frame_gpu, Mog_Mask_g, -1);
		pair<gpu::GpuMat, gpu::GpuMat> Mask = separateMaskMOG2_gpu(Mog_Mask_g);

		//use superresolution. MOG is run twice, with differnt downsamples. result is either joined with bitwise_and or bit_wise_or depending on "robustness_mode". And is robust. 
		if (use_superres){
				//run the secound instance of MOG
				pMOG2_g2.operator()(r_frame_gpu_cropped, Mog_Mask_g2, -1);
				pair<gpu::GpuMat, gpu::GpuMat> Mask2 = separateMaskMOG2_gpu(Mog_Mask_g2);

				//958x538
				gpu::GpuMat tempen;
				gpu::GpuMat tempen2;
				gpu::resize(Mask2.first, tempen, Size(958, 538), INTER_NEAREST);
				gpu::resize(Mask2.second, tempen2, Size(958, 538), INTER_NEAREST);

				//960x540
				gpu::GpuMat tempen3;
				gpu::GpuMat tempen4;
				gpu::resize(Mask.first, tempen3, Size(960, 540), INTER_NEAREST);
				gpu::resize(Mask.second, tempen4, Size(960, 540), INTER_NEAREST);

				full = (tempen3).clone();
				full.setTo(0);
				fulls = (tempen4).clone();
				fulls.setTo(0);

				//960x540
				(tempen).copyTo(full(print));
				(tempen2).copyTo(fulls(print));
				
				gpu::GpuMat temp5;
				gpu::GpuMat temp6;
				if (use_robust){
					bitwise_and(tempen4, fulls, temp6);
					bitwise_and(tempen3, full, temp5);

				} 
				else{
					bitwise_or(tempen3, full, temp5);
					bitwise_or(tempen4, fulls, temp6);
				}
				//480x270
				resize(temp5, Mask.first, Size(480, 270), INTER_NEAREST);
				resize(temp6, Mask.second, Size(480, 270), INTER_NEAREST);
		}

		//apply postprocessing
		gpu::morphologyEx(Mask.first, Mask.first, CV_MOP_OPEN, getKernel(3));
		gpu::morphologyEx(Mask.second, Mask.second, CV_MOP_OPEN, getKernel(2));

		//watershead	
		if (use_watershead){
			// Eliminate noise and smaller objects
			cv::gpu::GpuMat fg;
			gpu::erode(Mask.first, fg, cv::Mat(), cv::Point(-1, -1), 2); //3

			// Identify image pixels without objects
			gpu::GpuMat bg;
			gpu::dilate(Mask.first, bg, cv::Mat(), cv::Point(-1, -1), 4);
			gpu::threshold(Mask.second, Mask.second, 1, 128, cv::THRESH_BINARY_INV);
			gpu::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

			if (remove_shadow)
				gpu::bitwise_or(bg, Mask.second, bg);

			// Create markers image
			gpu::GpuMat markers_g;
			gpu::add(fg, bg, markers_g);

			if (watershead_native_size){
				gpu::GpuMat temptemp;
				gpu::resize(markers_g, temptemp, o_frame.size(), INTER_NEAREST); // on GPU you may use something else than LINEAR here...cubic for example
				temptemp.download(markers);

			}else{
			markers_g.download(markers);
			}

			//watershed segmentation
			WatershedSegmenter segmenter;
			segmenter.setMarkers(markers);

			if (watershead_native_size){
				result = segmenter.process(o_frame);
			}
			else{
				result = segmenter.process(r_frame);
				resize(result, result, o_frame.size(), INTER_NEAREST); //ON CPU you may use something else than LINEAR here...cubic for example

			}
			result.convertTo(result, CV_8UC1);
			if (show_result)
				imshow("final_result", result);
		}
		else if (use_grabcut){
			//grabcut
			Mask.first.download(Mog_Mask);
			gpu::GpuMat one;
			gpu::GpuMat two;

			gpu::morphologyEx(Mask.first, one, CV_MOP_DILATE, getKernel(10));
			one.download(Mog_Mask_padded);
			gpu::morphologyEx(Mask.first, two , CV_MOP_CLOSE, getKernel(10));
			two.download(Mog_Mask_extrapadded);
			Mask.second.download(Mog_Mask_shadow);

			Mat use_mask = getGrabCutMask2(Mog_Mask, Mog_Mask_padded, Mog_Mask_extrapadded, Mog_Mask_shadow);
			//imshow("input to GRABCUT",use_mask*255/3);
			if (countNonZero(use_mask)>(thres*thres*0.4)){
				//cout << r_frame.size() << endl;
				//cout << use_mask.size() << endl;

				grabCut(r_frame, use_mask, Rect(), bgModel, fgModel, 1, GC_INIT_WITH_MASK);
				//cout << "im here 2 " << endl;
				use_mask = (use_mask == 1) | (use_mask == 3); //make mask binary
				resize(use_mask, use_mask, o_frame.size(), INTER_NEAREST);
				result = use_mask.clone();
			}
		
		}else{
			//pure MOG2 output, discarded shadows as foreground. 
			Mask.first.download(Mog_Mask);
			resize(Mog_Mask, Mog_Mask, o_frame.size(), INTER_NEAREST);
			result = Mog_Mask;
			result.convertTo(result, CV_8UC1);
			if (show_result)
				imshow("final_result", result);
		}

		if (remove_grass){
			Mat grass;
			int sensitivity = 30;
			cv::cvtColor(o_frame, grass, CV_BGR2HSV);
			inRange(grass, Scalar(60 - sensitivity, 100, 50), Scalar(60 + sensitivity, 255, 255), grass);
			//imshow("grass", grass);
			result = result - grass;
			result.setTo(0, result < 0);
		}

		if (post_open){
			morphologyEx(result, result, CV_MOP_OPEN, getKernel(2));
			if (show_result)
				imshow("final_result", result);
		}

		if (use_depth){
			threshold(result, result, 155, 255, THRESH_BINARY);
			vector<Rect> ROI = getROIs(result, o_frame, 0, 0);
			vector<Rect> remove = DistanceEstimate(ROI, result, hor, thres);
			for (int u = 0; u < remove.size(); u++){
				result(remove[u]).setTo(0);
			}
			if (show_result)
				imshow("final_result", result);
		}


		if (save_fg){
			Mat temp;
			o_frame.copyTo(temp, result);
			outputVideo << temp;
			if (show_result)
				imshow("final_result", temp);
		}

		if (save_bg){
			Mat temp = Mat::zeros(1920, 1080, CV_8U);
			threshold(result, result, 130, 1, THRESH_BINARY_INV);
			o_frame.copyTo(temp, result);
			outputVideo << temp;
			if (show_result)
				imshow("final_result", temp);
		}


		BBtime = getTickCount();
		float pt = (BBtime - AAtime) / getTickFrequency();
		float fpt = 1 / pt;
		fps=approxRollingAverage(fps, fpt, itter);

		if (itter>5 && tv_average_smooth){ //dont save foreground or baclkgtound
			Mat temp;
			if (countNonZero(secound) > 0){
			third = secound;
			secound = first;
			first = result;
			bitwise_or(first, secound, temp);
			bitwise_or(temp, third, third);
			dilate(third, third, cv::Mat(), cv::Point(-1, -1), 4);
			//GaussianBlur(third, third, Size(21, 21), 11.0);
			result = third;

			}
			else if (countNonZero(first) > 0){
				secound = first;
				first = result;
			}
			else{
				first = result;
			}
			if (show_result)
				imshow("final_result", result);

		}

		//demo
		if (demo){
			Mat bg, fg, not_result, output;
			o_frame.copyTo(bg);			
			//line(bg, Point(0, 0), Point(RWIDTH, RHEIGHT), Scalar(255, 0, 0), 5);
			bg.setTo(cv::Scalar(204, 0, 40), trace);
			//bg = bg + trace;
			bitwise_not(result, not_result);
			o_frame.copyTo(fg, result);

			//bg.setTo(0, result);
			//fg.setTo(0, not_result);
			//bitwise_not(result, result);
			//result = fadeIn(fg, bg, 10);
			//r_frame.copyTo(bg, result);
			//bg.copyTo(r_frame, result);
			AlphaBlend(fg, bg, result);
			//result = r_frame;
			//result = bg;
			outputVideo << bg;
			if (show_result)
				imshow("final_result", bg);
		}

		//write to video
		if (record && (itter >5) && !demo && !save_bg && !save_fg) //if (record && itter> 500)
			outputVideo << result;
			//outputVideo.write(result);

		if (((itter % 100) == 0 && save_result) || not_golf){
			if (not_golf)
				resize(result, result, Size(320, 240));
			threshold(result, result, 200, 255, THRESH_BINARY);
			imwrite("Data/new/" + to_string(itter) + ".jpg", result);
			//imwrite("Data/new/" + to_string(itter) + "i.jpg", o_frame);


		}

		if (show_downscaled){
			Mat temp;
			resize(result, temp, Size(RWIDTH, RHEIGHT), INTER_NEAREST);
			imshow("small results", temp);

		}

		if (printout_status && itter % 50 == 0){
			printf("Current fps is %.4lf, this is not accurate if writing to file \n", fps);
		}

		if (show_origin){
			imshow("origin", r_frame);
		}

		itter++;

		if (waitKey(10) > 0)
			break;
		}
	//this should be in some other places also. lets make a function. TODO
	outputVideo.release();
	cap.release();
	return 0;
}

