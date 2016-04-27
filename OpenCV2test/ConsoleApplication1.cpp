
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


int main(){	
	////////////////////////////////////////////////////////////////////////
	//functionallity. BOOL. 1 is use. 0 is disable
	bool rescale_gpu = 0;
	bool use_blur = 0;
	bool use_watershead = 1;
	bool watershead_native_size = 0;
	bool record = 1;
	bool printout_status = 1;
	bool show_origin = 0;
	bool show_result = 1;
	bool save_result = 1;
	bool show_markers = 1;
	bool save_bg = 0;
	bool save_fg = 0;
	bool save_lowres = 1; //TODO might be a problem with framedrop.
	bool remove_shadow = 1;
	bool post_open = 1;
	bool tv_average_smooth = 1;
	bool demo = 1;
	bool use_depth = 1;
	//bool normalize_light = 0;

	//152
	double f = 25;
	int hor = 600 / 4;
	int play = 1100 / 4;
	int balldist = 3;
	int camera_height = 1;

	//video 142
	hor = 180;
	play = 600/4;

	//video 0002
	hor = 150;
	play = 170;


	//DISTANCE ESTIMATED
	//get rotation
	double assumed_height = 1.7;
	double to_pixels = 1080 / 5.6;
	double focal_length = (play*balldist) / assumed_height;
	cout << "assumed focal-length [mm]" << focal_length*to_pixels << endl;
	int diff = (RWIDTH / 2 - hor);
	double alpha = std::atan(diff/focal_length);

	//make matrix
	Mat transformation = Mat::zeros(3, 4, CV_8UC1);
	//rot
	transformation.at<double>(1,1) = 1;
	transformation.at<double>(2,2)= std::cos(alpha);
	transformation.at<double>(2,3)=-std::sin(alpha);
	transformation.at<double>(3,2)= std::sin(alpha);
	transformation.at<double>(3,3)= std::cos(alpha);
	//translate
	transformation.at<double>(3,4) = -camera_height;

	//thres
	double assumed_minheight = 1;

	//what is infered playerheight 2 meters behind tee?
	int player_min = focal_length*assumed_minheight / (balldist + 2);
	cout << "thres " << player_min << endl;
	int thres = 100;

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

	/////////////////////////////////////////////////////////////////////////
	//MOG2
	gpu::MOG2_GPU pMOG2_g(3);
	pMOG2_g.history = 1000; 
	pMOG2_g.varThreshold = 15; 
	pMOG2_g.bShadowDetection = true;
	pMOG2_g.backgroundRatio = 0.4; // 0.9 //0.7 
	pMOG2_g.varThreshold = 3;
	pMOG2_g.varThresholdGen = 9.0;
	pMOG2_g.fTau = 0.5;
	pMOG2_g.fCT = 0.05;
	pMOG2_g.fVarInit = 15.0;
	pMOG2_g.fVarMin = 0.0;
	pMOG2_g.fVarMax = 15.0;

	gpu::GpuMat Mog_Mask_g;
	Mat Mog_Mask;
	Mat Mog_Mask_shadow;

	/////////////////////////////////////////////////////////////////////////  
	//Intialize videocapture and read imput!
	//device – id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0.

	VideoCapture cap("Data/syn_0002.avi");
	if (!cap.isOpened()){
		cout << "video not opened" << endl;
		return -1;
	}
	VideoWriter outputVideo;   // Open the output. set 2nd arguemnt to -1 for popup select of codec. 
	//outputVideo.open("video.avi", -1, 25, Size(OWIDTH, OHEIGHT), true);
	Size sizen = Size(RWIDTH, RHEIGHT);
	if (!save_lowres){
		sizen = Size(OWIDTH, OHEIGHT);
	}
	//outputVideo.open("video.avi", CV_FOURCC('I', 'Y', 'U', 'V'), 25, sizen, true);
	outputVideo.open("Data/Results/video.avi", CV_FOURCC('I', 'Y', 'U', 'V'), 25, sizen, true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write " << endl;
		return -1;
	}

	/////////////////////////////////////////////////////////////////////////
	Mat o_frame;
	Mat r_frame;
	Mat result = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);
	Mat markers;
	Mat first = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);
	Mat secound = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);
	Mat third = Mat::zeros(RHEIGHT, RWIDTH, CV_8UC1);

	gpu::GpuMat o_frame_gpu;
	gpu::GpuMat r_frame_gpu;
	gpu::GpuMat rg_frame_gpu;
	gpu::GpuMat r_frame_blur_gpu;

	/////////////////////////////////////////////////////////////////////////  
	cap >> o_frame;
	if (o_frame.empty())
		return 0;
	vector< gpu::GpuMat> gpurgb(3);
	vector< gpu::GpuMat> gpurgb2(3);

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

		//Resize. 
		//on 4096×3072.jpg, 2,3 MB by 4x downscale excluded upload time CPU in average ~6.5 ms, the GPU ~1.65 ms. 
		//use rescale_gpu==0 if you want to decrease CPU load as a tradeoff for PCI-E bandwidth to GPU. 
		//INTER_AREA will merge many pixles to one. WITh RWIDTH and RHEIGHT => 4x downscale. Prefeable for MOG2 
		//according to reserach by Gustav Sandström Exjobb Protracer 2016. 
		if (rescale_gpu){
			o_frame_gpu.upload(o_frame);
			gpu::resize(o_frame_gpu, r_frame_gpu, Size(RWIDTH, RHEIGHT), INTER_AREA);
			r_frame_gpu.download(r_frame);
		}
		else{
			resize(o_frame, r_frame, Size(RWIDTH, RHEIGHT), INTER_AREA);
			r_frame_gpu.upload(r_frame);
		}

		////normalize_light ON GPU
		//if (normalize_light){
		//	gpu::cvtColor(r_frame_gpu, r_frame_gpu, CV_BGR2Lab);
		//	// Extract the L channel
		//	std::vector<cv::gpu::GpuMat> lab_planes(3);

		//	cv::gpu::split(r_frame_gpu, lab_planes);  // now we have the L image in lab_planes[0]

		//	// apply the CLAHE algorithm to the L channel
		//	cv::Ptr<cv::gpu::CLAHE> clahe = cv::gpu::createCLAHE();

		//	clahe->setClipLimit(4);

		//	cv::gpu::GpuMat dst;

		//	clahe->apply(lab_planes[0], dst);

		//	// Merge the the color planes back into an Lab image
		//	dst.copyTo(lab_planes[0]);

		//	cv::gpu::merge(lab_planes, r_frame_gpu);

		//	// convert back to RGB
		//	cv::gpu::cvtColor(r_frame_gpu, r_frame_gpu, CV_Lab2BGR);
		//	r_frame_gpu.download(r_frame);
		//	imshow("normalize colour", r_frame);
		//}

		//Preprocess with Gaussian blur per colour channel. ON GPU. 
		if (use_blur){
			gpu::split(r_frame_gpu, gpurgb);
			gpu::blur(gpurgb[0], gpurgb2[0], Size(3, 3));
			gpu::blur(gpurgb[1], gpurgb2[1], Size(3, 3));
			gpu::blur(gpurgb[2], gpurgb2[2], Size(3, 3));
			gpu::merge(gpurgb2, r_frame_gpu);
		}

		//use MOG2 and separate shadows and foreground. Shadows are tracked by MOG2 with fTau as threshold. Adjust if too much or too little is detected. 
		pMOG2_g.operator()(r_frame_gpu, Mog_Mask_g, -1);
		pair<gpu::GpuMat, gpu::GpuMat> Mask = separateMaskMOG2_gpu(Mog_Mask_g);

		//apply postprocessing
		gpu::morphologyEx(Mask.first, Mask.first, CV_MOP_OPEN, getKernel(2));
		gpu::morphologyEx(Mask.second, Mask.second, CV_MOP_OPEN, getKernel(1));
		Mask.first.download(Mog_Mask);
		Mask.second.download(Mog_Mask_shadow);

		if (use_watershead){
			if (!watershead_native_size){
				// Eliminate noise and smaller objects
				cv::Mat fg;
				cv::erode(Mog_Mask, fg, cv::Mat(), cv::Point(-1, -1), 4); //3

				// Identify image pixels without objects
				cv::Mat bg;
				cv::dilate(Mog_Mask, bg, cv::Mat(), cv::Point(-1, -1), 4);
				cv::threshold(Mog_Mask_shadow, Mog_Mask_shadow, 1, 128, cv::THRESH_BINARY_INV);
				cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);
				if (remove_shadow)
					cv::bitwise_and(bg, Mog_Mask_shadow, bg);

				// Create markers image
				markers = (Mog_Mask.size(), CV_8U, cv::Scalar(0));
				markers = fg + bg;
				if (show_markers)
					imshow("markers", markers);
			}
			else{
				// Eliminate noise and smaller objects
				cv::gpu::GpuMat fg;
				gpu::erode(Mask.first, fg, cv::Mat(), cv::Point(-1, -1), 3);

				// Identify image pixels without objects
				gpu::GpuMat bg;
				gpu::dilate(Mask.first, bg, cv::Mat(), cv::Point(-1, -1), 4);
				gpu::threshold(Mask.second, Mask.second, 1, 128, cv::THRESH_BINARY_INV);
				gpu::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);
				gpu::bitwise_and(bg, Mask.second, bg);

				// Create markers image
				gpu::GpuMat markers_g;
				Mat markers;
				gpu::add(fg, bg, markers_g);
				if (!save_lowres)
					gpu::resize(markers_g, markers_g, o_frame.size(), INTER_LINEAR); // on GPU you may use something else than LINEAR here...cubic for example
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
				if (!save_lowres)
					resize(result, result, o_frame.size(), INTER_LINEAR); //ON CPU you may use something else than LINEAR here...cubic for example

			}
			result.convertTo(result, CV_8UC1);
			if (show_result)
				imshow("final_result", result);
		}
		else
		{
			//pure MOG2 output, discarded shadows as foreground. 
			if (!save_lowres)
				resize(Mog_Mask, Mog_Mask, o_frame.size(), INTER_LINEAR);
			result = Mog_Mask;
			result.convertTo(result, CV_8UC1);
			if (show_result)
				imshow("final_result", result);
		}

		if (post_open){
			morphologyEx(result, result, CV_MOP_OPEN, getKernel(2));
			if (show_result)
				imshow("final_result", result);
		}

		if (save_fg){
			threshold(result, result, 130, 1, THRESH_BINARY);
			if (!save_lowres){
				o_frame.copyTo(result, result);
			}
			else{
				r_frame.copyTo(result, result);
			}
			if (show_result)
				imshow("final_result", result);
		}

		if (save_bg){
			threshold(result, result, 130, 1, THRESH_BINARY_INV);
			if (!save_lowres){
				o_frame.copyTo(result, result);
			}
			else{
				r_frame.copyTo(result, result);
			}
			if (show_result)
				imshow("final_result", result);
		}


		BBtime = getTickCount();
		float pt = (BBtime - AAtime) / getTickFrequency();
		float fpt = 1 / pt;
		fps=approxRollingAverage(fps, fpt, itter);

		if (itter>500 && tv_average_smooth){ //dont save foreground or baclkgtound
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

		if (use_depth){
			//Mat temp = Mat::zeros(result.size(), CV_8UC1);
			//temp += result;
			threshold(result, result, 155, 255, THRESH_BINARY);
			vector<Rect> ROI = getROIs(result, r_frame, 0, 0);
			vector<Rect> remove = DistanceEstimate(ROI, result, hor, thres);
			for (int u = 0; u < remove.size(); u++){
				result(remove[u]).setTo(0);
			}
			if (show_result)
				imshow("final_result", result);
		}

		//demo
		if (demo){
			Mat bg, fg, not_result, output;
			r_frame.copyTo(bg);			
			line(bg, Point(0, 0), Point(RWIDTH, RHEIGHT), Scalar(255, 0, 0), 5);

			bitwise_not(result, not_result);
			r_frame.copyTo(fg, result);

			//bg.setTo(0, result);
			//fg.setTo(0, not_result);
			//bitwise_not(result, result);
			//result = fadeIn(fg, bg, 10);
			//r_frame.copyTo(bg, result);
			//bg.copyTo(r_frame, result);
			AlphaBlend(fg, bg, result);
			//result = r_frame;
			//result = bg;
			if (show_result)
				imshow("final_result", bg);
		}

		//write to video
		if (record && itter> 500)
			outputVideo << result;
			//outputVideo.write(result);

		if (((itter%100) == 0 && save_result)){
			imwrite("Data/Results/" + to_string(itter) + ".jpg", result);

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

