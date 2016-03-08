//Header ==> Function Declarations

#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/video/background_segm.hpp>  
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;  
using namespace std;  

#ifndef PFS_H
#define PFS_H

class PFS{
public:
	//Defult Constructor
	PFS();

	//Overload Constructor
	PFS(Mat, Mat, Mat, Rect, Rect);

	//Accessor functions
	void update(Mat, Mat, Mat, Rect, Rect);
	void setFg(Mat);
	void setBg(Mat);
	void setMask(Mat);
	void setROI(Rect);
	void setlast_ROI(Rect);

	Mat getFg() const;
	Mat getBg() const;
	Mat getMask() const;
	Rect getROI() const;
	Rect getlast_ROI() const;
	
	//Destructor
	~PFS();

private:
	//Member variables
	Mat fgModel;
	Mat bgModel;
	Mat mask;
	Rect ROI;
	Rect last_ROI;

};

#endif