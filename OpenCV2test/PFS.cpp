//function defintions

#include "PFS.h"

PFS::PFS(){
	Mat fgModel;
	Mat bgModel;
	Mat mask;
	Rect ROI;
	Rect last_ROI;
}

PFS::PFS(Mat fg, Mat bg, Mat new_mask, Rect new_ROI, Rect ROI_last){
	fgModel=fg;
	bgModel=bg;
	mask=new_mask;
	ROI=new_ROI;
	last_ROI=ROI_last;
}

PFS::~PFS(){
}

void PFS::update(Mat fg, Mat bg, Mat new_mask, Rect new_ROI, Rect ROI_last){
	fgModel=fg;
	bgModel=bg;
	mask=new_mask;
	ROI=new_ROI;
	last_ROI=ROI_last;
}

void PFS::setFg(Mat fg){
	fgModel=fg;
}

void PFS::setBg(Mat bg){
	bgModel=bg;
}

void PFS::setMask(Mat new_mask){
	mask=new_mask;
}

void PFS::setROI(Rect new_ROI){
	ROI=new_ROI;
}

void PFS::setlast_ROI(Rect ROI_last){
	last_ROI=ROI_last;

}

Mat PFS::getFg() const{
	return fgModel;
};

Mat PFS::getBg() const{
	return bgModel;
};

Mat PFS::getMask() const{
	return mask;
};

Rect PFS::getROI() const{
	return ROI;
};

Rect PFS::getlast_ROI() const{
	return last_ROI;
};