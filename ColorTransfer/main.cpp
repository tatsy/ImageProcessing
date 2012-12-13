/***********************************************************
* Color Transfer Implementation
************************************************************
* This code is an implementation of the paper [Reinhard2000].
* The program transfers the color of one image (in this code
* reference image) to another image (in this code target image).
*
* usage: > ColorTransfer.exe [target image] [reference image]
*
* This code is this programmed by 'tatsy'. You can use this
* code for any purpose :-)
************************************************************/

#include <iostream>
#include <cstdio>
#include <cstdlib>
using namespace std;

#include <opencv2\opencv.hpp>

cv::Vec3d operator *(const cv::Mat& M, const cv::Vec3d& v) {
	cv::Vec3d u;
	for(int i=0; i<3; i++) {
		u(i) = 0.0;
		for(int j=0; j<3; j++) {
			u(i) += M.at<double>(i, j) * v(j);
		}
	}
	return u;
}

// Transformation from RGB to LMS
const double RGB2LMS[3][3] = {
	{ 0.3811, 0.5783, 0.0402 },
	{ 0.1967, 0.7244, 0.0782 },
	{ 0.0241, 0.1288, 0.8444 }
};

// Transformation from LMS to RGB
const double LMS2RGB[3][3] = {
	{  4.4679, -3.5873,  0.1193 },
	{ -1.2186,  2.3809, -0.1624 },
	{  0.0497, -0.2439,  1.2045 }
};

// First transformation from LMS to lab
const double LMS2lab1[3][3] = {
	{ 1.0 / sqrt(3.0), 0.0, 0.0 },
	{ 0.0, 1.0 / sqrt(6.0), 0.0 },
	{ 0.0, 0.0, 1.0 / sqrt(2.0) }
};

// Second transformation from LMS to lab
const double LMS2lab2[3][3] = {
	{ 1.0,  1.0,  1.0 },
	{ 1.0,  1.0, -2.0 },
	{ 1.0, -1.0,  0.0 }
};

int main(int argc, char** argv) {
	// Check number of arguments
	if(argc <= 2) {
		cout << "usage: > ColorTransfer.exe [target image] [reference image]" << endl;
		return -1;
	}

	// Load target image
	cv::Mat target = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	target.convertTo(target, CV_64FC3, 1.0 / 255.0);
	if(target.empty()) {
		cout << "Failed to load file \"" << argv[1] << "\"" << endl;
		return -1;
	}

	// Load reference image
	cv::Mat refer  = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
	refer.convertTo(refer, CV_64FC3, 1.0 / 255.0);
	if(refer.empty()) {
		cout << "Failed to load file \"" << argv[2] << "\"" << endl;
		return -1;
	}

	// Construct transformation matrix
	const size_t bufsize = sizeof(double) * 3 * 3;
	cv::Mat mRGB2LMS = cv::Mat(3, 3, CV_64FC1);
	memcpy(mRGB2LMS.data, &RGB2LMS[0][0], bufsize); 
	
	cv::Mat mLMS2RGB = cv::Mat(3, 3, CV_64FC1);
	memcpy(mLMS2RGB.data, &LMS2RGB[0][0], bufsize);

	cv::Mat mLMS2lab1 = cv::Mat(3, 3, CV_64FC1);
	memcpy(mLMS2lab1.data, &LMS2lab1[0][0], bufsize);

	cv::Mat mLMS2lab2 = cv::Mat(3, 3, CV_64FC1);
	memcpy(mLMS2lab2.data, &LMS2lab2[0][0], bufsize);
	
	// Transform images from RGB to LMS
	for(int y=0; y<target.rows; y++) {
		for(int x=0; x<target.cols; x++) {
			cv::Vec3d v = target.at<cv::Vec3d>(y,x);
			target.at<cv::Vec3d>(y, x) = mRGB2LMS * v;
		}
	}

	cv::namedWindow("target");
	cv::namedWindow("reference");
	cv::imshow("target", target);
	cv::imshow("reference", refer);
	cv::waitKey(0);
	cv::destroyAllWindows();

	target.release();
	refer.release();

}
