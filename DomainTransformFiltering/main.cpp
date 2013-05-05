/***********************************************************
* Domain transform filtering implementation
************************************************************
* This code is an implementation of the paper [Gastal and Oliveira 2011].
* This paper proposes an edge-preserving filter, 
* which is effectively parallelizable. This filter transforms
* the domain of the filter function, and performs linear filtering
* to the transfomed domain.
*
* usage: DomainTransformFiltering.exe [input_image] ([sigma_s] [sigma_r] [maxiter])
* (last three arguments are optional)
*
* This code is programmed by 'tatsy'. You can use this
* code for any purpose :-)
* If you are satisfied with the program and kind enough of
* cheering me up, please contact me from my github account
* "https://github.com/tatsy/". Thanks!
************************************************************/

#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
using namespace std;

#include "opencv2/opencv.hpp"

long t, elapsed;
#define CLOCK_START t = clock();
#define CLOCK_END cout << "[ Time ]" << endl << "  " << ((clock() - t) / 1000.0) << " sec" << endl;

// Fast pow function, referred from
// http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
double fastPow(double a, double b) {
	union {
		double d;
		int x[2];
	} u = { a };
	u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
	u.x[0] = 0;
	return u.d;
}

// Recursive filter for vertical direction
void recursiveFilterVertical(cv::Mat& out, cv::Mat& dct) {
	int width = out.cols;
	int height = out.rows;
	int dim = out.channels();

	// if openmp is available, compute in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int x=0; x<width; x++) {
		for(int y=1; y<height; y++) {
			double p = dct.at<float>(y-1, x);
			for(int c=0; c<dim; c++) {
				out.at<float>(y, x*dim+c) = (1.0 - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y-1, x*dim+c);
			}
		}

		for(int y=height-2; y>=0; y--) {
			double p = dct.at<float>(y, x);
			for(int c=0; c<dim; c++) {
				out.at<float>(y, x*dim+c) = p * out.at<float>(y+1, x*dim+c) + (1.0 - p) * out.at<float>(y, x*dim+c);
			}
		}
	}
}

// Recursive filter for horizontal direction
void recursiveFilterHorizontal(cv::Mat& out, cv::Mat& dct) {
	int width = out.cols;
	int height = out.rows;
	int dim = out.channels();

	// if openmp is available, compute in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int y=0; y<height; y++) {
		for(int x=1; x<width; x++) {
			double p = dct.at<float>(y, x-1);
			for(int c=0; c<dim; c++) {
				out.at<float>(y, x*dim+c) = (1.0 - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y, (x-1)*dim+c);
			}
		}

		for(int x=width-2; x>=0; x--) {
			double p = dct.at<float>(y, x);
			for(int c=0; c<dim; c++) {
				out.at<float>(y, x*dim+c) = p * out.at<float>(y, (x+1)*dim+c) + (1.0 - p) * out.at<float>(y, x*dim+c);
			}
		}
	}
}

// Domain transform filtering
void domainTransformFilter(cv::Mat& img, cv::Mat& out, double sigma_s, double sigma_r, int maxiter) {
	int width = img.cols;
	int height = img.rows;
	int dim = img.channels();
	img.convertTo(img, CV_MAKETYPE(CV_32F, dim), 1.0 / 255.0);

	// compute derivatives of transformed domain "dct"
	// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
	cv::Mat dctx = cv::Mat(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat(height-1, width, CV_32FC1);
	double ratio = sigma_s / sigma_r;

	double a = exp(-sqrt(2.0) / sigma_s);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width-1; x++) {
			float accum = 0.0f;
			for(int c=0; c<dim; c++) {
				accum += abs(img.at<float>(y, (x+1)*dim+c) - img.at<float>(y, x*dim+c)); 
			}
			dctx.at<float>(y, x) = fastPow(a, 1.0f + ratio * accum); 
		}
	}

	for(int x=0; x<width; x++) {
		for(int y=0; y<height-1; y++) {
			float accum = 0.0f;
			for(int c=0; c<dim; c++) {
				accum += abs(img.at<float>(y+1, x*dim+c) - img.at<float>(y, x*dim+c)); 
			}
			dcty.at<float>(y, x) = fastPow(a, 1.0f + ratio * accum); 
		}
	}

	// Apply recursive folter maxiter times
	img.convertTo(out, CV_MAKETYPE(CV_32F, dim));
	while(maxiter--) {
		recursiveFilterHorizontal(out, dctx);
		recursiveFilterVertical(out, dcty);
	}
}


// Main function
int main(int argc, char** argv) {
	// Check arguments
	if(argc <= 1) {
		cout << "usage: DomainTransformFiltering.exe [input_image] ([sigma_s] [sigma_r] [maxiter])" << endl;
		return -1;
	}

	// Load image
	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load image \"" << argv[1] << "\"" << endl;
		return -1;
	}

	// Parameter set
	const double sigma_s = argc <= 2 ? 25.0 : atof(argv[2]);
	const double sigma_r = argc <= 3 ? 0.1  : atof(argv[3]);
	const int    maxiter = argc <= 4 ? 10   : atoi(argv[4]);

	cout << "[ Parameters ]" << endl;
	cout << "  * sigma_s = " << sigma_s << endl; 
	cout << "  * sigma_r = " << sigma_r << endl; 
	cout << "  * maxiter = " << maxiter << endl; 
	cout << endl;

	// Call domain transform filter
CLOCK_START
	cv::Mat out;
	domainTransformFilter(img, out, sigma_s, sigma_r, maxiter);
CLOCK_END

	// Show results
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
