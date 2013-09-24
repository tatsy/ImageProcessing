/***********************************************************
* k-means++ Implementation
************************************************************
* This code is an implementation of the paper [Arthur et al. 2004].
* The program classifies input pixels into several clusters.
* In general, k-means method does not guaranteee that it converges
* on global optima. k-means++ is an improvement of k-means method
* that more reliably coverges on global optima by sampling initial centers
* with distances between already sampled centers and other samples.
* Please note that this code depends on OpenCV and Mersenne Twister.
*
* usage: KmeansPlusPlus.exe [input image] [output image] [ncluster] [maxiter]
*
* This code is this programmed by 'tatsy'. You can use this
* code for any purpose :-)
* If you are satisfied with the program and kind enough of 
* cheering me up, please contact me from my github account
* "https://github.com/tatsy/". Thanks!
************************************************************/

#include <iostream>
#include <ctime>
using namespace std;

#include <opencv2\opencv.hpp>

extern "C" {
#include "mt19937ar.h"
}

void kmeanspp(cv::Mat& samples, cv::Mat& centers, cv::Mat& indices, cv::Mat& count, int nclusters, int maxiter) {
	int randi;
	int N = samples.rows;
	int dim = samples.cols;
	
	// Intializing random seed
	init_genrand((unsigned long)time(NULL));
	
	// Sampling initial centers by k-means++
	centers = cv::Mat(nclusters, dim, CV_32FC1);
	randi = genrand_int31() % N;
	for(int d=0; d<dim; d++) centers.at<float>(0, d) = samples.at<float>(randi, d);

	vector<double> minval = vector<double>(N, HUGE_VAL);
	for(int k=1; k<nclusters; k++) {
		// Compute distances between already sampled centers and other input samples.
		// Update nearest distance if it is smaller than previous ones.
		double D = 0.0;
		for(int i=0; i<N; i++) {
			double dist = 0.0;
			for(int d=0; d<dim; d++) {
				double diff = centers.at<float>(k-1, d) - samples.at<float>(i, d);
				dist += diff * diff;
			}

			if(dist < minval[i]) {
				minval[i] = dist;
			}
			
			D += minval[i];
		}

		// Determine new initial center by roulette selection
		double rate = genrand_real2();
		double accum = 0.0;
		int j = 0;
		while(accum < rate) {
			accum += minval[j] / D;
			j++;
		}

		for(int d=0; d<dim; d++) {
			centers.at<float>(k, d) = samples.at<float>(j, d);
		}
	}

	// Perform general k-means method
	indices = cv::Mat(N, 1, CV_32SC1);
	count = cv::Mat(nclusters, 1, CV_32SC1);
	while(maxiter--) {
		count = cv::Mat::zeros(nclusters, 1, CV_32SC1);

		// Sample classification
		for(int i=0; i<N; i++) {
			double minidx = 0;
			double minval = HUGE_VAL;
			for(int k=0; k<nclusters; k++) {
				double dist = 0.0;
				for(int d=0; d<dim; d++) {
					double diff = centers.at<float>(k, d) - samples.at<float>(i, d);
					dist += diff * diff;
				}

				if(minval > dist) {
					minval = dist;
					minidx = k;
				}
			}

			indices.at<int>(i, 0) = minidx;
			count.at<int>(minidx, 0) += 1;
		}

		// Re-calculare cluster centers
		centers = cv::Mat::zeros(nclusters, dim, CV_32FC1);
		for(int i=0; i<N; i++) {
			int index = indices.at<int>(i, 0);
			for(int d=0; d<dim; d++) {
				centers.at<float>(index, d) += samples.at<float>(i, d);
			}
		}

		for(int k=0; k<nclusters; k++) {
			for(int d=0; d<dim; d++) {
				centers.at<float>(k, d) /= (float)count.at<int>(k, 0);
			}
		}
	}
}

int main(int argc, char** argv) {
	
	// Check input arguments etc.
	if(argc < 4) {
		cout << "usage: KmeansPlusPlus.exe [input image] [output image] [ncluster] [maxiter]" << endl;
		return -1;
	}

	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load file \"" << argv[1] << "\"." << endl;
		return -1;
	}
	const int width = img.cols;
	const int height = img.rows;
	cout << "[KmeansPlusPlus]: Image name \"" << argv[1] << "\" has been loaded." << endl;
	printf("[KmeansPlusPlus]: Image size = %d x %d\n", width, height);

	const int ncluster = atoi(argv[3]);
	const int maxiter = atoi(argv[4]);
	printf("[KmeansPlusPlus]: Classified into %d clusters by %d iterations.\n", ncluster, maxiter);

	// Reshape input pixels into a set of samples for classification
	const int N = width * height;
	const int dim = img.channels();
	cv::Mat samples = cv::Mat(N, dim, CV_32FC1);
	for(int x=0; x<width; x++) {
		for(int y=0; y<height; y++) {
			for(int d=0; d<dim; d++) {
				int index = y * width + x;
				samples.at<float>(index, d) = (float)img.at<uchar>(y, x*dim+d);
			}
		}
	}

	// Perform k-means++
	cv::Mat centers, indices, count;
	kmeanspp(samples, centers, indices, count, ncluster, maxiter);
	cout << "Kmeans++ finished." << endl;

	// Display computed centers
	printf("\n **** Centers **** \n");
	for(int k=0; k<ncluster; k++) {
		printf("  No. %2d: (", k+1);
		for(int d=0; d<dim; d++) {
			printf("%3d", (int)centers.at<float>(k, d));
			if(d != dim-1) printf(", ");
		}
		printf(")\n");
	}
	printf("\n");

	// Generate output image
	cv::Mat out = cv::Mat(height, width, CV_8UC3);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			int index = y * width + x;
			int ci = indices.at<int>(index, 0);
			for(int d=0; d<dim; d++) {
				out.at<uchar>(y, x*dim+d) = (uchar)centers.at<float>(ci, d);
			}
		}
	}

	// Display input and output images
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey();
	cout << "Output image has been saved in \"" << argv[2] << "\"." << endl;
	cv::imwrite(argv[2], out);
	cv::destroyAllWindows();
}
