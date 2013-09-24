#include <iostream>
#include <time.h>
#include <stdlib.h>
using namespace std;

#include "opencv2/opencv.hpp"

const float eps = 0.05f;
const int r = 4;
const int offset[4][2] = {
	{-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

void solvePoisson(cv::Mat& base, cv::Mat& laplace, cv::Mat& res, int tlx, int tly, int brx, int bry, cv::Mat& region, int depth=0, int maxDepth=5) {
	int width = base.cols;
	int height = base.rows;
	int dim = base.channels();

	if(depth == 0) {
		base.convertTo(res, CV_32FC3);
	}

	int subw = brx - tlx;
	int subh = bry - tly;
	for(int it=0; it<20; it++) {
		for(int c=0; c<dim; c++) {
			for(int y=tly; y<bry; y++) {
				for(int x=tlx; x<brx; x++) {
					float sum = 0.0;
					float w = 0.0;
					for(int k=0; k<4; k++) {
						int xx = x + offset[k][0];
						int yy = y + offset[k][1];
						if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
							sum += res.at<float>(yy, xx*dim+c);
							w += 1.0;
						}
					}
					res.at<float>(y, x*dim+c) = (sum - laplace.at<float>(y, x*dim+c)) / w;
				}
			}
		}
	}

	// compute difference
	if(depth != maxDepth) {
		for(int c=0; c<dim; c++) {
			for(int y=tly; y<bry; y++) {
				for(int x=tlx; x<brx; x++) {
					float r_val = res.at<float>(y, x*dim+c);
					float b_val = base.at<float>(y, x*dim+c);
					if(abs(r_val - b_val) > eps) {
						int midx = (tlx + brx) / 2;
						int midy = (tly + bry) / 2;
						solvePoisson(base, laplace, res, tlx, tly, midx, midy, region, depth+1, maxDepth);
						solvePoisson(base, laplace, res, tlx, midy, midx, bry, region, depth+1, maxDepth);
						solvePoisson(base, laplace, res, midx, tly, brx, midy, region, depth+1, maxDepth);
						solvePoisson(base, laplace, res, midx, midy, brx, bry, region, depth+1, maxDepth);
						return;
					}
				}
			}
		}
	}

	// paint computed region
	uchar color = rand() % 255;
	for(int y=tly; y<bry; y++) {
		for(int x=tlx; x<brx; x++) {
			region.at<uchar>(y, x) = color;
		}
	}
}

int main(int argc, char** argv) {
	if(argc <= 2) {
		cout << "usage: PoissonImageEditing.exe [base image] [blend image]" << endl;
		return -1;
	}

	cv::Mat base = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(base.empty()) {
		cout << "Failed to load image file \"" << argv[1] << "\"" << endl;
		return -1;
	}

	cv::Mat blend = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
	if(blend.empty()) {
		cout << "Failed to load image file \"" << argv[2] << "\"" << endl;
		return -1;
	}

	int width = base.cols;
	int height = base.rows;
	int dim = base.channels();
	if(width != blend.cols || height != blend.rows) {
		cout << "Size of base and blend is different." << endl;
		return -1;
	}

	base.convertTo(base, CV_32FC3, 1.0 / 255.0);
	blend.convertTo(blend, CV_32FC3, 1.0 / 255.0);

	cv::Mat laplace, lap_base;
	cv::Laplacian(blend, laplace, CV_32FC3);
	cv::Laplacian(base, lap_base, CV_32FC3);

	cv::Mat mask = cv::Mat(height, width, CV_8UC1);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			bool is_same = true;
			for(int c=0; c<dim; c++) {
				float base_val = base.at<float>(y, x*dim+c);
				float blend_val = blend.at<float>(y, x*dim+c);
				if(base_val != blend_val) {
					is_same =false;
					break;
				}
			}			

			if(is_same) {
				for(int dy=-r; dy<=r; dy++) {
					for(int dx=-r; dx<=r; dx++) {
						int xx = x + dx;
						int yy = y + dy;
						if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
							for(int c=0; c<dim; c++) {
								laplace.at<float>(yy, xx*dim+c) = lap_base.at<float>(yy, xx*dim+c);
							}
						}
					}
				}
			}
		}
	}

	// solve poisson equation with Gauss-Seidel method.
	srand((unsigned long)time(NULL));
	cv::Mat region = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat res;
	solvePoisson(base, laplace, res, 1, 1, width-1, height-1, region, 0, 8);

	cv::namedWindow("Base");
	cv::imshow("Base", base);

	cv::namedWindow("Blend");
	cv::imshow("Blend", blend);

	cv::namedWindow("Result");
	cv::imshow("Result", res);
	res.convertTo(res, CV_32FC3, 255.0);
	cv::imwrite("result.png", res);

	cv::namedWindow("Region");
	cv::imshow("Region", region);
	cv::imwrite("region.png", region);

	cv::waitKey(0);
	cv::destroyAllWindows();
}
