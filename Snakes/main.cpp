#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <cmath>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Vector2D.h"

const double R = 10.0;
const double EPS = 1.0e-12;
const double INF = 1.0e12;
double alpha = 0.9;
double beta  = 0.7;
double gamma = 0.2;

cv::Mat img;
cv::Mat out;
int prevx = 0;
int prevy = 0;
bool isPress = false;
const int winsize = 12;
const int neighbors = 2 * winsize * winsize;
const string winname = "Snakes";

vector<Vector2D> points;

void startSnakes() {
	const int width   = img.cols;
	const int height  = img.rows;
	const int dim     = img.channels();
	const int maxiter = 200;
	const int threshold = 0;

	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_32FC1);
	cv::GaussianBlur(gray, gray, cv::Size(), 2.0);

	cv::Mat gradX, gradY;
	cv::Sobel(gray, gradX, CV_32FC1, 1, 0);
	cv::Sobel(gray, gradY, CV_32FC1, 0, 1);

	cv::Mat grad = cv::Mat(gray.size(), CV_32FC1);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			double gx = gradX.at<float>(y, x);
			double gy = gradY.at<float>(y, x);
			grad.at<float>(y, x) = gx * gx + gy * gy;
		}
	}
	cv::imshow("gradient", grad / 255.0);
	
	vector<double> Econt(neighbors, INF);
	vector<double> Ecurv(neighbors, INF);
	vector<double> Eimag(neighbors, INF);

	int nseg = (int)points.size();
	int iter = 0;
	while(++iter < maxiter) {
		int    move = 0;
		double dAvg = 0.0;
		for(int i=0; i<points.size(); i++) {
			dAvg += (points[i] - points[(i+1)%nseg]).norm();
		}
		dAvg /= nseg;

		for(int i=0; i<nseg; i++) {
			double minEcont = INF;
			double minEcurv = INF;
			double minEimag = INF;
			double maxEcont = 0.0;
			double maxEcurv = 0.0;
			double maxEimag = 0.0;
			double maxCont = 0.0;
			double maxCurv = 0.0;

			int up     = max(0, (int)points[i].y - winsize/2);
			int bottom = min(height-1, (int)points[i].y + winsize/2);
			int left   = max(0, (int)points[i].x - winsize/2);
			int right  = min(width-1, (int)points[i].x + winsize/2);
			int count = 0;
			for(int yy=up; yy<=bottom; yy++) {
				for(int xx=left; xx<=right; xx++) {
					if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
						Vector2D next(xx, yy);
						Econt[count] = abs(dAvg - (next - points[(i+1)%nseg]).norm());
						Ecurv[count] = (points[(nseg+i-1)%nseg] - next * 2 + points[(i+1)%nseg]).norm2();
						Eimag[count] = grad.at<float>(yy, xx);
					
						minEcont = min(minEcont, Econt[count]);
						minEcurv = min(minEcurv, Ecurv[count]);
						minEimag = min(minEimag, Eimag[count]);
						maxEcont = max(maxEcont, Econt[count]);
						maxEcurv = max(maxEcurv, Ecurv[count]);
						maxEimag = max(maxEimag, Eimag[count]);
						count++;
					}
				}
			}

			double minE = INF;
			count = 0;
			int moveX = (int)points[i].x;
			int moveY = (int)points[i].y;
			for(int yy=up; yy<=bottom; yy++) {
				for(int xx=left; xx<=right; xx++) {
					if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
						Econt[count] = (Econt[count] - minEcont) / (maxEcont - minEcont + EPS);
						Ecurv[count] = (Ecurv[count] - minEcurv) / (maxEcurv - minEcurv + EPS);
						Eimag[count] = (minEimag - Eimag[count]) / (maxEimag - minEimag + EPS);

						double e = alpha * Econt[count] + beta * Ecurv[count] + gamma * Eimag[count];
						if(minE > e) {
							minE   = e;
							moveX  = xx;
							moveY  = yy;
						}
						count++;
					}
				}
			}
			
			if(moveX != (int)points[i].x || moveY != (int)points[i].y) {
				points[i].x = moveX;
				points[i].y = moveY;
				move++;
			}
		}

		if(move < threshold) {
			break;
		}

		img.convertTo(out, CV_8UC3);
		for(int i=0; i<nseg; i++) {
			cv::line(out, cv::Point(points[i].x, points[i].y), cv::Point(points[(i+1)%nseg].x, points[(i+1)%nseg].y), cv::Scalar(0.0, 255.0, 0), 1, CV_AA);
		}
		cv::imshow(winname, out);
		cv::waitKey(10);
	}
	printf("Finish in %d iterations", iter);

	img.convertTo(out, CV_8UC3);
	for(int i=0; i<nseg; i++) {
		cv::line(out, cv::Point(points[i].x, points[i].y), cv::Point(points[(i+1)%nseg].x, points[(i+1)%nseg].y), cv::Scalar(0.0, 255.0, 0), 1, CV_AA);
	}
	cout << "Finish" << endl;
	cv::imshow(winname, out);
}

void onMouse(int e, int x, int y, int flag, void* userdata) {
	if(e == CV_EVENT_LBUTTONDOWN) {
		points.clear();
		isPress = true;
		prevx = x;
		prevy = y;
		points.push_back(Vector2D(x, y));
	} else if(e == CV_EVENT_MOUSEMOVE) {
		if(isPress) {
			double dx = x - prevx;
			double dy = y - prevy;
			if(dx * dx + dy * dy >= R * R) {
				points.push_back(Vector2D(x, y));
				cv::line(out, cv::Point(prevx, prevy), cv::Point(x, y), cv::Scalar(0, 0, 255), 1, CV_AA);
				cv::imshow(winname, out);
				prevx = x;
				prevy = y;
			}
		}
	} else if(e == CV_EVENT_LBUTTONUP) {
		isPress = false;
		points.push_back(Vector2D(x, y));
		startSnakes();
	}
}

int main(int argc, char** argv) {
	if(argc <= 1) {
		cout << "usage: Snakes.exe [input image]" << endl;
		return -1;
	}

	img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load image file \"" << argv[1] << "\"" << endl;
		return -1;
	}

	cv::namedWindow(winname);
	cv::setMouseCallback(winname, onMouse);

	img.convertTo(out, CV_8UC3);
	cv::imshow(winname, out);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
