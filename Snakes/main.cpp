#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <cmath>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Vector2D.h"

const double INF = 1.0e12;
double alpha = 0.9;
double beta  = 0.8;
double gamma = 0.5;

cv::Mat img;
cv::Mat out;
int prevx = 0;
int prevy = 0;
bool isPress = false;
const int ndisc = 80;
const int winsize = 12;
const int neighbors = 2 * winsize * winsize;
const string winname = "Snakes";

vector<Vector2D> points;

void startSnakes() {
	const int width   = img.cols;
	const int height  = img.rows;
	const int dim     = img.channels();
	const int maxiter = 1000;
	const int threshold = 5;

	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_32FC1);
	cv::GaussianBlur(gray, gray, cv::Size(), 1.0);

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
	
	vector<double> cont(neighbors, INF);
	vector<double> curv(neighbors, INF);
	vector<double> imag(neighbors, INF);

	int iter = 0;
	while(++iter < maxiter) {
		int    move = 0;
		double dAvg = 0.0;
		for(int i=0; i<ndisc; i++) {
			dAvg += (points[i] - points[(i+1)%ndisc]).norm();
		}
		dAvg /= ndisc;

		for(int i=0; i<ndisc; i++) {
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
						cont[count] = abs(dAvg - (next - points[(i+1)%ndisc]).norm());
						curv[count] = (points[(ndisc+i-1)%ndisc] - next * 2 + points[(i+1)%ndisc]).norm2();
						imag[count] = grad.at<float>(yy, xx);
					
						minEcont = min(minEcont, cont[count]);
						minEcurv = min(minEcurv, curv[count]);
						minEimag = min(minEimag, imag[count]);
						maxEcont = max(maxEcont, cont[count]);
						maxEcurv = max(maxEcurv, curv[count]);
						maxEimag = max(maxEimag, imag[count]);
						count++;
					}
				}
			}

			double minE   = INF;
			count = 0;
			int moveX = (int)points[i].x;
			int moveY = (int)points[i].y;
			for(int yy=up; yy<=bottom; yy++) {
				for(int xx=left; xx<=right; xx++) {
					if(xx >= 0 && yy >= 0 && xx < width && yy < height) {
						cont[count] = (cont[count] - minEcont) / (maxEcont - minEcont);
						curv[count] = (curv[count] - minEcurv) / (maxEcurv - minEcurv);
						imag[count] = (minEimag - imag[count]) / (maxEimag - minEimag);

						double e = alpha * cont[count] + beta * curv[count] + gamma * imag[count];
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
	}
	printf("Finish in %d iterations", iter);

	img.convertTo(out, CV_8UC3);
	for(int i=0; i<ndisc; i++) {
		cv::line(out, cv::Point(points[i].x, points[i].y), cv::Point(points[(i+1)%ndisc].x, points[(i+1)%ndisc].y), cv::Scalar(0.0, 255.0, 0), 1, CV_AA);
	}
	cout << "Finish" << endl;
	cv::imshow(winname, out);
}

void onMouse(int e, int x, int y, int flag, void* userdata) {
	if(e == cv::EVENT_LBUTTONDOWN) {
		isPress = true;
		prevx = x;
		prevy = y;
	} else if(e == cv::EVENT_MOUSEMOVE) {
		if(isPress) {
			double dx = x - prevx;
			double dy = y - prevy;
			double r = hypot(dx, dy);
			img.convertTo(out, CV_8UC3);
			cv::circle(out, cv::Point(prevx, prevy), r, cv::Scalar(0, 0, 255), 1, CV_AA);
			cv::imshow(winname, out);
		}
	} else if(e == cv::EVENT_LBUTTONUP) {
		isPress = false;
		double dx = x - prevx;
		double dy = y - prevy;
		double r = hypot(dx, dy);
		points.clear();
		for(int i=0; i<ndisc; i++) {
			double theta = 2.0 * M_PI / ndisc * i;
			double px = prevx + r * cos(theta);
			double py = prevy + r * sin(theta);
			points.push_back(Vector2D(px, py));
		}
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
