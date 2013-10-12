#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <queue>
using namespace std;

#include "opencv2/opencv.hpp"

#include "Color3d.h"
#include "Grid.h"

const string winname = "Input";

bool isPress = false;
cv::Mat gray;
cv::Mat temp;
cv::Mat input;
int brush[3] = {0};

const int dim = 3;
const int dx[4] = {-1, 0, 0, 1};
const int dy[4] = {0, -1, 1, 0};

typedef pair<int, int> Pnt;
typedef pair<pair<double, int>, Pnt> Pair;

void onMouse(int e, int x, int y, int flag, void*userdata) {
	if(e == CV_EVENT_LBUTTONDOWN) {
		isPress = true;
	}

	if(e == CV_EVENT_MOUSEMOVE && isPress) {
		cv::circle(input, cv::Point(x, y), 5, cv::Scalar(brush[0], brush[1], brush[2]), -1);
			
		cv::cvtColor(gray, temp, CV_GRAY2BGR);
		for(int y=0; y<gray.rows; y++) {
			for(int x=0; x<gray.cols; x++) {
				uchar red   = input.at<uchar>(y, x*dim+2);
				uchar green = input.at<uchar>(y, x*dim+1);
				uchar blue  = input.at<uchar>(y, x*dim+0);
				if(red | green | blue) {
					temp.at<uchar>(y, x*dim+2) = red;
					temp.at<uchar>(y, x*dim+1) = green;
					temp.at<uchar>(y, x*dim+0) = blue;
				}
			}
		}
		cv::imshow(winname, temp);
	}

	if(e == CV_EVENT_LBUTTONUP) {
		isPress = false;
	}
}

void onChangeR(int pos, void* userdata) {
	brush[2] = pos;
	cv::rectangle(temp, cv::Rect(0, 0, 20, 20), cv::Scalar(brush[0], brush[1], brush[2]), -1);
	cv::imshow(winname, temp);
}

void onChangeG(int pos, void* userdata) {
	brush[1] = pos;
	cv::rectangle(temp, cv::Rect(0, 0, 20, 20), cv::Scalar(brush[0], brush[1], brush[2]), -1);
	cv::imshow(winname, temp);
}

void onChangeB(int pos, void* userdata) {
	brush[0] = pos;
	cv::rectangle(temp, cv::Rect(0, 0, 20, 20), cv::Scalar(brush[0], brush[1], brush[2]), -1);
	cv::imshow(winname, temp);
}

double wfunc(double r) {
	return 1.0 / (pow(abs(r), 3) + 1.0-8);
}

void colorize() {
	set<Color3d> S;
	const int width  = gray.cols;
	const int height = gray.rows;
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			uchar red   = input.at<uchar>(y, x*dim+2);
			uchar green = input.at<uchar>(y, x*dim+1);
			uchar blue  = input.at<uchar>(y, x*dim+0);
			S.insert(Color3d(red, green, blue));
		}
	}

	vector<Color3d> colors(S.begin(), S.end());
	map<Color3d, int> table;
	for(int i=0; i<colors.size(); i++) {
		table[colors[i]] = i;
	}
	
	priority_queue<Pair, vector<Pair>, greater<Pair> > que;				
	Grid<pair<int, double>, map<int, double> > grid(height, width);	
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			uchar red   = input.at<uchar>(y, x*dim+2);
			uchar green = input.at<uchar>(y, x*dim+1);
			uchar blue  = input.at<uchar>(y, x*dim+0);
			if(red | green | blue) {
				// dijkstra algorithm
				int color = table[Color3d(red, green, blue)];
				grid.ptrAt(y, x).insert(make_pair(color, 0.0));
				que.push(make_pair(make_pair(0.0, color), Pnt(x, y)));
			}
		}
	}

	while(!que.empty()) {
		double dist  = que.top().first.first;
		int    color = que.top().first.second;
		Pnt    pt    = que.top().second;
		que.pop();

		for(int k=0; k<4; k++) {
			int nx = pt.first  + dx[k];
			int ny = pt.second + dy[k];
			if(nx >= 0 && ny >= 0 && nx < width && ny < height) {
				double ndist = dist + abs(gray.at<uchar>(pt.second, pt.first) - gray.at<uchar>(ny, nx));
				if(grid.ptrAt(ny, nx).find(color) == grid.ptrAt(ny, nx).end()) {
					if(grid.ptrAt(ny, nx).size() < 3) {
						grid.ptrAt(ny, nx).insert(make_pair(color, ndist));
						que.push(make_pair(make_pair(ndist, color), Pnt(nx, ny)));
					}
				} else {
					if(grid.ptrAt(ny, nx)[color] > ndist) {
						grid.ptrAt(ny, nx)[color] = ndist;
						que.push(make_pair(make_pair(ndist, color), Pnt(nx, ny)));
					}
				}
			}
		}
	}

	cv::Mat out = cv::Mat(gray.size(), CV_8UC3, CV_RGB(0, 0, 0));
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			double weight = 0.0;
			Color3d color(0, 0, 0);
			map<int, double>::iterator it;
			for(it = grid.ptrAt(y, x).begin(); it != grid.ptrAt(y, x).end(); ++it) {
				double w = wfunc(it->second);
				color = color + colors[it->first].multiply(w);
				weight += w;
			}
			color = color.divide(weight);

			for(int c=0; c<dim; c++) {
				out.at<uchar>(y, x*dim+c) = color.v[dim-c-1];
			}
		}
	}

	cv::cvtColor(out, out, CV_BGR2YCrCb);
	for(int y=0; y<height; y++) {
		for(int x=0; x<width; x++) {
			out.at<uchar>(y, x*dim+0) = gray.at<uchar>(y, x);
		}
	}
	cv::cvtColor(out, out, CV_YCrCb2BGR);
	cv::imshow(winname, out);
}

int main(int argc, char** argv) {
	if(argc <= 1) {
		cout << "usage: Colorization.exe [input image]" << endl;
		return -1;
	}

	gray = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if(gray.empty()) {
		cout << "Failed to load image file \"" << argv[1] << "\"" << endl;
		return -1;
	}

	input = cv::Mat(gray.size(), CV_8UC3, CV_RGB(0, 0, 0));

	cv::namedWindow(winname);
	cv::setMouseCallback(winname, onMouse);
	cv::createTrackbar("Red", winname, &brush[2], 255, onChangeR);
	cv::createTrackbar("Green", winname, &brush[1], 255, onChangeG);
	cv::createTrackbar("Blue", winname, &brush[0], 255, onChangeB);

	cv::cvtColor(gray, temp, CV_GRAY2BGR);
	cv::imshow(winname, temp);
	int key = 0;
	while(key != 0x1b) {
		key = cv::waitKey(30);
		if(key == 'c') {
			colorize();
		}
	}
	cv::destroyAllWindows();
}
