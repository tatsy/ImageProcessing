/***********************************************************
* Seam Carving Implementation
************************************************************
* このコードはSeam Carvingを実装したもので、指定した分だけ横幅を
* Seam Carvingにより削ります。削っていく過程がわかるように、
* SeamをCarveするごとにその過程を表示するようにしています。
*
* 参考文献
* S.Avidan and A.Shamir, "Seam carving for content-aware
* image resizing", ACM TOG 2007.
*
* 使い方
* > SeamCarving.exe [input image] [No. of carved seams]
* 
* Copyright:
* This program is coded by tatsy. You can use this code
* for any purpose including commercial usage :-)
************************************************************/

#include <iostream>
#include <list>
#include <vector>
using namespace std;

#include <opencv2\opencv.hpp>

int main(int argc, char** argv) {
	if(argc <= 2) {
		cout << "usage: > SeamCarving.exe [input image] [No. of carved lines]" << endl;
		return -1;
	}

	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load image \"" << argv[1] << "\"." << endl;
		return -1;
	}

	int width = img.cols;
	int height = img.rows;
	int N = atoi(argv[2]);
	printf("width = %d, height = %d\n", width, height);

	// エッジを抽出する
	cv::Mat gray, edge;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::Laplacian(gray, edge, 0);

	// Carvingの実行
	cv::namedWindow("output");
	vector<vector<int> > table;
	vector<vector<int> > path;
	for(int i=0; i<N; i++) {
		// 動的計画法の実行
		table = vector<vector<int> >(height, vector<int>(width, 0));
		path = vector<vector<int> >(height, vector<int>(width, 0));
		for(int x=0; x<width; x++) table[0][x] = edge.at<uchar>(0, x);

		for(int y=1; y<height; y++) {
			for(int x=0; x<width; x++) {
				int id = -1;
				int minval = INT_MAX;
				for(int dx=-1; dx<=1; dx++) {
					int xx = x + dx;
					if(xx >= 0 && xx < width) {
						if(minval > edge.at<uchar>(y-1, xx)) {
							minval = edge.at<uchar>(y-1, xx);
							id = dx;
						}
					}
				}
				path[y][x] = id;
				table[y][x] = table[y-1][x+id] + edge.at<uchar>(y, x);
			}
		}

		// Carveされるパスを抽出
		list<int> cpath;
		int xid = 0;
		int minval = INT_MAX;
		for(int x=0; x<width; x++) {
			if(minval > table[height-1][x]) {
				minval = table[height-1][x];
				xid = x;
			}
		}

		cpath.push_front(xid);
		for(int y=height-1; y>=1; y--) {
			cpath.push_front(xid + path[y][xid]);
			xid = xid + path[y][xid];
		}
		
		// SeamをCarveする
		cv::Mat tmp = cv::Mat(height, width-i-1, CV_8UC3);
		int yid = 0;
		list<int>::iterator it = cpath.begin();
		while(it != cpath.end()) {
			for(int x=0; x<width-i-1; x++) {
				if(x < (*it)) {
					for(int c=0; c<3; c++) tmp.at<uchar>(yid, x*3+c) = img.at<uchar>(yid, x*3+c);
				} else {
					for(int c=0; c<3; c++) tmp.at<uchar>(yid, x*3+c) = img.at<uchar>(yid, (x+1)*3+c);
				}
			}
			++it;
			++yid;
		}

		// 画像の更新
		tmp.convertTo(img, CV_8UC3);
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		cv::Laplacian(gray, edge, 0);
		printf("%3d seams are carved!\r", i+1);

		cv::imshow("output", img);
		cv::waitKey(0);
	}
	printf("\n");
	cv::imwrite("output.png", img); 

	cv::destroyAllWindows();
}
