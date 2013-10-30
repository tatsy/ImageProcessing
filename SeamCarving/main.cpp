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

int mode;   // 縮小するか拡大するかのモード選択
int npix;   // 大きさを変更するピクセル数

const int INF = 1 << 16;

void description() {
    cout << "*** Seam Carving ***" << endl;
    cout << "  [1] shrinking" << endl;
    cout << "  [2] enlarging" << endl;
    cout << "  choose mode [1 or 2]: ";
    cin >> mode;
    cout << "  how many pixels?: ";
    cin >> npix;
}

void detectEdge(cv::InputArray img, cv::OutputArray edge) {
    cv::Mat  I = img.getMat();
    cv::Mat& E = edge.getMatRef();

    const int width  = I.cols;
    const int height = I.rows;

    cv::Mat gray, grad_x, grad_y;
	cv::cvtColor(I, gray, CV_BGR2GRAY);
    cv::Sobel(gray, grad_x, CV_8U, 1, 0);
    cv::Sobel(gray, grad_y, CV_8U, 0, 1);
    
    E = cv::Mat(height, width, CV_8UC1);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            double gx = grad_x.at<uchar>(y, x);
            double gy = grad_y.at<uchar>(y, x);
            double e = sqrt(gx * gx + gy * gy);
            E.at<uchar>(y, x) = cv::saturate_cast<uchar>(e);
        }
    }
}

void computeSeam(cv::InputArray edge, vector<int>& seam) {
    cv::Mat e = edge.getMat();
    const int width  = e.cols;
    const int height = e.rows;

    // 動的計画法の実行
	cv::Mat table = cv::Mat(height, width, CV_32SC1);
    cv::Mat prev  = cv::Mat(height, width, CV_32SC1);
	for(int x=0; x<width; x++) {
        table.at<int>(0, x) = e.at<uchar>(0, x);
         prev.at<int>(0, x) = 0;
    }

	for(int y=1; y<height; y++) {
		for(int x=0; x<width; x++) {
			int id = -1;
			int minval = INT_MAX;
			for(int dx=-1; dx<=1; dx++) {
				int xx = x + dx;
				if(xx >= 0 && xx < width) {
					if(minval > e.at<uchar>(y-1, xx)) {
						minval = e.at<uchar>(y-1, xx);
						id = dx;
					}
				}
			}
			prev.at<int>(y, x)  = id;
			table.at<int>(y, x) = table.at<int>(y-1, x+id);
		}
	}

    // バックトラックによるseamの決定
    seam.resize(height);
    int minval = INT_MAX;
    int cur_x  = 0;
    for(int x=0; x<width; x++) {
        if(minval > table.at<int>(height-1, x)) {
            minval = table.at<int>(height-1, x);
            cur_x = x;
        }
    }

    int cur_y = height-1;
    while(cur_y >= 0) {
        seam[cur_y] = cur_x;
        cur_x = cur_x + prev.at<int>(cur_y, cur_x);
        cur_y--;
    }
}

void carveSeam(cv::Mat& img, vector<int>& seam) {
    const int width  = img.cols;
    const int height = img.rows;

    cv::Mat tmp = cv::Mat(height, width-1, CV_8UC3);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width-1; x++) {
            int xx = (x < seam[y]) ? x : x+1;
            for(int c=0; c<3; c++) {             
                tmp.at<uchar>(y, x*3+c) = img.at<uchar>(y, xx*3+c);
            }
        }
    }
    tmp.convertTo(img, CV_8U);
}

void computeMultiSeams(cv::InputArray edge, cv::Mat& seam, int n_seam) {
    cv::Mat e = edge.getMat();
    const int width  = e.cols;
    const int height = e.rows;

    seam = cv::Mat::zeros(height, width, CV_32SC1);

    cv::Mat table, prev;
    for(int i=0; i<n_seam; i++) {
        table = cv::Mat::zeros(height, width, CV_32SC1);
        prev  = cv::Mat::zeros(height, width, CV_32SC1);
    
        // 動的計画法の実行
	    for(int x=0; x<width; x++) {
            table.at<int>(0, x) = e.at<uchar>(0, x);
             prev.at<int>(0, x) = 0;
        }

	    for(int y=1; y<height; y++) {
		    for(int x=0; x<width; x++) {
			    int id = -INF;
			    int minval = INT_MAX;
			    for(int dx=-1; dx<=1; dx++) {
				    int xx = x + dx;
				    if(xx >= 0 && xx < width) {
					    if(minval > e.at<uchar>(y-1, xx) && seam.at<int>(y-1, xx) == 0) {
						    minval = e.at<uchar>(y-1, xx);
						    id = dx;
					    }
				    }
			    }

                prev.at<int>(y, x)  = id;
                if(id != -INF) { 
    			    table.at<int>(y, x) = table.at<int>(y-1, x+id);
                } else {
                    table.at<int>(y, x) = INF;
                }
		    }
	    }

        // バックトラックによるseamの決定
        seam.resize(height);
        int minval = INT_MAX;
        int cur_x  = 0;
        for(int x=0; x<width; x++) {
            if(minval > table.at<int>(height-1, x)) {
                minval = table.at<int>(height-1, x);
                cur_x = x;
            }
        }

        int cur_y = height-1;
        while(cur_y >= 0) {
            seam.at<int>(cur_y, cur_x) = 1;
            if(abs(prev.at<int>(cur_y, cur_x)) <= 1) {
                cur_x = cur_x + prev.at<int>(cur_y, cur_x);
            } else {
                cout << "Too many seams!" << endl;
                return;
            }
            cur_y--;
        }
    }
}

void enlarge(cv::InputArray I, cv::OutputArray O) {
    cv::Mat  img = I.getMat();
    cv::Mat& out = O.getMatRef();
    const int width  = img.cols;
    const int height = img.rows;
    const int dim    = img.channels();

    cv::Mat edge, seams;
    detectEdge(img, edge);
    computeMultiSeams(edge, seams, npix);

    out = cv::Mat(height, width + npix, CV_8UC3);
    for(int y=0; y<height; y++) {
        int dx = 0;
        for(int x=0; x<width; x++) {
            int xx = min(x + dx, width+npix-1);
            for(int c=0; c<dim; c++) {
                out.at<uchar>(y, xx*dim+c) = img.at<uchar>(y, x*dim+c);
            }

            if(seams.at<int>(y, x) == 1) {
                for(int c=0; c<dim; c++) {
                    out.at<uchar>(y, (xx+1)*dim+c) = img.at<uchar>(y, x*dim+c);
                }
                dx++;
            }
        }
    }
}

int main(int argc, char** argv) {
	if(argc <= 2) {
		cout << "usage: > SeamCarving.exe [input image] [No. of carved lines]" << endl;
		return -1;
	}

    // 説明の表示とモード選択
    description();

    // 画像の読み込み
	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(img.empty()) {
		cout << "Failed to load image \"" << argv[1] << "\"." << endl;
		return -1;
	}
	int width  = img.cols;
	int height = img.rows;
	printf("width = %d, height = %d\n", width, height);


	// Carvingの実行
    cv::Mat out;
    if(mode == 1) {
	    cv::namedWindow("output");	
        cv::Mat edge;
        vector<int> seam;
        img.convertTo(out, CV_8U);
	    for(int i=0; i<npix; i++) {
            // エッジを抽出する
            detectEdge(img, edge);

            // seamの計算
            computeSeam(edge, seam);
		
		    // seamをcarveする
            carveSeam(out, seam);
		
		    // 画像の更新
		    cv::imshow("output", out);
		    cv::waitKey(30);
		    printf("%3d seams are carved!\r", i+1);
	    }
       	printf("\n");
    }
    else {
        enlarge(img, out);
    }

    cv::imshow("output", out);
    cv::waitKey(0);
	cv::destroyAllWindows();

	cv::imwrite("output.png", out); 
}
