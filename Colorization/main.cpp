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

cv::Mat gray;   // 入力グレースケール画像
cv::Mat temp;   // 一時退避用の画像
cv::Mat input;  // ユーザ入力保存用画像
cv::Mat out;    // 出力カラー画像

/* UI用の変数 */
const string winname = "Colorization";  // ウィンドウの名前
const double radius  = 5.0;            // ユーザ入力の線を引く間隔
bool isPress  = false;                  // マウスのボタンが押されているか
int  brush[3] = {0};                    // ブラシの色
int prevx, prevy;                       // 以前のマウス位置

/* 位置などの定数 */
const int dim = 3;
const int dx[4] = {-1, 0, 0, 1};
const int dy[4] = {0, -1, 1, 0};

/* Colorization用の画素構造体 */
struct Pixel {
    int x, y;
    double dist;
    int color;
    bool operator<(const Pixel& p) const { return dist < p.dist; }
    bool operator>(const Pixel& p) const { return dist > p.dist; }
};

/* OpenCVのマウスコールバック */
void onMouse(int e, int mx, int my, int flag, void*userdata) {
    // 左ボタンが押された
    if(e == CV_EVENT_LBUTTONDOWN) {
        isPress = true;
        prevx = mx;
        prevy = my;
    }

    // 左ボタンを押しながらドラックしている
    if(e == CV_EVENT_MOUSEMOVE && isPress) {
        double dx = mx - prevx;
        double dy = my - prevy;
        // 前の画素から一定以上動いたら線を引く
        if(dx * dx + dy * dy > radius * radius) {
            cv::line(input, cv::Point(prevx, prevy), cv::Point(mx, my), cv::Scalar(brush[0], brush[1], brush[2]), 5);

            // 描画処理
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
            prevx = mx;
            prevy = my;
        }
    }

    // 左ボタンが離された
    if(e == CV_EVENT_LBUTTONUP) {
        isPress = false;
    }
}

/* トラックバー用のコールバック関数群 */
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

/* 重みづけ関数 */
double wfunc(double r) {
    return 1.0 / (pow(abs(r), 3) + 1.0-8);
}

/* Colorizationの処理 */
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

    priority_queue<Pixel, vector<Pixel>, greater<Pixel> > que;				
    Grid<pair<int, double>, map<int, double> > grid(height, width);	
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            uchar red   = input.at<uchar>(y, x*dim+2);
            uchar green = input.at<uchar>(y, x*dim+1);
            uchar blue  = input.at<uchar>(y, x*dim+0);
            if(red | green | blue) {
                int color = table[Color3d(red, green, blue)];
                grid.ptrAt(y, x).insert(make_pair(color, 0.0));
                Pixel pix = { x, y, 0.0, color };
                que.push(pix);
            }
        }
    }

    while(!que.empty()) {
        Pixel pix = que.top();
        que.pop();

        for(int k=0; k<4; k++) {
            int nx = pix.x  + dx[k];
            int ny = pix.y + dy[k];
            if(nx >= 0 && ny >= 0 && nx < width && ny < height) {
                double ndist = pix.dist + abs(gray.at<uchar>(pix.y, pix.x) - gray.at<uchar>(ny, nx));
                if(grid.ptrAt(ny, nx).find(pix.color) == grid.ptrAt(ny, nx).end()) {
                    if(grid.ptrAt(ny, nx).size() < 3) {
                        grid.ptrAt(ny, nx).insert(make_pair(pix.color, ndist));
                        Pixel next = { nx, ny, ndist, pix.color };
                        que.push(next);
                    }
                } else {
                    if(grid.ptrAt(ny, nx)[pix.color] > ndist) {
                        grid.ptrAt(ny, nx)[pix.color] = ndist;
                        Pixel next = { nx, ny, ndist, pix.color };
                        que.push(next);
                    }
                }
            }
        }
    }

    out = cv::Mat(gray.size(), CV_8UC3, CV_RGB(0, 0, 0));
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

/* UIの説明文 */
void description() {
    cout << "*** Colorization ***" << endl;
    cout << "    [c]: colorize monochromatic image" << endl;
    cout << "    [s]: save output image" << endl;
    cout << "  [ESC]: exit" << endl;
    cout << endl;
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

    description();
    cv::cvtColor(gray, temp, CV_GRAY2BGR);
    cv::imshow(winname, temp);
    int key = 0;
    while(key != 0x1b) {
        key = cv::waitKey(30);
        if(key == 'c') {
            colorize();
        } else if(key == 's') {
            cv::imwrite("input.png", temp);
            cv::imwrite("output.png", out);
            cout << "Result images saved." << endl;
        }
    }
    cv::destroyAllWindows();
}
