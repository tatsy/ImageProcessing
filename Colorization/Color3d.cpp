#include "Color3d.h"

// デフォルトコンストラクタ
Color3d::Color3d()
	: v() 
{
	v[0] = 0.0;
	v[1] = 0.0;
	v[2] = 0.0;
}

// コンストラクタ
Color3d::Color3d(double v0, double v1, double v2)
	: v()
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;
}

// コピーコンストラクタ
Color3d::Color3d(const Color3d& c3d) 
	: v()
{
	v[0] = c3d.v[0];
	v[1] = c3d.v[1];
	v[2] = c3d.v[2];
}

// 演算子 =
Color3d& Color3d::operator=(const Color3d& c3d) {
	v[0] = c3d.v[0];
	v[1] = c3d.v[1];
	v[2] = c3d.v[2];
	return (*this);
}

// アクセス演算子
double& Color3d::operator()(int i) {
	assert(i >= 0 && i <= 2);
	return v[i];
}

// 演算子 +
Color3d Color3d::operator+(const Color3d& c3d) {
	Color3d ret = Color3d();
	ret.v[0] = v[0] + c3d.v[0];
	ret.v[1] = v[1] + c3d.v[1];
	ret.v[2] = v[2] + c3d.v[2];
	return ret;
}

// 演算子 -
Color3d Color3d::operator-(const Color3d& c3d) {
	Color3d ret = Color3d();
	ret.v[0] = v[0] - c3d.v[0];
	ret.v[1] = v[1] - c3d.v[1];
	ret.v[2] = v[2] - c3d.v[2];
	return ret;
}

// 演算子 +
Color3d Color3d::operator*(const Color3d& c3d) {
	Color3d ret = Color3d();
	ret.v[0] = v[0] * c3d.v[0];
	ret.v[1] = v[1] * c3d.v[1];
	ret.v[2] = v[2] * c3d.v[2];
	return ret;
}

bool Color3d::operator<(const Color3d& c3d) const {
	if(v[0] != c3d.v[0]) return v[0] < c3d.v[0];
	if(v[1] != c3d.v[1]) return v[1] < c3d.v[1];
	return v[2] < c3d.v[2];
}

bool Color3d::operator>(const Color3d& c3d) const {
	if(v[0] != c3d.v[0]) return v[0] > c3d.v[0];
	if(v[1] != c3d.v[1]) return v[1] > c3d.v[1];
	return v[2] > c3d.v[2];
}

// 値のスケーリング
Color3d Color3d::multiply(double d) {
	Color3d ret = Color3d();
	ret.v[0] = v[0] * d;
	ret.v[1] = v[1] * d;
	ret.v[2] = v[2] * d;
	return ret;
}

Color3d Color3d::divide(double d) {
	Color3d ret = Color3d();
	ret.v[0] = v[0] / d;
	ret.v[1] = v[1] / d;
	ret.v[2] = v[2] / d;
	return ret;
}
