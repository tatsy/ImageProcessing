#include <cmath>

#include "Vector2D.h"

// コンストラクタ
Vector2D::Vector2D()
	: x(0.0), y(0.0)
{
}

// コンストラクタ
Vector2D::Vector2D(double _x, double _y)
	: x(_x), y(_y)
{
}

// コピーコンストラクタ
Vector2D::Vector2D(const Vector2D& v2)
	: x(v2.x), y(v2.y)
{
}

// デストラクタ
Vector2D::~Vector2D()
{
}

// 演算子 =
Vector2D& Vector2D::operator=(const Vector2D& v2) {
	this->x = v2.x;
	this->y = v2.y;
	return (*this);
}

// 演算子 +=
Vector2D& Vector2D::operator+=(Vector2D v2) {
	this->x += v2.x;
	this->y += v2.y;
	return *this;
}

// 演算子 -=
Vector2D& Vector2D::operator-=(Vector2D v2) {
	this->x -= v2.x;
	this->y -= v2.y;
	return *this;
}

// 演算子 +
Vector2D Vector2D::operator+(Vector2D v2) {
	return Vector2D(this->x + v2.x, this->y + v2.y);
}

// 演算子 +
Vector2D Vector2D::operator-(Vector2D v2) {
	return Vector2D(this->x - v2.x, this->y - v2.y);
}

// 演算子 *
Vector2D Vector2D::operator*(double s) {
	return Vector2D(this->x*s, this->y*s);
}

// 演算子 /
Vector2D Vector2D::operator/(double s) {
	return Vector2D(this->x/s, this->y/s);
}

// ノルム
double Vector2D::norm() const {
	return hypot(x, y);
}

// ノルムの二乗
double Vector2D::norm2() const {
	return x * x + y * y;
}

// 標準出力
ostream& operator<<(ostream& os, const Vector2D& v) {
	ostringstream oss;
	oss << "[ " << v.x << ", " << v.y << " ]";
	return oss;
}