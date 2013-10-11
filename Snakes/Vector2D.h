#ifndef _VECTOR2D_H_
#define _VECTOR2D_H_

#include <iostream>
#include <sstream>
using namespace std;

class Vector2D {

public:
	double x, y;

	// コンストラクタ
	Vector2D();
	Vector2D(double x, double y);
	Vector2D(const Vector2D& v2);

	// デストラクタ
	virtual ~Vector2D();

	// 演算子定義
	Vector2D& operator =(const Vector2D& v2);
	Vector2D& operator +=(Vector2D v2);
	Vector2D& operator -=(Vector2D v2);
	Vector2D operator +(Vector2D v2);
	Vector2D operator -(Vector2D v2);
	Vector2D operator *(double s);
	Vector2D operator /(double s);

	// その他の処理
	double norm() const;
	double norm2() const;

	// 標準出力
	friend ostream& operator<<(ostream& os, const Vector2D& v);
};

ostream& operator<<(ostream& os, const Vector2D& v);

#endif