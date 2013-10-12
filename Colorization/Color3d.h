#ifndef _COLOR_3D_H_
#define _COLOR_3D_H_

#include <cassert>

class Color3d {
public:
	double v[3];

	// デフォルトコンストラクタ
	Color3d();

	// コンストラクタ
	Color3d(double r, double g, double b);

	// コピーコンストラクタ
	Color3d(const Color3d& c3d);

	// 演算子 =
	Color3d& operator=(const Color3d& c3d);

	// アクセス演算子
	double& operator()(int i);

	// 演算子 +
	Color3d operator+(const Color3d& c3d);

	// 演算子 -
	Color3d operator-(const Color3d& c3d);

	// 演算子 *
	Color3d operator*(const Color3d& c3d);

	// 演算子 <
	bool operator<(const Color3d& c3d) const;

	// 演算子 >
	bool operator>(const Color3d& c3d) const;

	/* メソッド定義 */
	// 値の定数倍
	Color3d multiply(double d);

	// 値の定数商
	Color3d divide(double d);
};

#endif
