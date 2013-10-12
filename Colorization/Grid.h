#ifndef _GRID_H_
#define _GRID_H_

#include <vector>
using namespace std;

template <class T, class V=vector<T> >
class Grid {
private:
	int rows;
	int cols;
	vector<V> data;

public:
	// * default constructor
	Grid() : data() {}

	// * constructor
	Grid(int r, int c)
		: rows(r), cols(c), data(r * c, V()) {}

	// * destructor
	virtual ~Grid() {}

	// * copy constructor
	Grid(const Grid& grid) 
		: rows(grid.rows),
		  cols(grid.cols),
		  data(data.begin(), data.end())
	{
	}

	// * operator =
	Grid& operator=(Grid& grid) {
		this->rows = grid.rows;
		this->cols = grid.cols;
		this->data = grid.data;
		return *this;
	}

	// * get number of rows
	int nrows() const {
		return this->rows;
	}

	// * get number of cols
	int ncols() const {
		return this->cols;
	}

	// * check (i, j) is in the grid range
	bool isin(int i, int j) {
		return i >= 0 && j >= 0 && i < rows && j < cols;
	}

	// * push at (i, j)
	void pushAt(int i, int j, T t) {
		data[i*cols+j].push_back(t);
	}

	// * access pointer of (i, j)
	V& ptrAt(int i, int j) {
		return data[i*cols+j];
	}
};

#endif
