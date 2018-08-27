/*
* Copyright 2017 Martha Dais Ferreira, Rodrigo Fernandes de Mello
* 
* This file is part of studyForwardCNN.
* 
* studyForwardCNN is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* studyForwardCNN is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with ImageFalseNearest.  If not, see <http://www.gnu.org/licenses/>.
* 
* See the file "COPYING" for the text of the license.
* 
* Contact: 
* 	Martha D. Ferreira: daismf@icmc.usp.br
* 	Rodrigo Fernandes de Mello: mello@icmc.usp.br
*/

#ifndef _POOLING_HPP_
#define _POOLING_HPP_

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class Pooling {
	private:
		int rowStride;
		int colStride;
		int rowPooling;
		int colPooling;

	public:

		/*
		 * Neuron constructor 
		 */
		Pooling(int rowStride, int colStride, int rowPooling, int colPooling) {

			this->rowStride = rowStride;
			this->colStride = colStride;
			this->rowPooling = rowPooling;
			this->colPooling = colPooling;
		}

		virtual ~Pooling() {

		}

	private:
		
		/***** MAX POOLING ******/
		// How pooling works
		// =================
		//
		// Now we apply pooling and strides together
		//
		//   0 1 2 3 4 5 6 7 8 9
		// 0 X X X
		// 1 X X X
		// 2 X X X
		// 3
		// 4
		// 5
		// 6
		// 7
		// 8
		// 9
		//
		//  nrowStride = 2
		//  ncolStride = 2
		//
		//   0 1 2 3 4 5 6 7 8 9
		// 0     X X X
		// 1     X X X
		// 2     X X X
		// 3
		// 4
		// 5
		// 6
		// 7
		// 8
		// 9
		//
		
		/*
		 * Funtion that return the maximun value in a matrix
		 */
		double maximum(Mat *output, int rowId, int colId) {
			int endRow = rowId + rowPooling;
			int endCol = colId + colPooling;
			double maxValue = DBL_MIN;

			for (rowId; rowId < endRow; rowId++) {
				for (colId; colId < endCol; colId++) {
					if (rowId < output->rows && colId < output->cols) {
						maxValue = fmax(maxValue, output->at<float>(rowId,colId));
					}
				}
			}

			return maxValue;
		}

	public:
		
		/*
		 * Compute the max pooling in a single image
		 */
		Mat *maxPooling(Mat *output) {
			 // We can use either max pooling or alpha_root(sum(x^alpha)) applying Pooling
			int nrows = (int) ceil(output->rows / (rowStride * 1.0));
			int ncols = (int) ceil(output->cols / (colStride * 1.0));
			Mat *pooling = new Mat(nrows, ncols, CV_32F);

			for(int i = 0; i < nrows; i++) {
				for(int j = 0; j < ncols; j++) {
					pooling->at<float>(i,j) = 
						maximum(output, i*rowStride, j*colStride);
				}
			}

			return pooling;
		}
		
		vector<Mat *> *applyPooling(vector<Mat *> *depthImage) {
			vector<Mat *> *DepthPooling = new vector<Mat *>();

			for (int d = 0; d < depthImage->size(); d++) {
				Mat *image = depthImage->at(d);				
				DepthPooling->push_back(maxPooling(image));
			}

			return DepthPooling;
		}
		
		int getRowStride() { return this->rowStride; }
		int getColStride() { return this->colStride; }
		int getRowPooling() { return this->rowPooling; }
		int getColPooling() { return this->colPooling; }
		
};

#endif
