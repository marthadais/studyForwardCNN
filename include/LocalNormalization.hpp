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

#ifndef _LOCALNORMALIZATION_HPP_
#define _LOCALNORMALIZATION_HPP_

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define SQRT_EPSILON	1e-7

class LocalNormalization {
	private:
		int normalizationRows;
		int normalizationCols;

	public:

		/*
		 * Local Normalization constructor 
		 */
		LocalNormalization(int rowNormalization, int colNormalization) {
			this->normalizationRows = rowNormalization;
			this->normalizationCols = colNormalization;
		}

		virtual ~LocalNormalization() {

		}

	private:

		/*
		 *  Compute the Euclidian norm to perform the normalization
		 */
		float norm(Mat *image, int rowId, int colId, int nRows, int nCols) {
			int startRow = rowId - (nRows / 2);
			int startCol = colId - (nCols / 2);
			int endRow = startRow + nRows;
			int endCol = startCol + nCols;
			float sum = 0;

			for (startRow; startRow < endRow; startRow++) {
				for (startCol; startCol < endCol; startCol++) {
					if (startRow >= 0 && startCol >= 0 && startRow < image->rows && startCol < image->cols) {
						sum += pow(image->at<float>(startRow, startCol), 2.0);
					}
				}
			}

			return sqrt(sum) + SQRT_EPSILON;
		}
		
		/*
		 *  Compute the Euclidian norm to perform the normalization for layer grater than 0 (1,2,...)
		 */
		float norm(vector<Mat *> *depthImage, int rowId, int colId, int nRows, int nCols) {
			int startRow = rowId - (nRows / 2);
			int startCol = colId - (nCols / 2);
			int endRow = startRow + nRows;
			int endCol = startCol + nCols;
			float sum = 0;

			for (int d = 0; d < depthImage->size(); d++) {
				Mat *image = depthImage->at(d);
				for (int i = startRow; i < endRow; i++) {
					for (int j = startCol; j < endCol; j++) {
						if (i >= 0 && j >= 0 &&
							i < image->rows && j < image->cols) {
							sum += pow(image->at<float>(i,j), 2.0);
						}
					}
				}
			}

			return sqrt(sum) + SQRT_EPSILON;
		}

	public:
		
		/*
		 *  Compute the local normalization
		 *  If the local has dimensions 0x0 the normalization is not computed
		 */
		Mat *normalizeImage(Mat *image) {
			Mat *newImage = new Mat(image->rows, image->cols, CV_32F);

			for (int i = 0; i < image->rows; i++) {
				for (int j = 0; j < image->cols; j++) {
					if(	this->normalizationCols > 0 && this->normalizationRows > 0){
						//divide the value by the euclidian norm of the region
						newImage->at<float>(i,j) = image->at<float>(i,j) / 
										norm(image, i, j, this->normalizationRows, this->normalizationCols);
					} else {
						//normalization is not computed
						newImage->at<float>(i,j) = image->at<float>(i,j);
					}
				}
			}

			return newImage;
		}
		
		/*
		 *  Compute the local normalization for layer grater than 0 (1,2,...)
		 *  If the local has dimensions 0x0 the normalization is not computed
		 */
		vector<Mat *> *normalizeImage(vector<Mat *> *depthImage) {

			vector<Mat *> *newDepthImage = new vector<Mat *>();

			for (int d = 0; d < depthImage->size(); d++) {
				Mat *image = depthImage->at(d);

				Mat *newImage = new Mat(image->rows, image->cols, CV_32F);
				for (int i = 0; i < image->rows; i++) {
					for (int j = 0; j < image->cols; j++) {
						if(this->normalizationCols > 0 && this->normalizationRows > 0){
							//divide the value by the euclidian norm of the region
							newImage->at<float>(i,j) = 
								image->at<float>(i,j) / norm(depthImage, i, j, this->normalizationRows, this->normalizationCols);
						} else {
							//normalization is not computed
							newImage->at<float>(i,j) = image->at<float>(i,j);
						}
					}
				}
				newDepthImage->push_back(newImage);
			}

			return newDepthImage;
		}
		
		int getRowNormalization() { return this->normalizationRows; }
		int getColNormalization() { return this->normalizationCols; }
		
};

#endif
