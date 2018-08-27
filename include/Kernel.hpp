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

#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Kernel"

#define MINVALUE 0
#define MAXVALUE 1
#define EPSILON	 1e-7

#define MICROSEC 1000000

#define DEFAULT_DEPTH 1

// Exceptions
#define NROWS_EXCEPTION 	0
#define NCOLS_EXCEPTION 	1
#define NDEPTH_EXCEPTION	2
#define MINVALUE_MAXVALUE_EXCEPTION	3

class Kernel {
	private:
		int nrows;
		int ncols;
		int ndepth;
		double minvalue; // 0 is indicated
		double maxvalue; // 1 is indicated
		vector<Mat *> *kernel;

	public:
		/*
		 * random single Kernel constructor
		 */
		Kernel(int nrows, int ncols) {

			if (nrows < 1) throw NROWS_EXCEPTION;
			if (ncols < 1) throw NCOLS_EXCEPTION;

			this->minvalue = MINVALUE;
			this->maxvalue = MAXVALUE;
			this->nrows = nrows;
			this->ncols = ncols;
			this->ndepth = DEFAULT_DEPTH;
			this->kernel = NULL;
			this->generateKernel();
		}

		/*
		 * random Kernel constructor in depth
		 */
		Kernel(int nrows, int ncols, int ndepth) {

			if (nrows < 1) throw NROWS_EXCEPTION;
			if (ncols < 1) throw NCOLS_EXCEPTION;
			if (ndepth < 1) throw NDEPTH_EXCEPTION;

			this->minvalue = MINVALUE;
			this->maxvalue = MAXVALUE;
			this->nrows = nrows;
			this->ncols = ncols;
			this->ndepth = ndepth;
			this->kernel = NULL;
			this->generateKernel();
		}

		/*
		 * random single Kernel constructor with maximum and minimum values defined
		 */
		Kernel(int nrows, int ncols, double minvalue, double maxvalue) {
			
			if (nrows < 1) throw NROWS_EXCEPTION;
			if (ncols < 1) throw NCOLS_EXCEPTION;
			if (minvalue >= maxvalue) throw MINVALUE_MAXVALUE_EXCEPTION;

			this->minvalue = minvalue;
			this->maxvalue = maxvalue;
			this->nrows = nrows;
			this->ncols = ncols;
			this->ndepth = DEFAULT_DEPTH;
			this->kernel = NULL;
			this->generateKernel();
		}

		/*
		 * random Kernel constructor in depth with maximum and minimum values defined
		 */
		Kernel(int nrows, int ncols, int ndepth, double minvalue, double maxvalue) {
			if (nrows < 1) throw NROWS_EXCEPTION;
			if (ncols < 1) throw NCOLS_EXCEPTION;
			if (ndepth < 1) throw NDEPTH_EXCEPTION;
			if (minvalue >= maxvalue) throw MINVALUE_MAXVALUE_EXCEPTION;

			this->minvalue = minvalue;
			this->maxvalue = maxvalue;
			this->nrows = nrows;
			this->ncols = ncols;
			this->ndepth = ndepth;
			this->kernel = NULL;
			this->generateKernel();
		}
		
		/*
		 * single Kernel constructor using values from a given file
		 */
		Kernel(double *weights, int nrows, int ncols) {

			if (nrows < 1) throw NROWS_EXCEPTION;
			if (ncols < 1) throw NCOLS_EXCEPTION;

			this->minvalue = MINVALUE;
			this->maxvalue = MAXVALUE;
			this->nrows = nrows;
			this->ncols = ncols;
			this->ndepth = DEFAULT_DEPTH;
			this->kernel = NULL;
			this->generateKernel(weights);
		}

		/*
		 * Kernel constructor in depth using values from a given file
		 */
		Kernel(double *weights, int nrows, int ncols, int ndepth) {

			if (nrows < 1) throw NROWS_EXCEPTION;
			if (ncols < 1) throw NCOLS_EXCEPTION;

			this->minvalue = MINVALUE;
			this->maxvalue = MAXVALUE;
			this->nrows = nrows;
			this->ncols = ncols;
			this->ndepth = ndepth;
			this->kernel = NULL;
			this->generateKernel(weights);
		}
		
		virtual ~Kernel() {
			if (this->kernel) {

				for (int d = 0; d < this->kernel->size(); d++) {
					Mat *depthKernel = this->kernel->at(d);
					delete depthKernel;
				}

				delete this->kernel;
			}
		}

	private:
		/*
		 * Generate a kernel using random values
		 */
		void generateKernel() {
			// INFORMATION:
			//
			// 	Kernels must be in range [0,1]
			// 	average = sum_i_1_n (x_i,j,k) / n
			// 	*x_i,j,k = x_i,j,k - average
			//	**x_i,j,k = *x_i,j,k / sqrt(sum((*x_i,j,k)^2))
			//
			double mean = 0;
			struct timeval tval;
			gettimeofday(&tval, NULL);
			srand(tval.tv_sec * MICROSEC + tval.tv_usec);

			this->kernel = new vector<Mat *>();

			for (int d = 0; d < this->ndepth; d++) {
				Mat *depthKernel = new Mat(this->nrows, this->ncols, CV_32F);

				for(int i = 0; i < depthKernel->rows; i++) {
					for(int j = 0; j < depthKernel->cols; j++) {
						depthKernel->at<float>(i,j) = (float)
							(this->minvalue + (rand() / (RAND_MAX * 1.0)) * 
							(this->maxvalue - this->minvalue));
						mean += depthKernel->at<float>(i,j);
					}
				}

				this->kernel->push_back(depthKernel);
			}

			mean = mean / (this->nrows * this->ncols * this->ndepth);

			for (int d = 0; d < this->ndepth; d++) {
				Mat *depthKernel = this->kernel->at(d);

				for(int i = 0; i < depthKernel->rows; i++) {
					for(int j = 0; j < depthKernel->cols; j++) {
						depthKernel->at<float>(i,j) = 
							depthKernel->at<float>(i,j) - mean;
					}
				}

				this->kernel->at(d) = depthKernel;
			}

			double sum = 0;
			for (int d = 0; d < this->ndepth; d++) {
				Mat *depthKernel = this->kernel->at(d);

				for(int i = 0; i < depthKernel->rows; i++) {
					for(int j = 0; j < depthKernel->cols; j++) {
						sum += pow(depthKernel->at<float>(i,j), 2.0);
					}
				}
			}

			for (int d = 0; d < this->ndepth; d++) {
				Mat *depthKernel = this->kernel->at(d);

				for(int i = 0; i < depthKernel->rows; i++) {
					for(int j = 0; j < depthKernel->cols; j++) {
						depthKernel->at<float>(i,j) =
							depthKernel->at<float>(i,j) / (sqrt(sum) + EPSILON);
					}
				}

				this->kernel->at(d) = depthKernel;
			}
		}

		/*
		 * Generate a kernel using values from a given file
		 */
		void generateKernel(double *weights) {
			int counter = 0;

			this->kernel = new vector<Mat *>();

			for (int d = 0; d < this->ndepth; d++) {
				Mat *depthKernel = new Mat(this->nrows, this->ncols, CV_32F);

				for(int i = 0; i < depthKernel->rows; i++)
					for(int j = 0; j < depthKernel->cols; j++)
						depthKernel->at<float>(i,j) = (float) weights[counter++];

				this->kernel->push_back(depthKernel);
			}
		}

	public:

		int getNRows() { return this->nrows; }
		int getNCols() { return this->ncols; }
		int getNDepth() { return this->ndepth; }
		double getMinValue() { return this->minvalue; }
		double getMaxValue() { return this->maxvalue; }
		vector<Mat *> *getKernel() { return this->kernel; }

		/*
		 * This method updates the Kernel and return the dCdK, i.e., the partial
		 * derivative of the resultant matrix C in the direction of the Kernel K
		 */
		double train(Mat *I, Mat *C, double eta) {
			double deltaTheta = 0.0;

			for (int i = 0; i < I->rows; i++)
				for (int j = 0; j < I->cols; j++)
					deltaTheta = deltaTheta + 2.0*C->at<float>(i,j)*I->at<float>(i,j);
			deltaTheta /= (I->rows * I->cols);

			for (int d = 0; d < this->ndepth; d++) {
				for (int i = 0; i < this->nrows; i++) {
					for (int j = 0; j < this->ncols; j++) {
						this->kernel->at(d)->at<float>(i,j) = 
							this->kernel->at(d)->at<float>(i,j) - eta * deltaTheta;
					}
				}
			}

			return deltaTheta;
		}

		void show() {
			if (this->ndepth == DEFAULT_DEPTH)
				imshow(WINDOW_NAME, *(this->kernel->at(0)));
			else
				cout << "void Kernel::show() -- I cannot show it" << endl;
		}

};

#endif
