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

#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// including other classes
#include <Kernel.hpp>

#define DEFAULT_INPUT		0
#define DEFAULT_KERNEL_INDEX	0

class Neuron {
	private:
		Kernel *kernel;

	public:

		/*
		 * Neuron constructor 
		 */
		Neuron(Kernel *kernel) {

			this->kernel = kernel;
		}

		virtual ~Neuron() {
			
		}

	private:

		/***** CONVOLUTION *****/
		/*
		 * Compute the convolution in a single image
		 */
		Mat *convolve(Mat *input, int kernelIndex) {
			
			Mat filtered(input->rows, input->cols, CV_32F);
			Mat *filtered_32F = NULL;

			#ifdef GPU

			#else			
				// convolving
				filter2D(*input, filtered, -1, *this->kernel->getKernel()->at(kernelIndex), 
						Point( -1, -1 ), 0, BORDER_DEFAULT);
			#endif

			filtered_32F = new Mat(filtered);

			return filtered_32F;
		}

	public:

		Mat *process(Mat *input) {

			// convolving...
			Mat *filtered_32F = this->convolve(input, DEFAULT_KERNEL_INDEX);
			
			// Apply threshold here on filtered_32F
			// ReLU function activation -- max(value,0)
			//int count = 0;
			for (int i = 0; i < filtered_32F->rows; i++) {
				for (int j = 0; j < filtered_32F->cols; j++) {
					if (filtered_32F->at<float>(i,j) < 0){
						filtered_32F->at<float>(i,j) = 0;
						//count++;
					}
				}
			}
			
			//cout << "l=0 Negative values = " << count << " of " << filtered_32F->rows*filtered_32F->cols << endl;

			return filtered_32F;
		}
		
		Mat *process(vector<Mat *> *input) {

			int nrows = input->at(DEFAULT_INPUT)->rows;
			int ncols = input->at(DEFAULT_INPUT)->cols;
			Mat sumConvolution = Mat::zeros(nrows, ncols, CV_32F);

			// convolving...
			for (int d = 0; d < input->size(); d++) {
				Mat *depthInput = input->at(d);
				Mat *filtered_32F = this->convolve(depthInput, d);
				sumConvolution += *filtered_32F;
				delete filtered_32F; // Releasing the filtered_32F
			}

			// Apply threshold here on sumConvolution
			// ReLU function activation -- max(value,0)
			//int count = 0;
			for (int i = 0; i < sumConvolution.rows; i++) {
				for (int j = 0; j < sumConvolution.cols; j++) {
					if (sumConvolution.at<float>(i,j) < 0){
						sumConvolution.at<float>(i,j) = 0;
						//count++;
					}
				}
			}
			//cout << "Negative values = " << count << " of " << sumConvolution.rows*sumConvolution.cols << endl;

			Mat *result = new Mat(sumConvolution);

			return result;
		}

		Kernel *getKernel() { return this->kernel; }
		double trainKernel(Mat *I, Mat *C, double eta) {
			return this->kernel->train(I, C, eta);
		}
		
};

#endif
