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

#ifndef _THREELAYERCNN_HPP_
#define _THREELAYERCNN_HPP_

#include <math.h>
#include <time.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define ENTER	10

#define SQRT_EPSILON	1e-7

// including other classes
#include <Neuron.hpp>
#include <Kernel.hpp>

typedef struct {
	string *filename;
	int normalizationRows, normalizationCols;
	vector<Neuron *> *firstlayer;
	int normalizationRowsLayer1, normalizationColsLayer1;
	vector<Neuron *> *secondlayer;
	int normalizationRowsLayer2, normalizationColsLayer2;
	vector<Neuron *> *thirdlayer;
	int normalizationRowsLayer3, normalizationColsLayer3;
} ThreadArgument;

class ThreeLayerCNN {
	private:
		vector<string *> *filelist;
		int firstLayerLength;
		int secondLayerLength;
		int thirdLayerLength;
		double minvalue;
		double maxvalue;
		vector<Neuron *> *firstlayer;
		vector<Neuron *> *secondlayer;
		vector<Neuron *> *thirdlayer;
		int normalizationRows;
		int normalizationCols;

		// layer 1
		int nrowsLayer1;
		int ncolsLayer1;
		int rowStrideLayer1;
		int colStrideLayer1;
		int rowPoolingLayer1;
		int colPoolingLayer1;
		int normalizationRowsLayer1;
		int normalizationColsLayer1;

		// layer 2
		int nrowsLayer2;
		int ncolsLayer2;
		int ndepthLayer2;
		int rowStrideLayer2;
		int colStrideLayer2;
		int rowPoolingLayer2;
		int colPoolingLayer2;
		int normalizationRowsLayer2;
		int normalizationColsLayer2;

		// layer 3
		int nrowsLayer3;
		int ncolsLayer3;
		int ndepthLayer3;
		int rowStrideLayer3;
		int colStrideLayer3;
		int rowPoolingLayer3;
		int colPoolingLayer3;
		int normalizationRowsLayer3;
		int normalizationColsLayer3;

		// kernelWeights
		FILE *kernelWeights;

		double *readWeights(int size) {
			double *buffer = NULL;
			int counter = 0;
			double value;

			for (int i = 0; i < size && !feof(this->kernelWeights); i++) {
				fscanf(this->kernelWeights, "%lf", &value);

				if (!feof(this->kernelWeights)) {
					buffer = (double *) realloc(buffer, sizeof(double) * (counter+1));
					buffer[counter++] = value;
				}
			}

			return buffer;
		}

		char *readFilename(FILE *stream) {
			char *buffer = NULL;
			int counter = 0;
			char character;

			do {
				character = fgetc(stream);
				if (!feof(stream) && character != ENTER) {
					buffer = (char *) realloc(buffer, sizeof(char) * (counter+1));
					buffer[counter++] = character;
				}
			} while (!feof(stream) && character != ENTER);

			buffer = (char *) realloc(buffer, sizeof(char) * (counter+1));
			buffer[counter] = '\0';

			return buffer;
		}

		void loadFilelist(string *filename) {
			this->filelist = new vector<string *>();
			FILE *fp = fopen(filename->c_str(), "r");

			while (!feof(fp)) {
				char *file = readFilename(fp);
				if (strlen(file) > 0) {
					this->filelist->push_back(new string(file));
				}
				free(file); // releasing the buffer
			}

			fclose(fp);
		}

	public:

		void writeWeights(string *filename) {
			int counter = 0;
			double value;

			FILE *fp = fopen(filename->c_str(), "w+");

			for (int i = 0; i < this->firstlayer->size(); i++) {
				Kernel *kernel = this->firstlayer->at(i)->getKernel();
				vector<Mat *> *imgDepth = kernel->getKernel();
				for (int d = 0; d < kernel->getNDepth(); d++) {
					Mat *img = imgDepth->at(d);
					for (int j = 0; j < img->rows; j++) {
						for (int k = 0; k < img->cols; k++) {
							fprintf(fp, "%f\t", img->at<float>(j, k));
						}
					}
					fprintf(fp, "\n");
				}
			}

			for (int i = 0; i < this->secondlayer->size(); i++) {
				Kernel *kernel = this->secondlayer->at(i)->getKernel();
				vector<Mat *> *imgDepth = kernel->getKernel();
				for (int d = 0; d < kernel->getNDepth(); d++) {
					Mat *img = imgDepth->at(d);
					for (int j = 0; j < img->rows; j++) {
						for (int k = 0; k < img->cols; k++) {
							fprintf(fp, "%f\t", img->at<float>(j, k));
						}
					}
					fprintf(fp, "\n");
				}
			}

			for (int i = 0; i < this->thirdlayer->size(); i++) {
				Kernel *kernel = this->thirdlayer->at(i)->getKernel();
				vector<Mat *> *imgDepth = kernel->getKernel();
				for (int d = 0; d < kernel->getNDepth(); d++) {
					Mat *img = imgDepth->at(d);
					for (int j = 0; j < img->rows; j++) {
						for (int k = 0; k < img->cols; k++) {
							fprintf(fp, "%f\t", img->at<float>(j, k));
						}
					}
					fprintf(fp, "\n");
				}
			}

			fclose(fp);
		}

		ThreeLayerCNN(string *filename, 
				int firstLayerLength, int secondLayerLength, int thirdLayerLength,
				double minvalue, double maxvalue, int normalizationRows, int normalizationCols,
				int nrowsLayer1, int ncolsLayer1,
				int rowStrideLayer1, int colStrideLayer1,
				int rowPoolingLayer1, int colPoolingLayer1,
				int normalizationRowsLayer1, int normalizationColsLayer1,
				int nrowsLayer2, int ncolsLayer2, int ndepthLayer2,
				int rowStrideLayer2, int colStrideLayer2,
				int rowPoolingLayer2, int colPoolingLayer2,
				int normalizationRowsLayer2, int normalizationColsLayer2,
				int nrowsLayer3, int ncolsLayer3, int ndepthLayer3,
				int rowStrideLayer3, int colStrideLayer3,
				int rowPoolingLayer3, int colPoolingLayer3,
				int normalizationRowsLayer3, int normalizationColsLayer3) {

			this->filelist = NULL;
			this->loadFilelist(filename);
			this->firstLayerLength = firstLayerLength;
			this->secondLayerLength = secondLayerLength;
			this->thirdLayerLength = thirdLayerLength;
			this->minvalue = minvalue;
			this->maxvalue = maxvalue;

			this->normalizationRows = normalizationRows;
			this->normalizationCols = normalizationCols;
			this->normalizationRowsLayer1 = normalizationRowsLayer1;
			this->normalizationColsLayer1 = normalizationColsLayer1;
			this->normalizationRowsLayer2 = normalizationRowsLayer2;
			this->normalizationColsLayer2 = normalizationColsLayer2;
			this->normalizationRowsLayer3 = normalizationRowsLayer3;
			this->normalizationColsLayer3 = normalizationColsLayer3;

			// layer 1
			this->nrowsLayer1 = nrowsLayer1;
			this->ncolsLayer1 = ncolsLayer1;
			this->rowStrideLayer1 = rowStrideLayer1;
			this->colStrideLayer1 = colStrideLayer1;
			this->rowPoolingLayer1 = rowPoolingLayer1;
			this->colPoolingLayer1 = colPoolingLayer1;

			// layer 2
			this->nrowsLayer2 = nrowsLayer2;
			this->ncolsLayer2 = ncolsLayer2;
			this->ndepthLayer2 = ndepthLayer2;
			this->rowStrideLayer2 = rowStrideLayer2;
			this->colStrideLayer2 = colStrideLayer2;
			this->rowPoolingLayer2 = rowPoolingLayer2;
			this->colPoolingLayer2 = colPoolingLayer2;

			// layer 3
			this->nrowsLayer3 = nrowsLayer3;
			this->ncolsLayer3 = ncolsLayer3;
			this->ndepthLayer3 = ndepthLayer3;
			this->rowStrideLayer3 = rowStrideLayer3;
			this->colStrideLayer3 = colStrideLayer3;
			this->rowPoolingLayer3 = rowPoolingLayer3;
			this->colPoolingLayer3 = colPoolingLayer3;

			// kernelWeights
			this->kernelWeights = NULL;
		}

		ThreeLayerCNN(string *filename, 
				int firstLayerLength, int secondLayerLength, int thirdLayerLength,
				double minvalue, double maxvalue, int normalizationRows, int normalizationCols,
				int nrowsLayer1, int ncolsLayer1,
				int rowStrideLayer1, int colStrideLayer1,
				int rowPoolingLayer1, int colPoolingLayer1,
				int normalizationRowsLayer1, int normalizationColsLayer1,
				int nrowsLayer2, int ncolsLayer2, int ndepthLayer2,
				int rowStrideLayer2, int colStrideLayer2,
				int rowPoolingLayer2, int colPoolingLayer2,
				int normalizationRowsLayer2, int normalizationColsLayer2,
				int nrowsLayer3, int ncolsLayer3, int ndepthLayer3,
				int rowStrideLayer3, int colStrideLayer3,
				int rowPoolingLayer3, int colPoolingLayer3,
				int normalizationRowsLayer3, int normalizationColsLayer3,
				FILE *kernelWeights) {

			this->filelist = NULL;
			this->loadFilelist(filename);
			this->firstLayerLength = firstLayerLength;
			this->secondLayerLength = secondLayerLength;
			this->thirdLayerLength = thirdLayerLength;
			this->minvalue = minvalue;
			this->maxvalue = maxvalue;

			this->normalizationRows = normalizationRows;
			this->normalizationCols = normalizationCols;
			this->normalizationRowsLayer1 = normalizationRowsLayer1;
			this->normalizationColsLayer1 = normalizationColsLayer1;
			this->normalizationRowsLayer2 = normalizationRowsLayer2;
			this->normalizationColsLayer2 = normalizationColsLayer2;
			this->normalizationRowsLayer3 = normalizationRowsLayer3;
			this->normalizationColsLayer3 = normalizationColsLayer3;

			// layer 1
			this->nrowsLayer1 = nrowsLayer1;
			this->ncolsLayer1 = ncolsLayer1;
			this->rowStrideLayer1 = rowStrideLayer1;
			this->colStrideLayer1 = colStrideLayer1;
			this->rowPoolingLayer1 = rowPoolingLayer1;
			this->colPoolingLayer1 = colPoolingLayer1;

			// layer 2
			this->nrowsLayer2 = nrowsLayer2;
			this->ncolsLayer2 = ncolsLayer2;
			this->ndepthLayer2 = ndepthLayer2;
			this->rowStrideLayer2 = rowStrideLayer2;
			this->colStrideLayer2 = colStrideLayer2;
			this->rowPoolingLayer2 = rowPoolingLayer2;
			this->colPoolingLayer2 = colPoolingLayer2;

			// layer 3
			this->nrowsLayer3 = nrowsLayer3;
			this->ncolsLayer3 = ncolsLayer3;
			this->ndepthLayer3 = ndepthLayer3;
			this->rowStrideLayer3 = rowStrideLayer3;
			this->colStrideLayer3 = colStrideLayer3;
			this->rowPoolingLayer3 = rowPoolingLayer3;
			this->colPoolingLayer3 = colPoolingLayer3;

			// kernelWeights
			this->kernelWeights = kernelWeights;
		}

		ThreeLayerCNN(vector<string *> *filelist, 
				int firstLayerLength, int secondLayerLength, int thirdLayerLength,
				double minvalue, double maxvalue, int normalizationRows, int normalizationCols,
				int nrowsLayer1, int ncolsLayer1,
				int rowStrideLayer1, int colStrideLayer1,
				int rowPoolingLayer1, int colPoolingLayer1,
				int normalizationRowsLayer1, int normalizationColsLayer1,
				int nrowsLayer2, int ncolsLayer2, int ndepthLayer2,
				int rowStrideLayer2, int colStrideLayer2,
				int rowPoolingLayer2, int colPoolingLayer2,
				int normalizationRowsLayer2, int normalizationColsLayer2,
				int nrowsLayer3, int ncolsLayer3, int ndepthLayer3,
				int rowStrideLayer3, int colStrideLayer3,
				int rowPoolingLayer3, int colPoolingLayer3,
				int normalizationRowsLayer3, int normalizationColsLayer3) {

			this->filelist = filelist;
			this->firstLayerLength = firstLayerLength;
			this->secondLayerLength = secondLayerLength;
			this->thirdLayerLength = thirdLayerLength;
			this->minvalue = minvalue;
			this->maxvalue = maxvalue;

			this->normalizationRows = normalizationRows;
			this->normalizationCols = normalizationCols;
			this->normalizationRowsLayer1 = normalizationRowsLayer1;
			this->normalizationColsLayer1 = normalizationColsLayer1;
			this->normalizationRowsLayer2 = normalizationRowsLayer2;
			this->normalizationColsLayer2 = normalizationColsLayer2;
			this->normalizationRowsLayer3 = normalizationRowsLayer3;
			this->normalizationColsLayer3 = normalizationColsLayer3;

			// layer 1
			this->nrowsLayer1 = nrowsLayer1;
			this->ncolsLayer1 = ncolsLayer1;
			this->rowStrideLayer1 = rowStrideLayer1;
			this->colStrideLayer1 = colStrideLayer1;
			this->rowPoolingLayer1 = rowPoolingLayer1;
			this->colPoolingLayer1 = colPoolingLayer1;

			// layer 2
			this->nrowsLayer2 = nrowsLayer2;
			this->ncolsLayer2 = ncolsLayer2;
			this->ndepthLayer2 = ndepthLayer2;
			this->rowStrideLayer2 = rowStrideLayer2;
			this->colStrideLayer2 = colStrideLayer2;
			this->rowPoolingLayer2 = rowPoolingLayer2;
			this->colPoolingLayer2 = colPoolingLayer2;

			// layer 3
			this->nrowsLayer3 = nrowsLayer3;
			this->ncolsLayer3 = ncolsLayer3;
			this->ndepthLayer3 = ndepthLayer3;
			this->rowStrideLayer3 = rowStrideLayer3;
			this->colStrideLayer3 = colStrideLayer3;
			this->rowPoolingLayer3 = rowPoolingLayer3;
			this->colPoolingLayer3 = colPoolingLayer3;

			// kernelWeights
			this->kernelWeights = NULL;
		}

		virtual ~ThreeLayerCNN() {

			// Releasing filelist
			for (int i = 0; i < this->filelist->size(); i++) {
				string *str = this->filelist->at(i);
				delete str;
			}
			delete this->filelist;

			// Releasing firstlayer
			for (int i = 0; i < this->firstlayer->size(); i++) {
				Neuron *neuron = this->firstlayer->at(i);
				delete neuron->getKernel();
				delete neuron;
			}
			delete this->firstlayer;

			// Releasing secondlayer
			for (int i = 0; i < this->secondlayer->size(); i++) {
				Neuron *neuron = this->secondlayer->at(i);
				delete neuron->getKernel();
				delete neuron;
			}
			delete this->secondlayer;

			// Releasing thirdlayer
			for (int i = 0; i < this->thirdlayer->size(); i++) {
				Neuron *neuron = this->thirdlayer->at(i);
				delete neuron->getKernel();
				delete neuron;
			}
			delete this->thirdlayer;
		}

		/*
		 TODO: We can create a method parallelForward to distribute workloads to pthreads
		*/
		vector<vector <Mat *> *> *parallelForward(int nthreads) {
			vector<vector <Mat *> *> *results = new vector<vector <Mat *> *>();
			vector <Mat *> *element = NULL;

			cout << "Building the network..." << endl;
			// building the network
			this->buildNetwork();

			// for every filename in this list
			int i = 0, j;
			pthread_t *thread_id = (pthread_t *) malloc(sizeof(pthread_t) * nthreads);

			cout << "Starting threads..." << endl;
			while (i < this->filelist->size()) {

				for (j = 0; j < nthreads && i+j < this->filelist->size(); j++) {

					// preparing thread arguments
					ThreadArgument *arg = (ThreadArgument *) malloc(sizeof(ThreadArgument));
					arg->filename 		= this->filelist->at(i+j);
					arg->normalizationRows	= this->normalizationRows;
					arg->normalizationCols	= this->normalizationCols;
					arg->firstlayer		= this->firstlayer;
					arg->normalizationRowsLayer1 = this->normalizationRowsLayer1;
					arg->normalizationColsLayer1 = this->normalizationColsLayer1;
					arg->secondlayer             = this->secondlayer;
					arg->normalizationRowsLayer2 = this->normalizationRowsLayer2;
					arg->normalizationColsLayer2 = this->normalizationColsLayer2;
					arg->thirdlayer              = this->thirdlayer;
					arg->normalizationRowsLayer3 = this->normalizationRowsLayer3;
					arg->normalizationColsLayer3 = this->normalizationColsLayer3;

					#ifdef DEBUG
						cout << "Processing file " << i+j << ": " << arg->filename->c_str() << endl;
					#endif

					// starting threads
					pthread_create(&thread_id[j], NULL, (void *(*) (void *)) &ThreeLayerCNN::forwardStepThread, 
								(void *) arg);
				}

				// waiting for thread results (in order)
				for (j = 0; j < nthreads && i+j < this->filelist->size(); j++) {
					pthread_join(thread_id[j], (void **) &element);

					#ifdef DEBUG
						cout << "\tDepth: " << element->size() << 
							" nrows: " << element->at(DEFAULT_DEPTH)->rows <<
							" ncols: " << element->at(DEFAULT_DEPTH)->cols << endl;
					#endif

					results->push_back(element);
				}

				i += j;
			}

			free(thread_id);

			return results;
		}

		vector<vector <Mat *> *> *forward() {
			vector<vector <Mat *> *> *results = new vector<vector <Mat *> *>();

			// building the network
			this->buildNetwork();

			// for every filename in this list
			for (int i = 0; i < this->filelist->size(); i++) {
				string *filename = this->filelist->at(i);

				#ifdef DEBUG
					cout << "Processing file " << i << ": " << filename->c_str() << endl;
				#endif

				vector <Mat *> *partialResult = this->forwardStep(filename);

				#ifdef DEBUG
					cout << "\tDepth: " << partialResult->size() << 
						" nrows: " << partialResult->at(DEFAULT_DEPTH)->rows <<
						" ncols: " << partialResult->at(DEFAULT_DEPTH)->cols << endl;
				#endif

				// we may save the results...
				results->push_back(partialResult);
			}

			return results;
		}

	private:

		static Mat *sumImageDepth(vector <Mat *> *imageInDepth) {
			Mat *result = NULL;
			for (int i = 0; i < imageInDepth->size(); i++) {
				if (result == NULL) {
					result = new Mat(imageInDepth->at(i)->rows, imageInDepth->at(i)->cols, CV_32F);
					*result = *(imageInDepth->at(i));
				} else {
					*result += *(imageInDepth->at(i));
				}
			}

			return result;
		}

		static float norm(Mat *image, int rowId, int colId, int nRows, int nCols) {
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

		static float norm(vector<Mat *> *depthImage, int rowId, int colId, int nRows, int nCols) {
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

		static Mat *normalizeImage(Mat *image, int nRows, int nCols) {

			Mat *newImage = new Mat(image->rows, image->cols, CV_32F);

			for (int i = 0; i < image->rows; i++) {
				for (int j = 0; j < image->cols; j++) {
					newImage->at<float>(i,j) = image->at<float>(i,j) / norm(image, i, j, nRows, nCols);
				}
			}

			return newImage;
		}

		static vector<Mat *> *normalizeImageDepth(vector<Mat *> *depthImage, int nRows, int nCols) {

			vector<Mat *> *newDepthImage = new vector<Mat *>();

			for (int d = 0; d < depthImage->size(); d++) {
				Mat *image = depthImage->at(d);

				Mat *newImage = new Mat(image->rows, image->cols, CV_32F);
				for (int i = 0; i < image->rows; i++) {
					for (int j = 0; j < image->cols; j++) {
						newImage->at<float>(i,j) = 
							image->at<float>(i,j) / norm(depthImage, i, j, nRows, nCols);
					}
				}
				newDepthImage->push_back(newImage);
			}

			return newDepthImage;
		}

		static vector<Mat *> *forwardStepThread(ThreadArgument *arg) {

			// loading image in grayscale
    			Mat image = imread(arg->filename->c_str(), CV_LOAD_IMAGE_GRAYSCALE);

			// Converting the image to CV_32F (float)
			Mat image32F(image.rows, image.cols, CV_32F);
			image.convertTo(image32F, CV_32F);

			// Normalizing (but requires conversion to float beforehand) -- From now on it is manipulated as CV_32F (float)
			Mat *normImage = ThreeLayerCNN::normalizeImage(&image32F, arg->normalizationRows, arg->normalizationCols);

			// processing first layer
			// 	- convolution
			// 	- threshold filtering
			// 	- pooling
			vector<Mat *> *firstLayerResults = new vector<Mat *>();
			for (int i = 0; i < arg->firstlayer->size(); i++) {
				firstLayerResults->push_back(arg->firstlayer->at(i)->process(normImage));
			}
			delete normImage;

			// Normalizing
			vector <Mat *> *normFirstLayerResults = ThreeLayerCNN::normalizeImageDepth(firstLayerResults, 
					arg->normalizationRowsLayer1, arg->normalizationColsLayer1);

			// Releasing firstlayer memory
			for (int i = 0; i < firstLayerResults->size(); i++) {
				Mat *result = firstLayerResults->at(i);
				delete result;
			}
			delete firstLayerResults;
			
			// processing second layer
			// 	- convolution
			// 	- threshold filtering
			// 	- pooling
			vector<Mat *> *secondLayerResults = new vector<Mat *>();
			for (int i = 0; i < arg->secondlayer->size(); i++) {
				secondLayerResults->push_back(arg->secondlayer->at(i)->process(normFirstLayerResults));
			}

			// Normalizing
			vector <Mat *> *normSecondLayerResults = ThreeLayerCNN::normalizeImageDepth(secondLayerResults,
						arg->normalizationRowsLayer2, arg->normalizationColsLayer2);

			// Releasing secondlayer memory
			for (int i = 0; i < secondLayerResults->size(); i++) {
				Mat *result = secondLayerResults->at(i);
				delete result;
			}
			delete secondLayerResults;

			// processing third layer
			// 	- convolution
			// 	- threshold filtering
			// 	- pooling
			vector<Mat *> *thirdLayerResults = new vector<Mat *>();
			for (int i = 0; i < arg->thirdlayer->size(); i++) {
				thirdLayerResults->push_back(arg->thirdlayer->at(i)->process(normSecondLayerResults));
			}

			// Normalizing
			vector<Mat *> *normThirdLayerResults = ThreeLayerCNN::normalizeImageDepth(thirdLayerResults,
						arg->normalizationRowsLayer3, arg->normalizationColsLayer3);

			// Releasing thirdlayer memory
			for (int i = 0; i < thirdLayerResults->size(); i++) {
				Mat *result = thirdLayerResults->at(i);
				delete result;
			}
			delete thirdLayerResults;

			// releasing all memory
			for (int i = 0; i < normFirstLayerResults->size(); i++) {
				Mat *result = normFirstLayerResults->at(i);
				delete result;
			}
			delete normFirstLayerResults;

			for (int i = 0; i < normSecondLayerResults->size(); i++) {
				Mat *result = normSecondLayerResults->at(i);
				delete result;
			}
			delete normSecondLayerResults;

			free(arg);

			// returning the thirlayer normalized results
			return normThirdLayerResults;
		}

		vector<Mat *> *forwardStep(string *filename) {
			// loading image in grayscale
    			Mat image = imread(filename->c_str(), CV_LOAD_IMAGE_GRAYSCALE);

			// Converting the image to CV_32F (float)
			Mat image32F(image.rows, image.cols, CV_32F);
			image.convertTo(image32F, CV_32F);

			// Normalizing (but requires conversion to float beforehand) -- From now on it is manipulated as CV_32F (float)
			Mat *normImage = ThreeLayerCNN::normalizeImage(&image32F, normalizationRows, normalizationCols);

			// processing first layer
			// 	- convolution
			// 	- threshold filtering
			// 	- pooling
			vector<Mat *> *firstLayerResults = new vector<Mat *>();
			for (int i = 0; i < this->firstlayer->size(); i++) {
				firstLayerResults->push_back(this->firstlayer->at(i)->process(normImage));
			}
			delete normImage;

			// Normalizing
			vector <Mat *> *normFirstLayerResults = ThreeLayerCNN::normalizeImageDepth(firstLayerResults, 
					normalizationRowsLayer1, normalizationColsLayer1);

			// Releasing firstlayer memory
			for (int i = 0; i < firstLayerResults->size(); i++) {
				Mat *result = firstLayerResults->at(i);
				delete result;
			}
			delete firstLayerResults;
			
			// processing second layer
			// 	- convolution
			// 	- threshold filtering
			// 	- pooling
			vector<Mat *> *secondLayerResults = new vector<Mat *>();
			for (int i = 0; i < this->secondlayer->size(); i++) {
				secondLayerResults->push_back(this->secondlayer->at(i)->process(normFirstLayerResults));
			}

			// Normalizing
			vector <Mat *> *normSecondLayerResults = ThreeLayerCNN::normalizeImageDepth(secondLayerResults,
						normalizationRowsLayer2, normalizationColsLayer2);

			// Releasing secondlayer memory
			for (int i = 0; i < secondLayerResults->size(); i++) {
				Mat *result = secondLayerResults->at(i);
				delete result;
			}
			delete secondLayerResults;

			// processing third layer
			// 	- convolution
			// 	- threshold filtering
			// 	- pooling
			vector<Mat *> *thirdLayerResults = new vector<Mat *>();
			for (int i = 0; i < this->thirdlayer->size(); i++) {
				thirdLayerResults->push_back(this->thirdlayer->at(i)->process(normSecondLayerResults));
			}

			// Normalizing
			vector<Mat *> *normThirdLayerResults = ThreeLayerCNN::normalizeImageDepth(thirdLayerResults,
						normalizationRowsLayer3, normalizationColsLayer3);

			// Releasing thirdlayer memory
			for (int i = 0; i < thirdLayerResults->size(); i++) {
				Mat *result = thirdLayerResults->at(i);
				delete result;
			}
			delete thirdLayerResults;

			// releasing all memory
			for (int i = 0; i < normFirstLayerResults->size(); i++) {
				Mat *result = normFirstLayerResults->at(i);
				delete result;
			}
			delete normFirstLayerResults;

			for (int i = 0; i < normSecondLayerResults->size(); i++) {
				Mat *result = normSecondLayerResults->at(i);
				delete result;
			}
			delete normSecondLayerResults;

			// returning the thirlayer normalized results
			return normThirdLayerResults;
		}

		void buildNetwork() {
			if (this->kernelWeights == NULL) {
				// layer 1
				// =======
				this->firstlayer = new vector<Neuron *>();
				for (int i = 0; i < this->firstLayerLength; i++) {
					Kernel *kernel = new Kernel(nrowsLayer1, ncolsLayer1, minvalue, maxvalue);
					this->firstlayer->push_back(new Neuron(kernel, rowStrideLayer1, 
								colStrideLayer1, rowPoolingLayer1, colPoolingLayer1));
				}

				// layer 2
				// =======
				this->secondlayer = new vector<Neuron *>();
				for (int i = 0; i < this->secondLayerLength; i++) {
					Kernel *kernel = new Kernel(nrowsLayer2, ncolsLayer2, ndepthLayer2,
								minvalue, maxvalue);
					this->secondlayer->push_back(new Neuron(kernel, rowStrideLayer2, 
								colStrideLayer2, rowPoolingLayer2, colPoolingLayer2));
				}

				// layer 3
				// =======
				this->thirdlayer = new vector<Neuron *>();
				for (int i = 0; i < this->thirdLayerLength; i++) {
					Kernel *kernel = new Kernel(nrowsLayer3, ncolsLayer3, ndepthLayer3,
								minvalue, maxvalue);
					this->thirdlayer->push_back(new Neuron(kernel, rowStrideLayer3,
								colStrideLayer3, rowPoolingLayer3, colPoolingLayer3));
				}
			} else {
				// layer 1
				// =======
				this->firstlayer = new vector<Neuron *>();
				for (int i = 0; i < this->firstLayerLength; i++) {
					double *weights = this->readWeights(nrowsLayer1 * ncolsLayer1);
					Kernel *kernel = new Kernel(weights, nrowsLayer1, ncolsLayer1);
					free(weights);
					this->firstlayer->push_back(new Neuron(kernel, rowStrideLayer1, 
								colStrideLayer1, rowPoolingLayer1, colPoolingLayer1));
				}

				// layer 2
				// =======
				this->secondlayer = new vector<Neuron *>();
				for (int i = 0; i < this->secondLayerLength; i++) {
					double *weights = this->readWeights(nrowsLayer2 * ncolsLayer2 * ndepthLayer2);
					Kernel *kernel = new Kernel(weights, nrowsLayer2, ncolsLayer2, ndepthLayer2);
					free(weights);
					this->secondlayer->push_back(new Neuron(kernel, rowStrideLayer2, 
								colStrideLayer2, rowPoolingLayer2, colPoolingLayer2));
				}

				// layer 3
				// =======
				this->thirdlayer = new vector<Neuron *>();
				for (int i = 0; i < this->thirdLayerLength; i++) {
					double *weights = this->readWeights(nrowsLayer3 * ncolsLayer3 * ndepthLayer3);
					Kernel *kernel = new Kernel(weights, nrowsLayer3, ncolsLayer3, ndepthLayer3);
					free(weights);
					this->thirdlayer->push_back(new Neuron(kernel, rowStrideLayer3,
								colStrideLayer3, rowPoolingLayer3, colPoolingLayer3));
				}

			}

		}
};

#endif
