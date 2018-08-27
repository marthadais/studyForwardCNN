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

#include <stdlib.h>
#include <Neuron.hpp>
#include <Kernel.hpp>
#include <ThreeLayerCNN.hpp>

#define FEATURE_FILE			"features.dat"
#define FEATURE_SELECTION_TYPE_NONE	-1
#define FEATURE_SELECTION_TYPE_RANDOM	 0
#define FEATURE_SELECTION_TYPE_VARIANCE	 1
#define FEATURE_SELECTION_TYPE_ALL	 2

enum {
	PROGNAME,
	FILENAME,
	FIRSTLAYERLENGTH,
	SECONDLAYERLENGTH,
	THIRDLAYERLENGTH,
	MIN_VALUE,
	MAX_VALUE,
	NORMALIZATION_NROWS,
	NORMALIZATION_NCOLS,
	NROWSLAYER1,
	NCOLSLAYER1,
	ROWSTRIDELAYER1,
	COLSTRIDELAYER1,
	ROWPOOLINGLAYER1,
	COLPOOLINGLAYER1,
	NORMALIZATION_NROWS_LAYER1,
	NORMALIZATION_NCOLS_LAYER1,
	NROWSLAYER2,
	NCOLSLAYER2,
	ROWSTRIDELAYER2,
	COLSTRIDELAYER2,
	ROWPOOLINGLAYER2,
	COLPOOLINGLAYER2,
	NORMALIZATION_NROWS_LAYER2,
	NORMALIZATION_NCOLS_LAYER2,
	NROWSLAYER3,
	NCOLSLAYER3,
	ROWSTRIDELAYER3,
	COLSTRIDELAYER3,
	ROWPOOLINGLAYER3,
	COLPOOLINGLAYER3,
	NORMALIZATION_NROWS_LAYER3,
	NORMALIZATION_NCOLS_LAYER3,
	FEATURE_SELECTION_TYPE,
	NFEATURES,
	CLASSID_FILENAME,
	NTHREADS,
	EXECUTION_MODE,
	ETA,
	THRESHOLD,
	NARGS
};

enum {
	SEQUENTIAL,
	PARALLEL,
	TRAINING
};

char *readClassId(FILE *stream) {
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

void writeAllData(FILE *featuresFile, Mat *feature, int minRows, int minCols) {
	
	for (int i = 0; i < minRows; i++)
		for (int j = 0; j < minCols; j++)
			fprintf(featuresFile, "%f ", feature->at<float>(i,j));
}

void writeDataAtIndex(FILE *featuresFile, Mat *feature, int minRows, int minCols, int index) {
	
	int rowId = (int) index / (minCols * 1.0);
	int colId = index - (rowId * minCols);
	fprintf(featuresFile, "%f ", feature->at<float>(rowId,colId));
}

int main(int argc, char *argv[]) {

	if (argc != NARGS) {
		cout <<" Usage: " << argv[PROGNAME] << " filelist.txt firstLayerLength secondLayerLength thirdLayerLength minvalue maxvalue normalizationNRows normalizationNCols nrowsLayer1 ncolsLayer1 rowStrideLayer1 colStrideLayer1 rowPoolingLayer1 colPoolingLayer1 normalizationNRowsLayer1 normalizationNColsLayer1 nrowsLayer2 ncolsLayer2 rowStrideLayer2 colStrideLayer2 rowPoolingLayer2 colPoolingLayer2 normalizationNRowsLayer2 normalizationNColsLayer2 nrowsLayer3 ncolsLayer3 rowStrideLayer3 colStrideLayer3 rowPoolingLayer3 colPoolingLayer3 normalizationNRowsLayer3 normalizationNColsLayer3 featureSelectionType nFeatures classIdFilename nthreads execution_mode eta threshold" << endl;
		return -1;
	}

	string *filename 	= new string(argv[FILENAME]);
	int firstLayerLength 	= atoi(argv[FIRSTLAYERLENGTH ]);
	int secondLayerLength 	= atoi(argv[SECONDLAYERLENGTH]);
	int thirdLayerLength 	= atoi(argv[THIRDLAYERLENGTH ]);
	double minvalue 	= atof(argv[MIN_VALUE        ]);
	double maxvalue 	= atof(argv[MAX_VALUE        ]);
	int normNrows		= atoi(argv[NORMALIZATION_NROWS]);
	int normNcols		= atoi(argv[NORMALIZATION_NCOLS]);
	int nrowsLayer1 	= atoi(argv[NROWSLAYER1      ]);
	int ncolsLayer1 	= atoi(argv[NCOLSLAYER1      ]);
	int rowStrideLayer1 	= atoi(argv[ROWSTRIDELAYER1  ]);
	int colStrideLayer1 	= atoi(argv[COLSTRIDELAYER1  ]);
	int rowPoolingLayer1 	= atoi(argv[ROWPOOLINGLAYER1 ]);
	int colPoolingLayer1 	= atoi(argv[COLPOOLINGLAYER1 ]);
	int normNrowsLayer1	= atoi(argv[NORMALIZATION_NROWS_LAYER1]);
	int normNcolsLayer1	= atoi(argv[NORMALIZATION_NCOLS_LAYER1]);
	int nrowsLayer2 	= atoi(argv[NROWSLAYER2      ]);
	int ncolsLayer2 	= atoi(argv[NCOLSLAYER2      ]);
	int ndepthLayer2 	= firstLayerLength;
	int rowStrideLayer2 	= atoi(argv[ROWSTRIDELAYER2 ]);
	int colStrideLayer2 	= atoi(argv[COLSTRIDELAYER2 ]);
	int rowPoolingLayer2 	= atoi(argv[ROWPOOLINGLAYER2]);
	int colPoolingLayer2 	= atoi(argv[COLPOOLINGLAYER2]);
	int normNrowsLayer2	= atoi(argv[NORMALIZATION_NROWS_LAYER2]);
	int normNcolsLayer2	= atoi(argv[NORMALIZATION_NCOLS_LAYER2]);
	int nrowsLayer3 	= atoi(argv[NROWSLAYER3     ]);
	int ncolsLayer3 	= atoi(argv[NCOLSLAYER3     ]);
	int ndepthLayer3 	= secondLayerLength;
	int rowStrideLayer3 	= atoi(argv[ROWSTRIDELAYER3 ]);
	int colStrideLayer3 	= atoi(argv[COLSTRIDELAYER3 ]);
	int rowPoolingLayer3 	= atoi(argv[ROWPOOLINGLAYER3]);
	int colPoolingLayer3 	= atoi(argv[COLPOOLINGLAYER3]);
	int normNrowsLayer3	= atoi(argv[NORMALIZATION_NROWS_LAYER3]);
	int normNcolsLayer3	= atoi(argv[NORMALIZATION_NCOLS_LAYER3]);
	int featureSelectionType= FEATURE_SELECTION_TYPE_NONE;
	int nthreads		= atoi(argv[NTHREADS]);
	int executionMode	= atoi(argv[EXECUTION_MODE]);
	double eta 		= atof(argv[ETA]);
	double threshold	= atof(argv[THRESHOLD]);

	if (strcmp(argv[FEATURE_SELECTION_TYPE], "-r") == 0) {
		featureSelectionType= FEATURE_SELECTION_TYPE_RANDOM;
	} else if (strcmp(argv[FEATURE_SELECTION_TYPE], "-v") == 0) {
		featureSelectionType= FEATURE_SELECTION_TYPE_VARIANCE;
	} else if (strcmp(argv[FEATURE_SELECTION_TYPE], "-a") == 0) {
		featureSelectionType= FEATURE_SELECTION_TYPE_ALL;
	} else {
		cout << "You must define -r (random) or -v (variance) for the feature selection type" << endl;
		return -1;
	}

	if (nthreads < 1) {
		cout << "nthreads must be greater than 0" << endl;
		return -1;
	}

	int nFeatures = atoi(argv[NFEATURES]);
	string *classIdFilename = new string(argv[CLASSID_FILENAME]);
	ThreeLayerCNN *cnn = NULL;

	FILE *kernelWeights = fopen("kernel-weights.dat", "rb");

	if (kernelWeights) { // if there is a file kernel-weights.dat, then we should load it

		#ifdef DEBUG
			cout << "Running with kernel-weights.dat" << endl;
		#endif

		cnn = new ThreeLayerCNN(filename, 
			firstLayerLength, secondLayerLength, thirdLayerLength,
			minvalue, maxvalue, normNrows, normNcols,
			nrowsLayer1, ncolsLayer1,
			rowStrideLayer1, colStrideLayer1,
			rowPoolingLayer1, colPoolingLayer1,
			normNrowsLayer1, normNcolsLayer1,
			nrowsLayer2, ncolsLayer2, ndepthLayer2,
			rowStrideLayer2, colStrideLayer2,
			rowPoolingLayer2, colPoolingLayer2,
			normNrowsLayer2, normNcolsLayer2,
			nrowsLayer3, ncolsLayer3, ndepthLayer3,
			rowStrideLayer3, colStrideLayer3,
			rowPoolingLayer3, colPoolingLayer3, 
			normNrowsLayer3, normNcolsLayer3,
			kernelWeights);
	} else {
		cnn = new ThreeLayerCNN(filename, 
			firstLayerLength, secondLayerLength, thirdLayerLength,
			minvalue, maxvalue, normNrows, normNcols,
			nrowsLayer1, ncolsLayer1,
			rowStrideLayer1, colStrideLayer1,
			rowPoolingLayer1, colPoolingLayer1,
			normNrowsLayer1, normNcolsLayer1,
			nrowsLayer2, ncolsLayer2, ndepthLayer2,
			rowStrideLayer2, colStrideLayer2,
			rowPoolingLayer2, colPoolingLayer2,
			normNrowsLayer2, normNcolsLayer2,
			nrowsLayer3, ncolsLayer3, ndepthLayer3,
			rowStrideLayer3, colStrideLayer3,
			rowPoolingLayer3, colPoolingLayer3,
			normNrowsLayer3, normNcolsLayer3);
	}

	vector<vector<Mat *> *> *results = NULL;
	switch (executionMode) {
		case SEQUENTIAL: results = cnn->forward();
			break;
		case PARALLEL: results = cnn->parallelForward(nthreads);
			break;
	}

	cout << results->size() << " images were processed." << endl;

	int minRows = INT_MAX;
	int minCols = INT_MAX;

	cout << "Discovering minRows/minCols" << endl;
	for (int i = 0; i < results->size(); i++) {
		Mat *feature = results->at(i)->at(0);
		if (feature->rows < minRows) minRows = feature->rows;
		if (feature->cols < minCols) minCols = feature->cols;
	}

	vector<Mat *> *variances = new vector<Mat *>();

	cout << "Computing Variance for " << minRows << "x" << minCols << " matrices" << endl;
	for (int i = 0; i < results->size(); i++) {
		Mat sum = Mat::zeros(minRows, minCols, CV_32F);
		Mat sumSquared = Mat::zeros(minRows, minCols, CV_32F);
		for (int j = 0; j < results->at(i)->size(); j++) {
			Mat *feature = results->at(i)->at(j);
			Mat roi(*feature, Rect(0, 0, minCols, minRows));
			sum = sum + roi;
			sumSquared = sumSquared + roi.mul(roi);
		}

		Mat variance = (sumSquared - (sum.mul(sum)/(results->at(i)->size()*1.0)))
					/(results->at(i)->size()*1.0-1.0);
		Mat *pVariance = new Mat(variance);
		variances->push_back(pVariance);
	}

	Mat setVariance = Mat::zeros(minRows, minCols, CV_32F);
	for (int i = 0; i < variances->size(); i++) {
		Mat copy(*variances->at(i), Rect(0, 0, minCols, minRows));
		setVariance += copy;
	}
	setVariance /= variances->size();

	if (featureSelectionType == FEATURE_SELECTION_TYPE_RANDOM) {
		cout << "The random selection of features is not implemented yet." << endl;
	} else if (featureSelectionType == FEATURE_SELECTION_TYPE_VARIANCE) {
		Mat varianceVector = setVariance.reshape(0, 1);
		Mat sortedVarianceVector, sortedIdxVarianceVector;

		cv::sort(varianceVector, sortedVarianceVector, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
		cv::sortIdx(varianceVector, sortedIdxVarianceVector, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

		/*
		#ifdef DEBUG
			for (int i = 0; i < sortedVarianceVector.cols; i++) {
				cout << "Idx: " << sortedIdxVarianceVector.at<int>(0,i) << " " <<
					sortedVarianceVector.at<float>(0,i) << endl;
			}
		#endif
		*/

		FILE *classId = fopen(classIdFilename->c_str(), "r+");
		FILE *featuresFile = fopen(FEATURE_FILE, "w+");

		// getting all feature vectors for every image
		for (int i = 0; i < results->size(); i++) {
			for (int j = 0; j < sortedIdxVarianceVector.cols && j < nFeatures; j++) {
				for (int k = 0; k < results->at(i)->size(); k++) {
					Mat *feature = results->at(i)->at(k);
					writeDataAtIndex(featuresFile, feature, minRows, minCols,
						sortedIdxVarianceVector.at<int>(j));
				}
			}

			char *classIdentifier = readClassId(classId);
			fprintf(featuresFile, " %s\n", classIdentifier);
			free(classIdentifier);
		}

		fclose(classId);
		fclose(featuresFile);

	} else if (featureSelectionType == FEATURE_SELECTION_TYPE_ALL) {

		FILE *classId = fopen(classIdFilename->c_str(), "r+");
		FILE *featuresFile = fopen(FEATURE_FILE, "w+");

		if (!featuresFile) {
			cout << "features file cannot be created." << endl;
		}

		// getting all feature vectors for every image
		for (int i = 0; i < results->size(); i++) {
			for (int j = 0; j < results->at(i)->size(); j++) {
				Mat *feature = results->at(i)->at(j);
				writeAllData(featuresFile, feature, minRows, minCols);
			}

			char *classIdentifier = readClassId(classId);
			fprintf(featuresFile, " %s\n", classIdentifier);
			free(classIdentifier);
		}

		fclose(classId);
		fclose(featuresFile);
	}

	// Releasing variances
	for (int i = 0; i < variances->size(); i++) {
		Mat *variance = variances->at(i);
		delete variance;
	}
	delete variances;

	// Releasing results
	for (int i = 0; i < results->size(); i++) {
		for (int j = 0; j < results->at(i)->size(); j++) {
			Mat *m = results->at(i)->at(j);
			delete m;
		}
		vector<Mat *> *v = results->at(i);
		delete v;
	}
	delete classIdFilename;
	delete results;
	delete filename;
	delete cnn;

	return 0;
}
