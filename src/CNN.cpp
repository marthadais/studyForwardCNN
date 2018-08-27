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

// including other classes
#include <IOfiles.hpp>
#include <Neuron.hpp>
#include <Kernel.hpp>
#include <CNN.hpp>

enum {
	PROGNAME, 
	FILENAME, // xml file
	FEATURES_FILE, // output file
	EXECUTION_MODE, // sequencial or parallel
	NTHREADS, // number of threads for parallel case
	NARGS = NTHREADS
};

/* execution mode */
enum {
	SEQUENTIAL,
	PARALLEL,
	TRAINING
};

int main(int argc, char *argv[]) {

	/* print the help information if some argument is missing */
	if (argc < NARGS) {
		cout <<" Usage: " << argv[PROGNAME] 
			<< " parameters.xml cnn-output-filename [0 | 1 #threads]" << endl;
		cout << "\tfirst parameter is XML config file" << endl;
		cout << "\tsecond parameter is the features file produced by CNN" << endl;
		cout << "\tthird parameter is the execution mode:" << endl;
		cout << "\t\t0: sequential" << endl;
		cout << "\t\t1: parallel" << endl;
		cout << "\t\t\twhen parallel you must define the number of threads" << endl;
		return -1;
	}
	
	/* create a new CNN with xml file */
	CNN *cnn = NULL;
	cnn = new CNN(argv[FILENAME]);
	int executionMode = atoi(argv[EXECUTION_MODE]); // obtain the execution mode
	
	vector<vector<Mat *> *> *results = NULL; // create the variable for receive the fetures produced
	switch (executionMode) {
		case SEQUENTIAL: 
			cout << "Remember all input images MUST have the same sizes." << endl;
			//Run sequencial CNN
		    results = cnn->forward();
			break;
		case PARALLEL:
			/* if the parallel was setted but the number of threads is missing
			 *  print the help information
			 */
			if (argc < (NARGS+1)) {
				cout <<" Usage: " << argv[PROGNAME] 
					<< " parameters.xml cnn-output-filename [0 | 1 #threads]" << endl;
				cout << "\tfirst parameter is XML config file" << endl;
				cout << "\tsecond parameter is the features file produced by CNN" << endl;
				cout << "\tthird parameter is the execution mode:" << endl;
				cout << "\t\t0: sequential" << endl;
				cout << "\t\t1: parallel" << endl;
				cout << "\t\t\twhen parallel you must define the number of threads" << endl;
				return -1;
			}
			int nthreads = atoi(argv[NTHREADS]); // obtain number of threads
			cout << "Remember all input images MUST have the same sizes." << endl;
			//Run parallel CNN
			results = cnn->parallelForward(nthreads);
			break;
	}

	cout << results->size() << " images were processed." << endl;

	/* If the input images have different sizes 
	 *	the features produced need to be cutted for the minimum size 
	 * 	to keep the same size
	 */
	/* obtain the minimum size in rows and cols */
	int minRows = INT_MAX;
	int minCols = INT_MAX;
	cout << "Discovering minRows/minCols for features" << endl;
	for (int i = 0; i < results->size(); i++) {
		for (int j = 0; j < results->at(i)->size(); j++) {
			Mat *feature = results->at(i)->at(j);
			if (feature->rows < minRows) minRows = feature->rows;
			if (feature->cols < minCols) minCols = feature->cols;
		}
	}
	
	cout << "Producing the output file " << argv[FEATURES_FILE] << " with all features" << endl;
	
	/* Open the class file passed in xml file 
	 * the class information is put on the last colunm of the features
	 * this information is not used on CNN 
	 */
	char* classesfile = cnn->getClassFileName();
	FILE *classId = fopen(classesfile, "r+");
	if (classId == NULL){
		cout << "ERROR: cannot open the file " << classesfile << endl;
		return 1;
	}
	
	/* open the output file */
	FILE *featuresFile = fopen(argv[FEATURES_FILE], "w+");
	if (!featuresFile) {
		cout << "ERROR: features file cannot be created." << endl;
	}

	// getting all feature vectors for every image
	for (int i = 0; i < results->size(); i++) {
		for (int j = 0; j < results->at(i)->size(); j++) {
			Mat *feature = results->at(i)->at(j);
			// save features in output file, cutting based in minimum size
			writeAllData(featuresFile, feature, minRows, minCols);
		}

		// read class information of current image
		char *classIdentifier = readClassId(classId);
		// save class information in output file
		fprintf(featuresFile, " %s\n", classIdentifier);
		free(classIdentifier);
	}

	fclose(classId);
	fclose(featuresFile);

	delete classesfile;
	delete results;
	delete cnn;

	return 0;
}
