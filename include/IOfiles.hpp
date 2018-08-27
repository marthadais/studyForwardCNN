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

#ifndef _IOfiles_HPP_
#define _IOfiles_HPP_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#define ENTER	10

/*
 * This function read the class information of a specific image
 */
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

/*
 * This function save the features produced for a image
 */
void writeAllData(FILE *featuresFile, Mat *feature, int minRows, int minCols) {
	
	for (int i = 0; i < minRows; i++)
		for (int j = 0; j < minCols; j++)
			fprintf(featuresFile, "%f ", feature->at<float>(i,j));
}


/*
 * This function is not being used 
 */
void writeDataAtIndex(FILE *featuresFile, Mat *feature, int minRows, int minCols, int index) {
	
	int rowId = (int) index / (minCols * 1.0);
	int colId = index - (rowId * minCols);
	fprintf(featuresFile, "%f ", feature->at<float>(rowId,colId));
}

#endif
