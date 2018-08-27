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

#ifndef _printImages_HPP_
#define _printImages_HPP_

#include <cfloat>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/*
 * Function to find the max value of a matrix
 */
float maxMatrix(Mat *m) {
	float max=FLT_MIN;
	//take the maximun and the minimum value of the matrix
	int i,j;
	for(i=0; i < m->rows; i++) {
		for(j=0; j < m->cols; j++) {
			//max value
			if(m->at<float>(i,j) > max) {
				max = m->at<float>(i,j);
			}
		}
	}
	return max;
}

/*
 * Function to find the min value of a matrix
 */
float minMatrix(Mat *m) {
	float min=FLT_MAX;
	//take the maximum and the minimum value of the matrix
	int i,j;
	for(i=0; i < m->rows; i++) {
		for(j=0; j < m->cols; j++) {
			//min value
			if(m->at<float>(i,j) < min) {
				min = m->at<float>(i,j);
			}
		}
	}
	return min;
}

/*
 * Normalize image between 0 and 255
 */
Mat *normalizeImage2print(Mat *m) {
	int i,j;
	float  min=minMatrix(m), max=maxMatrix(m);
	Mat *result = new Mat(m->rows, m->cols, CV_32F);
		//normalization
	for(i=0; i < m->rows; i++) {
		for(j=0; j < m->cols; j++) {
			result->at<float>(i,j) = (m->at<float>(i,j) - min)/(max-min) * 255.0;
		}
	}

	return result;
}

/*
 * Saving images in a folder
 * 		the result images from each CNN layer are save in a folder
 */
void printImages(vector<Mat *> * imagesResults, string filename, int layer){
	
	// create the folder name based on layer number
	stringstream convert;
	convert << layer;
	string folder = "images_layer_" + convert.str() + "/";
	
	// create a folder
	struct stat st = {0};
	if (stat(folder.c_str(), &st) == -1) {
		mkdir(folder.c_str(), 0700);
	}
	
	// spli the filename
	char * pch;
	char *name;
	char * dup = strdup(filename.c_str());
	pch = strtok (dup,"/");
	while (pch != NULL)
	{
		name = pch;
		pch = strtok (NULL, "/");
	}
	
	// create the base name of images
	folder = folder + name;
	
	for (int i = 0; i < imagesResults->size(); i++) {
		
		// create the filename of images
		stringstream convert2;
		convert2 << i;
		string imageFile = folder + convert2.str() + ".jpg";
		
		#ifdef DEBUG
			cout << "Saving images in " << imageFile << endl;
		#endif
		
		Mat *image = new Mat(imagesResults->at(i)->rows,imagesResults->at(i)->cols, CV_8U);
		// normalize and convert images
		normalizeImage2print(imagesResults->at(i))->convertTo(*image, CV_8U);
		//save images
		imwrite(imageFile, *image);
		image->release();
		delete image;
	}
}

#endif
