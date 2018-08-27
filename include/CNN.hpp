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

#ifndef _CNN_HPP_
#define _CNN_HPP_

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

//including classes for xml parser
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/parsers/SAXParser.hpp>
#include <SAXFunction.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>

#include <printImages.hpp>

//including variables for xml parser
static bool                     doNamespaces        = false;
static bool                     doSchema            = false;
static bool                     schemaFullChecking  = false;
static const char*              encodingName    = "LATIN1";
static XMLFormatter::UnRepFlags unRepFlags      = XMLFormatter::UnRep_CharRef;
static SAXParser::ValSchemes    valScheme       = SAXParser::Val_Auto;

using namespace cv;
using namespace std;

#define ENTER	10

#define SQRT_EPSILON	1e-7

// including other classes
#include <LocalNormalization.hpp>
#include <Neuron.hpp>
#include <Kernel.hpp>
#include <Pooling.hpp>

typedef struct {
	string *filename;
	vector<LocalNormalization *> *layersNorm;
	vector<vector<Neuron *> *> *layers;
	vector<Pooling *> *layersPooling;
	int nlayers;				// e.g. 3
} ThreadArgument;

class CNN {
	private:
		int nlayers;				// e.g. 3
		vector<LocalNormalization *> *layersNorm; // normalization layers
		vector<vector<Neuron *> *> *layers; // convolution layers		
		vector<Pooling *> *layersPooling; // pooling layers		
		
		double minvalue;			// 0
		double maxvalue;			// 1
		int *layerLength;			// e.g. number of kernels per layer	{10, 20, 20}
		int *nrowsLayer;			// e.g. kernel mask rows	{3, 5, 15}
		int *ncolsLayer;			// e.g. kernel mask cols	{3, 5, 15}
		int *rowStrideLayer;			
		int *colStrideLayer;
		int *rowPoolingLayer;
		int *colPoolingLayer;
		int *normalizationRows;		// e.g. if we have 3 layers, we will have 4 normalization values {10, 10, 10, 10}
		int *normalizationCols;		// e.g. if we have 3 layers, we will have 4 normalization values {10, 10, 10, 10}

		// list of images to process
		vector<string *> *filelist;
		
		// file that contains list of classes
		char* classesFile;

		// kernel weights file
		char *kernelFilename;
		FILE *kernelWeights;
		char *randomKernelFilename;
		
		/****** READ XML FILE ******/
		//read the xml file 
		void loadXMLFile(const char *xmlfilename) {
			try {
				 XMLPlatformUtils::Initialize();
			} catch (const XMLException& toCatch) {
				 XERCES_STD_QUALIFIER cerr << "Error during initialization! :\n"
					  << StrX(toCatch.getMessage()) << XERCES_STD_QUALIFIER endl;
				 return;
			}

			int errorCount = 0;

			//
			//  Create a SAX parser object. Then, according to what we were told on
			//  the command line, set it to validate or not.
			//
			SAXParser* parser = new SAXParser;
			parser->setValidationScheme(valScheme);
			parser->setDoNamespaces(doNamespaces);
			parser->setDoSchema(doSchema);
			parser->setHandleMultipleImports (true);
			parser->setValidationSchemaFullChecking(schemaFullChecking);

			//
			//  Create the handler object and install it as the document and error
			//  handler for the parser-> Then parse the file and catch any exceptions
			//  that propogate out
			//
			int errorCode = 0;
			try {
				SAXHandler handler(encodingName, unRepFlags);
				parser->setDocumentHandler(&handler);
				parser->setErrorHandler(&handler);
				parser->parse(xmlfilename);
				errorCount = parser->getErrorCount();

				// ----------------- Set information ------------------------------
				
				//files input
				int flag =0;
				flag = loadFilelist(handler.getInputs());
				if(flag == 1) return;
			
				//file classes
				classesFile = handler.getLabels();
				
				//file kernels
				this->kernelFilename = handler.getKernelWeightsFile();
				if(strcmp(kernelFilename,"") != 0){
					if((this->kernelWeights = fopen(this->kernelFilename,"r"))==NULL){
						cout << "ERROR: Could not open file: " << this->kernelFilename << endl;
					}
				}else{
					this->kernelWeights = NULL;
					// file to save the random kernels
					this->randomKernelFilename = handler.getFileRandomKernelWeights();
				}
				
				//other parameters
				this->nlayers = handler.getNlayers();
				this->minvalue = handler.getMinvalue();
				this->maxvalue = handler.getMaxvalue();
				
				layerLength = (int *)malloc(sizeof(int)*this->nlayers);
				for (int i = 0; i < this->nlayers; i++) {
					layerLength[i] = handler.getLayerNeurons()[i];
				}

				normalizationRows = (int *)malloc((sizeof(int)*this->nlayers)+1);
				normalizationCols = (int *)malloc((sizeof(int)*this->nlayers)+1);
				for (int i = 0; i < this->nlayers+1; i++) {
					normalizationRows[i] = handler.getNormalizationNrows()[i];
					normalizationCols[i] = handler.getNormalizationNcols()[i];
				}

				nrowsLayer = (int *)malloc(sizeof(int)*this->nlayers);
				ncolsLayer = (int *)malloc(sizeof(int)*this->nlayers);
				for (int i = 0; i < this->nlayers; i++) {
					nrowsLayer[i] = handler.getKernelNrows()[i];
					ncolsLayer[i] = handler.getKernelNcols()[i];
				}
				
				rowStrideLayer = (int *)malloc(sizeof(int)*this->nlayers);
				colStrideLayer = (int *)malloc(sizeof(int)*this->nlayers);
				rowPoolingLayer = (int *)malloc(sizeof(int)*this->nlayers);
				colPoolingLayer = (int *)malloc(sizeof(int)*this->nlayers);
				for (int i = 0; i < this->nlayers; i++) {
					rowStrideLayer[i] = handler.getMaxPoolingRowStride()[i];
					colStrideLayer[i] = handler.getMaxPoolingColStride()[i];
					rowPoolingLayer[i] = handler.getMaxPoolingRowPooling()[i];
					colPoolingLayer[i] = handler.getMaxPoolingColPooling()[i];
				}

			} catch (const OutOfMemoryException&) {
				XERCES_STD_QUALIFIER cerr << "OutOfMemoryException" << XERCES_STD_QUALIFIER endl;
				errorCode = 5;
			} catch (const XMLException& toCatch) {
				XERCES_STD_QUALIFIER cerr << "\nAn error occurred\n  Error: "
					 << StrX(toCatch.getMessage())
					 << "\n" << XERCES_STD_QUALIFIER endl;
				errorCode = 4;
			}

			if(errorCode) {
				XMLPlatformUtils::Terminate();
			}

			//  Delete the parser itself.  Must be done prior to calling Terminate, below.
			delete parser;

			XMLPlatformUtils::Terminate();
		}
		
		/****** READ INPUT IMAGES ******/
		/*
		 * read paths of input image
		 */
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
		
		/*
		 * read the file that contains paths of input images
		 */
		int loadFilelist(char *filename) {
			this->filelist = new vector<string *>();
			FILE *fp = fopen(filename, "r");
			
			if(fp==NULL){
				cout << "Could not open file: " << filename << endl;
				return 1;
			}

			while (!feof(fp)) {
				char *file = readFilename(fp);
				if (strlen(file) > 0) {
					this->filelist->push_back(new string(file));
				}
				free(file); // releasing the buffer
			}

			fclose(fp);
			return 0;
		}


	public:
		char * getClassFileName() { return this->classesFile; }
	
		/*
		 *  save the weights random ganerated
		 */
		void saveWeights(string filename) {
			int counter = 0;
			double value;

			FILE *fp = fopen(filename.c_str(), "w+");

			for (int l = 0; l < this->nlayers; l++) {

				// saving kernel weights for layer at index l
				vector<Neuron *> *layer = this->layers->at(l);

				for (int i = 0; i < layer->size(); i++) {
					Kernel *kernel = layer->at(i)->getKernel();
					vector<Mat *> *imgDepth = kernel->getKernel();
					for (int d = 0; d < kernel->getNDepth(); d++) {
						Mat *img = imgDepth->at(d);
						for (int j = 0; j < img->rows; j++) {
							for (int k = 0; k < img->cols; k++) {
								fprintf(fp, "%lf\t", img->at<float>(j, k));
							}
						}
						fprintf(fp, "\n");
					}
				}
			}

			fclose(fp);
		}

		/*
		 *  read the trained weigths
		 */
		double *loadWeights(int size) {
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

		/*
		 * CNN constructor that read the xml file
		 */
		CNN(const char *xmlfilename) {
			this->loadXMLFile(xmlfilename);
		}
		
		/*
		 * Virtual CNN desctructor
		 */
		virtual ~CNN() {

			// Releasing filelist
			for (int i = 0; i < this->filelist->size(); i++) {
				string *str = this->filelist->at(i);
				delete str;
			}
			delete this->filelist;

			// Releasing layers
			for (int l = 0; l < this->nlayers; l++) {

				vector <Neuron *> *layer = this->layers->at(l);

				for (int i = 0; i < layer->size(); i++) {
					Neuron *neuron = layer->at(i);
					delete neuron->getKernel();
					delete neuron;
				}
				delete layer;
			}

			delete this->layers;
		}

		/*
		 *  Execute the parallel CNN
		 */
		vector<vector <Mat *> *> *parallelForward(int nthreads) {
			vector<vector <Mat *> *> *results = new vector<vector <Mat *> *>();
			vector <Mat *> *element = NULL;

			#ifdef DEBUG
				cout << "Parallel process was chosen." << endl;
				cout << "Number of files = " << this->filelist->size() << endl;
			#endif
			
			// building the network
			this->buildNetwork();

			// for every filename in this list
			int i = 0, j;
			pthread_t *thread_id = (pthread_t *) malloc(sizeof(pthread_t) * nthreads);

			#ifdef DEBUG
				cout << "Starting threads..." << endl;
			#endif
			
			while (i < this->filelist->size()) {

				for (j = 0; j < nthreads && i+j < this->filelist->size(); j++) {

					// preparing thread arguments
					ThreadArgument *arg = (ThreadArgument *) malloc(sizeof(ThreadArgument));
					arg->filename 	= this->filelist->at(i+j);
					arg->layersNorm = this->layersNorm;
					arg->layers		= this->layers;
					arg->layersPooling = this->layersPooling;
					arg->nlayers	= this->nlayers;

					#ifdef DEBUG
						cout << "Processing file " << i+j << ": " << arg->filename->c_str() << endl;
					#endif

					// starting threads
					pthread_create(&thread_id[j], NULL, (void *(*) (void *)) &CNN::forwardStepThread, 
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

		/*
		 *  Execute the sequencial CNN
		 */
		vector<vector <Mat *> *> *forward() {
			vector<vector <Mat *> *> *results = new vector<vector <Mat *> *>();
			
			cout << "Sequential process chosen." << endl;
			cout << "Processing " << this->filelist->size() << " files." << endl;

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

		/***** CNN PROCESS *****/
		/*
		 * Process parallel CNN for a single image
		 */
		static vector<Mat *> *forwardStepThread(ThreadArgument *arg) {

			// loading image in grayscale
    		Mat image = imread(arg->filename->c_str(), CV_LOAD_IMAGE_GRAYSCALE);

			// Converting the image to CV_32F (float)
			Mat image32F(image.rows, image.cols, CV_32F);
			image.convertTo(image32F, CV_32F);

			vector<Mat *> *layerResults = NULL;

			for (int l = 0; l < arg->nlayers; l++) {

				vector <Neuron *> *layer = arg->layers->at(l);
				
				if (l == 0) {
					// compute local normalization for the input image
					Mat *normImage = arg->layersNorm->at(l)->normalizeImage(&image32F);
					
					// compute convolution for the input image
					vector<Mat *> *convolveResult = new vector<Mat *>();
					for (int i = 0; i < layer->size(); i++) { 
						convolveResult->push_back(layer->at(i)->process(normImage));
					}
					
					// compute max pooling for all output images produced by the previous layer
					layerResults = new vector<Mat *>();
					layerResults = arg->layersPooling->at(l)->applyPooling(convolveResult);
					
					delete normImage;
					
				} else {
					// compute local normalization for all output images produced by the previous layer
					vector <Mat *> *normLayerResults = arg->layersNorm->at(l)->normalizeImage(layerResults);
					
					// Releasing previous layer memory
					for (int i = 0; i < layerResults->size(); i++) {
						Mat *result = layerResults->at(i);
						delete result;
					}
					delete layerResults;

					// Compute convolution for all output images produced by the previous layer
					vector<Mat *> *convolveResult = new vector<Mat *>();
					for (int i = 0; i < layer->size(); i++) {
						convolveResult->push_back(layer->at(i)->process(normLayerResults));  
					}
					
					// compute max pooling for all output images produced by the previous layer
					layerResults = new vector<Mat *>();
					layerResults = arg->layersPooling->at(l)->applyPooling(convolveResult);
					
					// releasing normalization previous layer memory
					for (int i = 0; i < normLayerResults->size(); i++) {
						Mat *result = normLayerResults->at(i);
						delete result;
					}
					delete normLayerResults;
				}
			}

			// compute final local normalization for all output images produced by the previous layer
			vector <Mat *> *cnnResults = arg->layersNorm->at(arg->nlayers)->normalizeImage(layerResults);

			// Releasing layer memory
			for (int i = 0; i < layerResults->size(); i++) {
				Mat *result = layerResults->at(i);
				delete result;
			}
			delete layerResults;

			free(arg);

			return cnnResults;
		}
		
		/*
		 * Process sequencial CNN for a single image
		 */
		vector<Mat *> *forwardStep(string *filename) {
			// loading image in grayscale
    		Mat image = imread(filename->c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    			
    		// Converting the image to CV_32F (float)
			Mat image32F(image.rows, image.cols, CV_32F);
			image.convertTo(image32F, CV_32F);
			
			vector<Mat *> *layerResults = NULL;

			for (int l = 0; l < this->nlayers; l++) {
				
				vector <Neuron *> *layer = this->layers->at(l);

				if (l == 0) {
					// compute local normalization for the input image
					Mat *normImage = this->layersNorm->at(l)->normalizeImage(&image32F);
					
					// compute convolution for the input image
					vector<Mat *> *convolveResult = new vector<Mat *>();
					for (int i = 0; i < layer->size(); i++) { 
						convolveResult->push_back(layer->at(i)->process(normImage));
					}
					
					// compute max pooling for all output images produced by the previous layer
					layerResults = new vector<Mat *>();
					layerResults = this->layersPooling->at(l)->applyPooling(convolveResult);
					
					delete normImage;
					
					// save images produced
					printImages(layerResults,filename->c_str(),l);
					
				} else {
					// compute local normalization for all output images produced by the previous layer
					vector <Mat *> *normLayerResults = this->layersNorm->at(l)->normalizeImage(layerResults);
					
					// Releasing previous layer memory
					for (int i = 0; i < layerResults->size(); i++) {
						Mat *result = layerResults->at(i);
						delete result;
					}
					delete layerResults;

					// Compute convolution for all output images produced by the previous layer
					vector<Mat *> *convolveResult = new vector<Mat *>();
					for (int i = 0; i < layer->size(); i++) {
						convolveResult->push_back(layer->at(i)->process(normLayerResults));  
					}
					
					// compute max pooling for all output images produced by the previous layer
					layerResults = new vector<Mat *>();
					layerResults = this->layersPooling->at(l)->applyPooling(convolveResult);
					
					// releasing normalization previous layer memory
					for (int i = 0; i < normLayerResults->size(); i++) {
						Mat *result = normLayerResults->at(i);
						delete result;
					}
					delete normLayerResults;
					
					// save images produced
					printImages(layerResults,filename->c_str(),l);
				}
			}

			// compute final local normalization for all output images produced by the previous layer
			vector <Mat *> *cnnResults = this->layersNorm->at(this->nlayers)->normalizeImage(layerResults);

			// Releasing layer memory
			for (int i = 0; i < layerResults->size(); i++) {
				Mat *result = layerResults->at(i);
				delete result;
			}
			delete layerResults;

			return cnnResults;
		}

		/****** BUILD NETWORK ******/
		/*
		 * Build the network design according to the parameters passed in xml file 
		 */
		void buildNetwork() {
			
			#ifdef DEBUG
				cout << "Building Network..." << endl;
			#endif
			
			this->layersNorm = new vector<LocalNormalization *>();
			this->layers = new vector<vector<Neuron *> *>();
			this->layersPooling = new vector<Pooling *>();
			
			// Verify if the kernel values must be random generate or read from a file
			if (this->kernelWeights == NULL) {	
				
				for (int l = 0; l < this->nlayers; l++) {
					
					if (l == 0) {
						//for normalization layer
						LocalNormalization *newNormlayer = new LocalNormalization(this->normalizationRows[l],
										this->normalizationCols[l]);		
						this->layersNorm->push_back(newNormlayer);
						
						//for first layer just have one kernel
						vector<Neuron *> *newlayer = new vector<Neuron *>();

						for (int i = 0; i < this->layerLength[l]; i++) {
							Kernel *kernel = new Kernel(this->nrowsLayer[l], this->ncolsLayer[l], 
									this->minvalue, this->maxvalue);
							newlayer->push_back(new Neuron(kernel));
						}
						
						this->layers->push_back(newlayer);
						
						//for pooling layer
						Pooling *newPoollayer = new Pooling(this->rowStrideLayer[l], 
										this->colStrideLayer[l], 
										this->rowPoolingLayer[l], 
										this->colPoolingLayer[l]);		
						this->layersPooling->push_back(newPoollayer);
						
					} else {
						//for normalization layer
						LocalNormalization *newNormlayer = new LocalNormalization(this->normalizationRows[l],
										this->normalizationCols[l]);
						this->layersNorm->push_back(newNormlayer);
						
						//for others layers to include depth in kernel						
						vector<Neuron *> *newlayer = new vector<Neuron *>();
						
						for (int i = 0; i < this->layerLength[l]; i++) {
							Kernel *kernel = new Kernel(this->nrowsLayer[l], this->ncolsLayer[l], this->layerLength[l-1],
									this->minvalue, this->maxvalue);
							newlayer->push_back(new Neuron(kernel));
						}
				
						this->layers->push_back(newlayer);
						
						//for pooling layer
						Pooling *newPoollayer = new Pooling(this->rowStrideLayer[l], 
										this->colStrideLayer[l], 
										this->rowPoolingLayer[l], 
										this->colPoolingLayer[l]);		
						this->layersPooling->push_back(newPoollayer);
					}
				}
				
				//for final normalization layer
				LocalNormalization *newNormlayer = new LocalNormalization(this->normalizationRows[this->nlayers],
								this->normalizationCols[this->nlayers]);
				this->layersNorm->push_back(newNormlayer);
				
				// saving random kernel defined for current layer
				cout << "Saving random kernel values in " << this->randomKernelFilename << endl;
				
				string rkFilename (this->randomKernelFilename);
				this->saveWeights(rkFilename);
				
			} else {
				
				cout << "Reading kernel values from " << this->kernelFilename << endl;
				
				for (int l = 0; l < this->nlayers; l++) {
					
					if (l == 0) {
						//for normalization layer
						LocalNormalization *newNormlayer = new LocalNormalization(this->normalizationRows[l],
										this->normalizationCols[l]);
						this->layersNorm->push_back(newNormlayer);
						
						//for first layer just have one kernel				
						vector<Neuron *> *newlayer = new vector<Neuron *>();
						for (int i = 0; i < this->layerLength[l]; i++) {
							double *weights = this->loadWeights(this->nrowsLayer[l] * this->ncolsLayer[l]);
							Kernel *kernel = new Kernel(weights, this->nrowsLayer[l], this->ncolsLayer[l]);
							free(weights);
							newlayer->push_back(new Neuron(kernel));
						}
						
						this->layers->push_back(newlayer);
						
						//for pooling layer
						Pooling *newPoollayer = new Pooling(this->rowStrideLayer[l], 
										this->colStrideLayer[l], 
										this->rowPoolingLayer[l], 
										this->colPoolingLayer[l]);		
						this->layersPooling->push_back(newPoollayer);
						
					} else {
						//for normalization layer
						LocalNormalization *newNormlayer = new LocalNormalization(this->normalizationRows[l],
										this->normalizationCols[l]);
						this->layersNorm->push_back(newNormlayer);
										
						//for others layers to include depth in kernel						
						vector<Neuron *> *newlayer = new vector<Neuron *>();
						for (int i = 0; i < this->layerLength[l]; i++) {
							double *weights = this->loadWeights(this->nrowsLayer[l] 
									* this->ncolsLayer[l]
									* this->layerLength[l-1]);
							Kernel *kernel = new Kernel(weights, this->nrowsLayer[l], 
									this->ncolsLayer[l],
								        this->layerLength[l-1]);
							free(weights);
							newlayer->push_back(new Neuron(kernel));
						}
				
						this->layers->push_back(newlayer);
						
						//for pooling layer
						Pooling *newPoollayer = new Pooling(this->rowStrideLayer[l], 
										this->colStrideLayer[l], 
										this->rowPoolingLayer[l], 
										this->colPoolingLayer[l]);		
						this->layersPooling->push_back(newPoollayer);
					}
				}
				
				//for final normalization layer
				LocalNormalization *newNormlayer = new LocalNormalization(this->normalizationRows[this->nlayers],
								this->normalizationCols[this->nlayers]);
				this->layersNorm->push_back(newNormlayer);
				
			}

		}
};

#endif
