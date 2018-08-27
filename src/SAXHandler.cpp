#include <xercesc/util/XMLUniDefs.hpp>
#include <xercesc/sax/AttributeList.hpp>
#include <SAXFunction.hpp>
#include <iostream>
using namespace std;

// ---------------------------------------------------------------------------
//  Local const data
//
//  Note: This is the 'safe' way to do these strings. If you compiler supports
//        L"" style strings, and portability is not a concern, you can use
//        those types constants directly.
// ---------------------------------------------------------------------------
static const XMLCh  gEndElement[] = { chOpenAngle, chForwardSlash, chNull };
static const XMLCh  gEndPI[] = { chQuestion, chCloseAngle, chNull };
static const XMLCh  gStartPI[] = { chOpenAngle, chQuestion, chNull };
static const XMLCh  gXMLDecl1[] =
{
        chOpenAngle, chQuestion, chLatin_x, chLatin_m, chLatin_l
    ,   chSpace, chLatin_v, chLatin_e, chLatin_r, chLatin_s, chLatin_i
    ,   chLatin_o, chLatin_n, chEqual, chDoubleQuote, chDigit_1, chPeriod
    ,   chDigit_0, chDoubleQuote, chSpace, chLatin_e, chLatin_n, chLatin_c
    ,   chLatin_o, chLatin_d, chLatin_i, chLatin_n, chLatin_g, chEqual
    ,   chDoubleQuote, chNull
};

static const XMLCh  gXMLDecl2[] =
{
        chDoubleQuote, chQuestion, chCloseAngle
    ,   chLF, chNull
};

SAXHandler::SAXHandler( const   char* const              encodingName
                                    , const XMLFormatter::UnRepFlags unRepFlags) :

    fFormatter
    (
        encodingName
        , 0
        , this
        , XMLFormatter::NoEscapes
        , unRepFlags
    )
{
    //
    //  Go ahead and output an XML Decl with our known encoding. This
    //  is not the best answer, but its the best we can do until we
    //  have SAX2 support.
    //
    //fFormatter << gXMLDecl1 << fFormatter.getEncodingName() << gXMLDecl2;

    this->inputs = NULL;
    this->nlayers = -1;
    this->minvalue = 0;
    this->maxvalue = 1;
    this->kernel_weights_file = NULL;
    this->file_random_kernel_weights = NULL;
    this->layer_neurons = NULL;
    this->normalization_nrows = NULL;
    this->normalization_ncols = NULL;
    this->kernel_nrows = NULL;
    this->kernel_ncols = NULL;
    this->maxpooling_rowstride = NULL;
    this->maxpooling_colstride = NULL;
    this->maxpooling_rowpooling = NULL;
    this->maxpooling_colpooling = NULL;
    this->layerCounter = 0;
}

SAXHandler::~SAXHandler()
{
}

void SAXHandler::writeChars(const XMLByte* const /* toWrite */)
{
}

void SAXHandler::writeChars(const XMLByte* const toWrite,
                                  const XMLSize_t      count,
                                  XMLFormatter* const /* formatter */)
{
    // For this one, just dump them to the standard output
    // Surprisingly, Solaris was the only platform on which
    // required the char* cast to print out the string correctly.
    // Without the cast, it was printing the pointer value in hex.
    // Quite annoying, considering every other platform printed
    // the string with the explicit cast to char* below.
    XERCES_STD_QUALIFIER cout.write((char *) toWrite, (int) count);
	XERCES_STD_QUALIFIER cout.flush();
}


void SAXHandler::error(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void SAXHandler::fatalError(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nFatal Error at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void SAXHandler::warning(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nWarning at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}


void SAXHandler::unparsedEntityDecl(const     XMLCh* const /* name */
                                          , const   XMLCh* const /* publicId */
                                          , const   XMLCh* const /* systemId */
                                          , const   XMLCh* const /* notationName */)
{

}


void SAXHandler::notationDecl(const   XMLCh* const /* name */
                                    , const XMLCh* const /* publicId */
                                    , const XMLCh* const /* systemId */)
{

}

void SAXHandler::characters(const     XMLCh* const    chars
                                  , const   XMLSize_t       length)
{
    //fFormatter.formatBuf(chars, length, XMLFormatter::CharEscapes);
}


void SAXHandler::endDocument()
{
}

void SAXHandler::endElement(const XMLCh* const name)
{
    // No escapes are legal here
    //fFormatter << XMLFormatter::NoEscapes << gEndElement << name << chCloseAngle;

    char *element = XMLString::transcode(name);

    if (strcmp(element, "layer") == 0) {
		this->layerCounter++;
    }
}

void SAXHandler::ignorableWhitespace( const   XMLCh* const chars
                                            ,const  XMLSize_t    length)
{
    //fFormatter.formatBuf(chars, length, XMLFormatter::NoEscapes);
}

void SAXHandler::processingInstruction(const  XMLCh* const target
                                            , const XMLCh* const data)
{
	/*
    fFormatter << XMLFormatter::NoEscapes << gStartPI  << target;
    if (data)
        fFormatter << chSpace << data;
    fFormatter << XMLFormatter::NoEscapes << gEndPI;*/
}

void SAXHandler::startDocument()
{

}

void SAXHandler::startElement(const XMLCh* const name, AttributeList&  attributes) {

    //take the values of the attributes passed by xml file

    char *element = XMLString::transcode(name);

    if (strcmp(element, "cnn") == 0) {
	    XMLSize_t len = attributes.getLength();
	    
	    for (XMLSize_t index = 0; index < len; index++)
	    {
		    char *attrname = XMLString::transcode(attributes.getName(index));

		    if (strcmp(attrname, "inputs") == 0) {
				//list of images path
				inputs = XMLString::transcode(attributes.getValue(index));
		    } else if (strcmp(attrname, "labels") == 0) {
				//list of labels
				labels = XMLString::transcode(attributes.getValue(index));
		    } else if (strcmp(attrname, "nlayers") == 0) {
				//number of layers
				nlayers = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "minvalue") == 0) {
				//min value for kernel
				minvalue = atof(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "maxvalue") == 0) {
				//max value for kernel
				maxvalue = atof(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "kernel-weights-file") == 0) {
				//path of kernel file
				kernel_weights_file = XMLString::transcode(attributes.getValue(index));
		    } else if (strcmp(attrname, "file-random-kernel-weights") == 0) {
				//file of random kernel
				file_random_kernel_weights = XMLString::transcode(attributes.getValue(index));
		    }
	    }
    } else if (strcmp(element, "layer") == 0) {
		
		XMLSize_t len = attributes.getLength();
		for (XMLSize_t index = 0; index < len; index++)
	    {
		    char *attrname = XMLString::transcode(attributes.getName(index));
			//number os neurons per layer
		    if (strcmp(attrname, "nneurons") == 0) {
				layer_neurons = (int *) realloc(layer_neurons, sizeof(int) * (layerCounter+1));
				layer_neurons[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
			}
		}
			
    } else if (strcmp(element, "normalization") == 0) {
	    normalization_nrows = (int *) realloc(normalization_nrows, sizeof(int) * (layerCounter+1));
	    normalization_ncols = (int *) realloc(normalization_ncols, sizeof(int) * (layerCounter+1));
	    normalization_nrows[layerCounter] = 1;
	    normalization_ncols[layerCounter] = 1;

	    XMLSize_t len = attributes.getLength();
	    for (XMLSize_t index = 0; index < len; index++)
	    {
		    char *attrname = XMLString::transcode(attributes.getName(index));
			
			//number of rows and cols for normalization
		    if (strcmp(attrname, "nrows") == 0) {
				normalization_nrows[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "ncols") == 0) {
				normalization_ncols[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    }
	    }

    } else if (strcmp(element, "kernel") == 0) {
	    kernel_nrows = (int *) realloc(kernel_nrows, sizeof(int) * (layerCounter+1));
	    kernel_ncols = (int *) realloc(kernel_ncols, sizeof(int) * (layerCounter+1));
	    kernel_nrows[layerCounter] = 1;
	    kernel_ncols[layerCounter] = 1;

	    XMLSize_t len = attributes.getLength();
	    for (XMLSize_t index = 0; index < len; index++)
	    {
		    char *attrname = XMLString::transcode(attributes.getName(index));
			
			//number of rows and cols for kernel
		    if (strcmp(attrname, "nrows") == 0) {
				kernel_nrows[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "ncols") == 0) {
				kernel_ncols[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    }
	    }
    } else if (strcmp(element, "maxpooling") == 0) {
	    maxpooling_rowstride = (int *) realloc(maxpooling_rowstride, sizeof(int) * (layerCounter+1));
	    maxpooling_colstride = (int *) realloc(maxpooling_colstride, sizeof(int) * (layerCounter+1));
	    maxpooling_rowpooling = (int *) realloc(maxpooling_rowpooling, sizeof(int) * (layerCounter+1));
	    maxpooling_colpooling = (int *) realloc(maxpooling_colpooling, sizeof(int) * (layerCounter+1));

	    XMLSize_t len = attributes.getLength();
	    for (XMLSize_t index = 0; index < len; index++)
	    {
		    char *attrname = XMLString::transcode(attributes.getName(index));

			//number of rows and cols for mask and stride sizes of max pooling 
		    if (strcmp(attrname, "rowstride") == 0) {
				maxpooling_rowstride[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "colstride") == 0) {
				maxpooling_colstride[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "rowpooling") == 0) {
				maxpooling_rowpooling[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "colpooling") == 0) {
				maxpooling_colpooling[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    }
	    }
    } else if (strcmp(element, "final-normalization") == 0) {
	    normalization_nrows = (int *) realloc(normalization_nrows, sizeof(int) * (layerCounter+1));
	    normalization_ncols = (int *) realloc(normalization_ncols, sizeof(int) * (layerCounter+1));
	    normalization_nrows[layerCounter] = 1;
	    normalization_ncols[layerCounter] = 1;

	    XMLSize_t len = attributes.getLength();
	    for (XMLSize_t index = 0; index < len; index++)
	    {
		    char *attrname = XMLString::transcode(attributes.getName(index));

			//number of rows and cols for final normalization
		    if (strcmp(attrname, "nrows") == 0) {
				normalization_nrows[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    } else if (strcmp(attrname, "ncols") == 0) {
				normalization_ncols[layerCounter] = atoi(XMLString::transcode(attributes.getValue(index)));
		    }
	    }
    }
}
