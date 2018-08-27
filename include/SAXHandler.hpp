#include <iostream>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/framework/XMLFormatter.hpp>

XERCES_CPP_NAMESPACE_USE

class SAXHandler : public HandlerBase, private XMLFormatTarget
{
public:

    SAXHandler
    (
        const   char* const                 encodingName
        , const XMLFormatter::UnRepFlags    unRepFlags
    );
    ~SAXHandler();

    void writeChars
    (
        const   XMLByte* const  toWrite
    );

    virtual void writeChars
    (
        const   XMLByte* const  toWrite
        , const XMLSize_t       count
        , XMLFormatter* const   formatter
    );

    void endDocument();

    void endElement(const XMLCh* const name);

    void characters(const XMLCh* const chars, const XMLSize_t length);

    void ignorableWhitespace
    (
        const   XMLCh* const    chars
        , const XMLSize_t       length
    );

    void processingInstruction
    (
        const   XMLCh* const    target
        , const XMLCh* const    data
    );

    void startDocument();

    void startElement(const XMLCh* const name, AttributeList& attributes);

    void warning(const SAXParseException& exc);
    void error(const SAXParseException& exc);
    void fatalError(const SAXParseException& exc);

    void notationDecl
    (
        const   XMLCh* const    name
        , const XMLCh* const    publicId
        , const XMLCh* const    systemId
    );

    void unparsedEntityDecl
    (
        const   XMLCh* const    name
        , const XMLCh* const    publicId
        , const XMLCh* const    systemId
        , const XMLCh* const    notationName
    );

    char *getInputs() { return this->inputs; }
    char *getLabels() { return this->labels; }
    int getNlayers() { return this->nlayers; }
    double getMinvalue() { return this->minvalue; }
    double getMaxvalue() { return this->maxvalue; }
    char *getKernelWeightsFile() { return this->kernel_weights_file; }
    char *getFileRandomKernelWeights() { return this->file_random_kernel_weights; }
    int *getLayerNeurons() { return this->layer_neurons; }
    int *getNormalizationNrows() { return this->normalization_nrows; }
    int *getNormalizationNcols() { return this->normalization_ncols; }
    int *getKernelNrows() { return this->kernel_nrows; }
    int *getKernelNcols() { return this->kernel_ncols; }
    int *getMaxPoolingRowStride() { return this->maxpooling_rowstride; }
    int *getMaxPoolingColStride() { return this->maxpooling_colstride; }
    int *getMaxPoolingRowPooling() { return this->maxpooling_rowpooling; }
    int *getMaxPoolingColPooling() { return this->maxpooling_colpooling; }

private :
    XMLFormatter    fFormatter;
    char *inputs;
    char *labels;
    int nlayers;
    double minvalue;
    double maxvalue;
    char *kernel_weights_file;
    char *file_random_kernel_weights;
    int *layer_neurons;
    int *normalization_nrows;
    int *normalization_ncols;
    int *kernel_nrows;
    int *kernel_ncols;
    int *maxpooling_rowstride;
    int *maxpooling_colstride;
    int *maxpooling_rowpooling;
    int *maxpooling_colpooling;
    int layerCounter;
};
