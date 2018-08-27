#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/parsers/SAXParser.hpp>
#include <SAXFunction.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>
#include <iostream>
using namespace std;

// ---------------------------------------------------------------------------
//  Local data
//
//  doNamespaces
//      Indicates whether namespace processing should be enabled or not.
//      Defaults to disabled.
//
//  doSchema
//      Indicates whether schema processing should be enabled or not.
//      Defaults to disabled.
//
//  schemaFullChecking
//      Indicates whether full schema constraint checking should be enabled or not.
//      Defaults to disabled.
//
//  encodingName
//      The encoding we are to output in. If not set on the command line,
//      then it is defaulted to LATIN1.
//
//  xmlFile
//      The path to the file to parser. Set via command line.
//
//  valScheme
//      Indicates what validation scheme to use. It defaults to 'auto', but
//      can be set via the -v= command.
// ---------------------------------------------------------------------------
static bool                     doNamespaces        = false;
static bool                     doSchema            = false;
static bool                     schemaFullChecking  = false;
static const char*              encodingName    = "LATIN1";
static XMLFormatter::UnRepFlags unRepFlags      = XMLFormatter::UnRep_CharRef;
static SAXParser::ValSchemes    valScheme       = SAXParser::Val_Auto;

void loadXMLFile(const char *xmlFile) {
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
        parser->parse(xmlFile);
        errorCount = parser->getErrorCount();

	// ----------------- Printing information ------------------------------
	cout << "inputs: " << handler.getInputs() << endl;
	cout << "Labels: " << handler.getLabels() << endl;
	cout << "nlayers: " << handler.getNlayers() << endl;
	cout << "minvalue: " << handler.getMinvalue() << endl;
	cout << "maxvalue: " << handler.getMaxvalue() << endl;
	cout << "kernel weights file: " << handler.getKernelWeightsFile() << endl;
	
	cout << "Number of Neuron: " << endl;
	for (int i = 0; i < handler.getNlayers(); i++) {
		cout << "\t" << handler.getLayerNeurons()[i] << endl;
	}

	cout << "Normalization (rows and cols): " << endl;
	for (int i = 0; i < handler.getNlayers()+1; i++) {
		cout << "\t" << handler.getNormalizationNrows()[i] << " x " <<
				handler.getNormalizationNcols()[i] << endl;
	}

	cout << "Kernel (rows and cols): " << endl;
	for (int i = 0; i < handler.getNlayers(); i++) {
		cout << "\t" << handler.getKernelNrows()[i] << " x " <<
				handler.getKernelNcols()[i] << endl;
	}
	cout << "MaxPooling (rows and cols): " << endl;
	for (int i = 0; i < handler.getNlayers(); i++) {
		cout << "\t" << handler.getMaxPoolingRowStride()[i] << " x " << 
				handler.getMaxPoolingColStride()[i] << "   " <<
				handler.getMaxPoolingRowPooling()[i] << " x " << 
				handler.getMaxPoolingColPooling()[i] << endl;
	}
	// ----------------- Printing information ------------------------------

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

int main(int argc, char* argv[])
{
	// Check command line and extract arguments.
    if (argc < 2)
    {
        cout << "Usage: ./testXml <name>.xml" << endl;
        return 1;
    }
    
	loadXMLFile(argv[1]);

        return 0;
}
