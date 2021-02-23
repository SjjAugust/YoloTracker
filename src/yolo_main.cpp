#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <caffe/caffe.hpp>
#include "SimplePipeline.h"
// ----------------------------------------------------------------------

const char* keys =
        {
                "{help h usage ?  |                    | Print usage| }"
                "{ @input_video   |../data/VideoD.mkv  | Input video file | }"
                "{ ocl opencl     |1                   | Flag to use opencl | }"

                "{ sf start_frame |0                   | Frame modification parameter: Start a video from this position | }"
                "{ ef end_frame   |100000              | Frame modification parameter: Play a video to this position (if 0 then played to the end of file) | }"
    
                //models
                "{ m model        |../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt | Detection parameter: Model file | }"
                "{ w weight       |../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel  | Detection parameter: Weight file | }"
                "{ lm label_map   |../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel  | Detection parameter: Label map  file  | }"
                
                //detect parameters
                "{ th threshold   |0.7                 | Detection parameter: Confidence percentage of detected objects must exceed this value to be reported as a detected object. | }"
                "{ dd desired_detect |1                | Detection Parameter: Flag to detect only desired objects | }"
                "{ do desired_objects |15,2,12   | Detection Parameter: list of desired objects to detect | }"

                //output parameters
                "{ o output       |../data/D2.avi      | Writing parameter: Name of output video file | }"
                "{ save_video     |1                   | Writing parameter: Flag to enable writing to file | }"
        };

// ----------------------------------------------------------------------

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        //print help messeges
        parser.printMessage();
        return 0;
    }
    // Log set up
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = true;

    bool useOCL = parser.get<int>("opencl") != 0;
    cv::ocl::setUseOpenCL(useOCL);
    LOG(INFO) << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;


    SSDExample ssdExample(parser);
    ssdExample.Process();

    cv::destroyAllWindows();
    return 0;
}
