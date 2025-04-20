// Ref: https://blog.csdn.net/weixin_45824067/article/details/130618583

#include "YoloONNX.h"

namespace BodyHand {

    using namespace std;
    using namespace cv;
    using namespace cv::dnn;
    using namespace Ort;

    void LetterBox(
        const cv::Mat& image, cv::Mat& outImage,
        cv::Vec4d& params,
        const cv::Size& newShape,
        bool autoShape,
        bool scaleFill,
        bool scaleUp,
        int stride,
        const cv::Scalar& color
    ) {
        if (false) {
            int maxLen = MAX(image.rows, image.cols);
            outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
            image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
            params[0] = 1;
            params[1] = 1;
            params[3] = 0;
            params[2] = 0;
        }

        // ȡ��С�����ű���
        cv::Size shape = image.size();
        float r = std::min((float)newShape.height / (float)shape.height,
            (float)newShape.width / (float)shape.width);
        if (!scaleUp)
            r = std::min(r, 1.0f);
        //printf("ԭͼ�ߴ磺w:%d * h:%d, Ҫ��ߴ磺w:%d * h:%d, �������õ����űȣ�%f\n",
        //	shape.width, shape.height, newShape.width, newShape.height, r);

        // ����ǰ������ű�����ԭͼ�ĳߴ�
        float ratio[2]{ r,r };
        int new_un_pad[2] = { (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };
        //printf("�ȱ������ź�ĳߴ��Ϊ��w:%d * h:%d\n", new_un_pad[0], new_un_pad[1]);

        // �������Ŀ��ߴ��padding������
        auto dw = (float)(newShape.width - new_un_pad[0]);
        auto dh = (float)(newShape.height - new_un_pad[1]);
        if (autoShape)
        {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            new_un_pad[0] = newShape.width;
            new_un_pad[1] = newShape.height;
            ratio[0] = (float)newShape.width / (float)shape.width;
            ratio[1] = (float)newShape.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;
        //printf("���padding: dw=%f , dh=%f\n", dw, dh);

        // �ȱ�������
        if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
        {
            cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
        }
        else {
            outImage = image.clone();
        }

        // ͼ������padding��䣬����ԭͼ��Ŀ��ߴ�һ��
        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        params[0] = ratio[0]; // width�����ű���
        params[1] = ratio[1]; // height�����ű���
        params[2] = left; // ˮƽ�������ߵ�padding������
        params[3] = top; //��ֱ�������ߵ�padding������
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    bool Yolov8Onnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaId, bool warmUp) {
        if (_batchSize < 1) _batchSize = 1;
        try
        {
            std::vector<std::string> available_providers = GetAvailableProviders();
            auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");


            if (isCuda && (cuda_available == available_providers.end()))
            {
                //std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
                //std::cout << "************* Infer model on CPU! *************" << std::endl;
            }
            else if (isCuda && (cuda_available != available_providers.end()))
            {
                //std::cout << "************* Infer model on GPU! *************" << std::endl;
                //#if ORT_API_VERSION < ORT_OLD_VISON
                //			OrtCUDAProviderOptions cudaOption;
                //			cudaOption.device_id = cudaID;
                //            _OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
                //#else
                //			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
                //#endif
            }
            else
            {
                //std::cout << "************* Infer model on CPU! *************" << std::endl;
            }
            //

            _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
            std::wstring model_path(modelPath.begin(), modelPath.end());
            _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
            _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

            Ort::AllocatorWithDefaultOptions allocator;
            //init input
            _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
            _inputName = _OrtSession->GetInputName(0, allocator);
            _inputNodeNames.push_back(_inputName);
#else
            _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
            _inputNodeNames.push_back(_inputName.get());
#endif
            //cout << _inputNodeNames[0] << endl;
            Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
            auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
            _inputNodeDataType = input_tensor_info.GetElementType();
            _inputTensorShape = input_tensor_info.GetShape();

            if (_inputTensorShape[0] == -1)
            {
                _isDynamicShape = true;
                _inputTensorShape[0] = _batchSize;

            }
            if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
                _isDynamicShape = true;
                _inputTensorShape[2] = _netHeight;
                _inputTensorShape[3] = _netWidth;
            }
            //init output
            _outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
            _output_name0 = _OrtSession->GetOutputName(0, allocator);
            _outputNodeNames.push_back(_output_name0);
#else
            _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
            _outputNodeNames.push_back(_output_name0.get());
#endif
            Ort::TypeInfo type_info_output0(nullptr);
            type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

            auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
            _outputNodeDataType = tensor_info_output0.GetElementType();
            _outputTensorShape = tensor_info_output0.GetShape();

            //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
            //_outputMaskTensorShape = tensor_info_output1.GetShape();
            //if (_outputTensorShape[0] == -1)
            //{
            //	_outputTensorShape[0] = _batchSize;
            //	_outputMaskTensorShape[0] = _batchSize;
            //}
            //if (_outputMaskTensorShape[2] == -1) {
            //	//size_t ouput_rows = 0;
            //	//for (int i = 0; i < _strideSize; ++i) {
            //	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
            //	//}
            //	//_outputTensorShape[1] = ouput_rows;

            //	_outputMaskTensorShape[2] = _segHeight;
            //	_outputMaskTensorShape[3] = _segWidth;
            //}

            //warm up
            if (isCuda && warmUp) {
                //draw run
                //cout << "Start warming up" << endl;
                size_t input_tensor_length = VectorProduct(_inputTensorShape);
                float* temp = new float[input_tensor_length];
                std::vector<Ort::Value> input_tensors;
                std::vector<Ort::Value> output_tensors;
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
                    _inputTensorShape.size()));
                for (int i = 0; i < 3; ++i) {
                    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                        _inputNodeNames.data(),
                        input_tensors.data(),
                        _inputNodeNames.size(),
                        _outputNodeNames.data(),
                        _outputNodeNames.size());
                }

                delete[]temp;
            }
        }
        catch (const std::exception&) {
            return false;
        }
        return true;

    }

    int Yolov8Onnx::Preprocessing(const std::vector<cv::Mat>& SrcImgs,
        std::vector<cv::Mat>& OutSrcImgs,
        std::vector<cv::Vec4d>& params) {
        OutSrcImgs.clear();
        Size input_size = Size(_netWidth, _netHeight);

        // �ŷ⴦��
        for (size_t i = 0; i < SrcImgs.size(); ++i) {
            Mat temp_img = SrcImgs[i];
            Vec4d temp_param = { 1,1,0,0 };
            if (temp_img.size() != input_size) {
                Mat borderImg;
                LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
                OutSrcImgs.push_back(borderImg);
                params.push_back(temp_param);
            }
            else {
                OutSrcImgs.push_back(temp_img);
                params.push_back(temp_param);
            }
        }

        int lack_num = _batchSize - SrcImgs.size();
        if (lack_num > 0) {
            Mat temp_img = Mat::zeros(input_size, CV_8UC3);
            Vec4d temp_param = { 1,1,0,0 };
            OutSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
        return 0;
    }

    // ��������ķ���ֵ��ʾ��������Ƿ����ˣ����Ǵ��� bug��Ӧ��ֱ���ж����鳤�Ⱦ���
    bool Yolov8Onnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputPose>>& output)
    {
        vector<Vec4d> params;
        vector<Mat> input_images;
        cv::Size input_size(_netWidth, _netHeight);

        //preprocessing (�ŷ⴦��)
        Preprocessing(srcImgs, input_images, params);
        // [0~255] --> [0~1]; BGR2RGB
        Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

        // ǰ�򴫲��õ�������
        int64_t input_tensor_length = VectorProduct(_inputTensorShape);// ?
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data,
            input_tensor_length, _inputTensorShape.data(),
            _inputTensorShape.size()));

        output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
            _inputNodeNames.data(),
            input_tensors.data(),
            _inputNodeNames.size(),
            _outputNodeNames.data(),
            _outputNodeNames.size()
        );

        //post-process

        float* all_data = output_tensors[0].GetTensorMutableData<float>(); // ��һ��ͼƬ�����

        _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // һ��ͼƬ�����ά����Ϣ [1, 84, 8400]

        int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0]; // һ��ͼƬ�����ռ�ڴ泤�� 8400*84

        for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
            Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t(); // [1, 56 ,8400] -> [1, 8400, 56]

            all_data += one_output_length; //ָ��ָ����һ��ͼƬ�ĵ�ַ

            float* pdata = (float*)output0.data; // [classid,x,y,w,h,x,y,...21����]
            int rows = output0.rows; // Ԥ�������� 8400

            // һ��ͼƬ��Ԥ���

            vector<float> confidences;
            vector<Rect> boxes;
            vector<int> labels;
            vector<vector<float>> kpss;
            for (int r = 0; r < rows; ++r) {

                // �õ���������
                auto kps_ptr = pdata + 5;


                // Ԥ�������ӳ�䵽ԭͼ��
                float score = pdata[4];
                if (score > _classThreshold) {

                    // rect [x,y,w,h]
                    float x = (pdata[0] - params[img_index][2]) / params[img_index][0]; //x
                    float y = (pdata[1] - params[img_index][3]) / params[img_index][1]; //y
                    float w = pdata[2] / params[img_index][0]; //w
                    float h = pdata[3] / params[img_index][1]; //h

                    int left = MAX(int(x - 0.5 * w + 0.5), 0);
                    int top = MAX(int(y - 0.5 * h + 0.5), 0);

                    std::vector<float> kps;
                    for (int k = 0; k < 17; k++) {
                        float kps_x = (*(kps_ptr + 3 * k) - params[img_index][2]) / params[img_index][0];
                        float kps_y = (*(kps_ptr + 3 * k + 1) - params[img_index][3]) / params[img_index][1];
                        float kps_s = *(kps_ptr + 3 * k + 2);

                        // cout << *(kps_ptr + 3*k) << endl;

                        kps.push_back(kps_x);
                        kps.push_back(kps_y);
                        kps.push_back(kps_s);
                    }

                    confidences.push_back(score);
                    labels.push_back(0);
                    kpss.push_back(kps);
                    boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
                }
                pdata += _anchorLength; //��һ��Ԥ���
            }

            // ��һ��ͼ��Ԥ���ִ��NMS����
            vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThrehold, nms_result); // ����ҪclassThreshold��

            // ��һ��ͼƬ������NMS����õ����������õ����id��confidence��box�������ڽṹ��OutputDet��������
            vector<OutputPose> temp_output;
            for (size_t i = 0; i < nms_result.size(); ++i) {
                int idx = nms_result[i];
                OutputPose result;

                result.confidence = confidences[idx];
                result.box = boxes[idx];
                result.label = labels[idx];
                result.kps = kpss[idx];
                temp_output.push_back(result);
            }
            output.push_back(temp_output); // ����ͼƬ����������һ��ͼƬ��������ڴ�������
        }
        if (output.size())
            return true;
        else
            return false;

    }


    bool Yolov8Onnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputPose>& output) {
        vector<Mat> input_data = { srcImg };
        vector<vector<OutputPose>> temp_output;

        if (OnnxBatchDetect(input_data, temp_output)) {
            output = temp_output[0];
            return true;
        }
        else return false;
    }

}