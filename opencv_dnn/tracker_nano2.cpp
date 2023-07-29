// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the https://github.com/HonglinChu/NanoTrack/blob/master/ncnn_macos_nanotrack/nanotrack.cpp
// Author, HongLinChu, 1628464345@qq.com
// Adapt to OpenCV, ZihaoMu: zihaomu@outlook.com

// Link to original inference code: https://github.com/HonglinChu/NanoTrack
// Link to original training repo: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack

#include <opencv2/video/tracking.hpp>
#include "opencv2/core/cvstd_wrapper.hpp"
#include "tracker_nano2.hpp"
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cvx {

    TrackerNano2::TrackerNano2()
{
    // nothing
}

    TrackerNano2::~TrackerNano2()
{
    // nothing
}

    TrackerNano2::Params::Params()
{
    backbone = "backbone.onnx";
    neckhead = "neckhead.onnx";
#ifdef HAVE_OPENCV_DNN
    backend = cv::dnn::DNN_BACKEND_DEFAULT;
    target = cv::dnn::DNN_TARGET_CPU;
#else
    backend = -1;  // invalid value
    target = -1;  // invalid value
#endif
}

#ifdef HAVE_OPENCV_DNN
static void softmax(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat maxVal;
    cv::max(src.row(1), src.row(0), maxVal);

    src.row(1) -= maxVal;
    src.row(0) -= maxVal;

    exp(src, dst);

    cv::Mat sumVal = dst.row(0) + dst.row(1);
    dst.row(0) = dst.row(0) / sumVal;
    dst.row(1) = dst.row(1) / sumVal;
}

static float sizeCal(float w, float h)
{
    float pad = (w + h) * 0.5f;
    float sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

static cv::Mat sizeCal(const cv::Mat& w, const cv::Mat& h)
{
    cv::Mat pad = (w + h) * 0.5;
    cv::Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

// Similar python code: r = np.maximum(r, 1. / r) # r is matrix
static void elementReciprocalMax(cv::Mat& srcDst)
{
    size_t totalV = srcDst.total();
    float* ptr = srcDst.ptr<float>(0);
    for (size_t i = 0; i < totalV; i++)
    {
        float val = *(ptr + i);
        *(ptr + i) = std::max(val, 1.0f/val);
    }
}

class TrackerNanoImpl : public TrackerNano2
{
public:
    TrackerNanoImpl(const TrackerNano2::Params& parameters)
        : params(parameters)
    {
        backbone = cv::dnn::readNet(params.backbone);
        neckhead = cv::dnn::readNet(params.neckhead);

        CV_Assert(!backbone.empty());
        CV_Assert(!neckhead.empty());

        backbone.setPreferableBackend(params.backend);
        backbone.setPreferableTarget(params.target);
        neckhead.setPreferableBackend(params.backend);
        neckhead.setPreferableTarget(params.target);
    }

    void init(cv::InputArray image, const cv::Rect& boundingBox) CV_OVERRIDE;
    bool update(cv::InputArray image, cv::Rect& boundingBox) CV_OVERRIDE;
    float getTrackingScore() CV_OVERRIDE;

    // Save the target bounding box for each frame.
    std::vector<float> targetSz = {0, 0};  // H and W of bounding box
    std::vector<float> targetPos = {0, 0}; // center point of bounding box (x, y)
    float tracking_score;

    TrackerNano2::Params params;

    struct trackerConfig
    {
        float windowInfluence = 0.455f;
        float lr = 0.37f;
        float contextAmount = 0.5;
        bool swapRB = true;
        int totalStride = 16;
        float penaltyK = 0.055f;
    };

protected:
    const int exemplarSize = 127;
    const int instanceSize = 255;

    trackerConfig trackState;
    int scoreSize;
    cv::Size imgSize = {0, 0};
    cv::Mat hanningWindow;
    cv::Mat grid2searchX, grid2searchY;

    cv::dnn::Net backbone, neckhead;
    cv::Mat image;

    void getSubwindow(cv::Mat& dstCrop, cv::Mat& srcImg, int originalSz, int resizeSz);
    void generateGrids();
};

void TrackerNanoImpl::generateGrids()
{
    int sz = scoreSize;
    const int sz2 = sz / 2;

    std::vector<float> x1Vec(sz, 0);

    for (int i = 0; i < sz; i++)
    {
        x1Vec[i] = (float)(i - sz2);
    }

    cv::Mat x1M(1, sz, CV_32FC1, x1Vec.data());

    cv::repeat(x1M, sz, 1, grid2searchX);
    cv::repeat(x1M.t(), 1, sz, grid2searchY);

    grid2searchX *= trackState.totalStride;
    grid2searchY *= trackState.totalStride;

    grid2searchX += instanceSize/2;
    grid2searchY += instanceSize/2;
}

void TrackerNanoImpl::init(cv::InputArray image_, const cv::Rect &boundingBox_)
{
    scoreSize = (instanceSize - exemplarSize) / trackState.totalStride + 8;
    trackState = trackerConfig();
    image = image_.getMat().clone();

    // convert Rect2d from left-up to center.
    targetPos[0] = float(boundingBox_.x) + float(boundingBox_.width) * 0.5f;
    targetPos[1] = float(boundingBox_.y) + float(boundingBox_.height) * 0.5f;

    targetSz[0] = float(boundingBox_.width);
    targetSz[1] = float(boundingBox_.height);

    imgSize = image.size();

    // Extent the bounding box.
    float sumSz = targetSz[0] + targetSz[1];
    float wExtent = targetSz[0] + trackState.contextAmount * (sumSz);
    float hExtent = targetSz[1] + trackState.contextAmount * (sumSz);
    int sz = int(cv::sqrt(wExtent * hExtent));

    cv::Mat crop;
    getSubwindow(crop, image, sz, exemplarSize);
    cv::Mat blob = cv::dnn::blobFromImage(crop, 1.0, cv::Size(), cv::Scalar(), trackState.swapRB);

    backbone.setInput(blob);
    cv::Mat out = backbone.forward(); // Feature extraction.
    neckhead.setInput(out, "input1");

    createHanningWindow(hanningWindow, cv::Size(scoreSize, scoreSize), CV_32F);
    generateGrids();
}

void TrackerNanoImpl::getSubwindow(cv::Mat& dstCrop, cv::Mat& srcImg, int originalSz, int resizeSz)
{
    cv::Scalar avgChans = mean(srcImg);
    cv::Size imgSz = srcImg.size();
    int c = (originalSz + 1) / 2;

    int context_xmin = (int)(targetPos[0]) - c;
    int context_xmax = context_xmin + originalSz - 1;
    int context_ymin = (int)(targetPos[1]) - c;
    int context_ymax = context_ymin + originalSz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - imgSz.width + 1);
    int bottom_pad = std::max(0, context_ymax - imgSz.height + 1);

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat cropImg;
    if (left_pad == 0 && top_pad == 0 && right_pad == 0 && bottom_pad == 0)
    {
        // Crop image without padding.
        cropImg = srcImg(cv::Rect(context_xmin, context_ymin,
                                  context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else // Crop image with padding, and the padding value is avgChans
    {
        cv::Mat tmpMat;
        cv::copyMakeBorder(srcImg, tmpMat, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, avgChans);
        cropImg = tmpMat(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    resize(cropImg, dstCrop, cv::Size(resizeSz, resizeSz));
}

bool TrackerNanoImpl::update(cv::InputArray image_, cv::Rect &boundingBoxRes)
{
    image = image_.getMat().clone();
    int targetSzSum = (int)(targetSz[0] + targetSz[1]);

    float wc = targetSz[0] + trackState.contextAmount * targetSzSum;
    float hc = targetSz[1] + trackState.contextAmount * targetSzSum;
    float sz = cv::sqrt(wc * hc);
    float scale_z = exemplarSize / sz;
    float sx = sz * (instanceSize / exemplarSize);
    targetSz[0] *= scale_z;
    targetSz[1] *= scale_z;

    cv::Mat crop;
    getSubwindow(crop, image, int(sx), instanceSize);

    cv::Mat blob = cv::dnn::blobFromImage(crop, 1.0, cv::Size(), cv::Scalar(), trackState.swapRB);
    backbone.setInput(blob);
    cv::Mat xf = backbone.forward();
    neckhead.setInput(xf, "input2");
    std::vector<cv::String> outputName = {"output1", "output2"};
    std::vector<cv::Mat> outs;
    neckhead.forward(outs, outputName);

    CV_Assert(outs.size() == 2);

    cv::Mat clsScore = outs[0]; // 1x2x16x16
    cv::Mat bboxPred = outs[1]; // 1x4x16x16

    clsScore = clsScore.reshape(0, {2, scoreSize, scoreSize});
    bboxPred = bboxPred.reshape(0, {4, scoreSize, scoreSize});

    cv::Mat scoreSoftmax; // 2x16x16
    softmax(clsScore, scoreSoftmax);

    cv::Mat score = scoreSoftmax.row(1);
    score = score.reshape(0, {scoreSize, scoreSize});

    cv::Mat predX1 = grid2searchX - bboxPred.row(0).reshape(0, {scoreSize, scoreSize});
    cv::Mat predY1 = grid2searchY - bboxPred.row(1).reshape(0, {scoreSize, scoreSize});
    cv::Mat predX2 = grid2searchX + bboxPred.row(2).reshape(0, {scoreSize, scoreSize});
    cv::Mat predY2 = grid2searchY + bboxPred.row(3).reshape(0, {scoreSize, scoreSize});

    // size penalty
    // scale penalty
    cv::Mat sc = sizeCal(predX2 - predX1, predY2 - predY1)/sizeCal(targetPos[0], targetPos[1]);
    elementReciprocalMax(sc);

    // ratio penalty
    float ratioVal = targetSz[0] / targetSz[1];

    cv::Mat ratioM(scoreSize, scoreSize, CV_32FC1, cv::Scalar::all(ratioVal));
    cv::Mat rc = ratioM / ((predX2 - predX1) / (predY2 - predY1));
    elementReciprocalMax(rc);

    cv::Mat penalty;
    exp(((rc.mul(sc) - 1) * trackState.penaltyK * (-1)), penalty);
    cv::Mat pscore = penalty.mul(score);

    // Window penalty
    pscore = pscore * (1.0 - trackState.windowInfluence) + hanningWindow * trackState.windowInfluence;

    // get Max
    int bestID[2] = { 0, 0 };
    minMaxIdx(pscore, 0, 0, 0, bestID);

    tracking_score = pscore.at<float>(bestID);

    float x1Val = predX1.at<float>(bestID);
    float x2Val = predX2.at<float>(bestID);
    float y1Val = predY1.at<float>(bestID);
    float y2Val = predY2.at<float>(bestID);

    float predXs = (x1Val + x2Val)/2;
    float predYs = (y1Val + y2Val)/2;
    float predW = (x2Val - x1Val)/scale_z;
    float predH = (y2Val - y1Val)/scale_z;

    float diffXs = (predXs - instanceSize / 2) / scale_z;
    float diffYs = (predYs - instanceSize / 2) / scale_z;

    targetSz[0] /= scale_z;
    targetSz[1] /= scale_z;

    float lr = penalty.at<float>(bestID) * score.at<float>(bestID) * trackState.lr;

    float resX = targetPos[0] + diffXs;
    float resY = targetPos[1] + diffYs;
    float resW = predW * lr + (1 - lr) * targetSz[0];
    float resH = predH * lr + (1 - lr) * targetSz[1];

    resX = std::max(0.f, std::min((float)imgSize.width, resX));
    resY = std::max(0.f, std::min((float)imgSize.height, resY));
    resW = std::max(10.f, std::min((float)imgSize.width, resW));
    resH = std::max(10.f, std::min((float)imgSize.height, resH));

    targetPos[0] = resX;
    targetPos[1] = resY;
    targetSz[0] = resW;
    targetSz[1] = resH;

    // convert center to cv::Rect.
    boundingBoxRes = { int(resX - resW/2), int(resY - resH/2), int(resW), int(resH)};
    return true;
}

float TrackerNanoImpl::getTrackingScore()
{
    return tracking_score;
}

cv::Ptr<TrackerNano2> TrackerNano2::create(const TrackerNano2::Params& parameters)
{
    return cv::makePtr<TrackerNanoImpl>(parameters);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerNano> TrackerNano::create(const TrackerNano::Params& parameters)
{
    CV_UNUSED(parameters);
    CV_Error(cv::Error::StsNotImplemented, "to use NanoTrack, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN
}
