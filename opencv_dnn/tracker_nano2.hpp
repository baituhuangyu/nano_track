

#ifndef OPENCV_NANO_TRACK_TRACK_NANO_H
#define OPENCV_NANO_TRACK_TRACK_NANO_H
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

namespace cvx {

/** @brief the Nano tracker is a super lightweight dnn-based general object tracking.
 *
 *  Nano tracker is much faster and extremely lightweight due to special model structure, the whole model size is about 1.9 MB.
 *  Nano tracker needs two models: one for feature extraction (backbone) and the another for localization (neckhead).
 *  Model download link: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
 *  Original repo is here: https://github.com/HonglinChu/NanoTrack
 *  Author: HongLinChu, 1628464345@qq.com
 */
class CV_EXPORTS_W TrackerNano2 : public cv::Tracker
{
    protected:
    TrackerNano2();  // use ::create()
    public:
    virtual ~TrackerNano2() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
            {
                    CV_WRAP Params();
            CV_PROP_RW std::string backbone;
            CV_PROP_RW std::string neckhead;
            CV_PROP_RW int backend;
            CV_PROP_RW int target;
            };

    /** @brief Constructor
    @param parameters NanoTrack parameters TrackerNano::Params
    */
    static CV_WRAP
            cv::Ptr<TrackerNano2> create(const TrackerNano2::Params& parameters = TrackerNano2::Params());

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

}
#endif //OPENCV_NANO_TRACK_TRACK_NANO_H
