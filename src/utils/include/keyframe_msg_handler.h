#ifndef KEYFRAME_MSG_H
#define KEYFRAME_MSG_H
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <covis/KeyFrame.h>
#include "../3rdPartLib/DLib/DVision/DVision.h"
#include <include/common.h>
#include <include/vi_motion.h>
#include <include/camera_frame.h>
#include <include/feature_dem.h>
#include <thread>
#include <mutex>
#define KFMSG_CMD_NONE          (0)
#define KFMSG_CMD_RESET_LM      (1)

class BriefDescriptorExtractor
{
public:
  typedef  std::shared_ptr<BriefDescriptorExtractor> brief_ptr;
  void create(const string& pattern_path);
  void extract(const cv::Mat& img, vector<cv::KeyPoint>& keypoints, vector<DVision::BRIEF::bitset>& brief_descriptors);
  DVision::BRIEF m_brief;
};

struct KeyFrameStruct {
    cv::Mat         img0;
    cv::Mat         img1;
    int64_t         frame_id;
    size_t          AgentId_;
    int             lm_count;
    vector<int64_t> lm_id;
    vector<Vec2>    lm_2d;
    vector<Vec3>    lm_3d;
    vector<cv::Mat>     lm_descriptor;
    SE3             T_c_w;
};

class KeyFrameMsgHandler
{
    ros::Publisher kf_pub;
    ros::Publisher kf_odom_pub;
    //ros::Publisher kf_merge_pub;
    size_t AgentId_;
    std::mutex m_kf;
    DepthCamera d_camera;
    BriefDescriptorExtractor::brief_ptr BRIEF_;
public:
    KeyFrameMsgHandler();
    KeyFrameMsgHandler(ros::NodeHandle& nh, string topic_name, size_t AgentId_, int buffersize=2);
    void cmdLMResetPub(ros::Time stamp=ros::Time::now());//publish reset command to localmap thread
    void pub(CameraFrame& frame, ros::Time stamp=ros::Time::now());
    void preProcessKeyFramemsg(CameraFrame::Ptr& frame, VIMOTION::Ptr& vimotion, covis::KeyFrame& msg, ros::Time stamp=ros::Time::now());
    void ProcessKeyFramemsg(covis::KeyFrame& msg, const cv::Mat& img0);
    static void unpack(covis::KeyFrameConstPtr kf_const_ptr,
                       int64_t         &frame_id,
                       size_t          &AgentId_,
                       SE3             &T_c_w,
                       SE3             &T_c_i,
                       cv::Mat         &img0,
                       int             &lm_count,
                       vector<Vec3>    &lm_3d,
                       vector<Vec2>    &lm_2d,
                       vector<Vec2>    &lm_2d_post,
                       //vector<cv::Mat> &lm_2d_descriptor,
                       //vector<cv::Mat> &lm_2d_post_descriptor,
                       vector<DVision::BRIEF::bitset> &lm_2d_descriptor,
                       vector<DVision::BRIEF::bitset> &lm_2d_post_descriptor,
                       ros::Time       &time);
};

#endif // KEYFRAME_MSG_H
