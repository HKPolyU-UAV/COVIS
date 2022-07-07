#ifndef MERGING
#define MERGING
#include <include/common.h>
#include <iostream>
#include <fstream>
#include <deque>
#include <stdint.h>
#include <condition_variable>
#include <thread>

//COVIS
#include <include/yamlRead.h>
#include <include/triangulation.h>
#include <include/keyframe_msg_handler.h>
#include <covis/KeyFrame.h>
#include <include/camera_frame.h>
#include <include/vi_type.h>
#include <include/tic_toc_ros.h>
#include <include/rviz_path.h>
#include <include/rviz_edge.h>
//#include <include/rviz_odom.h>
#include <include/rviz_mesh.h>
#include <include/kinetic_math.h>

//OPENCV
#include <opencv2/opencv.hpp>


//ROS
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

//DBoW2
#include "../3rdPartLib/DLib/DVision/DVision.h"
#include "../3rdPartLib/DBow2/DBoW/DBoW2.h"
#include "../3rdPartLib/DBow2/DBoW/TemplatedDatabase.h"
#include "../3rdPartLib/DBow2/DBoW/TemplatedVocabulary.h"

//g2o
#include <g2o/config.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

using namespace DVision;
using namespace DBoW2;

struct LC_PARAS
{
    int lcKFStart;
    int lcKFDist;
    int lcKFMaxDist;
    int lcKFLast;
    int lcNKFClosest;
    double ratioMax;
    double ratioRansac;
    int minPts;
    double minScore;
    std::vector<string> ResultPaths;
};

struct KeyFrameMerge
{
    int64_t         frame_id;      // frame id
    int64_t         keyframe_id;   // global id in merging module
    size_t          AgentId_;      // Agent id
    int             lm_count;      // number of landmark
    ros::Time       t;             // timestamp of frame
    SE3             T_c_w;         // pose from VO * T_odom_map :w0 to w1
    SE3             T_c_w_odom;    // pose from VO: w0
    SE3             T_c_w_global;  // cam pose aligned in global frame
    SE3             T_c_i;
    SE3             T_w_i;
    vector<Vec3>    lm_3d;         // 3d landmark in camera frame
    vector<Vec2>    lm_2d;         // 2d landmark in pixel coordinate frame
    vector<Vec2>    lm_2d_post;    // 2d landmark redect
    vector<double>  lm_depth;      // landmark depth
    vector<DVision::BRIEF::bitset> lm_2d_descriptor, lm_2d_post_descriptor;
    DBoW2::BowVector kf_bv;          // BoWvector of img0
    cv::Mat img0;
};

//sort by descending order
struct sort_simmat_by_score
{
    inline bool operator()(const Vector2d& a, const Vector2d& b){
        return ( a(1) > b(1) );
    }
};
//sort by ascending order
struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

typedef std::pair<size_t, int> idpair;
class Merging
{
public:

    Merging(ros::NodeHandle &nh, LC_PARAS &lc_paras, const vector<DepthCamera> &d_cameras, int save_flag_=0, int number_of_Agent=1, bool IntraLoop=false) ;
    void setKeyFrame(const covis::KeyFrameConstPtr& msg);
    ~Merging();

private:
    ros::NodeHandle nh;
    ros::Subscriber sub_kf;
    image_transport::Publisher merge_Img_pub;   // publish matching keyframe
    RVIZEdge* edge_merge_pub; // visualize edge between merged pose
    //RVIZPose* pose_pub;
    vector<RVIZPath*> globalpaths_;
    vector<RVIZMesh*> dronepub_;

    tf2_ros::TransformBroadcaster br;

    vector<DepthCamera> d_cameras;
    //DepthCamera dc;
    //DBow related para
    LC_PARAS lc_paras;

    BriefVocabulary* voc;
    BriefDatabase db;// faster search

    vector<vector<double>> sim_matrix;//for similarity visualization
    vector<double> sim_vec;
    vector<double> sim_vec_covis;

    //KF database
    //KeyFrameMerge kf;
    queue<shared_ptr<KeyFrameMerge>> kf_queue;
    queue<int64_t> graph_queue;
    vector<shared_ptr<KeyFrameMerge>> kfs_all;

    vector<BowVector> kfbv_map;               // store all BowVectors
    vector<shared_ptr<KeyFrameMerge>> kfs_this;
    vector<shared_ptr<KeyFrameMerge>> kfs_other; // store all pointers to kf

    vector<cv::Mat> kf_img_merge; // store all unpacked left image of keyframe
    map<idpair, cv::Mat> idpairImg;
    map<int, cv::Mat> LoopImgs;
    vector<cv::DMatch> select_match;

    int kf_curr_idx_;   // curr kf keyframeId
    int kf_prev_idx_ = -1;   // loop candidate keyframeId
    map<size_t, bool> merged_agent;
    map<size_t, SE3> T_local_global;
    map<size_t, SE3> drifts;

    //loop info
    vector<Vec3I> loop_ids;  //store loop id pair [kf_prev_idx kf_curr_idx 1]
    vector<SE3> loop_poses;  // store pose from candidate keyframe to current keyframe


    //kf id
    int64_t kf_id = 0; // global id
    //last loop id
    int64_t last_pgo_id = -5000;

    // Agent info
    size_t AgentId_;
    string AgentFrameId;
    size_t number_of_Agent = 1;
    bool IntraLoop;

    // visualize parameter
    int SAVE_IMG;
    bool SAVE_LOOP = true;

    std::mutex m_graph;
    std::mutex m_vector;
    std::mutex m_imgs;
    std::mutex m_match;
    std::mutex m_db;
    std::mutex m_loop;
    std::mutex m_drift;
    std::mutex m_path;
    std::mutex m_pgo;
    std::thread pgothread_;
    vector <std::thread> Loopthreads_;

    std::condition_variable cv_pgo;

    bool Loop_running_ = false;
    bool pgo_running_  = false;

    void runLC();
    void startPGO();
    void startGBA();
    void PoseGraphOptimization();
    bool isMergeCandidate();
    int AddandDetectLoop(shared_ptr<KeyFrameMerge> kf);
    bool isLoopClosureKF(shared_ptr<KeyFrameMerge> kf0, shared_ptr<KeyFrameMerge> kf1, SE3 &se_ji);
    void updateGlobalPose(shared_ptr<KeyFrameMerge> kf0, shared_ptr<KeyFrameMerge> kf1, SE3 &se_ji);
    bool add_Loop_check(shared_ptr<KeyFrameMerge> kf0, shared_ptr<KeyFrameMerge> kf1, SE3 &loop_pose);
//    void searchByBRIEFDes(const std::vector<Vec2> &lm_2d_old,
//                          const std::vector<BRIEF::bitset> &lm_2d_des_old,
//                          const std::vector<Vec2> &lm_2d,
//                          const std::vector<BRIEF::bitset> &lm_2d_des,
//                          std::vector<cv::Point2f>& matched_2d, std::vector<uchar>& status);
    void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &point_des,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::Point2f> &feature_2d_old);
    bool searchInAera(const BRIEF::bitset window_descriptor,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::Point2f> &feature_2d_old,
                      cv::Point2f &best_match_norm);
    int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
    void expandGraph();
    void sim_mat_update();
    bool check_lastLC_close();
    void broadcastTF();
    void merge();



};


#endif
