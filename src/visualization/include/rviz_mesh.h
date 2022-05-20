#ifndef RVIZ_MESH_H
#define RVIZ_MESH_H
#include <ros/ros.h>
#include <include/common.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
class RVIZMesh
{
private:

    ros::Publisher mesh_pub;
    string GlobalFrameId;

    std::vector<visualization_msgs::Marker> markers;
    size_t AgentId_;
    string AgentFrameId;
    bool is_Server = false;

public:
    RVIZMesh();
    ~RVIZMesh();
    RVIZMesh(ros::NodeHandle& nh,
             string topicName, string GlobalFrameId,
             int bufferSize=10);
    RVIZMesh(ros::NodeHandle& nh,
             string topicName, size_t AgentId_, string AgentFrameId,
             int bufferSize=10);
    void PubT_w_c(SE3 cam_pose, ros::Time t, size_t AgentId=0);
    void PubT_w_i(SE3 drone_pose, ros::Time t, size_t AgentId=0);

    //void PubMarkerArray();
    //void clearMarkerArray();


};


#endif // RVIZ_MESH_H
