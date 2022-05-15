#ifndef RVIZ_EDGE_H
#define RVIZ_EDGE_H
#include <ros/ros.h>
#include <include/common.h>
#include <include/yamlRead.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
class RVIZEdge
{
private:

    ros::Publisher edge_pub;
    string GlobalframeId;
    std::vector<visualization_msgs::Marker> markers;
    size_t AgentId_;
    string AgentFrameId;

    double marker_scale = 1.0;

public:
    RVIZEdge();
    ~RVIZEdge();
    RVIZEdge(ros::NodeHandle& nh,
             string topicName, string GlobalframeId,
             int bufferSize=2);
//    RVIZEdge(ros::NodeHandle& nh,
//             string topicName, size_t &AgentId_, string &AgentFrameId,
//             int bufferSize=2);
    void AddLoopEdge(SE3 from, SE3 to, ros::Time t);
    void PubEdge();
    void clearEdge();


};


#endif // RVIZ_EDGE_H
