#include "include/rviz_edge.h"


RVIZEdge::RVIZEdge(){

}
RVIZEdge::~RVIZEdge(){

}

RVIZEdge::RVIZEdge(ros::NodeHandle& nh, string topicName, string frameId, int bufferSize)
{
    edge_pub = nh.advertise<visualization_msgs::MarkerArray>(topicName, bufferSize);
    frame_id = frameId;


}
#if 0
RVIZEdge::RVIZEdge(ros::NodeHandle& nh,
         string topicName, size_t &AgentId_, string &AgentFrameId,
                   int bufferSize)
{
  this->AgentId_ = AgentId_;
  this->AgentFrameId = AgentFrameId;

  stringstream* ss;
  ss = new stringstream;
  *ss << "/Agent" << this->AgentId_ << topicName;

  edge_pub = nh.advertise<visualization_msgs::MarkerArray>(ss->str(), bufferSize);
  delete ss;
  frame_id = this->AgentFrameId;

}
#endif

void RVIZEdge::AddLoopEdge(SE3 from, SE3 to, ros::Time t_to)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = t_to;
  marker.ns = "LoopEdges";
  marker.id = markers.size() + 1; //unique identifier of Loop Edge
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.lifetime = ros::Duration();

  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.03;

  std_msgs::ColorRGBA white, green;
  white.r = 1.0;
  white.g = 1.0;
  white.b = 1.0;
  white.a = 1.0;
  green.r = 0.0;
  green.g = 1.0;
  green.b = 0.0;
  green.a = 1.0;

  marker.color = white;


  Vec3 t1, t2;
  t1 = from.translation();
  t2 = to.translation();

  geometry_msgs::Point pt1;
  geometry_msgs::Point pt2;
  pt1.x = t1[0];
  pt1.y = t1[1];
  pt1.z = t1[2];
  pt2.x = t2[0];
  pt2.y = t2[1];
  pt2.z = t2[2];

  marker.points.push_back(pt1);
  marker.points.push_back(pt2);

  markers.push_back(marker);

}
void RVIZEdge::PubEdge()
{
  visualization_msgs::MarkerArray marker_array;
  for(auto &marker:markers)
  {
      marker_array.markers.push_back(marker);
  }

  edge_pub.publish(marker_array);
  cout << "pub marker size: " << markers.size() << endl;

}
void RVIZEdge::clearEdge()
{
  markers.clear();

}
