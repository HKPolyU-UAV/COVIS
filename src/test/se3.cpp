#include <include/common.h>
#include <iostream>
#include <include/kinetic_math.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <include/rviz_pose.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#define PI 3.1415926
using namespace std;
geometry_msgs::PoseStamped pose1;
geometry_msgs::PoseStamped pose2;
geometry_msgs::PoseStamped pose_1_2;
int main(int argc, char **argv)
{

    ros::init(argc, argv, "test");
    ros::NodeHandle nh("~");
    ros::Subscriber imu_sub;
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/imu_pose1", 2);
    ros::Publisher pose_pub_2 = nh.advertise<geometry_msgs::PoseStamped>("/imu_pose2", 2);
    ros::Publisher pose_pub_3 = nh.advertise<geometry_msgs::PoseStamped>("/relative_pose", 2);

    Vec3 v1(0 ,0, PI/4);
    Matrix3d R1= rpy2R(v1);
    Vec3 t1(1,0,0);
    SE3 pose_1(R1,t1);
    cout << pose_1.rotation_matrix() << endl;
    cout << pose_1.translation() << endl;
    Quaterniond q = rpy2Q(v1);

    pose1.header.stamp = ros::Time::now();
    pose1.header.frame_id = "map";
    pose1.pose.position.x = t1[0];
    pose1.pose.position.y = t1[1];
    pose1.pose.position.z = t1[2];
    pose1.pose.orientation.x = q.x();
    pose1.pose.orientation.y = q.y();
    pose1.pose.orientation.z = q.z();
    pose1.pose.orientation.w = q.w();
    //cout << (T_cam_imu.inverse()).rotation_matrix() << endl;
    //cout << (T_cam_imu.inverse()).translation() << endl;
    Vec3 rpy2(0 ,0, PI/2);
    Matrix3d R2= rpy2R(rpy2);
    Vec3 t2(1,0,0);
    SE3 pose_2(R2,t2);
    cout << pose_2.rotation_matrix() << endl;
    cout << pose_2.translation() << endl;
    Quaterniond q2 = rpy2Q(rpy2);

    pose2.header.stamp = ros::Time::now();
    pose2.header.frame_id = "map";
    pose2.pose.position.x = t2[0];
    pose2.pose.position.y = t2[1];
    pose2.pose.position.z = t2[2];
    pose2.pose.orientation.x = q2.x();
    pose2.pose.orientation.y = q2.y();
    pose2.pose.orientation.z = q2.z();
    pose2.pose.orientation.w = q2.w();

    SE3 T_1_2 = pose_1.inverse() * pose_2;  // 2 in 1 frames T_w_c
    T_1_2 = pose_1.inverse() * pose_2;
    SE3 T_2_1 = pose_2.inverse() * pose_1; // 1 in 2 frames T_w_c
    cout << T_2_1.rotation_matrix() << endl;
    cout << T_2_1.translation() << endl;

    pose_1_2.header.stamp = ros::Time::now();
    pose_1_2.header.frame_id = "map";
    pose_1_2.pose.position.x = T_2_1.translation().x();
    pose_1_2.pose.position.y = T_2_1.translation().y();
    pose_1_2.pose.position.z = T_2_1.translation().z();
    pose_1_2.pose.orientation.x = T_2_1.unit_quaternion().x();
    pose_1_2.pose.orientation.y = T_2_1.unit_quaternion().y();
    pose_1_2.pose.orientation.z = T_2_1.unit_quaternion().z();
    pose_1_2.pose.orientation.w = T_2_1.unit_quaternion().w();

    ros::Rate loop_rate(20);

    while(ros::ok())
    {
      pose_pub.publish(pose1);
      pose_pub_2.publish(pose2);
      pose_pub_3.publish(pose_1_2);
      ros::spinOnce();
      loop_rate.sleep();
    }




    return 0;
}
