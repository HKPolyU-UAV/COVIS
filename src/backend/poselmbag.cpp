#include "include/poselmbag.h"
#include <include/common.h>
#include <stdio.h>
//pose_buffer_size
PoseLMBag::PoseLMBag(int pose_buffer_size_in)
{
    this->pose_buffer_size = pose_buffer_size_in;
    this->lm_sub_bag.clear();
    this->pose_sub_bag.clear();
    POSE_ITEM pose;
    for(int i=0; i<pose_buffer_size; i++)
    {
        pose_sub_bag.push_back(pose);
    }
    wp_init=0;//write pointer for initialization
    pose_cnt_init=0;
    pose_sub_bag_initialized=false;
}

void PoseLMBag::reset()
{
    this->lm_sub_bag.clear();
    this->pose_sub_bag.clear();
    POSE_ITEM pose;
    for(int i=0; i<pose_buffer_size; i++)
    {
        pose_sub_bag.push_back(pose);
    }
    wp_init=0;//write pointer for initialization
    pose_cnt_init=0;
    pose_sub_bag_initialized=false;
}
/** @brief Check landmark exist. If lm_sub_bag has this landmark, the function return true, the index will be returned
@param id_in Landmark id to query.
@param idx Default as 0

*/
bool PoseLMBag::hasTheLM(int64_t id_in, int &idx)
{
    idx = 0;
    for(unsigned int i=0; i<lm_sub_bag.size(); i++)
    {
        if(lm_sub_bag.at(i).id == id_in)
        {
            idx = i;

            return true;
        }
    }

    return false;
}
/** @brief Add Keyframe Landmarks to lm_sub_bag. If the landmark is observed multiple times, the p3d_w will be updated
@param id_in Keyframe landmark id.
@param p3d_w_in Keyframe landmark 3d info p3d_w.

*/
bool PoseLMBag::addLMObservationSlidingWindow(int64_t id_in, Vec3 p3d_w_in)
{
    int idx; //index in lm_sub_bag
    bool add_lm_to_optimizer=false;
    if(this->hasTheLM(id_in,idx))
    {//update lm with average 3d inf
        Vec3 p3d_w;
        int cnt = this->lm_sub_bag.at(idx).count;
        //cout << "cnt: "<< cnt << endl;
        //cout << "p3d_w " << this->lm_sub_bag.at(idx).p3d_w.transpose() << endl;
        cnt++;
        this->lm_sub_bag.at(idx).count = cnt;
        //cout << "after cnt: "<< cnt << endl;

    }else{//
        LM_ITEM lm_item;
        lm_item.id = id_in;
        lm_item.count = 1;
        lm_item.p3d_w = p3d_w_in;
        this->lm_sub_bag.push_back(lm_item);
        add_lm_to_optimizer=true;
    }
    return add_lm_to_optimizer;
}
/** @brief Add Keyframe Landmarks to lm_sub_bag. If the landmark is observed multiple times, the p3d_w will be updated
@param id_in Keyframe landmark id.
@param p3d_w_in Keyframe landmark 3d info p3d_w.

*/
bool PoseLMBag::addLMObservation(int64_t id_in, Vec3 p3d_w_in)
{
    int idx;
    bool add_lm_to_optimizer=false;
    if(this->hasTheLM(id_in,idx))
    {//update lm with average 3d inf
        Vec3 p3d_w;
        int cnt = this->lm_sub_bag.at(idx).count;
        //cout << "cnt: "<< cnt << endl;
       // cout << "p3d_w " << this->lm_sub_bag.at(idx).p3d_w.transpose() << endl;
        p3d_w = static_cast<double>(cnt) * this->lm_sub_bag.at(idx).p3d_w + p3d_w_in;
        cnt++;
        p3d_w = (1.0/static_cast<double>(cnt))*p3d_w;
        //cout << "p3d_w new " << p3d_w.transpose() << endl;

        this->lm_sub_bag.at(idx).count = cnt;
        this->lm_sub_bag.at(idx).p3d_w = p3d_w;
    }else{// idx == 0 lm_sub_bag does not have landmark with same lm_id
        LM_ITEM lm_item;
        lm_item.id = id_in;
        lm_item.count = 1;
        lm_item.p3d_w = p3d_w_in;
        this->lm_sub_bag.push_back(lm_item);
        add_lm_to_optimizer=true;
    }
    return add_lm_to_optimizer;
}

bool PoseLMBag::removeLMObservation(int64_t id_in)
{
    bool remove_lm_from_optimizer=false;
    int idx;
    if(this->hasTheLM(id_in,idx))
    {
        lm_sub_bag.at(idx).count--;
        if(lm_sub_bag.at(idx).count==0)
        {
            //cout << "remove from bag" << endl;
            lm_sub_bag.erase(lm_sub_bag.begin() + idx);
            remove_lm_from_optimizer = true;
        }
    }
    return remove_lm_from_optimizer;
}
/** @brief Add Keyframe pose to pose_sub_bag
@param id_in Keyframe frame_id.
@param pose_in Keyframe pose T_c_w

*/
void PoseLMBag::addPose(int64_t id_in, SE3 pose_in)
{
    if(this->pose_sub_bag_initialized)
    {
        //cover the oldest pose with the newpose
        //cout << "pose_sub_bag_initialized: " << pose_sub_bag_initialized << endl;
        newest = oldest;
        this->pose_sub_bag[newest].relevent_frame_id = id_in;
        this->pose_sub_bag[newest].pose = pose_in;
        this->oldest++;
        if(this->oldest==pose_buffer_size)
        {
            this->oldest = 0;
        }
    }else
    {
        //cout << "pose_sub_bag_initialized: " << pose_sub_bag_initialized << endl;
        //cout << "wp_init: " << wp_init << endl;
        this->pose_sub_bag[wp_init].relevent_frame_id = id_in;
        this->pose_sub_bag[wp_init].pose = pose_in;
        this->pose_sub_bag[wp_init].pose_id = wp_init;
        wp_init++;
        //cout << "wp_init: " << wp_init << endl;

        if(this->wp_init==pose_buffer_size)
        {
            this->pose_sub_bag_initialized = true;
            this->oldest = 0;
            this->newest = (pose_buffer_size-1);
        }

    }
    //cout <<"oldest: " <<  oldest << " newest: " <<  newest << endl;
}
/** @brief return lm_sub_bag
*/
void PoseLMBag::getAllLMs(vector<LM_ITEM> &lms_out)
{
    lms_out = this->lm_sub_bag;
}
/** @brief return landmarks by multiviewed
*/
void PoseLMBag::getMultiViewLMs(vector<LM_ITEM> &lms_out, int view_cnt)
{
    lms_out.clear();
    for(auto lm_in_bag:lm_sub_bag)
    {
        if(lm_in_bag.count>=view_cnt)
        {
            lms_out.push_back(lm_in_bag);
        }
    }
}

/** @brief return pose_sub_bag
*/
void PoseLMBag::getAllPoses(vector<POSE_ITEM> &poses_out)
{
    poses_out.clear();
    for (int i=0; i<pose_buffer_size; i++) {
        poses_out.push_back(this->pose_sub_bag[i]);
    }
}

int PoseLMBag::getNewestPoseInOptimizerIdx(void)
{
    return this->newest;
}

int PoseLMBag::getOldestPoseInOptimizerIdx(void)
{
    return this->oldest;
}

/** @brief Query pose relevant_frame_id
@param id_in Keyframe frame_id.
@param pose_in Keyframe pose T_c_w

*/
int64_t PoseLMBag::getPoseIdByReleventFrameId(int64_t frame_id)
{
    int64_t ret_val=-1;
    for(int i=0; i<pose_buffer_size; i++)
    {
        if(pose_sub_bag[i].relevent_frame_id == frame_id)
        {
            ret_val = i;
            break;
        }
    }
    return ret_val;
}

void PoseLMBag::debug_output(void)
{
    cout << "debug output" << endl;
    cout << "Poses:" << endl;
    for(int i=0; i<pose_buffer_size; i++)
    {
        cout << "pose id " << pose_sub_bag[i].pose_id
             << " relevent frame id: " << pose_sub_bag[i].relevent_frame_id
             << " Twc: " << pose_sub_bag[i].pose.inverse().so3().log().transpose()
             << " | " << pose_sub_bag[i].pose.inverse().translation().transpose() << endl;;
    }
    cout << "LMs:" << endl;
    for(std::vector<LM_ITEM>::iterator it = this->lm_sub_bag.begin(); it != this->lm_sub_bag.end(); ++it)
    {
        cout << "lm id" << it->id
             << " count " << it->count
             << " p3d: " << it->p3d_w.transpose() << endl;
    }

}
