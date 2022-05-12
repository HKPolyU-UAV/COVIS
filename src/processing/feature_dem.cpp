#include <include/feature_dem.h>


// Driver function to sort the vector elements by
// second element of pair in descending order
static bool sortbysecdesc(const pair<cv::Point2f,float> &a,
                          const pair<cv::Point2f,float> &b)
{
    return a.second>b.second;
}

FeatureDEM::FeatureDEM(const int image_width,
                       const int image_height,
                       const Vec6 f_para)
{
    width=image_width;
    height=image_height;
    regionWidth  = static_cast<int>(floor(width/4.0));
    regionHeight = static_cast<int>(floor(height/4.0));
    boundary_dis = static_cast<int>(floor(f_para(2)/2.0));
    max_region_feature_num = static_cast<unsigned int>(f_para(0));
    min_region_feature_num = static_cast<unsigned int>(f_para(1));
    gftt_num = static_cast<int>(f_para(3));
    gftt_ql  = static_cast<double>(f_para(4));
    gftt_dis = static_cast<int>(f_para(5));

    int gridx[5],gridy[5];
    for(int i=0; i<5; i++)
    {
        gridx[i]=i*regionWidth;
        gridy[i]=i*regionHeight;
    }
    for(int i=0; i<16; i++)
    {
        detectorMask[i] = cv::Mat(image_height, image_width, CV_8S,cv::Scalar(0));
        int x_begin,x_end,y_begin,y_end;
        x_begin = gridx[i%4];
        x_end   = gridx[(i%4)+1];
        y_begin = gridy[i/4];
        y_end   = gridy[(i/4)+1];
        for(int xx=x_begin; xx<x_end; xx++)
        {
            for(int yy=y_begin; yy<y_end; yy++)
            {
                detectorMask[i].at<schar>(yy,xx)=1;

            }
        }
    }
//    for(int i = 0; i < 16; i++)
//    {
//        cout << "detectorMask[" << i << "]: " << endl;
//        cout << this->detectorMask[i].size << endl;
//    }

}

FeatureDEM::~FeatureDEM()
{;}

void FeatureDEM::calHarrisR(const cv::Mat& img,
                            cv::Point2f& Pt,
                            float &R)
{
    uchar patch[9];
    int xx = static_cast<int>(Pt.x);
    int yy = static_cast<int>(Pt.y);
    patch[0]=img.at<uchar>(cv::Point(xx-1,yy-1));
    patch[1]=img.at<uchar>(cv::Point(xx,yy-1));
    patch[2]=img.at<uchar>(cv::Point(xx+1,yy-1));
    patch[3]=img.at<uchar>(cv::Point(xx-1,yy));
    patch[4]=img.at<uchar>(cv::Point(xx,yy));
    patch[5]=img.at<uchar>(cv::Point(xx+1,yy)); // typo?
    patch[6]=img.at<uchar>(cv::Point(xx-1,yy+1));
    patch[7]=img.at<uchar>(cv::Point(xx,yy+1));
    patch[8]=img.at<uchar>(cv::Point(xx+1,yy+1));
    float IX,IY;
    float X2,Y2;
    float XY;
    IX = (patch[0]+patch[3]+patch[6]-(patch[2]+patch[5]+patch[8]))/3;
    IY = (patch[0]+patch[1]+patch[2]-(patch[6]+patch[7]+patch[8]))/3;
    X2 = IX*IX;
    Y2 = IY*IY;  // typo?
    XY = IX*IY;  // typo?
    //M = | X2  XY |
    //    | XY  Y2 |
    //R = det(M)-k(trace^2(M))
    //  = X2*Y2-XY*XY  - 0.05*(X2+Y2)*(X2+Y2)
    R = (X2*Y2)-(XY*XY) - static_cast<float>(0.05)*(X2+Y2)*(X2+Y2);
}


/** @brief Divided all features into 16 regions
@param region Each region is filled with keypoints and corresponding corner response
@param existed_features
*/
void FeatureDEM::fillIntoRegion(const cv::Mat& img, const vector<cv::Point2f>& pts,
                                vector<pair<cv::Point2f,float>> (&region)[16], bool existed_features)
{
    if(existed_features)
    {//do not calculate harris value
        //cout << "existed feature" << endl;
        for(size_t i=0; i<pts.size(); i++)
        {
            cv::Point2f pt = pts.at(i);
            if (pt.x>=3 && pt.x<(width-3) && pt.y>=3 && pt.y<(height-3))
            {
            int regionNum= static_cast<int>(4*floor(pt.y/regionHeight) + (pt.x/regionWidth));
            region[regionNum].push_back(make_pair(pt,99999.0));
            }
        }
    }
    else
    {
        //cout << "Not existed feature" << endl;

        for(size_t i=0; i<pts.size(); i++)
        {
            cv::Point2f pt = pts.at(i);
            if (pt.x>=3 && pt.x<(width-3) && pt.y>=3 && pt.y<(height-3))
            {
                float Harris_R;
                /// compute Harris_R of a 2-D keypoint
                calHarrisR(img,pt,Harris_R);
                //cout << "Harris_R calculated: " << Harris_R << " -------------------- " << endl;

                ///regionNum range: [0,15]. for every region[regionNum], fill in the pair of cv::Point2f
                /// and corresponding Harris_R
                int regionNum= static_cast<int>(4*floor(pt.y/regionHeight) + (pt.x/regionWidth));
                region[regionNum].push_back(make_pair(pt,Harris_R));
            }
        }
    }
}

/** @brief ReDetect corner. The existedPts will be filled into region first and compared with new features.
@param existedPts Input vector of detected keypoints.
@param newPts Output vector of added detected keypoints

*/

void FeatureDEM::redetect(const cv::Mat& img,
                          const vector<Vec2>& existedPts,
                          vector<cv::Point2f>& newPts,
                          int &newPtscount)
{

    //Clear
    newPts.clear();
    newPtscount=0;
    vector<cv::Point2f> new_features;
    new_features.clear();

    //fill the existed features into region
    vector<cv::Point2f> existedPts_cvP2f=vVec2_2_vcvP2f(existedPts); // from std::vector<Eigen::Vec2> to vector<cv::Point2f>
    for(int i=0; i<16; i++)
    {
        regionKeyPts[i].clear();
    }
    fillIntoRegion(img,existedPts_cvP2f,regionKeyPts,true);

    //
    cv::Mat mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
    //over exposure mask
//    int over_exp_block_cnt=0;
//    for(int xx=10; xx < img.cols-10; xx+=10)
//    {
//        for(int yy=10; yy < img.rows-10; yy+=10)
//        {
//            if(img.at<unsigned char>(yy,xx)==255)
//            {
//                over_exp_block_cnt++;
//                cv::circle(mask, cv::Point2f(xx,yy), 15, 0, -1);

//            }
//        }
//    }

    vector<cv::Point2f>  features;
    cv::goodFeaturesToTrack(img, features, gftt_num, gftt_ql, gftt_dis, mask);
    vector<pair<cv::Point2f,float>> regionKeyPts_prepare[16];
    for(int i=0; i<16; i++)
    {
        regionKeyPts_prepare[i].clear();
    }
    fillIntoRegion(img,features,regionKeyPts_prepare,false);

    //pith up new features: new detected features in regionKeyPts_prepare
    for(size_t i=0; i<16; i++)
    {
        sort(regionKeyPts_prepare[i].begin(), regionKeyPts_prepare[i].end(), sortbysecdesc);
        for(size_t j=0; j<regionKeyPts_prepare[i].size(); j++)
        {
            int noFeatureNearby = 1;
            cv::Point pt=regionKeyPts_prepare[i].at(j).first;
            for(size_t k=0; k<regionKeyPts[i].size(); k++)
            {
                float dis_x = fabs(pt.x-regionKeyPts[i].at(k).first.x);
                float dis_y = fabs(pt.y-regionKeyPts[i].at(k).first.y);

                if(dis_x <= boundary_dis || dis_y <= boundary_dis)
                {
                    noFeatureNearby=0;
                }
            }
            if(noFeatureNearby)
            {
                regionKeyPts[i].push_back(make_pair(pt,999999.0));
                new_features.push_back(pt);
                if(regionKeyPts[i].size() >= max_region_feature_num) break;
            }
        }
        //        if(regionKeyPts[i].size()<min_region_feature_num)
        //        {
        //            int cnt = min_region_feature_num-regionKeyPts[i].size();
        //            int x_begin = regionWidth*(i%4);
        //            int y_begin = regionHeight*(i/4);
        //            for(int i=0; i<cnt; i++)
        //            {
        //                int x=rand() % (regionWidth-10) + 5;
        //                int y=rand() % (regionHeight-10) + 5;
        //                cv::Point pt(x+x_begin,y+y_begin);

        //                new_features.push_back(pt);
        //            }
        //        }
    }
//    cout << "after regionKeyPts " << endl;

//    for(int i = 0; i < 16; i++)
//    {
//        cout << "region " << i << "'s size " << regionKeyPts[i].size() << endl;

//        for (int j = 0; j< regionKeyPts[i].size(); j++)
//        {
//            cout << regionKeyPts[i].at(j).first << " " << regionKeyPts[i].at(j).second <<  endl;
//        }
//    }
    //output
    if(new_features.size()>0)
    {
        newPts = new_features;
        //    trackedLMDescriptors.push_back(zeroDescriptor);
    }

}
/** @brief Detect corner and remove too much close keypoint to make a uniform distribution
@param img Input image from left camera
@param newPts Output vector of detected keypoints

*/
void FeatureDEM::detect(const cv::Mat& img, vector<cv::Point2f>& newPts)
{

    ///Clear
    newPts.clear();
    /// Harris Corner detecor, using cv::cornerMinEigenVal to reject corners, for the first time detect, accept two times maixmum
    /// number of corners
    vector<cv::Point2f>  features;
    cv::goodFeaturesToTrack(img,features, gftt_num*2, gftt_ql, gftt_dis);

    for(int i=0; i<16; i++)
    {
        regionKeyPts[i].clear();
    }
    /// after fillIntoRegion, the features are filled into regionKeyPts, but not sorted yet
    fillIntoRegion(img,features,regionKeyPts,false);
    ///For every region, select features by Harris response and boundary size
    for(int i=0; i<16; i++)
    {
        /// tmp = [tmp[0] tmp[1] tmp[2] ... ]  <-j
        ///
        /// regionKeyPts[i] = [tmp[0] ...]     <-k

        /// sort cv::Point2f in each region by compare Harris_R,
        /// std::vector<pair<cv::Point2f,float>> regionKeyPts[i]
        sort(regionKeyPts[i].begin(), regionKeyPts[i].end(), sortbysecdesc);
        vector<pair<cv::Point2f,float>> tmp = regionKeyPts[i]; // each region's candidate keypoints
//        cout << "i: " << i << endl;
//        for (auto v:tmp)
//        {
//          cout << "tmp:: pixel:" << v.first << " Harris_R:" << v.second << endl;
//        }
        regionKeyPts[i].clear();

        unsigned int count = 0;
        for(size_t j=0; j<tmp.size(); j++)
        {
            /// j is the index over all the keypoints in regionKeyPts[i]
            //cout <<  "j: " << j << endl;
            int outSideConflictBoundary = 1;
            for(size_t k=0; k<regionKeyPts[i].size(); k++)
            {

                /// k is the index over keypoints in regionKeyPts[i]
                //cout << " k: " << k << endl;
                ///
                float dis_x = fabs(tmp.at(j).first.x-regionKeyPts[i].at(k).first.x);
                float dis_y = fabs(tmp.at(j).first.y-regionKeyPts[i].at(k).first.y);

                if(dis_x<=boundary_dis || dis_y<=boundary_dis)
                {
                    /// the distance between keypoints is too close
                    outSideConflictBoundary=0;
                }
            }
            if(outSideConflictBoundary)
            {
                regionKeyPts[i].push_back(tmp.at(j));
                count++;
//                for (auto v: regionKeyPts[i])
//                {
//                  cout << "region pixel:" << v.first <<  " Harris_R:" << v.second << endl;
//                }
//                cout << "count: "<< count << endl;
//                cout << endl;
                if(count>=max_region_feature_num) break;
            }

        }

    }

    features.clear();
    for(int i=0; i<16; i++)
    {
        for(size_t j=0; j<regionKeyPts[i].size(); j++)
        {
            features.push_back(regionKeyPts[i].at(j).first);
        }
    }
    newPts = features;
//    cout << "after selecting, features size: " << features.size() << endl;
//    for(auto f:features)
//    {
//        cout << f << endl;
//    }
    //    trackedLMDescriptors.push_back(zeroDescriptor);

}



//void FeatureDEM::detect_conventional(const cv::Mat& img, vector<Vec2>& pts, vector<cv::Mat>& descriptors)
//{
//    //Clear
//    pts.clear();
//    descriptors.clear();
//    for(int i=0; i<16; i++)
//    {
//        regionKeyPts[i].clear();
//    }
//    //Detect FAST
//    cv::Ptr<cv::FastFeatureDetector> detector= cv::FastFeatureDetector::create();
//    //cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(4000);
//    vector<cv::KeyPoint> tmpKPs;
//    detector->detect(img, tmpKPs);
//    //Fill into region
//    vector<cv::Point2f>  tmpPts;
//    cv::KeyPoint::convert(tmpKPs,tmpPts);
//    vector<cv::Point2f>  output;
//    output.clear();
//    int range=tmpPts.size();
//    for(int i=0; i<900; i++)
//    {
//        int idx = rand() % range;
//        output.push_back(tmpPts.at(idx));
//    }
//    cv::Mat tmpDescriptors;
//    cv::KeyPoint::convert(output,tmpKPs);
//    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
//    extractor->compute(img, tmpKPs, tmpDescriptors);
//    for(size_t i=0; i<tmpKPs.size(); i++)
//    {
//        pts.push_back(Vec2(tmpKPs.at(i).pt.x,tmpKPs.at(i).pt.y));
//    }
//    descriptors_to_vMat(tmpDescriptors,descriptors);
//}
