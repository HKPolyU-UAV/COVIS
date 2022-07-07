#include "include/optimize_in_frame.h"



OptimizeInFrame::OptimizeInFrame()
{

}
/** @brief In-frame Optimization.
 * We optimize every frame camera pose to prevent from drift accumulation. We Fix all the landmarks nodes in the graph.
 * The function return false if
 * 1. number of valid landmarks lower than 10 (has depth and LK tracking success)
 * 2. number of valid edges lower than 10 (squared error of edge smaller than 3)
@param frame Current frame to optimize camera pose.
*/
bool OptimizeInFrame::optimize(CameraFrame &frame)
{
    //get all landmarks (has depth information and is inliers)
    double fx=frame.d_camera.cam0_fx;
    double fy=frame.d_camera.cam0_fy;
    double cx=frame.d_camera.cam0_cx;
    double cy=frame.d_camera.cam0_cy;
    vector<LandMarkInFrame> lms_in_frame;
    frame.getValidInliersPair(lms_in_frame);
    //cout << lms_in_frame.size() << "|" << frame.landmarks.size() << endl;
    //SE3 pose_befor_ba = frame.T_c_w;
    //cout << "pose_befor_ba: " << frame.T_c_w.translation() << endl;
    if(lms_in_frame.size()<10)
    {
        cout << "no enough tracking landmark " << endl;
        return false;
    }
    else {
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;

        std::unique_ptr<Block::LinearSolverType> linearSolver ( new g2o::LinearSolverEigen<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr ( new Block ( std::move(linearSolver)));
        //g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg ( std::move(solver_ptr));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm (solver);

        //add pose vertex

        g2o::VertexSE3Expmap* v_pose = new g2o::VertexSE3Expmap();
        v_pose->setId(0);
        v_pose->setEstimate(g2o::SE3Quat(frame.T_c_w.so3().unit_quaternion().toRotationMatrix(),
                                         frame.T_c_w.translation()));
        optimizer.addVertex(v_pose);


        vector<g2o::EdgeSE3ProjectXYZ*> edges;
        for(auto lm : lms_in_frame)
        {
            g2o::VertexSBAPointXYZ* v_point = new g2o::VertexSBAPointXYZ();
            v_point->setId (lm.lm_id);
            v_point->setEstimate (lm.lm_3d_w);
            v_point->setFixed(true);
            optimizer.addVertex (v_point);
            g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;
            edge->setId(lm.lm_id);
            edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(lm.lm_id)));
            edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>   (optimizer.vertex(0)));
            edge->setMeasurement(Eigen::Vector2d(lm.lm_2d_undistort));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setParameterId(0,0);
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
            edges.push_back(edge);
        }
        //optimizer.save("../before.g2o");
        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(2);

        for (auto e:edges)
        {
            e->computeError();
            if (e->chi2()>3.0){
                optimizer.removeEdge(e);
            }
        }
        if(optimizer.edges().size()<10)
        {
            cout << "no enough lk tracking, reproj error too high" << endl;
            return false;
        }
        optimizer.initializeOptimization();
        optimizer.optimize(2);

        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(0) );
        Eigen::Isometry3d pose = v->estimate();

        //update frame pose
        frame.T_c_w =  SE3(pose.rotation(),pose.translation());
        //cout << "pose_after_ba: "  << frame.T_c_w.translation() << endl;
        //optimizer.save("../after.g2o");
        return true;

    }
}
