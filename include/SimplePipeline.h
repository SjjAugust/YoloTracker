//
// Created by sumbal on 20/06/18.
//

#ifndef PROJECT_PIPELINE_H
#define PROJECT_PIPELINE_H

#pragma once

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <opencv/cxmisc.h>
#include "defines.h"
#include <math.h>
#include <algorithm>
#include "../yolo/detector.h"


class SimplePipeline
{
public:
    explicit SimplePipeline(const cv::CommandLineParser &parser)
    {
        
        inFile = parser.get<std::string>(0);
        startFrame = parser.get<int>("start_frame");
        endFrame = parser.get<int>("end_frame");
        detectThreshold = parser.get<float>("threshold");
        desiredDetect = parser.get<bool>("desired_detect");
        desiredObjectsString = parser.get<std::string>("desired_objects");
        saveVideo = parser.get<bool>("save_video");
        outFile = parser.get<std::string>("output");

        m_fps = 30;
        
        if (!parser.check())
        {
            parser.printErrors();
        }

        // Different color used for path lines in tracking
        // Add more if you are a colorful person.
        m_colors.emplace_back(cv::Scalar(255, 0, 0));
        m_colors.emplace_back(cv::Scalar(0, 255, 0));
        m_colors.emplace_back(cv::Scalar(0, 0, 255));
        m_colors.emplace_back(cv::Scalar(255, 255, 0));
        m_colors.emplace_back(cv::Scalar(0, 255, 255));
        m_colors.emplace_back(cv::Scalar(255, 0, 255));
        m_colors.emplace_back(cv::Scalar(255, 127, 255));
        m_colors.emplace_back(cv::Scalar(127, 0, 255));
        m_colors.emplace_back(cv::Scalar(127, 0, 127));
    }

    void Process(){

        // Prepossessing step. May be make a new function to do prepossessing (TODO)
        // Converting desired object into float.
        std::vector <float> desiredObjects;
        std::stringstream ss(desiredObjectsString);
        //get indexes of detection object 
        while( ss.good() )
        {
            string substring;
            getline( ss, substring, ',' );
            desiredObjects.push_back( std::stof(substring) );
        }

        LOG(INFO) << "Process start" << std::endl;

#ifndef GFLAGS_GFLAGS_H_
        namespace gflags = google;
#endif

        // Set up input
        cv::VideoCapture cap(inFile);
        if (!cap.isOpened()) {
            LOG(FATAL) << "Failed to open video: " << inFile;
        }
        cv::Mat frame;
        int frameCount = 0;

        // video output
        cv::VideoWriter writer;
        auto frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
        auto frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_fps, cv::Size(frame_width, frame_height), true);

        std::map<string, int> countObjects_LefttoRight;
        std::map<string, int> countObjects_RighttoLeft;
        double fontScale = CalculateRelativeSize(frame_width, frame_height);

        double tFrameModification = 0;
        double tDetection = 0;
        double tTracking = 0;
        double tDTC = 0;
        double tStart  = cv::getTickCount();

        // Process one frame at a time
        while (true) {

            bool success = cap.read(frame);
            if (!success) {
                LOG(INFO) << "Process " << frameCount << " frames from " << inFile;
                break;
            }
            if(frameCount < startFrame)
            {
                continue;
            }

            if (frameCount > endFrame)
            {
                std::cout << "Process: reached last " << endFrame << " frame" << std::endl;
                break;
            }
            CHECK(!frame.empty()) << "Error when read frame";

            // Get all the detected objects.
            double tStartDetection = cv::getTickCount();
            regions_t tmpRegions;
            std::vector<vector<float>> detections = detectframe(frame);

            // Filter out all the objects based
            // 1. Threshold
            // 2. Desired object classe
            for (auto const& detection : detections){
                const vector<float> &d = detection;
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];
                const int label_idx = static_cast<int>(d[1]);
                if(desiredDetect)
                {
                    if (!(std::find(desiredObjects.begin(), desiredObjects.end(), label_idx) != desiredObjects.end()))
                    {
                        continue;
                    }
                }

                std::string label;
                switch (label_idx)
                {
                case 1:
                    label = "pedestrian";
                    break;
                case 2:
                    label = "people";
                    break;
                case 3:
                    label = "bicycle";
                    break;
                case 4:
                    label = "car";
                    break;
                case 5:
                    label = "van";
                    break;
                case 6:
                    label = "truck";
                    break;
                case 7:
                    label = "tricycle";
                    break;
                case 8:
                    label = "awing-tricycle";
                    break;
                case 9:
                    label = "bus";
                    break;
                case 10:
                    label = "motor";
                    break;
                default:
                    label = std::to_string(static_cast<int>(label_idx));
                    break;
                }

                // auto xLeftBottom = static_cast<int>(d[3] * frame.cols);
                // auto yLeftBottom = static_cast<int>(d[4] * frame.rows);
                // auto xRightTop = static_cast<int>(d[5] * frame.cols);
                // auto yRightTop = static_cast<int>(d[6] * frame.rows);
                cv::Rect object(d[3], d[4], d[5], d[6]);
                tmpRegions.push_back(CRegion(object, label, score));
            }
            tDetection += cv::getTickCount() - tStartDetection;

            double tStartTracking = cv::getTickCount();
            // Update Tracker
            cv::UMat clFrame;
            clFrame = frame.getUMat(cv::ACCESS_READ);
            m_tracker->Update(tmpRegions, clFrame, m_fps);
            tTracking += cv::getTickCount() - tStartTracking;
            DrawData(frame, fontScale);
            if (writer.isOpened() and saveVideo)
            {
                writer << frame;
            }
            ++frameCount;
        }
        if (cap.isOpened()) {
            cap.release();
        }

        // Calculate Time for components
        double tEnd  = cv::getTickCount();
        double totalRunTime = (tEnd - tStart)/cv::getTickFrequency();
        double detectionRunTime = tDetection/cv::getTickFrequency();
        double trackingRunTime = tTracking/cv::getTickFrequency();
        double FDTCRuntime = detectionRunTime + trackingRunTime;

        // Display and write output
        std::ofstream csvFile;
        csvFile.open ("../data/D2.csv");
        csvFile << "Detection time" << ",";
        csvFile << "Tracking time" << ",";
        csvFile << "FDTC time" << ",";
        csvFile << "Total time" << ",";
        csvFile << "FDTC frame rate" << ",";
        csvFile << "Total frame rate" << "\n";
        LOG(INFO)  << "Detection time = " << detectionRunTime << " seconds" << std::endl;
        csvFile << detectionRunTime << ",";
        LOG(INFO)  << "Tracking time = " << trackingRunTime << " seconds" << std::endl;
        csvFile << trackingRunTime<< ",";
        LOG(INFO)  << "FDTC time = " << FDTCRuntime << " seconds " << std::endl;
        csvFile << FDTCRuntime << ",";
        LOG(INFO)  << "Total time = " << totalRunTime << " seconds " << std::endl;
        csvFile << totalRunTime << ",";
        LOG(INFO)  << " FDTC frame rate: "<< frameCount/FDTCRuntime << " fps" <<std::endl;
        csvFile << frameCount/FDTCRuntime  << ",";
        LOG(INFO)  << " Total frame rate: "<< frameCount/totalRunTime << " fps" << std::endl;
        csvFile << frameCount/totalRunTime << "\n";
        LOG(INFO)  << "Left to Right or Top to Bottom ";
        csvFile << "Object label" << "," << "count Left to Right" << "\n";
        for(auto elem : countObjects_LefttoRight)
        {
            LOG(INFO) << elem.first << " " << elem.second << "\n";
            csvFile << elem.first << "," << elem.second << "\n";
        }
        LOG(INFO)  << "Right to Left or Bottom to Top";
        csvFile << "Object label" << "," << "count Right to Left" << "\n";
        for(auto elem : countObjects_RighttoLeft)
        {
            LOG(INFO) << elem.first << " " << elem.second << "\n";
            csvFile << elem.first << "," << elem.second << "\n";
        }

        csvFile.close();
    }
protected:
    std::unique_ptr<CTracker> m_tracker;
    float m_fps;
    int direction;

    virtual std::vector<vector<float> > detectframe(cv::Mat frame)= 0;
    virtual void DrawData(cv::Mat frame, double fontScale) = 0;

    void DrawTrack(cv::Mat frame,
                   const CTrack& track,
                   bool drawTrajectory = true,
                   bool isStatic = false
    )
    {

        cv::rectangle(frame, track.GetLastRect(), cv::Scalar(119, 102, 39), 1, CV_AA);

        if (drawTrajectory)
        {
            cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

            for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
            {
                const TrajectoryPoint& pt1 = track.m_trace.at(j);
                const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);

                cv::line(frame, pt1.m_prediction, pt2.m_prediction, cv::Scalar(119, 102, 39), 1, CV_AA);
                
            }
        }
    }

    double CalculateRelativeSize(int frame_width, int frame_height)
    {
        int baseLine = 0;
        double countBoxWidth = frame_width * 0.1;
        double countBoxHeight = frame_height * 0.1;
        std::string counterLabel_Left = "Count : " + std::to_string(0);
        cv::Size rect = cv::getTextSize(counterLabel_Left, cv::FONT_HERSHEY_PLAIN, 1.0, 1, &baseLine);
        double scalex = countBoxWidth / (double)rect.width;
        double scaley = countBoxHeight / (double)rect.height;
        return std::min(scalex, scaley);
    }
private:
    bool saveVideo;
    int endFrame;
    int startFrame;
    cv::Rect cropRect;
    bool desiredDetect;
    std::string inFile;
    std::string outFile;
    float detectThreshold;
    std::vector<cv::Scalar> m_colors;
    std::string desiredObjectsString;


};

class YoloExample : public SimplePipeline{
public:
    explicit YoloExample(const cv::CommandLineParser &parser) : Pipeline(parser){
        modelFile = parser.get<std::string>("model");
        weightsFile = parser.get<std::string>("weight");

        // Initialize the Detector
        detector = Detector(modelFile, weightsFile, 0);

        // Initialize the tracker
        config_t config;

        // TODO: put these variables in main
        TrackerSettings settings;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 100;                          // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = (size_t)(1 * m_fps);  // Maximum allowed skipped frames
        settings.m_maxTraceLength = (size_t)(5 * m_fps);               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);
    }
private:
    std::string modelFile;
    std::string weightsFile;
    Detector detector;
    int frame_count_ = 0;

protected:
    std::vector<vector<float> > detectframe(cv::Mat frame){
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax]
        auto result = detector.detect(frame, detectThreshold);
        std::vector<vector<float>> ret;
        for(auto bbox : result){
            float xmin = static_cast<float>(bbox.x);
            float ymin = static_cast<float>(bbox.y);
            float xmax = static_cast<float>(bbox.w) + xmin;
            float ymax = static_cast<float>(bbox.h) + ymin;
            ret.emplace_back({frame_count_, static_cast<float>(bbox.obj_id+1), bbox.prob, xmin, ymin, xmax, ymax});
        }
        ++frame_count_;
        return ret;
    }
    void DrawData(cv::Mat frame, double fontScale){
        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
                std::string label = track->m_lastRegion.m_type + ": " + std::to_string((int)(track->m_lastRegion.m_confidence * 100)) + " %";
                //std::string label = std::to_string(track->m_trace.m_firstPass) + " | " + std::to_string(track->m_trace.m_secondPass);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0),1);
            }
        }

    }



   
};

#endif //PROJECT_PIPELINE_H