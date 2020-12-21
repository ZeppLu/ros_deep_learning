/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>

#include <jetson-inference/detectNet.h>
#include <jetson-utils/cudaMappedMemory.h>

#include "image_converter.h"

#include <unordered_map>

#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>


namespace ros_deep_learning {

class DetectNetNode : public nodelet::Nodelet {
public:
	// XXX: if cannot run nodelet, try change to `virtual void onInit()'
	void onInit() override;

private:
	detectNet* net = nullptr;
	imageConverter* cvt = nullptr;
	ros::Publisher detect_pub, info_pub;
	ros::Subscriber img_sub;
	vision_msgs::VisionInfoPtr info_msg = nullptr;  // this field remains unchanged after initialized

	void info_connect(const ros::SingleSubscriberPublisher& pub);
	void img_callback(const sensor_msgs::ImageConstPtr& input);
};


// callback triggered when a new subscriber connected to vision_info topic
void DetectNetNode::info_connect(const ros::SingleSubscriberPublisher& pub)
{
	throw ros::Exception("don't use");
	NODELET_INFO("new subscriber '%s' connected to vision_info topic '%s', sending VisionInfo msg", pub.getSubscriberName().c_str(), pub.getTopic().c_str());
	pub.publish(info_msg);
}


// input image subscriber callback
void DetectNetNode::img_callback(const sensor_msgs::ImageConstPtr& input)
{
	ros::Time time_received = ros::Time::now();
	NODELET_INFO("received image with %fs delay",
			(ros::Time::now() - input->header.stamp).toSec());

	// convert the image to reside on GPU
	if( !cvt || !cvt->Convert(input) )
	{
		NODELET_INFO("failed to convert %ux%u %s image", input->width, input->height, input->encoding.c_str());
		return;
	}

	// classify the image
	detectNet::Detection* detections = NULL;

	const int numDetections = net->Detect(cvt->ImageGPU(), cvt->GetWidth(), cvt->GetHeight(), &detections, detectNet::OVERLAY_NONE);

	// verify success
	if( numDetections < 0 )
	{
		NODELET_ERROR("failed to run object detection on %ux%u image", input->width, input->height);
		return;
	}

	// create a detection for each bounding box
	// TODO: change to pointer
	vision_msgs::Detection2DArrayPtr msg(new vision_msgs::Detection2DArray());

	// if objects were detected, send out message
	if( numDetections > 0 )
	{
		NODELET_INFO("detected %i objects in %ux%u image", numDetections, input->width, input->height);

		for( int n=0; n < numDetections; n++ )
		{
			detectNet::Detection* det = detections + n;

			NODELET_INFO("object %i class #%u (%s)  confidence=%f\n", n, det->ClassID, net->GetClassDesc(det->ClassID), det->Confidence);
			NODELET_INFO("object %i bounding box (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, det->Left, det->Top, det->Right, det->Bottom, det->Width(), det->Height());

			// create a detection sub-message
			vision_msgs::Detection2D detMsg;

			detMsg.bbox.size_x = det->Width();
			detMsg.bbox.size_y = det->Height();

			float cx, cy;
			det->Center(&cx, &cy);

			detMsg.bbox.center.x = cx;
			detMsg.bbox.center.y = cy;

			detMsg.bbox.center.theta = 0.0f;		// TODO optionally output object image

			// create classification hypothesis
			vision_msgs::ObjectHypothesisWithPose hyp;

			hyp.id = det->ClassID;
			hyp.score = det->Confidence;

			detMsg.results.push_back(hyp);
			if (hyp.id == 1) {  // person
			detMsg.header.stamp = input->header.stamp;
			msg->detections.push_back(detMsg);
			}
		}
	}

	// populate timestamp filed in header
	msg->header.stamp = input->header.stamp;

	// delay
	ros::Duration diff1 = time_received - input->header.stamp;
	ros::Duration diff2 = ros::Time::now() - time_received;
	NODELET_INFO("image delay: %fs; DNN delay: %fs", diff1.toSec(), diff2.toSec());

	// publish the detection message
	detect_pub.publish(msg);
}


// node main loop
void DetectNetNode::onInit()
{
	ros::NodeHandle& nh = this->getNodeHandle();
	ros::NodeHandle& private_nh = this->getPrivateNodeHandle();

	/*
	 * retrieve parameters
	 */
	std::string class_labels_path;
	std::string prototxt_path;
	std::string model_path;
	std::string model_name;

	bool use_model_name = false;

	// determine if custom model paths were specified
	if( !private_nh.getParam("prototxt_path", prototxt_path) ||
	    !private_nh.getParam("model_path", model_path) )
	{
		// without custom model, use one of the built-in pretrained models
		private_nh.param<std::string>("model_name", model_name, "ssd-mobilenet-v2");
		use_model_name = true;
	}

	// set mean pixel and threshold defaults
	float mean_pixel = 0.0f;
	float threshold  = 0.5f;

	private_nh.param<float>("mean_pixel_value", mean_pixel, mean_pixel);
	private_nh.param<float>("threshold", threshold, threshold);

	// input and output blob for DetectNet model
	std::string input_blob = DETECTNET_DEFAULT_INPUT;
	std::string output_cvg = DETECTNET_DEFAULT_COVERAGE;
	std::string output_bbox = DETECTNET_DEFAULT_BBOX;

	private_nh.param<std::string>("input_blob", input_blob, input_blob);
	private_nh.param<std::string>("output_cvg", output_cvg, output_cvg);
	private_nh.param<std::string>("output_bbox", output_bbox, output_bbox);


	/*
	 * load object detection network
	 */
	if( use_model_name )
	{
		// determine which built-in model was requested
		detectNet::NetworkType model = detectNet::NetworkTypeFromStr(model_name.c_str());

		if( model == detectNet::CUSTOM )
		{
			NODELET_ERROR("invalid built-in pretrained model name '%s', defaulting to pednet", model_name.c_str());
			model = detectNet::PEDNET;
		}

		// create network using the built-in model
		net = detectNet::Create(model);
	}
	else
	{
		// get the class labels path (optional)
		private_nh.getParam("class_labels_path", class_labels_path);

		// create network using custom model paths
		net = detectNet::Create(prototxt_path.c_str(), model_path.c_str(), mean_pixel, class_labels_path.c_str(), threshold, input_blob.c_str(), output_cvg.c_str(), output_bbox.c_str());
	}

	if( !net )
	{
		NODELET_ERROR("failed to load detectNet model");
		throw ros::Exception("failed to load detectNet model");
	}


	/*
	 * create the class labels parameter vector
	 */
	std::hash<std::string> model_hasher;  // hash the model path to avoid collisions on the param server
	std::string model_hash_str = std::string(net->GetModelPath()) + std::string(net->GetClassPath());
	const size_t model_hash = model_hasher(model_hash_str);

	NODELET_INFO("model hash => %zu", model_hash);
	NODELET_INFO("hash string => %s", model_hash_str.c_str());

	// obtain the list of class descriptions
	std::vector<std::string> class_descriptions;
	const uint32_t num_classes = net->GetNumClasses();

	for( uint32_t n=0; n < num_classes; n++ )
		class_descriptions.push_back(net->GetClassDesc(n));

	// create the key on the param server
	std::string class_key = std::string("class_labels_") + std::to_string(model_hash);
	private_nh.setParam(class_key, class_descriptions);

	// populate the vision info msg
	std::string node_namespace = private_nh.getNamespace();
	NODELET_INFO("node namespace => %s", node_namespace.c_str());

	info_msg.reset(new vision_msgs::VisionInfo());
	info_msg->database_location = node_namespace + std::string("/") + class_key;
	info_msg->database_version  = 0;
	info_msg->method 		  = net->GetModelPath();

	NODELET_INFO("class labels => %s", info_msg->database_location.c_str());


	/*
	 * create an image converter object
	 */
	cvt = new imageConverter();

	if( !cvt )
	{
		NODELET_ERROR("failed to create imageConverter object");
		throw ros::Exception("failed to create imageConverter object");
	}


	/*
	 * advertise publisher topics
	 */
	this->detect_pub = private_nh.advertise<vision_msgs::Detection2DArray>("detections", 25);

	// message published in vision info topic is fixed
	this->info_pub = private_nh.advertise<vision_msgs::VisionInfo>("vision_info", 1, true);
	info_pub.publish(info_msg);


	/*
	 * subscribe to image topic
	 */
	//image_transport::ImageTransport it(nh);	// BUG - stack smashing on TX2?
	//image_transport::Subscriber img_sub = it.subscribe("image", 1, img_callback);
	this->img_sub = private_nh.subscribe("image_in", 1, &DetectNetNode::img_callback, this);


	/*
	 * wait for messages
	 */
	NODELET_INFO("detectnet node initialized, waiting for messages");
}

}  // namespace


PLUGINLIB_EXPORT_CLASS(ros_deep_learning::DetectNetNode, nodelet::Nodelet);
