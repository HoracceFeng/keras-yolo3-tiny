sudo nvidia-docker run -it --rm \
	-v /data/01366808/hnfeng/hnfeng_205/docker_learn/keras-yolo3:/code \
	-v /data/01366808/hnfeng/DATA_HUB/traffic_cognition/Dataset_TC-version1.x:/data \
	10.202.107.19/sfai/ubuntu16_cuda8_cudnn6_py2_sklearn_cv3.4:keras2.2.4 bash
