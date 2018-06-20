:: batch file for windows

echo "begining..."

echo "===="

python main.py -c config\conf_dist1-resnet50-with_bbox-da-class_weigths-imagenet-francho.py -a train > logs\conf_dist1-resnet50-with_bbox-da-class_weigths-imagenet-francho-train 2>&1


#python main.py -c config\conf_dist1-resnet50-with_bbox-crop-da-francho.py -a train > logs\dist1-resnet50-with_bbox-crop-da-francho-train 2>&1


echo "finished!"
