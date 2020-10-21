torchserve --stop
torchserve \
--start \
--model-store /home/ubuntu/work/InferenceModels/model_store \
--models all \
--ts-config config.properties