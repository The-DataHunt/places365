torch-model-archiver \
--model-name places365 \
--version 1.0 \
--serialized-file ./wideresnet18_places365.pth.tar \
--extra-files ../wideresnet.py,./IO_places365.txt,./W_sceneattribute_wideresnet18.npy,./categories_places365.txt,./labels_sunattribute.txt \
--handler PlacesHandler:handle \
--export-path /home/ubuntu/work/InferenceModels/model_store \
-f