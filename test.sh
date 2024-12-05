LOGNAME='test'
CONFIG='./configs/r34.json'
DATA_DIR='/home/lyf/下载/OneDrive'
DATASET_NAME='facial_components'
TEST_TYPE='adaptive'
RESULTS_DIR='./results'

HOST='127.0.0.1'
PORT='12345'
NUM_GPU=1

python test.py \
--log_name ${LOGNAME} \
--cfg ${CONFIG} \
--data_dir ${DATA_DIR} \
--dataset_name ${DATASET_NAME} \
--test_type ${TEST_TYPE} \
--dist-url tcp://${HOST}:${PORT} \
--world_size ${NUM_GPU} \
--rank 0 \
--launcher none \
--results_dir ${RESULTS_DIR}

# dataset_name: facial_components facial_attributes
# test_type: fixed adaptive

