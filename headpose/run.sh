# VARIABLES
example=39
encode_dim=24
decode_dim=24
dataset_dir="obama_processed"
chkpt_dir="chkpt_obama"

steps=()

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

model_name="new-headpose${encode_dim}.hdf5"
# model_name="headpose6-ep100.hdf5"
example_name=ex${example}--dim${encode_dim}
dir="headpose/results/$(date +%s)"

echo "example name:" ${example_name}
echo "encoding dim:"  ${encode_dim}
echo "processed dataset dir:"  ${dataset_dir}
echo "DAE checkpoint dir:" ${chkpt_ir}
echo "results dir:"  ${dir}
echo "steps"  ${steps}

mkdir -p ${chkpt_dir}
mkdir -p ${dir}

# PIPELINE

# 0. pre prepare
if [[ " ${steps[@]} " =~ "zero" ]]; then
  echo "Step 0: Setting up data folder"
  python data_processing/prepare_obama.py ${dataset_dir}/
fi

# 1. prepare data - data dir, N_CONTEXT
if [[ " ${steps[@]} " =~ "one" ]]; then
  echo "Step 1: Preparing data for training"
  python data_processing/create_vector_headpose.py ${dataset_dir} 60
fi

# 2. train DAE
if [[ " ${steps[@]} " =~ "two" ]]; then
  echo "Step 2: Training the Autoencoder"
  python motion_repr_learning/ae/learn_dataset_encoding.py \
    ${dataset_dir} \
    -chkpt_dir=${chkpt_dir} \
    -layer1_width=${encode_dim} \
    -frame_size=${decode_dim} \
    -batch_size=128 \
    -training_epochs=80 \
    -early_stopping=false
fi

# 3. encode dataset
if [[ " ${steps[@]} " =~ "three" ]]; then
  echo "Step 3: Encoding dataset"
  python motion_repr_learning/ae/encode_dataset.py \
    ${dataset_dir} \
    -chkpt_dir=${chkpt_dir} \
    -restore=True \
    -frame_size=${decode_dim} \
    -pretrain=False \
    -layer1_width=${encode_dim}
fi

# 4. train model - model name, epochs, data_dir, speech features, encode, encode_dim
if [[ " ${steps[@]} " =~ "four" ]]; then
  echo "Step 4: Train model"
  python train.py ${model_name} 20 ${dataset_dir} 26 True ${encode_dim}
# python train.py headpose.hdf5 10 ${dataset_dir} 26 False ${encode_dim}
fi

# 5. encode test data
if [[ " ${steps[@]} " =~ "five" ]]; then
  echo "Step 5: predict encoding with model"
  # python predict.py ${model_name} ${dataset_dir}/test_inputs/X_test_audio${example}.npy ${dir}/${example_name}_ENC.txt
  python predict.py ${model_name} new_data/audio32.npy ${dir}/model32_ENC.txt
fi

# 6. decode model results
if [[ " ${steps[@]} " =~ "six" ]]; then
  echo "Step 6: Decode prediction"
  python motion_repr_learning/ae/decode.py ${dataset_dir} \
    ${dir}/model32_ENC.txt \
    ${dir}/model32_DEC.txt \
    -restore=True -pretrain=False -layer1_width=${encode_dim} -chkpt_dir=${chkpt_dir} -batch_size=8 -frame_size=${decode_dim}
fi

# 7. animate
if [[ " ${steps[@]} " =~ "seven" ]]; then
  echo "Step 7: Animate result"
  # python headpose/animate_headpose.py --input ${dir}/${example_name}_DEC.txt --output ${dir}/${example_name}_ANIM.mp4

  # python headpose/animate_headpose.py --input new_data/audio32_DEC.txt --output new_data/audio32_ANIM.mp4
  # python headpose/renderhead.py --input new_data/Y_32_dim18_DEC.txt --output new_data/dae_test_DEC.mp4
  python headpose/renderhead.py --input ${dir}/model32_DEC.txt --output ${dir}/model32_ANIM.mp4

  # python headpose/animate_headpose.py --input ${dir}/${example_name}_ENC.txt --output ${dir}/${example_name}_ANIM.mp4
fi

# 8. stitch video and predicted headpose
if [[ " ${steps[@]} " =~ "eight" ]]; then
  echo "Step 8: Stitching video and predicted headpose"
  ffmpeg \
    -i obama/annotated/annotated32.mp4 \
    -i ${dir}/model32_ANIM.mp4 \
    -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
    -map [vid] \
    -c:v libx264 \
    -crf 23 \
    -preset veryfast \
    ${dir}/model32_VID.mp4
fi

# 9. combine animation and audio
if [[ " ${steps[@]} " =~ "nine" ]]; then
  echo "Step 9: Combining animation and audio"
  ffmpeg -i ${dir}/model32_VID.mp4 -i ${dataset_dir}/train/inputs/audio32.wav \
    -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k ${dir}/model32_ALL.mp4
fi

# 10. test autoencoder
if [[ " ${steps[@]} " =~ "ten" ]]; then
  echo "Step 10"
  # python motion_repr_learning/ae/decode.py ${dataset_dir} \
  #   ${dataset_dir}/${encode_dim}/Y_39_encoded.txt \
  #   ${dir}/dae_test_DEC.txt \
  #   -restore=True -pretrain=False -layer1_width=${encode_dim} -chkpt_dir=${chkpt_dir} -batch_size=8 -frame_size=${decode_dim}

  python motion_repr_learning/ae/decode.py ${dataset_dir} \
    new_data/Y_32_encoded.txt \
    new_data/Y_32_dim18_DEC.txt \
    -restore=True -pretrain=False -layer1_width=${encode_dim} -chkpt_dir=${chkpt_dir} -batch_size=8 -frame_size=${decode_dim}

  # python headpose/animate_headpose.py --input ${dir}/dae_test_DEC.txt --output ${dir}/dae_test_DEC.mp4
  # python headpose/animate_headpose.py --input obama_processed/test/labels/pose39.csv --openface --output ${dir}/dae_test_ORIG.mp4

  python headpose/renderhead.py --input new_data/Y_32_dim18_DEC.txt --output new_data/dae_test_DEC.mp4
  # python headpose/renderhead.py --input obama_processed/train/labels/pose32.csv --openface --output new_data/dae_test_ORIG.mp4

  ffmpeg \
    -y \
    -i new_data/dae_test_ORIG.mp4 \
    -i new_data/dae_test_DEC.mp4 \
    -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
    -map [vid] \
    -c:v libx264 \
    -crf 23 \
    -preset veryfast \
    new_data/dae_test_COMPARE.mp4
fi

NEWLINE=$'\n'
echo "${model_name} ${NEWLINE}encoding dims: ${encode_dim} ${NEWLINE} example: ${example}" > ${dir}/info.txt
