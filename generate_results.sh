# # desc format = final, domain, box or point, tta yes or no, method (naive, kd, kmean), checkpoint
# # domains = ['bg_change', 'blood', 'low_brightness', 'regular', 'smoke'] -> comma separated
# # box prompt = True or False
# # tta = True or False
# # model = 'naive'/ 'kd'/'kmean' +  lora_bg_change / lora_regular + .pth

# # smoke
# # naive
# # python generate_result.py \
# #     --model_path 'checkpoints/naive/lora_bg_change.pth' \
# #     --perform_tta 'True' \
# #     --box_prompt 'True' \
# #     --domains 'smoke' \
# #     --desc 'final-smoke-box-yes-tta-naive-bg-change-checkpoint'

# # python generate_result.py \
# #     --model_path 'checkpoints/naive/lora_bg_change.pth' \
# #     --perform_tta 'False' \
# #     --box_prompt 'True' \
# #     --domains 'smoke' \
# #     --desc 'final-smoke-box-no-tta-naive-bg-change-checkpoint'

# # kmean

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'smoke' \
#     --desc 'final-smoke-box-yes-tta-kmean-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'smoke' \
#     --desc 'final-smoke-box-no-tta-kmean-bg-change-checkpoint'

# # kd

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'smoke' \
#     --desc 'final-smoke-box-yes-tta-kd-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'smoke' \
#     --desc 'final-smoke-box-no-tta-kd-bg-change-checkpoint'


# # low_brightness

# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-box-yes-tta-naive-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-box-no-tta-naive-bg-change-checkpoint'

# # kmean

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-box-yes-tta-kmean-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-box-no-tta-kmean-bg-change-checkpoint'

# # kd

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-box-yes-tta-kd-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-box-no-tta-kd-bg-change-checkpoint'

# # blood

# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'blood' \
#     --desc 'final-blood-box-yes-tta-naive-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'blood' \
#     --desc 'final-blood-box-no-tta-naive-bg-change-checkpoint'

# # kmean

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'blood' \
#     --desc 'final-blood-box-yes-tta-kmean-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'blood' \
#     --desc 'final-blood-box-no-tta-kmean-bg-change-checkpoint'

# # kd

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'blood' \
#     --desc 'final-blood-box-yes-tta-kd-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'blood' \
#     --desc 'final-blood-box-no-tta-kd-bg-change-checkpoint'

# # bg_change
# # naive

# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-box-yes-tta-naive-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-box-no-tta-naive-bg-change-checkpoint'

# # kmean

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-box-yes-tta-kmean-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-box-no-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-box-yes-tta-kd-bg-change-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-box-no-tta-kd-bg-change-checkpoint'

# # regular
# # naive

# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_regular.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'regular' \
#     --desc 'final-regular-box-yes-tta-naive-regular-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_regular.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'regular' \
#     --desc 'final-regular-box-no-tta-naive-regular-checkpoint'

# # kmean

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_regular.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'regular' \
#     --desc 'final-regular-box-yes-tta-kmean-regular-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_regular.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'regular' \
#     --desc 'final-regular-box-no-tta-kmean-regular-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_regular.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'True' \
#     --domains 'regular' \
#     --desc 'final-regular-box-yes-tta-kd-regular-checkpoint'

# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_regular.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'True' \
#     --domains 'regular' \
#     --desc 'final-regular-box-no-tta-kd-regular-checkpoint'

# # ----------------------------------------------------------------------------------

# # for each domain, for each method, we need to run box as False and tta as False

# # smoke
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'smoke' \
#     --desc 'final-smoke-point-no-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'smoke' \
#     --desc 'final-smoke-point-no-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'smoke' \
#     --desc 'final-smoke-point-no-tta-kd-bg-change-checkpoint'

# # low_brightness
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-point-no-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-point-no-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-point-no-tta-kd-bg-change-checkpoint'

# # blood
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'blood' \
#     --desc 'final-blood-point-no-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'blood' \
#     --desc 'final-blood-point-no-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'blood' \
#     --desc 'final-blood-point-no-tta-kd-bg-change-checkpoint'

# # bg_change
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-point-no-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-point-no-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-point-no-tta-kd-bg-change-checkpoint'

# # regular
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_regular.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'regular' \
#     --desc 'final-regular-point-no-tta-naive-regular-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_regular.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'regular' \
#     --desc 'final-regular-point-no-tta-kmean-regular-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_regular.pth' \
#     --perform_tta 'False' \
#     --box_prompt 'False' \
#     --domains 'regular' \
#     --desc 'final-regular-point-no-tta-kd-regular-checkpoint'


# # ----------------------------------------------------------------------------------

# # for each domain, for each method, we need to run box as False and tta as True
# # smoke
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'smoke' \
#     --desc 'final-smoke-point-yes-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'smoke' \
#     --desc 'final-smoke-point-yes-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'smoke' \
#     --desc 'final-smoke-point-yes-tta-kd-bg-change-checkpoint'

# # low_brightness
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-point-yes-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-point-yes-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'low_brightness' \
#     --desc 'final-low-brightness-point-yes-tta-kd-bg-change-checkpoint'

# # blood
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'blood' \
#     --desc 'final-blood-point-yes-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'blood' \
#     --desc 'final-blood-point-yes-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'blood' \
#     --desc 'final-blood-point-yes-tta-kd-bg-change-checkpoint'

# # bg_change
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-point-yes-tta-naive-bg-change-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-point-yes-tta-kmean-bg-change-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_bg_change.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'bg_change' \
#     --desc 'final-bg-change-point-yes-tta-kd-bg-change-checkpoint'

# # regular
# # naive
# python generate_result.py \
#     --model_path 'checkpoints/naive/lora_regular.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'regular' \
#     --desc 'final-regular-point-yes-tta-naive-regular-checkpoint'

# # kmean
# python generate_result.py \
#     --model_path 'checkpoints/kmean/lora_regular.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'regular' \
#     --desc 'final-regular-point-yes-tta-kmean-regular-checkpoint'

# # kd
# python generate_result.py \
#     --model_path 'checkpoints/kd/lora_regular.pth' \
#     --perform_tta 'True' \
#     --box_prompt 'False' \
#     --domains 'regular' \
#     --desc 'final-regular-point-yes-tta-kd-regular-checkpoint'

#!/usr/bin/env bash
set -euo pipefail

# Domains and methods
domains=(smoke low_brightness blood bg_change regular)
methods=(naive kd kmean)

# Generate the descriptive suffix for each run
gen_desc() {
  local domain=$1 method=$2 box=$3 tta=$4
  # map boolean flags to labels
  local prompt_label tta_label ckpt_id
  if [[ $box  == "true" ]]; then  prompt_label="box";  else  prompt_label="point";  fi
  if [[ $tta  == "true" ]]; then  tta_label="yes-tta"; else  tta_label="no-tta";  fi
  # checkpoint identifier depends on domain
  if [[ $domain == "regular" ]]; then ckpt_id="regular"; else ckpt_id="bg-change"; fi
  echo "final-${domain//_/-}-${prompt_label}-${tta_label}-${method}-${ckpt_id}-checkpoint"
}

# Generate the model_path for each method/domain
gen_model_path() {
  local method=$1 domain=$2
  if [[ $domain == "regular" ]]; then
    echo "checkpoints/${method}/lora_regular.pth"
  else
    echo "checkpoints/${method}/lora_bg_change.pth"
  fi
}

# Emit one command
run_cmd() {
  local domain=$1 method=$2 box=$3 tta=$4
  local desc model
  desc=$(gen_desc       "$domain" "$method" "$box" "$tta")
  model=$(gen_model_path "$method" "$domain")

  python generate_result.py \
    --model_path "$model" \
    --perform_tta "$tta" \
    --box_prompt "$box" \
    --domains "$domain" \
    --desc "$desc"
}

echo_cmd() {
  local domain=$1 method=$2 box=$3 tta=$4
  local desc model
  desc=$(gen_desc       "$domain" "$method" "$box" "$tta")
  model=$(gen_model_path "$method" "$domain")
  echo "python generate_result.py --model_path '$model' --perform_tta '$tta' --box_prompt '$box' --domains '$domain' --desc '$desc'"
}


# # 1) First set: box_prompt=true, tta=yes/no
# for domain in "${domains[@]}"; do
#   for method in "${methods[@]}"; do

#     if [[ $domain == "smoke" ]]; then
#       continue
#     fi

#     for tta in true false; do
#       echo_cmd "$domain" "$method" true  "$tta"
#       run_cmd "$domain" "$method" true  "$tta"
#     done
#   done
# done

echo "----------------------------------------"

# 2) Second set: box_prompt=false, tta=false
for domain in "${domains[@]}"; do
  for method in "${methods[@]}"; do
    echo_cmd "$domain" "$method" false false
    run_cmd "$domain" "$method" false false
  done
done

echo "----------------------------------------"

# 3) Third set: box_prompt=false, tta=true
for domain in "${domains[@]}"; do
  for method in "${methods[@]}"; do
    echo_cmd "$domain" "$method" false true
    run_cmd "$domain" "$method" false true
  done
done