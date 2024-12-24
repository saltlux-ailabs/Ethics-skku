### Binary_Kor
batch_size=32
path="/nlp_data/yumin/data/total/multi/"
export CUDA_LAUNCH_BLOCKING=1

for model in "klue/bert-base"; do
    for mode in "problems"; do
        for hyper_param in "3e-5,0.1,0.01"; do
            for c_debias in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do  
                for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do 
                    echo "*&*&*&*&*&*&*&*&%%%${model}%%%${mode}%%%${hyper_param}%%%${c_debias}%%%${alpha}%%%"
                    
                    # Define a unique save directory for each combination
                    save_dir="/nlp_data/yumin/ethic_baseline/models/${model}_${mode}_${hyper_param}_${c_debias}_${alpha}"
                    
                    # Run training with unique save_dir
                    CUDA_VISIBLE_DEVICES=3 python3 train_multi_debiasing_copy.py \
                    --model "${model}" \
                    --hyper_param ${hyper_param} \
                    --batch_size ${batch_size} \
                    --num_epochs 5 \
                    --train_data "${path}${mode}/train_merge.json" \
                    --valid_data "${path}${mode}/test_merge.json"  \
                    --test_data "${path}${mode}/test_merge.json" \
                    --mode ${mode} \
                    --c_debias ${c_debias} \
                    --alpha ${alpha} \
                    --save_dir "${save_dir}"
                    
                done
            done
        done
    done
done
