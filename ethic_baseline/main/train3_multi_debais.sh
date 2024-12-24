### Binary_Kor
batch_size=32
path="/nlp_data/yumin/data/total/multi/"
for model in "klue/bert-base"; do
    for mode in "problems"; do # "topic"
        for alpha in "0.1" "0.3" "0.5" "0.7" "0.9"; do
            for lambda in "1" "3" "5" "10" "20" "30"; do
                for temp in "0.5" "1" "2"; do
                    for hyper_param in "3e-5,0.1,0.01"; do
                        CUDA_VISIBLE_DEVICES=3 python3 train_multi_debiasing.py --model "${model}" \
                        --hyper_param ${hyper_param} --batch_size ${batch_size} --num_epochs 10 --train_data "${path}${mode}/train_merge.json" \
                        --valid_data "${path}${mode}/test_merge.json"  --test_data "${path}${mode}/test_merge.json" --mode ${mode} \
                        --loss_alpha ${alpha} --loss_lambda ${lambda} --loss_temp ${temp} --find_hyper
                    done
                done
            done
        done
    done
done